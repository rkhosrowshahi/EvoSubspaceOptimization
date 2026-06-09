"""Dual-EA alternating optimization with ask-eval-tell for both EAs.

Same m+k cycle as ``scripts/main_dual_ea.py``, but both optimizers use an
explicit **ask-eval-tell** loop:

- **Full-space** — PyMOO ``ask()`` / ``tell()`` (e.g. DE).
- **Subspace** — evosax ``ask()`` / ``tell()`` (e.g. CMA-ES).

See README.md and ``scripts/main_dual_ea.py`` for the all-PyMOO entry point.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from scripts.main import (
    build_parser as _build_base_parser,
    effective_subspace_param,
    optimizer_search_dim,
    subspace_method_is_block_lora,
    subspace_method_is_fullspace,
    subspace_method_is_lora,
    validate_lora_blocks,
)
from scripts.main_dual_ea import (
    DualEALoggingCallback,
    _best_fullspace_solution,
    _best_subspace_solution,
    _budget_left,
    _evaluate_batch,
    _inject_into_fullspace,
    _refresh_subspace_after_anchor,
    _refresh_subspace_cost,
    _sub_pop_size,
    _total_nfe,
    _track_best,
)
from scripts.main_two_phase import CenteredSampling
from subspace import build_subspace
from problems import LSGOProblem
from optimizers import build_algorithm
from optimizers.evosax_builder import (
    EvosaxSubspaceOptimizer,
    build_evosax_optimizer,
    evosax_optimizer_choices,
)
from utils import SubspaceProblem


# ---------------------------------------------------------------------------
# Ask-eval-tell generation steps
# ---------------------------------------------------------------------------

def _pymoo_ask_tell_generation(optim, problem: SubspaceProblem) -> bool:
    """One PyMOO generation: ask candidates, evaluate, tell."""
    infills = optim.ask()
    if infills is None:
        return False
    optim.evaluator.eval(problem, infills, algorithm=optim)
    optim.tell(infills)
    return True


def _evosax_ask_tell_generation(sub_optim: EvosaxSubspaceOptimizer, problem: SubspaceProblem) -> bool:
    """One evosax generation: ask candidates, evaluate, tell."""
    population = sub_optim.ask()
    fitness = _evaluate_batch(problem, population).reshape(-1)
    sub_optim.evaluator.n_eval += len(fitness)
    sub_optim.tell(fitness)
    return True


def _advance_pymoo_generations(
    optim,
    problem: SubspaceProblem,
    *,
    n_gens: int,
    pop_size: int,
    full_optim,
    sub_optim,
    max_nfe: int,
) -> int:
    """Run up to ``n_gens`` PyMOO ask-eval-tell steps."""
    completed = 0
    for _ in range(n_gens):
        if _budget_left(full_optim, sub_optim, max_nfe) < pop_size:
            break
        if not optim.has_next():
            break
        if not _pymoo_ask_tell_generation(optim, problem):
            break
        completed += 1
    return completed


def _advance_evosax_generations(
    sub_optim: EvosaxSubspaceOptimizer,
    problem: SubspaceProblem,
    *,
    n_gens: int,
    pop_size: int,
    full_optim,
    max_nfe: int,
) -> int:
    """Run up to ``n_gens`` evosax ask-eval-tell steps."""
    completed = 0
    for _ in range(n_gens):
        if _budget_left(full_optim, sub_optim, max_nfe) < pop_size:
            break
        if not sub_optim.has_next():
            break
        if not _evosax_ask_tell_generation(sub_optim, problem):
            break
        completed += 1
    return completed


# ---------------------------------------------------------------------------
# Alternating optimization loop (ask-eval-tell)
# ---------------------------------------------------------------------------

def run_dual_ea_ask_tell(
    *,
    full_optim,
    sub_optim: EvosaxSubspaceOptimizer,
    full_problem: SubspaceProblem,
    sub_problem: SubspaceProblem,
    full_subspace,
    sub_subspace,
    args: argparse.Namespace,
    max_nfe: int,
    callback: DualEALoggingCallback | None = None,
) -> tuple[np.ndarray, float, int, int, int, int]:
    """Run alternating cycles with explicit ask-eval-tell for both EAs."""
    global_best_f = float("inf")
    global_best_x: np.ndarray | None = None
    n_cycles = 0
    full_improve_count = 0
    sub_improve_count = 0

    while full_optim.has_next() and _budget_left(full_optim, sub_optim, max_nfe) > 0:
        n_cycles += 1
        global_best_at_cycle_start = global_best_f

        if _budget_left(full_optim, sub_optim, max_nfe) < args.pop_size:
            break
        full_ran = _advance_pymoo_generations(
            full_optim,
            full_problem,
            n_gens=args.full_iters,
            pop_size=args.pop_size,
            full_optim=full_optim,
            sub_optim=sub_optim,
            max_nfe=max_nfe,
        )
        if full_ran == 0:
            break

        best_x_full, f_full = _best_fullspace_solution(full_optim, full_subspace)
        global_best_x, global_best_f = _track_best(
            best_x_full, f_full, global_best_x, global_best_f
        )
        full_improved_cycle = global_best_f < global_best_at_cycle_start
        if full_improved_cycle:
            full_improve_count += 1
        global_best_after_full = global_best_f

        sub_subspace.set_x0(best_x_full)
        refresh_cost = _refresh_subspace_cost(sub_optim, args, sub_subspace)
        if _budget_left(full_optim, sub_optim, max_nfe) < refresh_cost:
            break
        _refresh_subspace_after_anchor(
            sub_optim,
            sub_problem,
            sub_subspace,
            args,
            mode=args.sub_anchor_update,
            best_x=best_x_full,
        )

        sub_pop = _sub_pop_size(args)
        if _budget_left(full_optim, sub_optim, max_nfe) < sub_pop:
            break
        sub_ran = _advance_evosax_generations(
            sub_optim,
            sub_problem,
            n_gens=args.sub_iters,
            pop_size=sub_pop,
            full_optim=full_optim,
            max_nfe=max_nfe,
        )
        if sub_ran == 0:
            break

        global_best_before_sub = global_best_after_full
        best_x_sub, f_sub, _ = _best_subspace_solution(sub_optim, sub_subspace)
        global_best_x, global_best_f = _track_best(
            best_x_sub, f_sub, global_best_x, global_best_f
        )

        if _budget_left(full_optim, sub_optim, max_nfe) >= 1:
            f_inj, injected = _inject_into_fullspace(
                full_optim, full_problem, best_x_sub, f_sub
            )
            if injected and f_inj is not None:
                global_best_x, global_best_f = _track_best(
                    best_x_sub, f_inj, global_best_x, global_best_f
                )

        sub_improved_cycle = global_best_f < global_best_before_sub
        if sub_improved_cycle:
            sub_improve_count += 1

        if callback is not None:
            callback.notify(
                full_optim,
                full_optim=full_optim,
                sub_optim=sub_optim,
                full_improve_count=full_improve_count,
                sub_improve_count=sub_improve_count,
            )

    if global_best_x is None:
        global_best_x, global_best_f = _best_fullspace_solution(
            full_optim, full_subspace
        )

    return (
        global_best_x,
        global_best_f,
        n_cycles,
        _total_nfe(full_optim, sub_optim),
        full_improve_count,
        sub_improve_count,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = _build_base_parser()
    parser.description = (
        "Dual-EA alternating optimization with ask-eval-tell: PyMOO full-space "
        "and evosax subspace EAs exchange best solutions each cycle (m+k generations)."
    )

    dual = parser.add_argument_group("Dual-EA")
    dual.add_argument(
        "--full_iters",
        type=int,
        default=1,
        help=(
            "Full-space EA generations per cycle (m in an m+k cycle). "
            "Must be >= 1."
        ),
    )
    dual.add_argument(
        "--sub_iters",
        type=int,
        default=1,
        help=(
            "Subspace EA generations per cycle (k in an m+k cycle). "
            "Must be >= 1."
        ),
    )
    dual.add_argument(
        "--sub_anchor_update",
        type=str,
        default="reeval",
        choices=["resample", "reeval"],
        help=(
            "After updating the subspace anchor x0 from the full-space best: "
            "'resample' re-initializes the subspace population near z=0; "
            "'reeval' keeps current z and re-evaluates fitness under the new x0."
        ),
    )
    dual.add_argument(
        "--fullspace_assignment",
        type=str,
        default="absolute",
        choices=["absolute", "additive"],
        help="Assignment mode for the full-space EA (typically absolute).",
    )

    evosax_group = parser.add_argument_group("Subspace evosax optimizer")
    evosax_group.add_argument(
        "--sub_optimizer",
        type=str,
        default="cma_es",
        choices=evosax_optimizer_choices(),
        help="evosax algorithm for the subspace EA (see evosax README).",
    )
    evosax_group.add_argument(
        "--sub_pop_size",
        type=int,
        default=None,
        help=(
            "Population size for the subspace evosax optimizer. "
            "Defaults to --pop_size."
        ),
    )

    return parser


def init_wandb_dual_ea_evosax(args: argparse.Namespace) -> None:
    import wandb  # type: ignore

    eff_d = effective_subspace_param(args)
    sub_pop = args.sub_pop_size if args.sub_pop_size is not None else args.pop_size
    if args.wandb_name in (None, "", "__auto__"):
        args.wandb_name = (
            f"{args.problem}-dim{args.dim}-dual_ea_evosax-{args.subspace_method}"
        )
        if subspace_method_is_lora(args.subspace_method):
            args.wandb_name += f"-lora_rank{args.lora_rank}"
            if subspace_method_is_block_lora(args.subspace_method):
                args.wandb_name += f"-blocks{args.lora_blocks}"
        else:
            args.wandb_name += f"-subdim{eff_d}"
        args.wandb_name += (
            f"-{args.subspace_assignment}-{args.optimizer}+{args.sub_optimizer}"
            f"-seed{args.seed}"
        )

    if args.wandb_group:
        placeholder_pattern = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")

        placeholder_values = vars(args).copy()
        placeholder_values["subspace_dim"] = eff_d
        placeholder_values["sub_optimizer"] = args.sub_optimizer
        if not subspace_method_is_lora(args.subspace_method):
            placeholder_values["lora_rank"] = ""

        def _replace_placeholder(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in placeholder_values:
                return match.group(0)
            value = placeholder_values[key]
            return "" if value is None else str(value)

        s = placeholder_pattern.sub(_replace_placeholder, args.wandb_group)
        if not subspace_method_is_lora(args.subspace_method):
            while "--" in s:
                s = s.replace("--", "-")
        args.wandb_group = s

    config = {k: v for k, v in vars(args).items() if k != "wandb"}
    config["approach"] = "dual_ea_evosax"
    config["fullspace_assignment"] = args.fullspace_assignment
    config["search_dim_full"] = args.dim
    config["search_dim_sub"] = optimizer_search_dim(args)
    config["sub_pop_size"] = sub_pop
    if subspace_method_is_lora(args.subspace_method):
        config["subspace_dim"] = None
    wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        group=args.wandb_group,
        name=args.wandb_name,
        config=config,
    )


def _setup_sub_evosax(
    sub_optim: EvosaxSubspaceOptimizer,
    sub_problem: SubspaceProblem,
    sub_subspace,
    args: argparse.Namespace,
) -> None:
    """Initialize evosax subspace optimizer (distribution or population-based)."""
    n_var = sub_subspace.search_dim
    pop_size = args.sub_pop_size if args.sub_pop_size is not None else args.pop_size
    sub_optim.setup(sub_problem, seed=args.seed + 1, verbose=False)

    if sub_optim.is_distribution_based:
        sub_optim.initialize_distribution(np.zeros(n_var, dtype=float))
        return

    sampling = CenteredSampling(
        center=np.zeros(n_var, dtype=float),
        method=args.init_pop,
        scale=args.pop_sigma,
    )
    X = sampling._do(sub_problem, pop_size)
    F = _evaluate_batch(sub_problem, X)
    sub_optim.initialize_population(X, F)
    sub_optim.evaluator.n_eval += pop_size


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.subspace_method == "none":
        args.subspace_method = "fullspace"

    if subspace_method_is_fullspace(args.subspace_method):
        parser.error(
            "Dual-EA requires a reduced subspace for the subspace EA; "
            "choose random_projection, random_blocking, or lora."
        )

    if subspace_method_is_lora(args.subspace_method) and args.lora_rank is None:
        parser.error(
            "--lora_rank is required when the subspace method uses LoRA"
        )

    if subspace_method_is_block_lora(args.subspace_method):
        try:
            validate_lora_blocks(args.lora_blocks, args.dim)
        except ValueError as exc:
            parser.error(str(exc))

    if args.subspace_dim is None and not subspace_method_is_lora(args.subspace_method):
        parser.error(
            "--subspace_dim is required for random_projection and random_blocking"
        )

    if args.subspace_assignment != "additive":
        print(
            "Warning: dual-EA is designed for additive subspace assignment "
            f"(x = x0 + f(z)); got {args.subspace_assignment!r}."
        )

    if args.full_iters < 1 or args.sub_iters < 1:
        parser.error("--full_iters and --sub_iters must be >= 1")

    try:
        build_evosax_optimizer(
            args,
            search_dim=optimizer_search_dim(args),
            gen_cap=1,
        )
    except ValueError as exc:
        parser.error(str(exc))

    np.random.seed(args.seed)

    if args.wandb:
        init_wandb_dual_ea_evosax(args)

    eff_sub = effective_subspace_param(args)
    sub_pop = args.sub_pop_size if args.sub_pop_size is not None else args.pop_size
    print("=" * 70)
    print("Dual-EA ask-eval-tell (PyMOO full / evosax sub)")
    print("=" * 70)
    print(f"  Problem           : {args.problem} (dim={args.dim})")
    print(
        f"  Full-space EA     : fullspace "
        f"({args.fullspace_assignment}, search_dim={args.dim})"
    )
    if subspace_method_is_lora(args.subspace_method):
        sub_stat = f"rank r={eff_sub}"
        if subspace_method_is_block_lora(args.subspace_method):
            sub_stat += f", blocks={args.lora_blocks}"
    else:
        sub_stat = f"d={eff_sub}"
    print(
        f"  Subspace EA       : {args.subspace_method} "
        f"({sub_stat}, {args.subspace_assignment}, "
        f"search_dim={optimizer_search_dim(args)})"
    )
    print(f"  Full optimizer    : {args.optimizer} (pop={args.pop_size})")
    print(f"  Sub optimizer     : {args.sub_optimizer} (pop={sub_pop})")
    print(f"  Cycle schedule    : {args.full_iters}+{args.sub_iters} (full+sub gens)")
    print(f"  Max NFE (shared)  : {args.max_nfe}")
    print(f"  Sub anchor update : {args.sub_anchor_update}")
    print(f"  Optimizer seed    : {args.seed}")
    print(f"  Benchmark seed    : {args.benchmark_seed}")
    print("=" * 70)

    lsgo = LSGOProblem(
        func_id=args.problem,
        D=args.dim,
        seed=args.benchmark_seed,
    )
    print(f"  Optimum known     : {lsgo.optimum}")
    print("=" * 70)

    full_subspace = build_subspace(
        method="fullspace",
        D=args.dim,
        d=args.dim,
        subspace_assignment=args.fullspace_assignment,
        seed=args.seed,
        lb=lsgo.lb,
        ub=lsgo.ub,
    )
    full_problem = SubspaceProblem(lsgo=lsgo, subspace=full_subspace)

    sub_subspace = build_subspace(
        method=args.subspace_method,
        D=args.dim,
        d=eff_sub,
        subspace_assignment=args.subspace_assignment,
        seed=args.seed,
        lb=lsgo.lb,
        ub=lsgo.ub,
        device=args.subspace_device,
        lora_blocks=args.lora_blocks,
    )
    sub_problem = SubspaceProblem(lsgo=lsgo, subspace=sub_subspace)

    full_optim = build_algorithm(args)

    gen_cap = max(1, args.max_nfe // max(1, args.pop_size) + 2)
    sub_optim = build_evosax_optimizer(
        args,
        search_dim=sub_subspace.search_dim,
        gen_cap=gen_cap,
    )

    from pymoo.termination import get_termination

    full_optim.setup(
        full_problem,
        termination=get_termination("n_gen", gen_cap),
        seed=args.seed,
        verbose=False,
    )
    _setup_sub_evosax(sub_optim, sub_problem, sub_subspace, args)

    callback = DualEALoggingCallback(
        eval_fn=lsgo.evaluate,
        full_subspace=full_subspace,
        sub_subspace=sub_subspace,
        use_wandb=args.wandb,
        log_every=args.log_every,
    )

    t0 = time.perf_counter()
    best_x, best_f, n_cycles, total_nfe, full_improve_count, sub_improve_count = run_dual_ea_ask_tell(
        full_optim=full_optim,
        sub_optim=sub_optim,
        full_problem=full_problem,
        sub_problem=sub_problem,
        full_subspace=full_subspace,
        sub_subspace=sub_subspace,
        args=args,
        max_nfe=args.max_nfe,
        callback=callback,
    )
    elapsed = time.perf_counter() - t0

    print("=" * 70)
    print(f"Dual-EA ask-eval-tell finished in {elapsed:.2f}s")
    print(f"  Cycles completed  : {n_cycles}")
    print(f"  Full improved best: {full_improve_count} cycles")
    print(f"  Sub improved best : {sub_improve_count} cycles")
    print(f"  Best fitness      : {best_f:.6e}")
    print(f"  Total NFE         : {total_nfe}")
    print(f"  ||best_x||_2      : {float(np.linalg.norm(best_x)):.4f}")
    if lsgo.optimum is not None:
        gap = best_f - lsgo.optimum
        print(f"  Gap to optimum    : {gap:.6e}")
    print("=" * 70)

    if args.wandb:
        import wandb  # type: ignore

        wandb.summary["best_fitness"] = best_f
        wandb.summary["total_nfe"] = total_nfe
        wandb.summary["n_cycles"] = n_cycles
        wandb.summary["full_improve_count"] = full_improve_count
        wandb.summary["sub_improve_count"] = sub_improve_count
        wandb.summary["elapsed_seconds"] = elapsed
        wandb.finish()


if __name__ == "__main__":
    main()
