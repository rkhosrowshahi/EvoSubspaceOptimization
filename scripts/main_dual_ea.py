"""Dual-EA alternating optimization: full-space and subspace EAs in lockstep.

Two evolutionary algorithms run in parallel:

- **Full-space EA** — searches ``R^D`` with absolute assignment (``x = z``).
- **Subspace EA** — searches a reduced subspace (e.g. LoRA) with additive
  assignment (``x = x0 + f(z)``).

Each **cycle** runs ``m`` full-space generations, then ``k`` subspace generations
(``--full_iters m``, ``--sub_iters k``; default ``1+1``):

1. Advance the full-space EA by ``m`` generations; take its best full-space solution.
2. Set the subspace anchor ``x0`` to that solution, refresh the subspace population
   (re-sample around ``z = 0`` or re-evaluate under the new anchor).
3. Advance the subspace EA by ``k`` generations; inject the subspace best into the
   full-space population (replace worst).

Total NFE is ``full_evaluator.n_eval + sub_evaluator.n_eval`` and is capped by
``--max_nfe``.

See README.md for single-phase (``scripts/main.py``) and sequential two-phase
(``scripts/main_two_phase.py``) entry points.
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
from pymoo.core.individual import Individual
from pymoo.core.population import Population

from scripts.main import (
    build_parser as _build_base_parser,
    effective_subspace_param,
    optimizer_search_dim,
    subspace_method_is_fullspace,
    subspace_method_is_lora,
)
from scripts.main_two_phase import CenteredSampling, _set_algorithm_sampling
from subspace import build_subspace
from subspace.base import Subspace
from problems import LSGOProblem
from optimizers import build_algorithm
from utils import LoggingCallback, SubspaceProblem


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = _build_base_parser()
    parser.description = (
        "Dual-EA alternating optimization: full-space and subspace EAs "
        "exchange best solutions each cycle (m+k generations per cycle)."
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

    return parser


# ---------------------------------------------------------------------------
# Population / anchor helpers
# ---------------------------------------------------------------------------

def _total_nfe(full_algo, sub_algo) -> int:
    return int(full_algo.evaluator.n_eval + sub_algo.evaluator.n_eval)


def _budget_left(full_algo, sub_algo, max_nfe: int) -> int:
    return max_nfe - _total_nfe(full_algo, sub_algo)


def _best_fullspace_solution(algo, subspace) -> tuple[np.ndarray, float]:
    F = algo.pop.get("F").flatten()
    X = algo.pop.get("X")
    best_idx = int(np.argmin(F))
    x = np.asarray(subspace.expand(X[best_idx]), dtype=float).reshape(-1)
    return x, float(F[best_idx])


def _best_subspace_solution(algo, subspace) -> tuple[np.ndarray, float, np.ndarray]:
    F = algo.pop.get("F").flatten()
    X = algo.pop.get("X")
    best_idx = int(np.argmin(F))
    z = np.asarray(X[best_idx], dtype=float).reshape(-1)
    x = np.asarray(subspace.expand(z), dtype=float).reshape(-1)
    return x, float(F[best_idx]), z


def _population_from_arrays(X: np.ndarray, F: np.ndarray) -> Population:
    inds = [
        Individual(X=np.asarray(X[i], dtype=float), F=np.asarray(F[i], dtype=float))
        for i in range(len(X))
    ]
    return Population.create(*inds)


def _evaluate_batch(problem: SubspaceProblem, X: np.ndarray) -> np.ndarray:
    out: dict = {}
    problem._evaluate(X, out)
    return np.asarray(out["F"], dtype=float).reshape(-1, 1)


def _inject_into_fullspace(
    full_algo,
    full_problem: SubspaceProblem,
    x: np.ndarray,
) -> float:
    """Replace the worst full-space individual with ``x``; return new fitness."""
    pop = full_algo.pop
    F = pop.get("F").flatten()
    X = pop.get("X").copy()
    worst_idx = int(np.argmax(F))

    xl, xu = full_problem.bounds()
    x_clip = np.clip(np.asarray(x, dtype=float).reshape(-1), xl, xu)
    f_new = float(_evaluate_batch(full_problem, x_clip.reshape(1, -1))[0, 0])

    X[worst_idx] = x_clip
    F_new = F.copy()
    F_new[worst_idx] = f_new
    full_algo.pop = _population_from_arrays(X, F_new.reshape(-1, 1))
    full_algo.evaluator.n_eval += 1
    return f_new


def _refresh_subspace_after_anchor(
    sub_algo,
    sub_problem: SubspaceProblem,
    subspace,
    args: argparse.Namespace,
    *,
    mode: str,
) -> int:
    """Apply anchor update side effects on the subspace population; return NFE used."""
    n_var = subspace.search_dim
    pop_size = args.pop_size

    if mode == "resample":
        sampling = CenteredSampling(
            center=np.zeros(n_var, dtype=float),
            method=args.init_pop,
            scale=args.pop_sigma,
        )
        X = sampling._do(sub_problem, pop_size)
        F = _evaluate_batch(sub_problem, X)
        n_eval = pop_size
    elif mode == "reeval":
        pop = sub_algo.pop
        if pop is None:
            sampling = CenteredSampling(
                center=np.zeros(n_var, dtype=float),
                method=args.init_pop,
                scale=args.pop_sigma,
            )
            X = sampling._do(sub_problem, pop_size)
            F = _evaluate_batch(sub_problem, X)
            n_eval = pop_size
        else:
            X = pop.get("X")
            F = _evaluate_batch(sub_problem, X)
            n_eval = len(X)
    else:
        raise ValueError(f"Unknown sub_anchor_update mode {mode!r}")

    sub_algo.pop = _population_from_arrays(X, F)
    sub_algo.evaluator.n_eval += n_eval
    return n_eval


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class DualEALoggingCallback(LoggingCallback):
    """Log per-cycle metrics with the same W&B keys as ``LoggingCallback``.

    Standard keys (full-space population unless noted):
    ``generation``, ``nfe``, ``best_fitness``, ``mean_fitness``, ``center_fitness``.

    ``best_fitness`` is the minimum over both EA populations. ``nfe`` is the
    shared budget across both evaluators. Dual-specific keys ``cycle``,
    ``full_*``, and ``sub_*`` are also logged.
    """

    def __init__(
        self,
        *,
        eval_fn,
        full_subspace,
        sub_subspace,
        use_wandb: bool = False,
        log_every: int = 1,
    ) -> None:
        super().__init__(
            eval_fn=eval_fn,
            subspace=full_subspace,
            use_wandb=use_wandb,
            log_every=log_every,
        )
        self._sub_subspace = sub_subspace
        self._cycle = 0

    def notify(self, algorithm, **kwargs) -> None:
        self._cycle += 1
        if self._cycle % self._log_every != 0:
            return

        full_algo = kwargs["full_algo"]
        sub_algo = kwargs["sub_algo"]
        nfe = _total_nfe(full_algo, sub_algo)

        full_pop = full_algo.pop
        sub_pop = sub_algo.pop
        full_F = full_pop.get("F").flatten()
        sub_F = sub_pop.get("F").flatten()
        full_X = full_pop.get("X")
        sub_X = sub_pop.get("X")

        best_fitness = float(min(full_F.min(), sub_F.min()))
        mean_fitness = float(full_F.mean())
        center_fitness = self._compute_center_fitness(full_X)
        generation = int(full_algo.n_gen)

        metrics = {
            "generation": generation,
            "nfe": nfe,
            "best_fitness": best_fitness,
            "mean_fitness": mean_fitness,
            "center_fitness": center_fitness,
            "cycle": self._cycle,
            "full_best_fitness": float(full_F.min()),
            "full_mean_fitness": float(full_F.mean()),
            "full_center_fitness": center_fitness,
            "sub_best_fitness": float(sub_F.min()),
            "sub_mean_fitness": float(sub_F.mean()),
            "sub_center_fitness": self._center_fitness_for(sub_X, self._sub_subspace),
        }
        if generation % 1000 == 0:
            self._log_console(metrics)
        if self._use_wandb:
            self._log_wandb(metrics)

    def _center_fitness_for(self, X: np.ndarray, subspace) -> float:
        centroid_z = X.mean(axis=0)
        centroid_x = subspace.expand(centroid_z)
        try:
            return float(self._eval_fn(centroid_x))
        except Exception:
            return float("nan")


# ---------------------------------------------------------------------------
# W&B
# ---------------------------------------------------------------------------

def init_wandb_dual_ea(args: argparse.Namespace) -> None:
    import wandb  # type: ignore

    eff_d = effective_subspace_param(args)
    if args.wandb_name in (None, "", "__auto__"):
        args.wandb_name = (
            f"{args.problem}-dim{args.dim}-dual_ea-{args.subspace_method}"
        )
        if subspace_method_is_lora(args.subspace_method):
            args.wandb_name += f"-lora_rank{args.lora_rank}"
        else:
            args.wandb_name += f"-subdim{eff_d}"
        args.wandb_name += (
            f"-{args.subspace_assignment}-{args.optimizer}-seed{args.seed}"
        )

    if args.wandb_group:
        placeholder_pattern = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
        placeholder_values = vars(args).copy()
        placeholder_values["subspace_dim"] = eff_d
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
    config["approach"] = "dual_ea"
    config["fullspace_assignment"] = args.fullspace_assignment
    config["search_dim_full"] = args.dim
    config["search_dim_sub"] = optimizer_search_dim(args)
    if subspace_method_is_lora(args.subspace_method):
        config["subspace_dim"] = None
    wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        group=args.wandb_group,
        name=args.wandb_name,
        config=config,
    )


def _refresh_subspace_cost(sub_algo, args: argparse.Namespace) -> int:
    """NFE cost of refreshing the subspace population after an anchor update."""
    if args.sub_anchor_update == "resample":
        return args.pop_size
    if sub_algo.pop is None:
        return args.pop_size
    return len(sub_algo.pop)


def _advance_generations(
    algo,
    *,
    n_gens: int,
    pop_size: int,
    full_algo,
    sub_algo,
    max_nfe: int,
) -> int:
    """Run up to ``n_gens`` optimizer steps; return the number completed."""
    completed = 0
    for _ in range(n_gens):
        if _budget_left(full_algo, sub_algo, max_nfe) < pop_size:
            break
        if not algo.has_next():
            break
        algo.next()
        completed += 1
    return completed


def _track_best(
    x: np.ndarray,
    f: float,
    global_best_x: np.ndarray | None,
    global_best_f: float,
) -> tuple[np.ndarray, float]:
    if f < global_best_f:
        return x.copy(), f
    return global_best_x, global_best_f


# ---------------------------------------------------------------------------
# Alternating optimization loop
# ---------------------------------------------------------------------------

def run_dual_ea(
    *,
    full_algo,
    sub_algo,
    full_problem: SubspaceProblem,
    sub_problem: SubspaceProblem,
    full_subspace,
    sub_subspace,
    args: argparse.Namespace,
    max_nfe: int,
    callback: DualEALoggingCallback | None = None,
) -> tuple[np.ndarray, float, int, int]:
    """Run alternating cycles until the shared NFE budget is exhausted.

    Returns ``(best_x, best_fitness, n_cycles, total_nfe)``.
    """
    global_best_f = float("inf")
    global_best_x: np.ndarray | None = None
    n_cycles = 0

    while full_algo.has_next() and _budget_left(full_algo, sub_algo, max_nfe) > 0:
        n_cycles += 1

        # ---- Step 1: m full-space generations ----
        if _budget_left(full_algo, sub_algo, max_nfe) < args.pop_size:
            break
        full_ran = _advance_generations(
            full_algo,
            n_gens=args.full_iters,
            pop_size=args.pop_size,
            full_algo=full_algo,
            sub_algo=sub_algo,
            max_nfe=max_nfe,
        )
        if full_ran == 0:
            break

        best_x_full, f_full = _best_fullspace_solution(full_algo, full_subspace)
        global_best_x, global_best_f = _track_best(
            best_x_full, f_full, global_best_x, global_best_f
        )

        # ---- Step 2: anchor update + k subspace generations ----
        sub_subspace.set_x0(best_x_full)
        refresh_cost = _refresh_subspace_cost(sub_algo, args)
        if _budget_left(full_algo, sub_algo, max_nfe) < refresh_cost:
            break
        _refresh_subspace_after_anchor(
            sub_algo,
            sub_problem,
            sub_subspace,
            args,
            mode=args.sub_anchor_update,
        )

        if _budget_left(full_algo, sub_algo, max_nfe) < args.pop_size:
            break
        sub_ran = _advance_generations(
            sub_algo,
            n_gens=args.sub_iters,
            pop_size=args.pop_size,
            full_algo=full_algo,
            sub_algo=sub_algo,
            max_nfe=max_nfe,
        )
        if sub_ran == 0:
            break

        best_x_sub, f_sub, _ = _best_subspace_solution(sub_algo, sub_subspace)
        global_best_x, global_best_f = _track_best(
            best_x_sub, f_sub, global_best_x, global_best_f
        )

        # ---- Step 3: inject subspace best into full-space population ----
        if _budget_left(full_algo, sub_algo, max_nfe) >= 1:
            f_inj = _inject_into_fullspace(full_algo, full_problem, best_x_sub)
            global_best_x, global_best_f = _track_best(
                best_x_sub, f_inj, global_best_x, global_best_f
            )

        if callback is not None:
            callback.notify(
                full_algo,
                full_algo=full_algo,
                sub_algo=sub_algo,
            )

    if global_best_x is None:
        global_best_x, global_best_f = _best_fullspace_solution(
            full_algo, full_subspace
        )

    return (
        global_best_x,
        global_best_f,
        n_cycles,
        _total_nfe(full_algo, sub_algo),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

    np.random.seed(args.seed)

    if args.wandb:
        init_wandb_dual_ea(args)

    eff_sub = effective_subspace_param(args)
    print("=" * 70)
    print("Dual-EA Alternating Full-Space / Subspace Optimization")
    print("=" * 70)
    print(f"  Problem           : {args.problem} (dim={args.dim})")
    print(
        f"  Full-space EA     : fullspace "
        f"({args.fullspace_assignment}, search_dim={args.dim})"
    )
    if subspace_method_is_lora(args.subspace_method):
        sub_stat = f"rank r={eff_sub}"
    else:
        sub_stat = f"d={eff_sub}"
    print(
        f"  Subspace EA       : {args.subspace_method} "
        f"({sub_stat}, {args.subspace_assignment}, "
        f"search_dim={optimizer_search_dim(args)})"
    )
    print(f"  Optimizer         : {args.optimizer} (pop={args.pop_size})")
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

    # ---- Full-space EA ----
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

    # ---- Subspace EA (additive; x0 updated each cycle from full-space best) ----
    sub_subspace = build_subspace(
        method=args.subspace_method,
        D=args.dim,
        d=eff_sub,
        subspace_assignment=args.subspace_assignment,
        seed=args.seed,
        lb=lsgo.lb,
        ub=lsgo.ub,
        device=args.subspace_device,
    )
    sub_problem = SubspaceProblem(lsgo=lsgo, subspace=sub_subspace)

    full_algo = build_algorithm(args)
    sub_algo = build_algorithm(args)

    # Subspace EA starts with a standard initial population; anchor x0 is set
    # from the full-space best before the first subspace step each cycle.
    _set_algorithm_sampling(
        sub_algo,
        CenteredSampling(
            center=np.zeros(sub_subspace.search_dim, dtype=float),
            method=args.init_pop,
            scale=args.pop_sigma,
        ),
    )

    from pymoo.termination import get_termination

    # Generous per-algorithm gen cap; the shared NFE budget stops the outer loop.
    gen_cap = max(1, args.max_nfe // max(1, args.pop_size) + 2)
    full_algo.setup(
        full_problem,
        termination=get_termination("n_gen", gen_cap),
        seed=args.seed,
        verbose=False,
    )
    sub_algo.setup(
        sub_problem,
        termination=get_termination("n_gen", gen_cap),
        seed=args.seed + 1,
        verbose=False,
    )

    callback = DualEALoggingCallback(
        eval_fn=lsgo.evaluate,
        full_subspace=full_subspace,
        sub_subspace=sub_subspace,
        use_wandb=args.wandb,
        log_every=args.log_every,
    )

    t0 = time.perf_counter()
    best_x, best_f, n_cycles, total_nfe = run_dual_ea(
        full_algo=full_algo,
        sub_algo=sub_algo,
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
    print(f"Dual-EA optimization finished in {elapsed:.2f}s")
    print(f"  Cycles completed  : {n_cycles}")
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
        wandb.summary["elapsed_seconds"] = elapsed
        wandb.finish()


if __name__ == "__main__":
    main()
