"""Evolutionary Subspace Optimization - CLI entry point.

See README.md for a full usage example; run ``python main.py --help`` for arguments.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time

import numpy as np
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from subspace import build_subspace
from problems import LSGOProblem
from optimizers import build_algorithm
from utils import LoggingCallback, SubspaceProblem


def subspace_method_is_lora(subspace_method: str) -> bool:
    """True when the reduction is LoRA-based (method id contains ``lora``)."""
    return "lora" in subspace_method


def subspace_method_is_fullspace(subspace_method: str) -> bool:
    """True when the EA searches the full problem dimension ``D`` (no reduction)."""
    return subspace_method == "fullspace"


def effective_subspace_param(args: argparse.Namespace) -> int:
    """Subspace size used by the active method: d for RP/RB; LoRA rank r for LoRA; D for fullspace."""
    if subspace_method_is_fullspace(args.subspace_method):
        return args.dim
    if subspace_method_is_lora(args.subspace_method):
        if args.lora_rank is None:
            raise RuntimeError(
                "lora_rank is unset; main() must validate CLI args before calling this"
            )
        return args.lora_rank
    return args.subspace_dim


def optimizer_search_dim(args: argparse.Namespace) -> int:
    """Dimensionality of z passed to the evolutionary algorithm (matches ``Subspace.search_dim``)."""
    if subspace_method_is_fullspace(args.subspace_method):
        return args.dim
    if subspace_method_is_lora(args.subspace_method):
        if args.lora_rank is None:
            raise RuntimeError(
                "lora_rank is unset; main() must validate CLI args before calling this"
            )
        m = math.ceil(math.sqrt(args.dim))
        return 2 * m * args.lora_rank
    return args.subspace_dim


def _wandb_bool(value: object) -> bool:
    """Bool compatible with W&B sweep agents (they often emit ``--flag=True``)."""
    if isinstance(value, bool):
        return value
    if value is None:
        return True
    v = str(value).strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean string, got {value!r}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evolutionary Subspace Optimization on CEC-2013 LSGO benchmarks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Problem ----
    parser.add_argument(
        "--problem",
        type=str,
        default="cec2013_lsgo_f1",
        help=(
            "Benchmark problem id (e.g. cec2013_lsgo_f1 ... cec2013_lsgo_f15). "
            "Must match a registered problem."
        ),
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=1000,
        choices=[1000, 5000, 10000, 100000, 1000000],
        dest="dim",
        help="Full-space dimensionality D.",
    )

    # ---- Subspace ----
    parser.add_argument(
        "--subspace_method",
        type=str,
        default="random_projection",
        choices=[
            "random_projection",
            "random_blocking",
            "lora",
            "fullspace",
            "none",
        ],
        help=(
            "Subspace reduction method. Use ``fullspace`` or ``none`` to run the EA "
            "directly in the full problem space (dimension D)."
        ),
    )
    parser.add_argument(
        "--subspace_dim",
        type=int,
        default=None,
        help=(
            "Subspace dimensionality d for random_projection and random_blocking."
        ),
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=None,
        help=(
            "LoRA rank r. Required when the subspace method uses LoRA; ignored otherwise."
        ),
    )
    parser.add_argument(
        "--subspace_assignment",
        type=str,
        default="absolute",
        choices=["absolute", "additive"],
        help=(
            "Subspace assignment: absolute (x=z@P / grouping / LoRA map) or additive "
            "(x=x0+f(z)); for additive, x0 is uniform in [lb, ub], deterministic from "
            "--seed together with the subspace RNG stream."
        ),
    )
    parser.add_argument(
        "--subspace_device",
        type=str,
        default="cuda:0",
        help=(
            "PyTorch device for random_projection / LoRA matmul in expand() "
            "(e.g. cuda, cuda:0, cpu)."
        ),
    )

    # ---- Optimizer ----
    parser.add_argument(
        "--optimizer",
        type=str,
        default="de",
        choices=["de", "pso", "es", "cmaes"],
        help="Evolutionary algorithm.",
    )
    parser.add_argument("--pop_size", type=int, default=100, help="Population size.")
    parser.add_argument(
        "--init_pop",
        type=str,
        default="uniform",
        choices=["uniform", "gaussian", "lhs"],
        help="Initial population sampling method.",
    )
    parser.add_argument(
        "--pop_sigma",
        type=float,
        default=1.0,
        help="Sigma scale for --init_pop gaussian sampling.",
    )

    # ---- DE parameters ----
    de = parser.add_argument_group("DE parameters")
    de.add_argument(
        "--de_mut_rate",
        type=float,
        default=0.8,
        help="DE mutation scale factor F in (0, 2].",
    )
    de.add_argument(
        "--de_cr_rate",
        type=float,
        default=0.9,
        help="DE crossover rate CR in [0, 1].",
    )
    de.add_argument(
        "--de_selection",
        type=str,
        default="rand",
        choices=["rand", "best", "target-to-best"],
        help="DE selection method.",
    )
    de.add_argument(
        "--de_n_diffs",
        type=int,
        default=1,
        help="DE number of differences.",
    )
    de.add_argument(
        "--de_jitter",
        type=bool,
        default=False,
        help="DE jitter.",
    )
    de.add_argument(
        "--de_crossover",
        type=str,
        default="bin",
        choices=["bin", "exp", "hypercube", "line"],
        help="DE crossover method.",
    )
    de.add_argument(
        "--de_evolving",
        nargs="?",
        const=True,
        default=False,
        type=_wandb_bool,
        help=(
            "Enable PyMOO evolutionary adaptation of DE parameters (F, CR); "
            "otherwise F and CR stay fixed at de_mut_rate / de_cr_rate."
        ),
    )

    # ---- PSO parameters ----
    pso = parser.add_argument_group("PSO parameters")
    pso.add_argument("--pso_w", type=float, default=0.9, help="PSO inertia weight.")
    pso.add_argument(
        "--pso_c1",
        type=float,
        default=2.0,
        help="PSO cognitive (personal best) weight.",
    )
    pso.add_argument(
        "--pso_c2",
        type=float,
        default=2.0,
        help="PSO social (global best) weight.",
    )
    pso.add_argument(
        "--pso_evolving",
        nargs="?",
        const=True,
        default=False,
        type=_wandb_bool,
        help=(
            "Enable PyMOO adaptive PSO: w, c1, and c2 are updated each generation "
            "from swarm spread (fuzzy strategy). Omit to keep pso_w, pso_c1, pso_c2 fixed."
        ),
    )

    # ---- ES parameters ----
    es = parser.add_argument_group("ES parameters")
    es.add_argument(
        "--es_sigma",
        type=float,
        default=0.3,
        help="ES initial step-size sigma (used to seed Gaussian sampling spread).",
    )

    # ---- CMA-ES parameters ----
    cmaes = parser.add_argument_group("CMA-ES parameters")
    cmaes.add_argument(
        "--cmaes_sigma",
        type=float,
        default=0.5,
        help="CMA-ES initial step-size sigma.",
    )

    # ---- Termination ----
    parser.add_argument(
        "--max_nfe",
        type=int,
        default=3_000_000,
        help="Maximum number of function evaluations (NFE budget).",
    )

    # ---- Misc ----
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "RNG seed for the EA (PyMOO), subspace matrices, and NumPy. "
            "Sweep this for repeat runs; use --benchmark_seed for the LSGO instance."
        ),
    )
    parser.add_argument(
        "--benchmark_seed",
        type=int,
        default=0,
        help=(
            "Seed for CEC-2013 LSGO structural data (shifts, rotations, weights). "
            "Keep fixed when averaging over --seed for the same benchmark instance."
        ),
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=1,
        help="Log metrics every N generations.",
    )

    # ---- W&B ----
    wb = parser.add_argument_group("Weights & Biases")
    wb.add_argument(
        "--wandb",
        nargs="?",
        const=True,
        default=False,
        type=_wandb_bool,
        help="Enable W&B logging.",
    )
    wb.add_argument("--wandb_entity", type=str, default=None, help="W&B entity.")
    wb.add_argument(
        "--wandb_project",
        type=str,
        default="evo-subspace-opt",
        help="W&B project name.",
    )
    wb.add_argument(
        "--wandb_group",
        type=str,
        default=None,
        help=(
            "W&B group. Placeholders: {dim}, {problem}, {subspace_method}, "
            "{subspace_assignment}, "
            "{optimizer}, {subspace_dim} (RP/RB d or LoRA rank r), {lora_rank} (LoRA only; "
            "otherwise omitted)."
        ),
    )
    wb.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help=(
            "W&B run name. Use __auto__ or omit for a deterministic name from "
            "problem, full dimension (--dim), subspace_assignment, subspace, optimizer, "
            "DE F and CR, and optimizer seed."
        ),
    )

    return parser


# ---------------------------------------------------------------------------
# W&B initialization
# ---------------------------------------------------------------------------

def init_wandb(args: argparse.Namespace) -> None:
    """Initialize a W&B run with the full config."""
    import wandb  # type: ignore

    eff_d = effective_subspace_param(args)
    if args.wandb_name in (None, "", "__auto__"):
        args.wandb_name = (f"{args.problem}-dim{args.dim}-{args.subspace_method}")
        if subspace_method_is_lora(args.subspace_method):
            args.wandb_name += f"-lora_rank{args.lora_rank}"
        elif not subspace_method_is_fullspace(args.subspace_method):
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
    config["search_dim"] = optimizer_search_dim(args)
    if subspace_method_is_lora(args.subspace_method):
        # CLI default is unused for LoRA; rank r drives structure and search_dim is 2*M*r.
        config["subspace_dim"] = None
    elif subspace_method_is_fullspace(args.subspace_method):
        config["subspace_dim"] = args.dim
    wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        group=args.wandb_group,
        name=args.wandb_name,
        config=config,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.subspace_method == "none":
        args.subspace_method = "fullspace"

    if subspace_method_is_lora(args.subspace_method) and args.lora_rank is None:
        parser.error(
            "--lora_rank is required when the subspace method uses LoRA"
        )

    if (
        not subspace_method_is_fullspace(args.subspace_method)
        and not subspace_method_is_lora(args.subspace_method)
        and args.subspace_dim is None
    ):
        parser.error(
            "--subspace_dim is required for random_projection and random_blocking"
        )

    # Seed global RNG for reproducibility
    np.random.seed(args.seed)

    # -- W&B --
    if args.wandb:
        init_wandb(args)
    print("=" * 70)
    print("Evolutionary Subspace Optimization")
    print("=" * 70)
    print(f"  Problem        : {args.problem} (dim={args.dim})")
    eff_sub = effective_subspace_param(args)
    if subspace_method_is_fullspace(args.subspace_method):
        sub_stat = f"D={eff_sub} (full space)"
        dev_suffix = ""
    elif subspace_method_is_lora(args.subspace_method):
        sub_stat = f"rank r={eff_sub}"
        dev_suffix = f", device={args.subspace_device}"
    else:
        sub_stat = f"d={eff_sub}"
        dev_suffix = f", device={args.subspace_device}"
    print(
        f"  Subspace       : {args.subspace_method} "
        f"({sub_stat}, {args.subspace_assignment}{dev_suffix})"
    )
    print(f"  Optimizer      : {args.optimizer} (pop={args.pop_size})")
    print(f"  Max NFE        : {args.max_nfe}")
    print(f"  Optimizer seed : {args.seed}")
    print(f"  Benchmark seed : {args.benchmark_seed}")
    print("=" * 70)

    # -- Benchmark (bounds needed before additive subspace init) --
    lsgo = LSGOProblem(
        func_id=args.problem,
        D=args.dim,
        seed=args.benchmark_seed,
    )

    # -- Subspace --
    subspace = build_subspace(
        method=args.subspace_method,
        D=args.dim,
        d=effective_subspace_param(args),
        subspace_assignment=args.subspace_assignment,
        seed=args.seed,
        lb=lsgo.lb,
        ub=lsgo.ub,
        device=args.subspace_device,
    )
    print(f"  Search dim     : {subspace.search_dim}")
    if args.subspace_assignment == "additive":
        xa = subspace.x0
        assert xa is not None
        print(
            f"  Additive x0    : uniform in [lb, ub] (same seed as subspace); "
            f"||x0||_2={float(np.linalg.norm(xa)):.4f}"
        )

    problem = SubspaceProblem(lsgo=lsgo, subspace=subspace)
    print(f"  Optimum known  : {lsgo.optimum}")
    print("=" * 70)

    # -- Callback --
    callback = LoggingCallback(
        eval_fn=lsgo.evaluate,
        subspace=subspace,
        use_wandb=args.wandb,
        log_every=args.log_every,
    )

    # -- Build algorithm --
    algorithm = build_algorithm(args)

    # -- Termination --
    termination = get_termination("n_eval", args.max_nfe)

    # -- Run optimization --
    t0 = time.perf_counter()
    result = minimize(
        problem,
        algorithm,
        termination,
        seed=args.seed,
        callback=callback,
        verbose=False,
        save_history=False,
    )
    elapsed = time.perf_counter() - t0

    # -- Report result --
    print("=" * 70)
    print(f"Optimization finished in {elapsed:.2f}s")
    print(f"  Best fitness   : {float(result.F.flatten()[0]):.6e}")
    print(f"  Total NFE      : {result.algorithm.evaluator.n_eval}")
    if lsgo.optimum is not None:
        gap = float(result.F.flatten()[0]) - lsgo.optimum
        print(f"  Gap to optimum : {gap:.6e}")
    print("=" * 70)

    if args.wandb:
        import wandb  # type: ignore

        wandb.summary["best_fitness"] = float(result.F.flatten()[0])
        wandb.summary["total_nfe"] = result.algorithm.evaluator.n_eval
        wandb.summary["elapsed_seconds"] = elapsed
        wandb.finish()


if __name__ == "__main__":
    main()
