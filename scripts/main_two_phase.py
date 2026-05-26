"""Two-phase evolutionary optimization: full-space then subspace.

Phase 1 always optimizes in full space with **absolute** assignment ($x = z$).
Phase 2 continues in a reduced subspace for ``sub_nfe`` evaluations, warm-started
from the phase-1 best solution (``x0`` for additive assignment, or ``reduce(x*)``
when supported for absolute assignment). ``--subspace_assignment`` applies to phase 2 only.

Budget constraint: ``full_nfe + sub_nfe == max_nfe``.

See README.md for the classic single-phase CLI (``scripts/main.py``).
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
from pymoo.core.sampling import Sampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from scripts.main import (
    build_parser as _build_base_parser,
    effective_subspace_param,
    optimizer_search_dim,
    subspace_method_is_fullspace,
    subspace_method_is_lora,
)
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
        "Two-phase evolutionary optimization: full-space then subspace "
        "(full_nfe + sub_nfe = max_nfe)."
    )

    phase = parser.add_argument_group("Two-phase budget")
    phase.add_argument(
        "--full_nfe",
        type=int,
        default=None,
        help=(
            "NFE budget for phase 1 (full-space). If omitted, derived as "
            "max_nfe - sub_nfe."
        ),
    )
    phase.add_argument(
        "--sub_nfe",
        type=int,
        default=None,
        help=(
            "NFE budget for phase 2 (subspace). If omitted, derived as "
            "max_nfe - full_nfe."
        ),
    )
    phase.add_argument(
        "--phase2_init",
        type=str,
        default="from_phase1",
        choices=["from_phase1", "uniform"],
        help=(
            "Phase-2 initial population: warm-start around the phase-1 best "
            "(from_phase1) or standard sampling (uniform)."
        ),
    )

    return parser


def resolve_nfe_budgets(args: argparse.Namespace) -> tuple[int, int]:
    """Return (full_nfe, sub_nfe) and validate against max_nfe."""
    full = args.full_nfe
    sub = args.sub_nfe

    if full is None and sub is None:
        raise ValueError(
            "Specify at least one of --full_nfe or --sub_nfe "
            "(the other is derived from --max_nfe)."
        )
    if full is None:
        full = args.max_nfe - sub
    if sub is None:
        sub = args.max_nfe - full

    if full <= 0 or sub <= 0:
        raise ValueError(
            f"Both phase budgets must be positive; got full_nfe={full}, sub_nfe={sub}"
        )
    if full + sub != args.max_nfe:
        raise ValueError(
            f"full_nfe + sub_nfe must equal max_nfe "
            f"({full} + {sub} = {full + sub} != {args.max_nfe})"
        )
    return full, sub


# ---------------------------------------------------------------------------
# Phase-2 warm-start sampling
# ---------------------------------------------------------------------------

class CenteredSampling(Sampling):
    """Sample around a fixed center in search space (for phase-2 warm start).

    The first individual is always the clipped ``center`` (e.g. ``z = 0`` in additive
    mode, mapping to the phase-1 best via ``x0``). Remaining individuals are sampled
    around that anchor so the initial population cannot regress far below phase-1
    fitness purely from initialization.
    """

    def __init__(
        self,
        center: np.ndarray,
        method: str = "gaussian",
        scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.center = np.asarray(center, dtype=float).reshape(-1)
        self.method = method.lower()
        self.scale = scale

    def _sample_around(
        self,
        problem,
        n_samples: int,
        center: np.ndarray,
        xl: np.ndarray,
        xu: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        if n_samples <= 0:
            return np.empty((0, center.shape[0]), dtype=float)

        if self.method == "uniform":
            half = self.scale * (xu - xl) / 2.0
            low = np.maximum(xl, center - half)
            high = np.minimum(xu, center + half)
            return np.random.uniform(low, high, size=(n_samples, center.shape[0]))

        if self.method == "gaussian":
            sigma = self.scale * (xu - xl) / 2.0
            X = np.random.normal(center, sigma, size=(n_samples, center.shape[0]))
            return np.clip(X, xl, xu)

        if self.method == "lhs":
            from pymoo.operators.sampling.lhs import LatinHypercubeSampling

            lhs = LatinHypercubeSampling()
            unit = lhs._do(problem, n_samples, **kwargs)
            half = self.scale * (xu - xl) / 2.0
            low = np.maximum(xl, center - half)
            high = np.minimum(xu, center + half)
            return low + unit * (high - low)

        raise ValueError(
            f"Unknown init method {self.method!r}. Choose from: uniform, gaussian, lhs."
        )

    def _do(self, problem, n_samples: int, **kwargs) -> np.ndarray:
        xl, xu = problem.bounds()
        n_var = problem.n_var
        if self.center.shape[0] != n_var:
            raise ValueError(
                f"center length {self.center.shape[0]} != problem n_var {n_var}"
            )

        anchor = np.clip(self.center, xl, xu)
        if n_samples <= 0:
            return np.empty((0, n_var), dtype=float)
        if n_samples == 1:
            return anchor.reshape(1, -1)

        X_rest = self._sample_around(
            problem, n_samples - 1, anchor, xl, xu, **kwargs
        )
        return np.vstack([anchor.reshape(1, -1), X_rest])


class TwoPhaseLoggingCallback(LoggingCallback):
    """Logging callback with cumulative NFE offset and phase label."""

    def __init__(
        self,
        *args,
        nfe_offset: int = 0,
        phase: str = "",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._nfe_offset = nfe_offset
        self._phase = phase

    def notify(self, algorithm, **kwargs) -> None:
        gen: int = algorithm.n_gen
        if gen % self._log_every != 0:
            return

        pop = algorithm.pop
        F: np.ndarray = pop.get("F").flatten()
        X: np.ndarray = pop.get("X")
        nfe: int = algorithm.evaluator.n_eval + self._nfe_offset

        best_fitness = float(F.min())
        mean_fitness = float(F.mean())
        center_fitness = self._compute_center_fitness(X)

        metrics = {
            "generation": gen,
            "nfe": nfe,
            "best_fitness": best_fitness,
            "mean_fitness": mean_fitness,
            "center_fitness": center_fitness,
            "phase": self._phase,
        }
        if gen % 1000 == 0:
            self._log_console(metrics)
        if self._use_wandb:
            self._log_wandb(metrics)


# ---------------------------------------------------------------------------
# Optimization helpers
# ---------------------------------------------------------------------------

def _best_fullspace_x(result, subspace) -> np.ndarray:
    """Extract the best full-space solution from a PyMOO result."""
    best_z = np.asarray(result.X, dtype=float).reshape(-1)
    return np.asarray(subspace.expand(best_z), dtype=float).reshape(-1)


def _phase2_search_center(subspace, best_x: np.ndarray) -> np.ndarray | None:
    """Return the z-vector to center phase-2 initialization, or None if unavailable."""
    if subspace.subspace_assignment == "additive":
        return np.zeros(subspace.search_dim, dtype=float)
    try:
        return np.asarray(subspace.reduce(best_x), dtype=float).reshape(-1)
    except NotImplementedError:
        return None


def _set_algorithm_sampling(algorithm, sampling: Sampling) -> None:
    """Attach sampling for initial population creation.

    PyMOO genetic algorithms copy ``sampling`` into ``Initialization`` at
    construction time; assigning ``algorithm.sampling`` alone does not affect
    the population actually evaluated in generation 1.
    """
    algorithm.sampling = sampling
    initialization = getattr(algorithm, "initialization", None)
    if initialization is not None:
        initialization.sampling = sampling


def _run_phase(
    *,
    label: str,
    problem: SubspaceProblem,
    args: argparse.Namespace,
    nfe_budget: int,
    nfe_offset: int,
    subspace,
    warm_center: np.ndarray | None = None,
) -> tuple[object, float, np.ndarray]:
    """Run one optimization phase and return (result, elapsed, best_x)."""
    callback = TwoPhaseLoggingCallback(
        eval_fn=problem.lsgo.evaluate,
        subspace=subspace,
        use_wandb=args.wandb,
        log_every=args.log_every,
        nfe_offset=nfe_offset,
        phase=label,
    )

    algorithm = build_algorithm(args)
    if warm_center is not None:
        _set_algorithm_sampling(
            algorithm,
            CenteredSampling(
                center=warm_center,
                method=args.init_pop,
                scale=args.pop_sigma,
            ),
        )

    termination = get_termination("n_eval", nfe_budget)

    print("=" * 70)
    print(f"Phase: {label}")
    print(f"  Search dim     : {subspace.search_dim}")
    print(f"  NFE budget     : {nfe_budget}")
    print(f"  Cumulative NFE : {nfe_offset} -> {nfe_offset + nfe_budget}")
    print("=" * 70)

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
    best_x = _best_fullspace_x(result, subspace)
    return result, elapsed, best_x


def init_wandb_two_phase(args: argparse.Namespace, full_nfe: int, sub_nfe: int) -> None:
    """Initialize W&B with two-phase-specific naming and config."""
    import wandb  # type: ignore

    eff_d = effective_subspace_param(args)
    if args.wandb_name in (None, "", "__auto__"):
        args.wandb_name = (
            f"{args.problem}-dim{args.dim}-two_phase-full{full_nfe}-sub{sub_nfe}"
            f"-{args.subspace_method}"
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
        placeholder_values["full_nfe"] = full_nfe
        placeholder_values["sub_nfe"] = sub_nfe
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
    config["approach"] = "two_phase"
    config["phase1_subspace_assignment"] = Subspace.ABSOLUTE
    config["full_nfe"] = full_nfe
    config["sub_nfe"] = sub_nfe
    config["search_dim"] = optimizer_search_dim(args)
    if subspace_method_is_lora(args.subspace_method):
        config["subspace_dim"] = None
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

    if subspace_method_is_fullspace(args.subspace_method):
        parser.error(
            "Two-phase optimization requires a reduced subspace for phase 2; "
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

    try:
        full_nfe, sub_nfe = resolve_nfe_budgets(args)
    except ValueError as exc:
        parser.error(str(exc))

    np.random.seed(args.seed)

    if args.wandb:
        init_wandb_two_phase(args, full_nfe, sub_nfe)

    eff_sub = effective_subspace_param(args)
    print("=" * 70)
    print("Two-Phase Evolutionary Subspace Optimization")
    print("=" * 70)
    print(f"  Problem        : {args.problem} (dim={args.dim})")
    print(f"  Phase 1        : fullspace, absolute (NFE={full_nfe})")
    if subspace_method_is_lora(args.subspace_method):
        sub_stat = f"rank r={eff_sub}"
    else:
        sub_stat = f"d={eff_sub}"
    print(
        f"  Phase 2        : {args.subspace_method} "
        f"({sub_stat}, {args.subspace_assignment}, NFE={sub_nfe})"
    )
    print(f"  Optimizer      : {args.optimizer} (pop={args.pop_size})")
    print(f"  Total NFE      : {args.max_nfe} (= {full_nfe} + {sub_nfe})")
    print(f"  Phase-2 init   : {args.phase2_init}")
    print(f"  Optimizer seed : {args.seed}")
    print(f"  Benchmark seed : {args.benchmark_seed}")
    print("=" * 70)

    lsgo = LSGOProblem(
        func_id=args.problem,
        D=args.dim,
        seed=args.benchmark_seed,
    )
    print(f"  Optimum known  : {lsgo.optimum}")
    print("=" * 70)

    # ---- Phase 1: full space ----
    fullspace = build_subspace(
        method="fullspace",
        D=args.dim,
        d=args.dim,
        subspace_assignment=Subspace.ABSOLUTE,
        seed=args.seed,
        lb=lsgo.lb,
        ub=lsgo.ub,
    )
    phase1_problem = SubspaceProblem(lsgo=lsgo, subspace=fullspace)

    result1, elapsed1, best_x = _run_phase(
        label="fullspace",
        problem=phase1_problem,
        args=args,
        nfe_budget=full_nfe,
        nfe_offset=0,
        subspace=fullspace,
    )
    phase1_nfe = int(result1.algorithm.evaluator.n_eval)
    phase1_fitness = float(result1.F.flatten()[0])

    print("=" * 70)
    print(f"Phase 1 finished in {elapsed1:.2f}s")
    print(f"  Best fitness   : {phase1_fitness:.6e}")
    print(f"  Phase NFE      : {phase1_nfe}")
    print(f"  ||best_x||_2   : {float(np.linalg.norm(best_x)):.4f}")
    print("=" * 70)

    # ---- Phase 2: subspace (warm-started from phase-1 best) ----
    phase2_x0 = best_x if args.subspace_assignment == "additive" else None
    subspace = build_subspace(
        method=args.subspace_method,
        D=args.dim,
        d=eff_sub,
        subspace_assignment=args.subspace_assignment,
        seed=args.seed,
        lb=lsgo.lb,
        ub=lsgo.ub,
        x0=phase2_x0,
        device=args.subspace_device,
    )

    warm_center: np.ndarray | None = None
    if args.phase2_init == "from_phase1":
        warm_center = _phase2_search_center(subspace, best_x)
        if warm_center is None:
            print(
                "Warning: phase-2 subspace does not support reduce(); "
                "falling back to uniform initialization. "
                "Consider --subspace_assignment additive for LoRA two-phase runs."
            )
        elif args.subspace_assignment == "additive":
            print(
                f"  Phase-2 x0     : phase-1 best (||x0||_2="
                f"{float(np.linalg.norm(subspace.x0)):.4f})"
            )
            print("  Phase-2 center : z = 0 (additive perturbation)")
        else:
            print("  Phase-2 center : reduce(phase-1 best)")

    phase2_problem = SubspaceProblem(lsgo=lsgo, subspace=subspace)
    result2, elapsed2, final_x = _run_phase(
        label="subspace",
        problem=phase2_problem,
        args=args,
        nfe_budget=sub_nfe,
        nfe_offset=phase1_nfe,
        subspace=subspace,
        warm_center=warm_center,
    )
    phase2_nfe = int(result2.algorithm.evaluator.n_eval)
    phase2_fitness = float(result2.F.flatten()[0])
    total_nfe = phase1_nfe + phase2_nfe
    total_elapsed = elapsed1 + elapsed2

    print("=" * 70)
    print(f"Phase 2 finished in {elapsed2:.2f}s")
    print(f"  Best fitness   : {phase2_fitness:.6e}")
    print(f"  Phase NFE      : {phase2_nfe}")
    print("=" * 70)
    print("=" * 70)
    print(f"Two-phase optimization finished in {total_elapsed:.2f}s")
    print(f"  Final fitness  : {phase2_fitness:.6e}")
    print(f"  Phase 1 fitness: {phase1_fitness:.6e}")
    print(f"  Total NFE      : {total_nfe}")
    if lsgo.optimum is not None:
        gap = phase2_fitness - lsgo.optimum
        print(f"  Gap to optimum : {gap:.6e}")
    print("=" * 70)

    if args.wandb:
        import wandb  # type: ignore

        wandb.summary["phase1_best_fitness"] = phase1_fitness
        wandb.summary["phase1_nfe"] = phase1_nfe
        wandb.summary["phase2_best_fitness"] = phase2_fitness
        wandb.summary["phase2_nfe"] = phase2_nfe
        wandb.summary["best_fitness"] = phase2_fitness
        wandb.summary["total_nfe"] = total_nfe
        wandb.summary["elapsed_seconds"] = total_elapsed
        wandb.finish()


if __name__ == "__main__":
    main()
