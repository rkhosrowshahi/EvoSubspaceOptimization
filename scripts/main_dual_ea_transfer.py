"""Dual-DE with population transfer from the best additive subspace direction.

This PyMOO-only variant keeps the usual dual-EA anchor logic:

- the full-space DE evolves absolute full-space solutions;
- the subspace DE evolves additive directions around the current full-space best;
- after each subspace phase, the best anchor direction is applied as a common
  scaled displacement to the full-space population.

The transfer step evaluates ``x_i + alpha * (best_x_sub - best_x_full)`` for a
budgeted subset of full-space individuals and replaces only individuals whose
fitness improves. The per-cycle success rate measures how often the
anchor-learned direction acts as a useful tuner for the current population.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from scripts.main import (
    effective_subspace_param,
    optimizer_search_dim,
    subspace_method_is_block_lora,
    subspace_method_is_fullspace,
    subspace_method_is_lora,
    validate_lora_blocks,
)
from scripts.main_dual_ea import (
    _advance_generations,
    _best_fullspace_solution,
    _best_subspace_solution,
    _budget_left,
    _evaluate_batch,
    _population_from_arrays,
    _refresh_subspace_cost,
    _sub_pop_size,
    _total_nfe,
    _track_best,
    build_parser as _build_dual_parser,
)
from scripts.main_two_phase import CenteredSampling, _set_optim_sampling
from subspace import build_subspace
from problems import LSGOProblem
from utils import LoggingCallback, SubspaceProblem
from optimizers import build_algorithm


@dataclass(frozen=True)
class TransferResult:
    success_count: int
    candidate_count: int
    population_size: int
    eval_count: int
    best_x: np.ndarray | None
    best_f: float | None
    mean_delta: float = 0.0
    best_delta: float = 0.0
    direction_norm: float = 0.0
    clip_fraction: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.candidate_count == 0:
            return 0.0
        return self.success_count / self.candidate_count


@dataclass(frozen=True)
class TransferRunResult:
    best_x: np.ndarray
    best_f: float
    n_cycles: int
    total_nfe: int
    full_improve_count: int
    subspace_improve_count: int
    transfer_improve_count: int
    transfer_attempt_count: int
    transfer_success_total: int
    transfer_candidate_total: int
    transfer_eval_total: int
    full_step_nfe_total: int
    sub_refresh_nfe_total: int
    sub_step_nfe_total: int

    @property
    def transfer_success_rate_avg(self) -> float:
        if self.transfer_candidate_total == 0:
            return 0.0
        return self.transfer_success_total / self.transfer_candidate_total


def build_parser() -> argparse.ArgumentParser:
    parser = _build_dual_parser()
    parser.description = (
        "PyMOO dual-DE with population-wide transfer of the best additive "
        "subspace direction."
    )

    transfer = parser.add_argument_group("Population transfer")
    transfer.add_argument(
        "--transfer_alpha",
        type=float,
        default=1.0,
        help=(
            "Scale applied to the anchor-learned direction before testing it "
            "on the full-space population: x_i + alpha * (x_sub_best - x_anchor)."
        ),
    )
    transfer.add_argument(
        "--transfer_every",
        type=int,
        default=1,
        help="Run population transfer every N dual-EA cycles.",
    )
    transfer.add_argument(
        "--transfer_fraction",
        type=float,
        default=1.0,
        help=(
            "Fraction of the full-space population tested by transfer. "
            "Use values below 1.0 to reduce per-cycle transfer NFE."
        ),
    )
    transfer.add_argument(
        "--transfer_selection",
        type=str,
        default="random",
        choices=["random", "best", "worst"],
        help=(
            "Which full-space individuals to test when transfer_fraction < 1.0. "
            "At transfer_fraction=1.0 this evaluates the whole population."
        ),
    )
    transfer.add_argument(
        "--transfer_alpha_mode",
        type=str,
        default="fixed",
        choices=["fixed", "adaptive"],
        help=(
            "Use a fixed transfer alpha or adapt it from the recent transfer "
            "success rate."
        ),
    )
    transfer.add_argument(
        "--transfer_success_target",
        type=float,
        default=0.2,
        help="Target transfer success rate used by --transfer_alpha_mode adaptive.",
    )
    transfer.add_argument(
        "--transfer_alpha_growth",
        type=float,
        default=1.2,
        help="Multiplier applied to alpha when adaptive transfer is above target.",
    )
    transfer.add_argument(
        "--transfer_alpha_shrink",
        type=float,
        default=0.7,
        help="Multiplier applied to alpha when adaptive transfer is below target.",
    )
    transfer.add_argument(
        "--transfer_alpha_min",
        type=float,
        default=1.0e-3,
        help="Lower bound for adaptive transfer alpha.",
    )
    transfer.add_argument(
        "--transfer_alpha_max",
        type=float,
        default=10.0,
        help="Upper bound for adaptive transfer alpha.",
    )
    transfer.add_argument(
        "--sub_pop_size",
        type=int,
        default=None,
        help="Population size for the subspace PyMOO DE. Defaults to --pop_size.",
    )

    return parser


def _sub_algorithm_args(args: argparse.Namespace) -> argparse.Namespace:
    """Build a PyMOO args namespace for the subspace optimizer."""
    sub_args = argparse.Namespace(**vars(args))
    sub_args.pop_size = _sub_pop_size(args)
    return sub_args


def _refresh_subspace_after_anchor_with_zero(
    sub_optim,
    sub_problem: SubspaceProblem,
    subspace,
    args: argparse.Namespace,
    *,
    mode: str,
) -> int:
    """Refresh subspace population and force the anchor-preserving zero vector.

    In additive assignment, ``z = 0`` expands to the current anchor exactly.
    Keeping it in the population prevents the subspace phase from starting with
    only directions that are worse under the new anchor.
    """
    n_var = subspace.search_dim
    pop_size = _sub_pop_size(args)
    zero_z = np.zeros(n_var, dtype=float)

    if mode == "resample":
        sampling = CenteredSampling(
            center=zero_z,
            method=args.init_pop,
            scale=args.pop_sigma,
        )
        X = sampling._do(sub_problem, pop_size)
        X[0] = zero_z
    elif mode in ("reeval", "reeval_with_zero_elite"):
        pop = sub_optim.pop
        if pop is None:
            sampling = CenteredSampling(
                center=zero_z,
                method=args.init_pop,
                scale=args.pop_sigma,
            )
            X = sampling._do(sub_problem, pop_size)
        else:
            X = np.asarray(pop.get("X"), dtype=float).copy()
        if mode == "reeval_with_zero_elite":
            X[0] = zero_z
    else:
        raise ValueError(f"Unknown sub_anchor_update mode {mode!r}")

    F = _evaluate_batch(sub_problem, X)
    n_eval = len(X)
    sub_optim.pop = _population_from_arrays(X, F)
    sub_optim.evaluator.n_eval += n_eval
    return n_eval


def _transfer_candidate_indices(
    F: np.ndarray,
    *,
    fraction: float,
    selection: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Choose which full-space individuals should spend transfer evaluations."""
    pop_size = len(F)
    if pop_size == 0:
        return np.empty(0, dtype=int)

    n_candidates = max(1, int(np.ceil(pop_size * fraction)))
    n_candidates = min(pop_size, n_candidates)
    if n_candidates == pop_size:
        return np.arange(pop_size, dtype=int)

    if selection == "random":
        return np.asarray(rng.choice(pop_size, size=n_candidates, replace=False), dtype=int)
    order = np.argsort(F)
    if selection == "best":
        return np.asarray(order[:n_candidates], dtype=int)
    if selection == "worst":
        return np.asarray(order[-n_candidates:], dtype=int)
    raise ValueError(f"Unknown transfer_selection {selection!r}")


def _adapt_transfer_alpha(
    alpha: float,
    transfer: TransferResult,
    args: argparse.Namespace,
) -> float:
    """Simple success-rate controller for the transfer step size."""
    if args.transfer_alpha_mode != "adaptive" or transfer.candidate_count == 0:
        return alpha

    factor = (
        args.transfer_alpha_growth
        if transfer.success_rate >= args.transfer_success_target
        else args.transfer_alpha_shrink
    )
    next_alpha = alpha * factor
    return float(np.clip(next_alpha, args.transfer_alpha_min, args.transfer_alpha_max))


def _transfer_direction_to_full_population(
    full_optim,
    full_problem: SubspaceProblem,
    direction: np.ndarray,
    candidate_indices: np.ndarray,
) -> TransferResult:
    """Apply one common direction to selected full-space individuals."""
    pop = full_optim.pop
    X = np.asarray(pop.get("X"), dtype=float).copy()
    F = np.asarray(pop.get("F"), dtype=float).reshape(-1)
    pop_size = len(X)
    candidate_indices = np.asarray(candidate_indices, dtype=int).reshape(-1)
    candidate_count = len(candidate_indices)

    if candidate_count == 0:
        return TransferResult(
            success_count=0,
            candidate_count=0,
            population_size=pop_size,
            eval_count=0,
            best_x=None,
            best_f=None,
            direction_norm=float(np.linalg.norm(direction)),
        )

    xl, xu = full_problem.bounds()
    raw_candidates = X[candidate_indices] + direction.reshape(1, -1)
    candidates = np.clip(raw_candidates, xl, xu)
    candidate_F = _evaluate_batch(full_problem, candidates).reshape(-1)
    full_optim.evaluator.n_eval += candidate_count

    old_F = F[candidate_indices]
    delta = old_F - candidate_F
    improved = candidate_F < old_F
    success_count = int(np.count_nonzero(improved))
    clipped_rows = np.any(~np.isclose(raw_candidates, candidates), axis=1)
    clip_fraction = float(np.mean(clipped_rows)) if candidate_count > 0 else 0.0
    mean_delta = float(np.mean(delta)) if candidate_count > 0 else 0.0
    best_delta = float(np.max(delta)) if candidate_count > 0 else 0.0

    if success_count == 0:
        return TransferResult(
            success_count=0,
            candidate_count=candidate_count,
            population_size=pop_size,
            eval_count=candidate_count,
            best_x=None,
            best_f=None,
            mean_delta=mean_delta,
            best_delta=best_delta,
            direction_norm=float(np.linalg.norm(direction)),
            clip_fraction=clip_fraction,
        )

    improved_indices = candidate_indices[improved]
    X[improved_indices] = candidates[improved]
    F[improved_indices] = candidate_F[improved]
    full_optim.pop = _population_from_arrays(X, F.reshape(-1, 1))

    best_idx = int(np.argmin(F))
    return TransferResult(
        success_count=success_count,
        candidate_count=candidate_count,
        population_size=pop_size,
        eval_count=candidate_count,
        best_x=X[best_idx].copy(),
        best_f=float(F[best_idx]),
        mean_delta=mean_delta,
        best_delta=best_delta,
        direction_norm=float(np.linalg.norm(direction)),
        clip_fraction=clip_fraction,
    )


class TransferDualEALoggingCallback(LoggingCallback):
    """Log dual-EA metrics plus population transfer success statistics."""

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

    def notify(self, optim, **kwargs) -> None:
        self._cycle += 1
        if self._cycle % self._log_every != 0:
            return

        full_optim = kwargs["full_optim"]
        sub_optim = kwargs["sub_optim"]
        full_pop = full_optim.pop
        sub_pop = sub_optim.pop
        full_F = full_pop.get("F").flatten()
        sub_F = sub_pop.get("F").flatten()
        full_X = full_pop.get("X")
        sub_X = sub_pop.get("X")

        transfer_success_total = int(kwargs.get("transfer_success_total", 0))
        transfer_candidate_total = int(kwargs.get("transfer_candidate_total", 0))
        transfer_success_rate_avg = (
            transfer_success_total / transfer_candidate_total
            if transfer_candidate_total > 0
            else 0.0
        )

        metrics = {
            "generation": int(full_optim.n_gen),
            "nfe": _total_nfe(full_optim, sub_optim),
            "best_fitness": float(min(full_F.min(), sub_F.min())),
            "mean_fitness": float(full_F.mean()),
            "center_fitness": self._compute_center_fitness(full_X),
            "cycle": self._cycle,
            "full_best_fitness": float(full_F.min()),
            "full_mean_fitness": float(full_F.mean()),
            "full_center_fitness": self._compute_center_fitness(full_X),
            "sub_best_fitness": float(sub_F.min()),
            "sub_mean_fitness": float(sub_F.mean()),
            "sub_center_fitness": self._center_fitness_for(sub_X, self._sub_subspace),
            "full_improve_count": int(kwargs.get("full_improve_count", 0)),
            "sub_improve_count": int(kwargs.get("subspace_improve_count", 0)),
            "subspace_improve_count": int(kwargs.get("subspace_improve_count", 0)),
            "transfer_improve_count": int(kwargs.get("transfer_improve_count", 0)),
            "transfer_attempt_count": int(kwargs.get("transfer_attempt_count", 0)),
            "transfer_alpha": float(kwargs.get("transfer_alpha", 1.0)),
            "transfer_success_count": int(kwargs.get("transfer_success_count", 0)),
            "transfer_candidate_count": int(kwargs.get("transfer_candidate_count", 0)),
            "transfer_population_size": int(kwargs.get("transfer_population_size", 0)),
            "transfer_success_rate": float(kwargs.get("transfer_success_rate", 0.0)),
            "transfer_eval_count": int(kwargs.get("transfer_eval_count", 0)),
            "transfer_mean_delta": float(kwargs.get("transfer_mean_delta", 0.0)),
            "transfer_best_delta": float(kwargs.get("transfer_best_delta", 0.0)),
            "transfer_direction_norm": float(kwargs.get("transfer_direction_norm", 0.0)),
            "transfer_clip_fraction": float(kwargs.get("transfer_clip_fraction", 0.0)),
            "transfer_success_total": transfer_success_total,
            "transfer_candidate_total": transfer_candidate_total,
            "transfer_eval_total": int(kwargs.get("transfer_eval_total", 0)),
            "transfer_success_rate_avg": float(transfer_success_rate_avg),
            "full_step_nfe_total": int(kwargs.get("full_step_nfe_total", 0)),
            "sub_refresh_nfe_total": int(kwargs.get("sub_refresh_nfe_total", 0)),
            "sub_step_nfe_total": int(kwargs.get("sub_step_nfe_total", 0)),
            "transfer_nfe_total": int(kwargs.get("transfer_nfe_total", 0)),
        }
        if metrics["generation"] % 1000 == 0:
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


def init_wandb_transfer(args: argparse.Namespace) -> None:
    import wandb  # type: ignore

    eff_d = effective_subspace_param(args)
    sub_pop = _sub_pop_size(args)
    if args.wandb_name in (None, "", "__auto__"):
        args.wandb_name = f"{args.problem}-xfer"
        if subspace_method_is_lora(args.subspace_method):
            args.wandb_name += f"-r{args.lora_rank}"
            if subspace_method_is_block_lora(args.subspace_method):
                args.wandb_name += f"-b{args.lora_blocks}"
        else:
            args.wandb_name += f"-d{eff_d}"
        args.wandb_name += f"-a{args.transfer_alpha}-s{args.seed}"

    if args.wandb_group:
        placeholder_pattern = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
        placeholder_values = vars(args).copy()
        placeholder_values["subspace_dim"] = eff_d
        placeholder_values["sub_pop_size"] = sub_pop
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
    config["approach"] = "dual_ea_transfer"
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


def run_dual_ea_transfer(
    *,
    full_optim,
    sub_optim,
    full_problem: SubspaceProblem,
    sub_problem: SubspaceProblem,
    full_subspace,
    sub_subspace,
    args: argparse.Namespace,
    max_nfe: int,
    callback: TransferDualEALoggingCallback | None = None,
) -> TransferRunResult:
    """Run alternating dual-DE cycles with population transfer."""
    global_best_f = float("inf")
    global_best_x: np.ndarray | None = None
    n_cycles = 0
    full_improve_count = 0
    subspace_improve_count = 0
    transfer_improve_count = 0
    transfer_attempt_count = 0
    transfer_success_total = 0
    transfer_candidate_total = 0
    transfer_eval_total = 0
    full_step_nfe_total = 0
    sub_refresh_nfe_total = 0
    sub_step_nfe_total = 0
    transfer_alpha = float(args.transfer_alpha)
    transfer_rng = np.random.default_rng(args.seed + 7919)

    while full_optim.has_next() and _budget_left(full_optim, sub_optim, max_nfe) > 0:
        n_cycles += 1
        global_best_at_cycle_start = global_best_f

        if _budget_left(full_optim, sub_optim, max_nfe) < args.pop_size:
            break
        nfe_before_full = _total_nfe(full_optim, sub_optim)
        full_ran = _advance_generations(
            full_optim,
            n_gens=args.full_iters,
            pop_size=args.pop_size,
            full_optim=full_optim,
            sub_optim=sub_optim,
            max_nfe=max_nfe,
        )
        if full_ran == 0:
            break
        full_step_eval_count = _total_nfe(full_optim, sub_optim) - nfe_before_full
        full_step_nfe_total += full_step_eval_count

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
        nfe_before_refresh = _total_nfe(full_optim, sub_optim)
        _refresh_subspace_after_anchor_with_zero(
            sub_optim,
            sub_problem,
            sub_subspace,
            args,
            mode=args.sub_anchor_update,
        )
        sub_refresh_eval_count = (
            _total_nfe(full_optim, sub_optim) - nfe_before_refresh
        )
        sub_refresh_nfe_total += sub_refresh_eval_count

        sub_pop = _sub_pop_size(args)
        if _budget_left(full_optim, sub_optim, max_nfe) < sub_pop:
            break
        nfe_before_sub = _total_nfe(full_optim, sub_optim)
        sub_ran = _advance_generations(
            sub_optim,
            n_gens=args.sub_iters,
            pop_size=sub_pop,
            full_optim=full_optim,
            sub_optim=sub_optim,
            max_nfe=max_nfe,
        )
        if sub_ran == 0:
            break
        sub_step_eval_count = _total_nfe(full_optim, sub_optim) - nfe_before_sub
        sub_step_nfe_total += sub_step_eval_count

        global_best_before_sub = global_best_after_full
        best_x_sub, f_sub, _ = _best_subspace_solution(sub_optim, sub_subspace)
        global_best_x, global_best_f = _track_best(
            best_x_sub, f_sub, global_best_x, global_best_f
        )
        if global_best_f < global_best_before_sub:
            subspace_improve_count += 1

        transfer = TransferResult(0, 0, 0, 0, None, None)
        if n_cycles % args.transfer_every == 0 and _budget_left(
            full_optim, sub_optim, max_nfe
        ) > 0:
            full_F = np.asarray(full_optim.pop.get("F"), dtype=float).reshape(-1)
            candidate_indices = _transfer_candidate_indices(
                full_F,
                fraction=args.transfer_fraction,
                selection=args.transfer_selection,
                rng=transfer_rng,
            )
            budget = _budget_left(full_optim, sub_optim, max_nfe)
            if len(candidate_indices) > budget:
                candidate_indices = candidate_indices[:budget]

            direction = transfer_alpha * (best_x_sub - best_x_full)
            global_best_before_transfer = global_best_f
            transfer = _transfer_direction_to_full_population(
                full_optim,
                full_problem,
                direction,
                candidate_indices,
            )
            transfer_attempt_count += 1
            transfer_success_total += transfer.success_count
            transfer_candidate_total += transfer.candidate_count
            transfer_eval_total += transfer.eval_count
            if transfer.best_x is not None and transfer.best_f is not None:
                global_best_x, global_best_f = _track_best(
                    transfer.best_x, transfer.best_f, global_best_x, global_best_f
                )
            if global_best_f < global_best_before_transfer:
                transfer_improve_count += 1
            transfer_alpha = _adapt_transfer_alpha(transfer_alpha, transfer, args)

        if callback is not None:
            callback.notify(
                full_optim,
                full_optim=full_optim,
                sub_optim=sub_optim,
                full_improve_count=full_improve_count,
                subspace_improve_count=subspace_improve_count,
                transfer_improve_count=transfer_improve_count,
                transfer_attempt_count=transfer_attempt_count,
                transfer_alpha=transfer_alpha,
                transfer_success_count=transfer.success_count,
                transfer_candidate_count=transfer.candidate_count,
                transfer_population_size=transfer.population_size,
                transfer_success_rate=transfer.success_rate,
                transfer_eval_count=transfer.eval_count,
                transfer_mean_delta=transfer.mean_delta,
                transfer_best_delta=transfer.best_delta,
                transfer_direction_norm=transfer.direction_norm,
                transfer_clip_fraction=transfer.clip_fraction,
                transfer_success_total=transfer_success_total,
                transfer_candidate_total=transfer_candidate_total,
                transfer_eval_total=transfer_eval_total,
                full_step_nfe_total=full_step_nfe_total,
                sub_refresh_nfe_total=sub_refresh_nfe_total,
                sub_step_nfe_total=sub_step_nfe_total,
                transfer_nfe_total=transfer_eval_total,
            )

    if global_best_x is None:
        global_best_x, global_best_f = _best_fullspace_solution(
            full_optim, full_subspace
        )

    return TransferRunResult(
        best_x=global_best_x,
        best_f=global_best_f,
        n_cycles=n_cycles,
        total_nfe=_total_nfe(full_optim, sub_optim),
        full_improve_count=full_improve_count,
        subspace_improve_count=subspace_improve_count,
        transfer_improve_count=transfer_improve_count,
        transfer_attempt_count=transfer_attempt_count,
        transfer_success_total=transfer_success_total,
        transfer_candidate_total=transfer_candidate_total,
        transfer_eval_total=transfer_eval_total,
        full_step_nfe_total=full_step_nfe_total,
        sub_refresh_nfe_total=sub_refresh_nfe_total,
        sub_step_nfe_total=sub_step_nfe_total,
    )


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.subspace_method == "none":
        args.subspace_method = "fullspace"

    if args.optimizer != "de":
        parser.error("main_dual_ea_transfer.py is intended for dual PyMOO DEs; use --optimizer de")

    if args.transfer_alpha <= 0:
        parser.error("--transfer_alpha must be > 0")

    if args.transfer_every < 1:
        parser.error("--transfer_every must be >= 1")

    if not (0.0 < args.transfer_fraction <= 1.0):
        parser.error("--transfer_fraction must be in (0, 1]")

    if not (0.0 <= args.transfer_success_target <= 1.0):
        parser.error("--transfer_success_target must be in [0, 1]")

    if args.transfer_alpha_growth <= 0 or args.transfer_alpha_shrink <= 0:
        parser.error("--transfer_alpha_growth and --transfer_alpha_shrink must be > 0")

    if args.transfer_alpha_min <= 0 or args.transfer_alpha_max <= 0:
        parser.error("--transfer_alpha_min and --transfer_alpha_max must be > 0")

    if args.transfer_alpha_min > args.transfer_alpha_max:
        parser.error("--transfer_alpha_min must be <= --transfer_alpha_max")

    if args.sub_pop_size is not None and args.sub_pop_size < 1:
        parser.error("--sub_pop_size must be >= 1")

    if args.fullspace_assignment != "absolute":
        parser.error("Population transfer requires --fullspace_assignment absolute")

    if subspace_method_is_fullspace(args.subspace_method):
        parser.error(
            "Dual-EA requires a reduced subspace for the subspace EA; "
            "choose random_projection, random_blocking, or lora."
        )

    if subspace_method_is_lora(args.subspace_method) and args.lora_rank is None:
        parser.error("--lora_rank is required when the subspace method uses LoRA")

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
            "Warning: dual-EA transfer is designed for additive subspace assignment "
            f"(x = x0 + f(z)); got {args.subspace_assignment!r}."
        )

    if args.full_iters < 1 or args.sub_iters < 1:
        parser.error("--full_iters and --sub_iters must be >= 1")

    np.random.seed(args.seed)

    if args.wandb:
        init_wandb_transfer(args)

    eff_sub = effective_subspace_param(args)
    sub_pop = _sub_pop_size(args)
    print("=" * 70)
    print("Dual-DE Population Transfer")
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
    print(f"  Optimizer         : PyMOO DE (full_pop={args.pop_size}, sub_pop={sub_pop})")
    print(f"  Cycle schedule    : {args.full_iters}+{args.sub_iters} (full+sub gens)")
    print(f"  Transfer alpha    : {args.transfer_alpha} ({args.transfer_alpha_mode})")
    print(
        f"  Transfer policy   : every {args.transfer_every} cycle(s), "
        f"fraction={args.transfer_fraction}, selection={args.transfer_selection}"
    )
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
    sub_optim = build_algorithm(_sub_algorithm_args(args))

    _set_optim_sampling(
        sub_optim,
        CenteredSampling(
            center=np.zeros(sub_subspace.search_dim, dtype=float),
            method=args.init_pop,
            scale=args.pop_sigma,
        ),
    )

    from pymoo.termination import get_termination

    gen_cap = max(1, args.max_nfe // max(1, max(args.pop_size, sub_pop)) + 2)
    full_optim.setup(
        full_problem,
        termination=get_termination("n_gen", gen_cap),
        seed=args.seed,
        verbose=False,
    )
    sub_optim.setup(
        sub_problem,
        termination=get_termination("n_gen", gen_cap),
        seed=args.seed + 1,
        verbose=False,
    )

    callback = TransferDualEALoggingCallback(
        eval_fn=lsgo.evaluate,
        full_subspace=full_subspace,
        sub_subspace=sub_subspace,
        use_wandb=args.wandb,
        log_every=args.log_every,
    )

    t0 = time.perf_counter()
    result = run_dual_ea_transfer(
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
    print(f"Dual-DE transfer finished in {elapsed:.2f}s")
    print(f"  Cycles completed     : {result.n_cycles}")
    print(f"  Full improved best   : {result.full_improve_count} cycles")
    print(f"  Subspace improved    : {result.subspace_improve_count} cycles")
    print(f"  Transfer improved    : {result.transfer_improve_count} cycles")
    print(f"  Transfer attempts    : {result.transfer_attempt_count}")
    print(
        f"  Transfer successes   : "
        f"{result.transfer_success_total}/{result.transfer_candidate_total}"
    )
    print(f"  Avg transfer success : {result.transfer_success_rate_avg:.4f}")
    print(f"  Full-step NFE        : {result.full_step_nfe_total}")
    print(f"  Sub-refresh NFE      : {result.sub_refresh_nfe_total}")
    print(f"  Sub-step NFE         : {result.sub_step_nfe_total}")
    print(f"  Transfer NFE         : {result.transfer_eval_total}")
    print(f"  Best fitness         : {result.best_f:.6e}")
    print(f"  Total NFE            : {result.total_nfe}")
    print(f"  ||best_x||_2         : {float(np.linalg.norm(result.best_x)):.4f}")
    if lsgo.optimum is not None:
        gap = result.best_f - lsgo.optimum
        print(f"  Gap to optimum       : {gap:.6e}")
    print("=" * 70)

    if args.wandb:
        import wandb  # type: ignore

        wandb.summary["best_fitness"] = result.best_f
        wandb.summary["total_nfe"] = result.total_nfe
        wandb.summary["n_cycles"] = result.n_cycles
        wandb.summary["full_improve_count"] = result.full_improve_count
        wandb.summary["sub_improve_count"] = result.subspace_improve_count
        wandb.summary["subspace_improve_count"] = result.subspace_improve_count
        wandb.summary["transfer_improve_count"] = result.transfer_improve_count
        wandb.summary["transfer_attempt_count"] = result.transfer_attempt_count
        wandb.summary["transfer_success_total"] = result.transfer_success_total
        wandb.summary["transfer_candidate_total"] = result.transfer_candidate_total
        wandb.summary["transfer_eval_total"] = result.transfer_eval_total
        wandb.summary["transfer_success_rate_avg"] = (
            result.transfer_success_rate_avg
        )
        wandb.summary["full_step_nfe_total"] = result.full_step_nfe_total
        wandb.summary["sub_refresh_nfe_total"] = result.sub_refresh_nfe_total
        wandb.summary["sub_step_nfe_total"] = result.sub_step_nfe_total
        wandb.summary["elapsed_seconds"] = elapsed
        wandb.finish()


if __name__ == "__main__":
    main()
