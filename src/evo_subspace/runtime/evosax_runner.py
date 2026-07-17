"""Ask-eval-tell optimization loop for evosax-backed subspace optimizers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

from evo_subspace.runtime.problem import SubspaceProblem

if TYPE_CHECKING:
    from evo_subspace.optimizers.evosax_builder import EvosaxSubspaceOptimizer


def _evaluate_batch(problem: SubspaceProblem, X: np.ndarray) -> np.ndarray:
    out: dict = {}
    problem._evaluate(X, out)
    return np.asarray(out["F"], dtype=float).reshape(-1)


def setup_evosax_optimizer(
    args,
    problem: SubspaceProblem,
    *,
    search_dim: int,
    max_nfe: int,
    seed: int | None = None,
) -> EvosaxSubspaceOptimizer:
    """Build and initialize an evosax optimizer on ``problem``."""
    from evo_subspace.optimizers.evosax_builder import build_evosax_optimizer

    optim = build_evosax_optimizer(
        args,
        search_dim=search_dim,
        gen_cap=max_nfe,
    )
    optim.setup(problem, seed=seed if seed is not None else args.seed, verbose=False)

    if optim.is_distribution_based:
        optim.initialize_distribution(np.zeros(search_dim, dtype=float))
        return optim

    from evo_subspace.optimizers import build_sampling

    sampling = build_sampling(args.init_pop, gaussian_scale=args.pop_sigma)
    pop_size = optim._pop_size
    X = sampling._do(problem, pop_size)
    F = _evaluate_batch(problem, X)
    optim.initialize_population(X, F)
    optim.evaluator.n_eval += pop_size
    return optim


def run_evosax_optimization(
    optim: EvosaxSubspaceOptimizer,
    problem: SubspaceProblem,
    *,
    max_nfe: int,
    on_generation: Callable[[EvosaxSubspaceOptimizer], None] | None = None,
) -> tuple[float, int]:
    """Run ask-eval-tell until the NFE budget is exhausted."""
    while optim.evaluator.n_eval < max_nfe and optim.has_next():
        population = optim.ask()
        fitness = _evaluate_batch(problem, population).reshape(-1)
        optim.evaluator.n_eval += len(fitness)
        optim.tell(fitness)
        if on_generation is not None:
            on_generation(optim)

    if optim._state is None:
        raise RuntimeError("evosax optimizer state was not initialized")
    return float(optim._state.best_fitness), int(optim.evaluator.n_eval)
