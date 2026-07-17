"""Factory and PyMOO-compatible wrapper for evosax subspace optimizers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from evosax.algorithms import algorithms as EVOSAX_REGISTRY

from evo_subspace.runtime.problem import SubspaceProblem


POPULATION_BASED = {
    "DifferentialEvolution",
    "DiffusionEvolution",
    "GESMR_GA",
    "LGA",
    "MR15_GA",
    "PSO",
    "SAMR_GA",
    "SimpleGA",
}

EVEN_POPULATION_REQUIRED = {"Open_ES", "SV_Open_ES"}


def _optimizer_lookup() -> dict[str, tuple[Any, str]]:
    lookup: dict[str, tuple[Any, str]] = {}
    for canonical, cls in EVOSAX_REGISTRY.items():
        lookup[canonical.lower()] = (cls, canonical)
    aliases = {
        "de": "DifferentialEvolution",
        "differential_evolution": "DifferentialEvolution",
        "ga": "SimpleGA",
        "simple_ga": "SimpleGA",
        "cmaes": "CMA_ES",
    }
    for alias, canonical in aliases.items():
        if canonical in EVOSAX_REGISTRY:
            lookup[alias] = (EVOSAX_REGISTRY[canonical], canonical)
    return lookup


def evosax_optimizer_choices() -> list[str]:
    """CLI-friendly lowercase names for all registered evosax algorithms."""
    return sorted(_optimizer_lookup())


def _resolve_algorithm_class(name: str):
    lookup = _optimizer_lookup()
    key = name.strip().lower()
    if key in lookup:
        return lookup[key]
    raise ValueError(
        f"Unknown evosax optimizer {name!r}. "
        f"Choose from: {', '.join(evosax_optimizer_choices())}."
    )


@dataclass
class _PopArrays:
    """Minimal population view compatible with ``Population.get``."""

    X: np.ndarray
    F: np.ndarray

    def get(self, key: str) -> np.ndarray:
        if key == "X":
            return self.X
        if key == "F":
            return self.F
        raise KeyError(key)


class _Evaluator:
    def __init__(self) -> None:
        self.n_eval = 0


class EvosaxSubspaceOptimizer:
    """Thin adapter exposing a pymoo-like interface over an evosax ES."""

    def __init__(
        self,
        *,
        es: Any,
        algorithm_name: str,
        params: Any,
        pop_size: int,
        seed: int,
        gen_cap: int,
    ) -> None:
        self._es = es
        self._algorithm_name = algorithm_name
        self._params = params
        self._pop_size = pop_size
        self._key = jax.random.key(seed)
        self._gen_cap = gen_cap
        self._population_based = algorithm_name in POPULATION_BASED
        self._problem: SubspaceProblem | None = None
        self._state: Any = None
        self.evaluator = _Evaluator()
        self.n_gen = 0
        self._pop: _PopArrays | None = None
        self._pending_population: Any = None
        self._tell_key: jax.Array | None = None

    @property
    def is_distribution_based(self) -> bool:
        """True for ES variants (CMA-ES, Open-ES, …) that track a distribution, not a pop."""
        return not self._population_based

    @property
    def pop(self) -> _PopArrays | None:
        """Last evaluated batch (offspring) or single mean point for logging only."""
        return self._pop

    @pop.setter
    def pop(self, value: Any) -> None:
        if self.is_distribution_based:
            raise TypeError(
                f"{self._algorithm_name} is distribution-based and does not use a "
                "stored population; use refresh_after_anchor() instead."
            )
        if hasattr(value, "get"):
            X = np.asarray(value.get("X"), dtype=float)
            F = np.asarray(value.get("F"), dtype=float)
        else:
            X = np.asarray(value.get("X") if hasattr(value, "get") else value.X, dtype=float)
            F = np.asarray(value.get("F"), dtype=float)
        self._set_population(X, F, sync_state=True)

    def get_mean_z(self) -> np.ndarray:
        """Current search center in subspace (distribution mean or pop centroid)."""
        if self._state is None:
            raise RuntimeError("Optimizer state is not initialized.")
        if self.is_distribution_based and hasattr(self._es, "get_mean"):
            return np.asarray(self._es.get_mean(self._state), dtype=float).reshape(-1)
        if self._pop is not None:
            return np.asarray(self._pop.X.mean(axis=0), dtype=float).reshape(-1)
        raise RuntimeError("No mean available before initialization.")

    def anchor_refresh_cost(self, mode: str) -> int:
        """NFE cost of refreshing a distribution-based ES after an anchor update."""
        if mode == "resample":
            return 0
        if mode in ("reeval", "reeval_with_zero_elite"):
            return 1
        raise ValueError(f"Unknown anchor refresh mode {mode!r}")

    def initialize_distribution(self, mean_z: np.ndarray | None = None) -> None:
        """Init a distribution-based ES at ``mean_z`` (default z=0). No evaluations."""
        if not self.is_distribution_based:
            raise TypeError(
                f"{self._algorithm_name} is population-based; use initialize_population()."
            )
        if self._problem is None:
            raise RuntimeError("Call setup() before initialize_distribution().")
        if mean_z is None:
            mean_z = np.zeros(self._es.num_dims, dtype=float)
        mean_z = np.asarray(mean_z, dtype=float).reshape(-1)
        key, subkey = jax.random.split(self._key)
        self._key = key
        mean_j = jnp.asarray(mean_z, dtype=float)
        self._state = self._es.init(subkey, mean_j, self._params)
        self.n_gen = 0
        self._pending_population = None
        self._pop = None

    def refresh_after_anchor(
        self,
        problem: SubspaceProblem,
        *,
        mode: str,
        center_z: np.ndarray,
    ) -> int:
        """Re-init distribution-based ES after an anchor / center update."""
        if not self.is_distribution_based:
            raise TypeError(
                f"{self._algorithm_name} is population-based; use pop refresh instead."
            )
        center_z = np.asarray(center_z, dtype=float).reshape(-1)
        if mode == "resample":
            self.initialize_distribution(center_z)
            return 0
        if mode in ("reeval", "reeval_with_zero_elite"):
            f = float(_evaluate_batch(problem, center_z.reshape(1, -1))[0, 0])
            key, subkey = jax.random.split(self._key)
            self._key = key
            mean_j = jnp.asarray(center_z, dtype=float)
            self._state = self._es.init(subkey, mean_j, self._params)
            self._state = self._state.replace(
                best_solution=self._es._ravel_solution(mean_j),
                best_fitness=f,
            )
            self._pop = _PopArrays(
                X=center_z.reshape(1, -1),
                F=np.array([[f]], dtype=float),
            )
            return 1
        raise ValueError(f"Unknown anchor refresh mode {mode!r}")

    def setup(
        self,
        problem: SubspaceProblem,
        *,
        termination: Any = None,
        seed: int | None = None,
        verbose: bool = False,
    ) -> None:
        del verbose
        self._problem = problem
        if termination is not None and hasattr(termination, "n_max_gen"):
            self._gen_cap = int(termination.n_max_gen)
        if seed is not None:
            self._key = jax.random.key(seed)

    def has_next(self) -> bool:
        return self.n_gen < self._gen_cap

    def initialize_population(self, X: np.ndarray, F: np.ndarray) -> None:
        """Seed a population-based ES from an evaluated population."""
        if self.is_distribution_based:
            raise TypeError(
                f"{self._algorithm_name} is distribution-based; use initialize_distribution()."
            )
        self._set_population(X, F, sync_state=True)
        self.n_gen = 0
        self._pending_population = None

    def ask(self) -> np.ndarray:
        """Generate candidate solutions in subspace (z) to evaluate."""
        if self._problem is None or self._state is None:
            raise RuntimeError(
                "Call setup() and initialize_distribution() or "
                "initialize_population() before ask()."
            )

        key, key_ask, key_tell = jax.random.split(self._key, 3)
        self._key = key
        self._tell_key = key_tell
        population, self._state = self._es.ask(key_ask, self._state, self._params)
        self._pending_population = population
        return np.asarray(population, dtype=float)

    def tell(self, fitness: np.ndarray) -> None:
        """Update evosax state from evaluated fitness."""
        if self._pending_population is None:
            raise RuntimeError("Call ask() before tell().")

        fitness = np.asarray(fitness, dtype=float).reshape(-1)
        self._state, _metrics = self._es.tell(
            self._tell_key,
            self._pending_population,
            jnp.asarray(fitness),
            self._state,
            self._params,
        )
        pop_np = np.asarray(self._pending_population, dtype=float)
        self._pop = _PopArrays(X=pop_np, F=fitness.reshape(-1, 1))
        self.n_gen += 1
        self._pending_population = None

    def next(self) -> None:
        """One ask-eval-tell generation."""
        population = self.ask()
        fitness = _evaluate_batch(self._problem, population).reshape(-1)
        self.evaluator.n_eval += len(fitness)
        self.tell(fitness)

    def _set_population(
        self,
        X: np.ndarray,
        F: np.ndarray,
        *,
        sync_state: bool,
    ) -> None:
        X = np.asarray(X, dtype=float)
        F = np.asarray(F, dtype=float).reshape(-1, 1)
        self._pop = _PopArrays(X=X, F=F)
        if sync_state:
            self._sync_state_from_pop()

    def _sync_state_from_pop(self) -> None:
        if self._pop is None or self.is_distribution_based:
            return
        X = self._pop.X
        F = self._pop.F.reshape(-1)
        key, subkey = jax.random.split(self._key)
        self._key = key
        self._state = self._es.init(
            subkey,
            jnp.asarray(X),
            jnp.asarray(F),
            self._params,
        )


def _evaluate_batch(problem: SubspaceProblem, X: np.ndarray) -> np.ndarray:
    out: dict = {}
    problem._evaluate(X, out)
    return np.asarray(out["F"], dtype=float).reshape(-1, 1)


def _evosax_optimizer_name(args) -> str:
    """Resolve evosax algorithm id from CLI args."""
    sub_optimizer = getattr(args, "sub_optimizer", None)
    if sub_optimizer is not None:
        return sub_optimizer
    optimizer = getattr(args, "optimizer", "cmaes")
    if optimizer == "cmaes":
        return "cmaes"
    return optimizer


def build_evosax_optimizer(args, *, search_dim: int, gen_cap: int) -> EvosaxSubspaceOptimizer:
    """Construct an evosax-backed subspace optimizer from CLI args."""
    pop_size = (
        args.sub_pop_size
        if getattr(args, "sub_pop_size", None) is not None
        else args.pop_size
    )
    cls, algorithm_name = _resolve_algorithm_class(_evosax_optimizer_name(args))

    if algorithm_name in EVEN_POPULATION_REQUIRED and pop_size % 2 != 0:
        raise ValueError(
            f"{algorithm_name} requires an even population size; got {pop_size}."
        )

    solution = jnp.zeros(search_dim, dtype=jnp.float32)
    lr_steps = max(1, int(getattr(args, "max_nfe", gen_cap)) // int(pop_size))
    es = _instantiate_algorithm(
        cls, algorithm_name, pop_size, solution, args, lr_steps=lr_steps
    )
    params = _build_params(es, algorithm_name, args)

    return EvosaxSubspaceOptimizer(
        es=es,
        algorithm_name=algorithm_name,
        params=params,
        pop_size=pop_size,
        seed=getattr(args, "seed", 0) + 1,
        gen_cap=gen_cap,
    )


def _learning_rate(args, *, lr_steps: int):
    """Scalar LR or optax schedule from CLI args."""
    lr = float(getattr(args, "es_lr", 1e-3))
    schedule = str(getattr(args, "es_lr_schedule", "constant")).strip().lower()
    if schedule == "constant":
        return lr
    if schedule == "cosine":
        return optax.cosine_decay_schedule(
            init_value=lr,
            decay_steps=max(1, int(lr_steps)),
            alpha=0.0,
        )
    raise ValueError(
        f"Unknown es_lr_schedule {schedule!r}. Choose from: constant, cosine."
    )


def _optax_optimizer(args, *, lr_steps: int) -> optax.GradientTransformation:
    """Build the mean-update optimizer for Open-ES / SNES / xNES."""
    lr = _learning_rate(args, lr_steps=lr_steps)
    name = str(getattr(args, "es_opt", "adam")).strip().lower()
    if name == "adam":
        return optax.adam(learning_rate=lr)
    if name == "sgd":
        return optax.sgd(learning_rate=lr)
    raise ValueError(f"Unknown es_opt {name!r}. Choose from: adam, sgd.")


def _instantiate_algorithm(
    cls, algorithm_name: str, pop_size: int, solution, args, *, lr_steps: int
):
    if algorithm_name == "Open_ES":
        return cls(
            population_size=pop_size,
            solution=solution,
            optimizer=_optax_optimizer(args, lr_steps=lr_steps),
            std_schedule=optax.constant_schedule(args.es_sigma),
        )
    if algorithm_name == "SV_Open_ES":
        return cls(
            population_size=pop_size,
            solution=solution,
            optimizer=_optax_optimizer(args, lr_steps=lr_steps),
            std_schedule=optax.constant_schedule(args.es_sigma),
        )
    if algorithm_name in ("SNES", "xNES"):
        return cls(
            population_size=pop_size,
            solution=solution,
            optimizer=_optax_optimizer(args, lr_steps=lr_steps),
        )
    return cls(population_size=pop_size, solution=solution)


def _build_params(es, algorithm_name: str, args):
    params = es.default_params
    if algorithm_name in ("CMA_ES", "Sep_CMA_ES", "SV_CMA_ES"):
        return params.replace(std_init=args.cmaes_sigma)
    if algorithm_name in ("SNES", "xNES"):
        return params.replace(std_init=args.es_sigma)
    return params
