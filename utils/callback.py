"""PyMOO callback for W&B (and console) logging."""

from __future__ import annotations

from typing import Callable

import numpy as np
from pymoo.core.callback import Callback


class LoggingCallback(Callback):
    """Logs per-generation statistics to W&B and/or the console.

    Logged metrics (at every generation):
    - **best_fitness**   : minimum F in the current population.
    - **mean_fitness**   : mean F over the current population.
    - **center_fitness** : F evaluated at the population centroid (mean X).
                           This is an extra evaluation *not* counted toward
                           the NFE budget tracked by PyMOO's evaluator.
    - **nfe**            : cumulative number of function evaluations.

    Parameters
    ----------
    eval_fn : Callable[[np.ndarray], float]
        A function that maps a full-space solution (shape D) to a scalar
        fitness value.  Used to compute the centroid fitness.
    subspace : Subspace
        The active subspace; used to map the search-space centroid back to
        the full space before calling eval_fn.
    use_wandb : bool
        Whether to log to W&B.
    log_every : int
        Log every N generations (default 1 = every generation).
    """

    def __init__(
        self,
        eval_fn: Callable[[np.ndarray], float],
        subspace,
        use_wandb: bool = False,
        log_every: int = 1,
    ) -> None:
        super().__init__()
        self._eval_fn = eval_fn
        self._subspace = subspace
        self._use_wandb = use_wandb
        self._log_every = log_every

    # ------------------------------------------------------------------
    # PyMOO hook
    # ------------------------------------------------------------------

    def notify(self, algorithm, **kwargs) -> None:
        gen: int = algorithm.n_gen
        if gen % self._log_every != 0:
            return

        pop = algorithm.pop
        F: np.ndarray = pop.get("F").flatten()  # (pop_size,)
        X: np.ndarray = pop.get("X")            # (pop_size, search_dim)
        nfe: int = algorithm.evaluator.n_eval

        best_fitness = float(F.min())
        mean_fitness = float(F.mean())
        center_fitness = self._compute_center_fitness(X)

        metrics = {
            "generation": gen,
            "nfe": nfe,
            "best_fitness": best_fitness,
            "mean_fitness": mean_fitness,
            "center_fitness": center_fitness,
        }
        if gen % 1000 == 0:
            self._log_console(metrics)
        if self._use_wandb:
            self._log_wandb(metrics)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_center_fitness(self, X: np.ndarray) -> float:
        """Evaluate fitness at the centroid of the search-space population."""
        centroid_z = X.mean(axis=0)  # (search_dim,)
        centroid_x = self._subspace.expand(centroid_z)  # (D,)
        try:
            return float(self._eval_fn(centroid_x))
        except Exception:
            return float("nan")

    @staticmethod
    def _log_console(metrics: dict) -> None:
        print(
            f"[gen {metrics['generation']:>6d} | nfe {metrics['nfe']:>10d}]  "
            f"best={metrics['best_fitness']:.6e}  "
            f"mean={metrics['mean_fitness']:.6e}  "
            f"center={metrics['center_fitness']:.6e}"
        )

    @staticmethod
    def _log_wandb(metrics: dict) -> None:
        try:
            import wandb  # type: ignore

            wandb.log(metrics, step=metrics["nfe"])
        except ImportError:
            pass
