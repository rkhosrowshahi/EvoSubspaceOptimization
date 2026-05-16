"""PyMOO Problem wrapper that applies a subspace to an LSGO benchmark."""

from __future__ import annotations

import numpy as np
from pymoo.core.problem import Problem

from problems.lsgo import LSGOProblem
from subspace.base import Subspace


class SubspaceProblem(Problem):
    """PyMOO-compatible problem that optimises in a reduced subspace.

    The optimizer sees a ``search_dim``-dimensional problem with bounds
    inherited from the LSGO benchmark.  Each candidate in the search space
    is expanded to the full D-dimensional space by the subspace before the
    true objective is evaluated.

    Parameters
    ----------
    lsgo : LSGOProblem
        The benchmark problem instance.
    subspace : Subspace
        The active subspace mapping.
    """

    def __init__(self, lsgo: LSGOProblem, subspace: Subspace) -> None:
        self.lsgo = lsgo
        self.subspace = subspace

        # Use the full-space bounds as the search-space bounds.
        # For random projection and blocking this is a reasonable
        # approximation; for LoRA the bounds control A and B magnitudes.
        lb = float(lsgo.lb[0])
        ub = float(lsgo.ub[0])
        n_var = subspace.search_dim

        super().__init__(
            n_var=n_var,
            n_obj=1,
            xl=np.full(n_var, lb),
            xu=np.full(n_var, ub),
        )

    # ------------------------------------------------------------------
    # PyMOO evaluation hook
    # ------------------------------------------------------------------

    def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs) -> None:
        """Evaluate a batch of search-space candidates.

        Args:
            X: (n_samples, search_dim) array of search-space candidates.
            out: Output dictionary; we write ``out["F"]`` of shape (n, 1).
        """
        X_full = self.subspace.expand(X)  # (n, D)
        F = np.array(
            [self.lsgo.evaluate(x) for x in X_full], dtype=float
        ).reshape(-1, 1)
        out["F"] = F
