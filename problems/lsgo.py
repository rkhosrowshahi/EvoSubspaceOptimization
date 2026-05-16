"""CEC-2013 LSGO problem wrapper used by the rest of the project.

This thin wrapper delegates to the seed-based pure-Python implementation in
``problems/cec2013lsgo/``, which was derived from the original C++ benchmark by
Wenxiang Chen and adapted to:
  - support arbitrary dimensionality D (not just D=1000)
  - generate all structural data from an integer seed instead of fixed text files
  - cover only F1-F15 (F16-F25 removed as they were hardcoded for specific D)
"""

from __future__ import annotations

import sys
import pathlib

# Make the local cec2013lsgo package importable regardless of install state.
_HERE = pathlib.Path(__file__).parent
_CEC_PKG = _HERE / "cec2013lsgo"
if str(_CEC_PKG) not in sys.path:
    sys.path.insert(0, str(_CEC_PKG))

import numpy as np
from cec2013lsgo import LSGO2013, VALID_FUNC_IDS


class LSGOProblem:
    """Thin adapter around :class:`LSGO2013` for the rest of the pipeline.

    Parameters
    ----------
    func_id : str
        CEC-2013 LSGO benchmark id, e.g. ``"cec2013_lsgo_f1"`` ... ``"cec2013_lsgo_f15"``.
        Other suites (e.g. CEC 2017) may register different ids later.
    D : int
        Problem dimensionality.
    seed : int
        RNG seed used to generate the benchmark's structural data (shift
        vector, permutation, rotation matrices, weights).
    group_size : int
        Sub-component size for non-separable groups (default 50).
    """

    VALID_IDS = VALID_FUNC_IDS

    def __init__(
        self,
        func_id: str,
        D: int = 1000,
        seed: int = 0,
        group_size: int = 50,
    ) -> None:
        if func_id not in self.VALID_IDS:
            raise ValueError(f"Problem not found: {func_id!r}.")
        self.func_id = func_id
        self.D = D
        self.seed = seed

        self._func = LSGO2013(
            func_id=self.func_id,
            D=D,
            seed=seed,
            group_size=group_size,
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the benchmark at *x* (shape ``(D,)``)."""
        return self._func.evaluate(x)

    # ------------------------------------------------------------------
    # Bounds
    # ------------------------------------------------------------------

    @property
    def lb(self) -> np.ndarray:
        """Lower bounds, shape ``(D,)``."""
        return self._func.lb_array

    @property
    def ub(self) -> np.ndarray:
        """Upper bounds, shape ``(D,)``."""
        return self._func.ub_array

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def optimum(self) -> float | None:
        """Known optimal value.  CEC-2013 LSGO global optimum is 0
        (all functions are shift-based with optimum at the shifted origin),
        but the seed-based weights make the exact value unknown."""
        return None

    def __repr__(self) -> str:
        return (
            f"LSGOProblem(func_id={self.func_id!r}, D={self.D}, seed={self.seed})"
        )
