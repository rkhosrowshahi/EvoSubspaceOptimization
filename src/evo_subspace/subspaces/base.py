"""Abstract base class for subspace methods."""

from abc import ABC, abstractmethod
import numpy as np


class Subspace(ABC):
    """Base class for dimension-reduction subspace methods.

    Subclasses implement two complementary operations:
    - expand: map a low-dimensional search vector z -> full D-dimensional x
    - reduce: map a full D-dimensional x -> low-dimensional z (optional)

    Two assignment modes are supported:
    - absolute: x = f(z)                (pure subspace solution)
    - additive: x = x0 + f(z)           (perturbation around a base point)
    """

    ABSOLUTE = "absolute"
    ADDITIVE = "additive"

    def __init__(
        self,
        D: int,
        d: int,
        subspace_assignment: str = "absolute",
        seed: int | None = None,
        lb: np.ndarray | None = None,
        ub: np.ndarray | None = None,
        x0: np.ndarray | None = None,
    ) -> None:
        """
        Args:
            D: Full problem dimensionality.
            d: Subspace dimensionality (meaning is subclass-specific;
               for LoRA this is the rank r).
            subspace_assignment: 'absolute' or 'additive'.
            seed: RNG seed for subspace structure (projection, blocking, ...) and,
                in additive mode without an explicit ``x0``, for sampling **x0**
                ``uniform(lb, ub)`` (drawn before ``init()`` so the stream is
                deterministic).
            lb, ub: Full-space lower / upper bounds (shape ``(D,)``). When both
                are given, expanded vectors are **clipped** into ``[lb, ub]``.
            x0: Explicit additive anchor (shape ``(D,)``). Must be finite; if
                ``lb``/``ub`` are set, every coordinate must lie in the box.
                If ``None`` and ``subspace_assignment`` is additive, **x0** is sampled
                using ``seed`` as above (requires ``lb`` and ``ub``).
        """
        if subspace_assignment not in (self.ABSOLUTE, self.ADDITIVE):
            raise ValueError(
                f"subspace_assignment must be 'absolute' or 'additive', got {subspace_assignment!r}"
            )
        self.D = D
        self.d = d
        self.subspace_assignment = subspace_assignment
        self._seed = seed
        self.rng = np.random.default_rng(seed)
        self._x0: np.ndarray | None = None
        self._lb: np.ndarray | None = None
        self._ub: np.ndarray | None = None
        self._set_bounds(lb, ub)
        self._init_x0_from_args(x0)
        self.init()

    def _validated_x0_copy(self, x0: np.ndarray) -> np.ndarray:
        """Return a float copy of ``x0`` or raise if invalid for dimension ``D``."""
        v = np.asarray(x0, dtype=float).reshape(-1)
        if v.shape != (self.D,):
            raise ValueError(f"x0 must have shape (D,)={(self.D,)}, got {v.shape}")
        if not np.all(np.isfinite(v)):
            raise ValueError("x0 must contain only finite values")
        if self._lb is not None:
            if np.any(v < self._lb) or np.any(v > self._ub):
                raise ValueError("x0 must satisfy lb <= x0 <= ub elementwise")
        return v.copy()

    def _init_x0_from_args(self, x0: np.ndarray | None) -> None:
        """Set ``_x0`` from an explicit vector or sample it (additive only)."""
        if x0 is not None:
            self._x0 = self._validated_x0_copy(x0)
            return
        if self.subspace_assignment != self.ADDITIVE:
            return
        if self._lb is None or self._ub is None:
            raise ValueError("additive mode without explicit x0 requires lb and ub")
        self._x0 = self.rng.uniform(self._lb, self._ub)

    def _set_bounds(self, lb: np.ndarray | None, ub: np.ndarray | None) -> None:
        """Store full-space box constraints for clipping after ``expand``."""
        if lb is None and ub is None:
            return
        if lb is None or ub is None:
            raise ValueError("lb and ub must both be provided or both omitted")
        lb_a = np.asarray(lb, dtype=float).reshape(-1)
        ub_a = np.asarray(ub, dtype=float).reshape(-1)
        if lb_a.shape != (self.D,) or ub_a.shape != (self.D,):
            raise ValueError(
                f"lb and ub must have shape (D,)={(self.D,)}, got {lb_a.shape} and {ub_a.shape}"
            )
        self._lb = lb_a.copy()
        self._ub = ub_a.copy()

    def _clip_to_bounds(self, x: np.ndarray) -> np.ndarray:
        """Clip full-space vector(s) to ``[lb, ub]`` when bounds were provided."""
        if self._lb is None:
            return x
        return np.clip(x, self._lb, self._ub)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def search_dim(self) -> int:
        """Effective dimensionality of the search space the optimizer uses."""

    @abstractmethod
    def init(self) -> None:
        """Initialize internal structures (projection matrices, groupings, ...)."""

    @abstractmethod
    def expand(self, z: np.ndarray, x0: np.ndarray | None = None) -> np.ndarray:
        """Expand search-space vector(s) to the full D-dimensional space.

        Args:
            z: Search-space solution(s). Shape (search_dim,) or (n, search_dim).
            x0: Base point for additive assignment. Shape (D,). Overrides the
                stored x0 set via :meth:`set_x0`.

        Returns:
            Full-space solution(s). Shape (D,) or (n, D).
        """

    def reduce(self, x: np.ndarray) -> np.ndarray:
        """Project a full-space solution to the search space (optional).

        Not all subspace methods support an exact inverse; those raise
        ``NotImplementedError``.

        Args:
            x: Full-space solution. Shape (D,).

        Returns:
            Search-space approximation. Shape (search_dim,).
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement reduce().")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def set_x0(self, x0: np.ndarray) -> None:
        """Set the base point used for additive assignment."""
        self._x0 = np.asarray(x0, dtype=float).copy()

    @property
    def x0(self) -> np.ndarray | None:
        """Base vector **x0** for additive assignment (shape ``(D,)``).

        Returns ``None`` when unset; expansion then behaves as if **x0 = 0**.
        Returns a **copy** so callers cannot mutate internal storage.
        """
        if self._x0 is None:
            return None
        return self._x0.copy()

    def _apply_assignment(
        self, x_proj: np.ndarray, x0: np.ndarray | None = None
    ) -> np.ndarray:
        """Apply the assignment mode to the projected vector(s).

        Args:
            x_proj: Projected solution(s). Shape (..., D).
            x0: Override base point (uses stored _x0 or zeros if None).

        Returns:
            Final solution(s). Shape (..., D).
        """
        if self.subspace_assignment == self.ADDITIVE:
            base = (
                x0
                if x0 is not None
                else (self._x0 if self._x0 is not None else np.zeros(self.D))
            )
            out = base + x_proj
        else:
            out = x_proj
        return self._clip_to_bounds(out)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(D={self.D}, d={self.d}, "
            f"search_dim={self.search_dim}, subspace_assignment={self.subspace_assignment!r})"
        )
