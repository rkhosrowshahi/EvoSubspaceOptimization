"""Direct optimization in the full problem space (identity map)."""

from __future__ import annotations

import numpy as np

from .base import Subspace


class FullSpace(Subspace):
    """Search space equals full dimension ``D``; ``expand`` is the identity (plus bounds clipping).

    Use ``d=D`` when constructing via :func:`build_subspace`; ``subspace_assignment`` behaves like other
    subspaces (absolute: optimize ``x`` directly; additive: optimize perturbations around ``x0``).
    """

    def init(self) -> None:
        if self.d != self.D:
            raise ValueError(
                f"FullSpace requires d == D for consistency; got d={self.d}, D={self.D}"
            )

    @property
    def search_dim(self) -> int:
        return self.D

    def expand(self, z: np.ndarray, x0: np.ndarray | None = None) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        if z.ndim == 1:
            if z.shape[0] != self.D:
                raise ValueError(
                    f"z must have shape ({self.D},), got {z.shape}"
                )
            x_proj = z
            return self._apply_assignment(x_proj, x0)
        if z.ndim == 2:
            if z.shape[1] != self.D:
                raise ValueError(
                    f"z must have shape (n, {self.D}), got {z.shape}"
                )
            return self._apply_assignment(z, x0)
        raise ValueError(f"z must be 1d or 2d, got shape {z.shape}")
