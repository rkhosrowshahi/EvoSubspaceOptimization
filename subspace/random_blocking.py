"""Random blocking (dimension-sharing / grouping) subspace."""

import numpy as np
from .base import Subspace


class RandomBlocking(Subspace):
    """Random blocking assigns each of the D full-space dimensions to one of
    d groups.  All dimensions in the same group share the same scalar value
    from the search vector z.

    This is also known as *random grouping* or *cooperative coevolution*.

    Mapping:
        x[j] = z[group[j]]          (absolute)
        x[j] = x0[j] + z[group[j]] (additive)
    """

    def init(self) -> None:
        # Randomly assign each of the D dims to one of d groups
        self.groups: np.ndarray = self.rng.integers(0, self.d, size=self.D)

    @property
    def search_dim(self) -> int:
        return self.d

    def expand(self, z: np.ndarray, x0: np.ndarray | None = None) -> np.ndarray:
        """z (..., d) -> x (..., D) by indexing into the group assignment."""
        z = np.asarray(z, dtype=float)
        x = z[..., self.groups]  # (..., D)
        return self._apply_assignment(x, x0)

    def reduce(self, x: np.ndarray) -> np.ndarray:
        """Aggregate full-space values to group representatives via mean."""
        x = np.asarray(x, dtype=float)
        counts = np.bincount(self.groups, minlength=self.d).astype(float)
        z = np.zeros(self.d, dtype=float)
        np.add.at(z, self.groups, x)
        return z / np.maximum(counts, 1.0)
