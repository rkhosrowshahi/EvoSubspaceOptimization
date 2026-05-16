"""Low-Rank Adaptation (LoRA) subspace."""

from __future__ import annotations

import math
import numpy as np
import torch
from .base import Subspace
from .torch_backend import parse_torch_device


class LoRA(Subspace):
    """Low-Rank Adaptation subspace for black-box optimization.

    The D-dimensional full-space vector is treated as (part of) an MxM matrix
    where M = ceil(sqrt(D)).  The matrix is parameterized by two low-rank factors:

        A  in  R^{M x r}
        B  in  R^{r x M}

    so that the full-space matrix X_mat = A @ B  in R^{M x M}.  Flattening and
    taking the first D elements gives the D-dimensional solution x.

    Because M^2 >= D (by the ceiling), the remaining M^2 - D entries are discarded,
    ensuring every full-space dimension is covered.

    The search (optimization) vector z  in  R^{2*M*r}  concatenates the
    flattened A and B:

        z = [A.flatten(), B.flatten()]

    Parameters
    ----------
    d : int
        Rank *r* of the factorization.  The effective search-space
        dimensionality is then 2*M*r where M = ceil(sqrt(D)).
    device : str, optional
        If set, batched matmul in ``expand`` uses PyTorch on that device.

    Notes
    -----
    - This class sets ``search_dim = 2*M*r``, which may differ from ``d``.
    - LoRA does not implement ``reduce()`` (non-invertible compression).
    """

    def __init__(
        self,
        D: int,
        d: int,
        subspace_assignment: str = "absolute",
        seed: int | None = None,
        lb: np.ndarray | None = None,
        ub: np.ndarray | None = None,
        x0: np.ndarray | None = None,
        *,
        device: str | None = "cuda:0",
    ) -> None:
        self._torch_device = parse_torch_device(device)
        super().__init__(
            D=D,
            d=d,
            subspace_assignment=subspace_assignment,
            seed=seed,
            lb=lb,
            ub=ub,
            x0=x0,
        )

    def init(self) -> None:
        self.M: int = math.ceil(math.sqrt(self.D))
        self.r: int = self.d  # user-supplied d is interpreted as rank r
        self._split: int = self.M * self.r  # boundary between A and B in z

    @property
    def search_dim(self) -> int:
        return 2 * self.M * self.r

    def expand(self, z: np.ndarray, x0: np.ndarray | None = None) -> np.ndarray:
        """z (..., 2*M*r) -> x (..., D).

        z is split into A-part (first M*r elements) and B-part (last M*r
        elements), reshaped into matrices, multiplied, flattened, and
        truncated to D dimensions.
        """
        if self._torch_device is None:
            return self._expand_numpy(z, x0)
        return self._expand_torch(z, x0)

    def _expand_numpy(self, z: np.ndarray, x0: np.ndarray | None) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        M, r, split = self.M, self.r, self._split
        batched = z.ndim == 2

        if batched:
            n = z.shape[0]
            A = z[:, :split].reshape(n, M, r)
            B = z[:, split:].reshape(n, r, M)
            x_mat = A @ B
            x = x_mat.reshape(n, M * M)[:, : self.D]
        else:
            A = z[:split].reshape(M, r)
            B = z[split:].reshape(r, M)
            x_mat = A @ B
            x = x_mat.flatten()[: self.D]

        return self._apply_assignment(x, x0)

    def _expand_torch(self, z: np.ndarray, x0: np.ndarray | None) -> np.ndarray:
        z_t = torch.as_tensor(z, dtype=torch.float64, device=self._torch_device)
        M, r, split = self.M, self.r, self._split

        if z_t.ndim == 2:
            n = z_t.shape[0]
            A = z_t[:, :split].reshape(n, M, r)
            B = z_t[:, split:].reshape(n, r, M)
            x_mat = torch.bmm(A, B)
            x = x_mat.reshape(n, M * M)[:, : self.D]
        else:
            A = z_t[:split].reshape(M, r)
            B = z_t[split:].reshape(r, M)
            x_mat = A @ B
            x = x_mat.flatten()[: self.D]

        x_np = x.detach().cpu().numpy()
        return self._apply_assignment(x_np, x0)

    def reduce(self, x: np.ndarray) -> np.ndarray:  # type: ignore[override]
        raise NotImplementedError(
            "LoRA does not support reduce(): the mapping z -> x is many-to-one."
        )
