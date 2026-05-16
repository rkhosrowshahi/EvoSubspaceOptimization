"""Gaussian random projection subspace."""

from __future__ import annotations

import numpy as np
import torch
from .base import Subspace
from .torch_backend import parse_torch_device


class RandomProjection(Subspace):
    """Gaussian random projection from d-dimensional search space to D-dimensional full space.

    The projection matrix P in R^{d x D} is built by QR-decomposing a D x d standard
    Gaussian matrix and transposing Q: rows of P form an orthonormal basis of a
    random d-plane in R^D (equivalently P @ P^T = I_d).

    Mapping:
        x = z @ P          (absolute,  x in R^D)
        x = x0 + z @ P    (additive)

    Parameters
    ----------
    device : str, optional
        PyTorch device for sampling ``P`` and for ``expand`` matmul (e.g.
        ``cuda:0``, ``cpu``). If ``None``, ``cpu`` is used.
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
        self._torch_device = parse_torch_device(device) if device is not None else torch.device("cpu")
        self._P_t = None  # populated in init(); tensor on ``_torch_device``
        super().__init__(
            D=D,
            d=d,
            subspace_assignment=subspace_assignment,
            seed=seed,
            lb=lb,
            ub=ub,
            x0=x0,
        )

    def _sample_projection_P(self) -> torch.Tensor:
        """Return P in R^{d x D} with orthonormal rows (P @ P.T = I_d) via Gaussian + QR."""
        dev = self._torch_device
        generator = torch.Generator(device=dev)
        if self._seed is not None:
            generator.manual_seed(int(self._seed))
        G_t = torch.randn(
            self.D,
            self.d,
            generator=generator,
            device=dev,
            dtype=torch.float64,
        )
        Q_t, _ = torch.linalg.qr(G_t, mode="reduced")
        return Q_t.T

    def init(self) -> None:
        self._P_t = self._sample_projection_P()

    @property
    def P(self) -> np.ndarray:
        """Dense projection matrix ``(d, D)`` on CPU NumPy for inspection or legacy use."""
        assert self._P_t is not None
        return self._P_t.detach().cpu().numpy()

    @property
    def search_dim(self) -> int:
        return self.d

    def expand(self, z: np.ndarray, x0: np.ndarray | None = None) -> np.ndarray:
        """z (..., d) -> x (..., D)."""
        assert self._P_t is not None
        z_t = torch.as_tensor(z, dtype=torch.float64, device=self._torch_device)
        x = z_t @ self._P_t
        x_np = x.detach().cpu().numpy()
        return self._apply_assignment(x_np, x0)

    def reduce(self, x: np.ndarray) -> np.ndarray:
        """Back-projection x (D,) -> z (d,). With orthonormal rows, z = P @ x."""
        assert self._P_t is not None
        x_t = torch.as_tensor(x, dtype=torch.float64, device=self._torch_device)
        z_t = self._P_t @ x_t
        return z_t.detach().cpu().numpy()
