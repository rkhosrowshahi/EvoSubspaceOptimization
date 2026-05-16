"""Subspace methods for dimensionality reduction in evolutionary optimization."""

import numpy as np

from .base import Subspace
from .random_projection import RandomProjection
from .random_blocking import RandomBlocking
from .lora import LoRA

REGISTRY: dict[str, type[Subspace]] = {
    "random_projection": RandomProjection,
    "random_blocking": RandomBlocking,
    "lora": LoRA,
}


def build_subspace(
    method: str,
    D: int,
    d: int,
    assignment: str = "absolute",
    seed: int | None = None,
    lb: np.ndarray | None = None,
    ub: np.ndarray | None = None,
    x0: np.ndarray | None = None,
    device: str | None = "cuda:0",
) -> Subspace:
    """Factory function to instantiate a subspace by name.

    Args:
        method: One of 'random_projection', 'random_blocking', 'lora'.
        D: Full problem dimensionality.
        d: Subspace dimensionality for ``random_projection`` / ``random_blocking``;
            LoRA rank *r* when ``method=='lora'``.
        assignment: 'absolute' or 'additive'.
        seed: RNG seed for subspace random structure and default additive **x0**.
        lb, ub: Full-space bounds; both required together for box clipping after
            ``expand``.
        x0: Optional explicit additive anchor (see Subspace).
        device: PyTorch device string (e.g. ``cuda:0``) for ``random_projection``
            and ``lora`` matmul; ignored for ``random_blocking``.

    Returns:
        Initialised Subspace instance.
    """
    if method not in REGISTRY:
        raise ValueError(
            f"Unknown subspace method {method!r}. Choose from: {list(REGISTRY)}"
        )
    kw = dict(
        D=D,
        d=d,
        assignment=assignment,
        seed=seed,
        lb=lb,
        ub=ub,
        x0=x0,
    )
    if method in ("random_projection", "lora"):
        kw["device"] = device
    return REGISTRY[method](**kw)


__all__ = [
    "Subspace",
    "RandomProjection",
    "RandomBlocking",
    "LoRA",
    "REGISTRY",
    "build_subspace",
]
