"""Subspace methods for dimensionality reduction in evolutionary optimization."""

import numpy as np

from .base import Subspace
from .fullspace import FullSpace
from .random_projection import RandomProjection
from .random_blocking import RandomBlocking
from .lora import (
    LoRA,
    LoRADiagBlock,
    LoRAGatedBlock,
    LoRAIndependentBlock,
    LoRARank1Block,
    LoRASharedBlock,
    LORA_BLOCK_METHODS,
    lora_method_is_block,
    lora_search_dim,
    validate_lora_blocks,
)

REGISTRY: dict[str, type[Subspace]] = {
    "random_projection": RandomProjection,
    "random_blocking": RandomBlocking,
    "lora": LoRA,
    "lora_ib": LoRAIndependentBlock,
    "lora_shared": LoRASharedBlock,
    "lora_gated": LoRAGatedBlock,
    "lora_diag": LoRADiagBlock,
    "lora_rank1": LoRARank1Block,
    "fullspace": FullSpace,
}


def build_subspace(
    method: str,
    D: int,
    d: int,
    subspace_assignment: str = "absolute",
    seed: int | None = None,
    lb: np.ndarray | None = None,
    ub: np.ndarray | None = None,
    x0: np.ndarray | None = None,
    device: str | None = "cuda:0",
    lora_blocks: int = 1,
) -> Subspace:
    """Factory function to instantiate a subspace by name.

    Args:
        method: One of the keys in ``REGISTRY``.
        D: Full problem dimensionality.
        d: Subspace dimensionality for ``random_projection`` / ``random_blocking``;
            LoRA rank *r* for LoRA methods; must equal ``D`` for ``fullspace``.
        subspace_assignment: 'absolute' or 'additive'.
        seed: RNG seed for subspace random structure and default additive **x0**.
        lb, ub: Full-space bounds; both required together for box clipping after
            ``expand``.
        x0: Optional explicit additive anchor (see Subspace).
        device: PyTorch device string (e.g. ``cuda:0``) for ``random_projection``
            and global ``lora`` matmul; block LoRA variants use NumPy ``expand``.
        lora_blocks: Number of contiguous blocks for block LoRA variants (``>= 1``).

    Returns:
        Initialized Subspace instance.
    """
    if method not in REGISTRY:
        raise ValueError(
            f"Unknown subspace method {method!r}. Choose from: {list(REGISTRY)}"
        )
    kw = dict(
        D=D,
        d=d,
        subspace_assignment=subspace_assignment,
        seed=seed,
        lb=lb,
        ub=ub,
        x0=x0,
    )
    if method in ("random_projection", "lora"):
        kw["device"] = device
    if method in ("lora", *LORA_BLOCK_METHODS):
        kw["blocks"] = lora_blocks
    if lora_method_is_block(method):
        validate_lora_blocks(lora_blocks, D)
    if method in LORA_BLOCK_METHODS:
        kw["device"] = device
    return REGISTRY[method](**kw)


__all__ = [
    "Subspace",
    "FullSpace",
    "RandomProjection",
    "RandomBlocking",
    "LoRA",
    "LoRAIndependentBlock",
    "LoRASharedBlock",
    "LoRAGatedBlock",
    "LoRADiagBlock",
    "LoRARank1Block",
    "LORA_BLOCK_METHODS",
    "REGISTRY",
    "build_subspace",
    "lora_method_is_block",
    "lora_search_dim",
    "validate_lora_blocks",
]
