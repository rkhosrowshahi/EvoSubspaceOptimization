"""Low-Rank Adaptation (LoRA) subspace and block variants."""

from __future__ import annotations

import math
from abc import abstractmethod

import numpy as np
import torch

from .base import Subspace
from .torch_backend import parse_torch_device

LORA_BLOCK_METHODS: frozenset[str] = frozenset(
    {"lora_ib", "lora_shared", "lora_gated", "lora_diag", "lora_rank1"}
)


def lora_method_is_block(method: str) -> bool:
    """True when ``method`` is a block LoRA variant."""
    return method in LORA_BLOCK_METHODS


def block_matrix_side(D: int, blocks: int) -> int:
    """Side length ``M_s`` for block LoRA matrix factors."""
    return math.ceil(math.sqrt(math.ceil(D / blocks)))


def balanced_contiguous_blocks(D: int, blocks: int) -> list[tuple[int, int]]:
    """Return ``blocks`` contiguous slices ``(start, end)`` covering ``[0, D)``."""
    validate_lora_blocks(blocks, D)
    base = D // blocks
    rem = D % blocks
    slices: list[tuple[int, int]] = []
    start = 0
    for b in range(blocks):
        size = base + (1 if b < rem else 0)
        slices.append((start, start + size))
        start += size
    return slices


def validate_lora_blocks(blocks: int, D: int) -> None:
    """Raise ``ValueError`` when the block count is invalid for dimension ``D``."""
    if blocks < 1:
        raise ValueError(f"lora_blocks must be >= 1, got {blocks}")
    if blocks > D:
        raise ValueError(f"lora_blocks must be <= D={D}, got {blocks}")


def lora_search_dim(method: str, D: int, rank: int, blocks: int = 1) -> int:
    """Effective optimizer search dimension for a LoRA method."""
    if method == "lora":
        m = math.ceil(math.sqrt(D))
        return 2 * m * rank
    if method not in LORA_BLOCK_METHODS:
        raise ValueError(f"Unknown LoRA method {method!r}")
    validate_lora_blocks(blocks, D)
    m_s = block_matrix_side(D, blocks)
    if method == "lora_ib":
        return 2 * blocks * m_s * rank
    if method == "lora_shared":
        return 2 * m_s * rank
    if method == "lora_gated":
        return 2 * m_s * rank + blocks
    if method == "lora_diag":
        return 2 * m_s * rank + blocks * m_s
    if method == "lora_rank1":
        return 2 * m_s * rank + 2 * blocks * m_s
    raise ValueError(f"Unknown LoRA method {method!r}")


def _unpack_ab(z: np.ndarray, split: int, m: int, r: int) -> tuple[np.ndarray, np.ndarray]:
    """Split ``z`` into ``A`` and ``B`` with trailing batch dimensions."""
    a = z[..., :split].reshape(*z.shape[:-1], m, r)
    b = z[..., split : 2 * split].reshape(*z.shape[:-1], r, m)
    return a, b


def _block_matvec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Batch-aware ``A @ B`` for shapes ``(..., M, r)`` and ``(..., r, M)``."""
    return a @ b


def _truncate_block(x_mat: np.ndarray, block_size: int) -> np.ndarray:
    """Flatten ``(..., M, M)`` and keep the first ``block_size`` entries."""
    return x_mat.reshape(*x_mat.shape[:-2], -1)[..., :block_size]


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

    METHOD_ID = "lora"

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
        blocks: int = 1,
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
        return lora_search_dim(self.METHOD_ID, self.D, self.r)

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
        m, r, split = self.M, self.r, self._split
        a, b = _unpack_ab(z, split, m, r)
        x = _truncate_block(_block_matvec(a, b), self.D)
        return self._apply_assignment(x, x0)

    def _expand_torch(self, z: np.ndarray, x0: np.ndarray | None) -> np.ndarray:
        z_t = torch.as_tensor(z, dtype=torch.float64, device=self._torch_device)
        m, r, split = self.M, self.r, self._split

        if z_t.ndim == 2:
            n = z_t.shape[0]
            a = z_t[:, :split].reshape(n, m, r)
            b = z_t[:, split:].reshape(n, r, m)
            x_mat = torch.bmm(a, b)
            x = x_mat.reshape(n, m * m)[:, : self.D]
        else:
            a = z_t[:split].reshape(m, r)
            b = z_t[split:].reshape(r, m)
            x_mat = a @ b
            x = x_mat.flatten()[: self.D]

        x_np = x.detach().cpu().numpy()
        return self._apply_assignment(x_np, x0)

    def reduce(self, x: np.ndarray) -> np.ndarray:  # type: ignore[override]
        raise NotImplementedError(
            "LoRA does not support reduce(): the mapping z -> x is many-to-one."
        )


class BlockLoRABase(Subspace):
    """Shared setup for block LoRA variants (NumPy ``expand`` path only)."""

    METHOD_ID: str

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
        blocks: int = 1,
    ) -> None:
        self._blocks = blocks
        # Block variants use the NumPy path; ``device`` is accepted for CLI parity.
        _ = parse_torch_device(device)
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
        validate_lora_blocks(self._blocks, self.D)
        self.B = self._blocks
        self.r = self.d
        self.block_slices = balanced_contiguous_blocks(self.D, self.B)
        self.block_sizes = [end - start for start, end in self.block_slices]
        self.M_s = block_matrix_side(self.D, self.B)
        self._split = self.M_s * self.r

    @property
    def search_dim(self) -> int:
        return lora_search_dim(self.METHOD_ID, self.D, self.r, self.B)

    def expand(self, z: np.ndarray, x0: np.ndarray | None = None) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        if z.shape[-1] != self.search_dim:
            raise ValueError(
                f"Expected search vector with last dim {self.search_dim}, "
                f"got {z.shape[-1]}"
            )
        x_proj = self._expand_blocks(z)
        return self._apply_assignment(x_proj, x0)

    @abstractmethod
    def _expand_blocks(self, z: np.ndarray) -> np.ndarray:
        """Map ``z`` to the projected full-space vector(s) before assignment."""

    def reduce(self, x: np.ndarray) -> np.ndarray:  # type: ignore[override]
        raise NotImplementedError(
            f"{type(self).__name__} does not support reduce(): "
            "the mapping z -> x is many-to-one."
        )


class LoRAIndependentBlock(BlockLoRABase):
    """Independent block LoRA: each block has its own ``(A_b, B_b)`` pair."""

    METHOD_ID = "lora_ib"

    def _expand_blocks(self, z: np.ndarray) -> np.ndarray:
        m, r, split = self.M_s, self.r, self._split
        per_block = 2 * split
        batched = z.ndim == 2
        out_shape = (z.shape[0], self.D) if batched else (self.D,)
        x_out = np.zeros(out_shape, dtype=float)

        for b, (start, end) in enumerate(self.block_slices):
            block_size = end - start
            offset = b * per_block
            z_b = z[..., offset : offset + per_block]
            a, b_mat = _unpack_ab(z_b, split, m, r)
            block_x = _truncate_block(_block_matvec(a, b_mat), block_size)
            if batched:
                x_out[:, start:end] = block_x
            else:
                x_out[start:end] = block_x
        return x_out


class LoRASharedBlock(BlockLoRABase):
    """Shared block LoRA: one ``(A, B)`` pair applied to every block."""

    METHOD_ID = "lora_shared"

    def _expand_blocks(self, z: np.ndarray) -> np.ndarray:
        m, r, split = self.M_s, self.r, self._split
        a, b = _unpack_ab(z[..., : 2 * split], split, m, r)
        batched = z.ndim == 2
        out_shape = (z.shape[0], self.D) if batched else (self.D,)
        x_out = np.zeros(out_shape, dtype=float)

        for start, end in self.block_slices:
            block_size = end - start
            block_x = _truncate_block(_block_matvec(a, b), block_size)
            if batched:
                x_out[:, start:end] = block_x
            else:
                x_out[start:end] = block_x
        return x_out


class LoRAGatedBlock(BlockLoRABase):
    """Gated shared block LoRA: shared ``(A, B)`` scaled by one gate per block."""

    METHOD_ID = "lora_gated"

    def _expand_blocks(self, z: np.ndarray) -> np.ndarray:
        m, r, split = self.M_s, self.r, self._split
        ab_end = 2 * split
        a, b = _unpack_ab(z[..., :ab_end], split, m, r)
        gates = z[..., ab_end : ab_end + self.B]
        base = _block_matvec(a, b)
        batched = z.ndim == 2
        out_shape = (z.shape[0], self.D) if batched else (self.D,)
        x_out = np.zeros(out_shape, dtype=float)

        for b_idx, (start, end) in enumerate(self.block_slices):
            block_size = end - start
            gate = gates[..., b_idx]
            if batched:
                block_x = _truncate_block(base * gate[:, None, None], block_size)
                x_out[:, start:end] = block_x
            else:
                block_x = _truncate_block(base * gate, block_size)
                x_out[start:end] = block_x
        return x_out


class LoRADiagBlock(BlockLoRABase):
    """Diagonal-offset block LoRA: ``A @ B + diag(q_b)`` per block."""

    METHOD_ID = "lora_diag"

    def _expand_blocks(self, z: np.ndarray) -> np.ndarray:
        m, r, split = self.M_s, self.r, self._split
        ab_end = 2 * split
        a, b = _unpack_ab(z[..., :ab_end], split, m, r)
        base = _block_matvec(a, b)
        diag_start = ab_end
        batched = z.ndim == 2
        out_shape = (z.shape[0], self.D) if batched else (self.D,)
        x_out = np.zeros(out_shape, dtype=float)

        for b_idx, (start, end) in enumerate(self.block_slices):
            block_size = end - start
            q = z[..., diag_start + b_idx * m : diag_start + (b_idx + 1) * m]
            if batched:
                diag = np.zeros((z.shape[0], m, m), dtype=float)
                diag[:, np.arange(m), np.arange(m)] = q
                block_x = _truncate_block(base + diag, block_size)
                x_out[:, start:end] = block_x
            else:
                block_x = _truncate_block(base + np.diag(q), block_size)
                x_out[start:end] = block_x
        return x_out


class LoRARank1Block(BlockLoRABase):
    """Rank-one-offset block LoRA: ``A @ B + u_b v_b.T`` per block."""

    METHOD_ID = "lora_rank1"

    def _expand_blocks(self, z: np.ndarray) -> np.ndarray:
        m, r, split = self.M_s, self.r, self._split
        ab_end = 2 * split
        a, b = _unpack_ab(z[..., :ab_end], split, m, r)
        base = _block_matvec(a, b)
        rank1_start = ab_end
        batched = z.ndim == 2
        out_shape = (z.shape[0], self.D) if batched else (self.D,)
        x_out = np.zeros(out_shape, dtype=float)

        for b_idx, (start, end) in enumerate(self.block_slices):
            block_size = end - start
            u = z[..., rank1_start + 2 * b_idx * m : rank1_start + (2 * b_idx + 1) * m]
            v = z[
                ...,
                rank1_start + (2 * b_idx + 1) * m : rank1_start + (2 * b_idx + 2) * m,
            ]
            if batched:
                offset = np.einsum("...i,...j->...ij", u, v)
                block_x = _truncate_block(base + offset, block_size)
                x_out[:, start:end] = block_x
            else:
                block_x = _truncate_block(base + np.outer(u, v), block_size)
                x_out[start:end] = block_x
        return x_out
