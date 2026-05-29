"""Tests for global and block LoRA subspace variants."""

from __future__ import annotations

import unittest

import numpy as np

from subspace import build_subspace, lora_search_dim
from subspace.lora import validate_lora_blocks

D = 10
B = 3
R = 2
LB = np.full(D, -5.0)
UB = np.full(D, 5.0)
X0 = np.linspace(-1.0, 1.0, D)

BLOCK_METHODS = [
    "lora_ib",
    "lora_shared",
    "lora_gated",
    "lora_diag",
    "lora_rank1",
]

EXPECTED_SEARCH_DIM = {
    "lora": lora_search_dim("lora", D, R),
    "lora_ib": 24,
    "lora_shared": 8,
    "lora_gated": 11,
    "lora_diag": 14,
    "lora_rank1": 20,
}


class LoRAVariantTests(unittest.TestCase):
    def test_search_dim_matches_formula(self) -> None:
        for method in ["lora", *BLOCK_METHODS]:
            with self.subTest(method=method):
                sub = build_subspace(
                    method=method,
                    D=D,
                    d=R,
                    lora_blocks=B,
                    device="cpu",
                )
                self.assertEqual(sub.search_dim, EXPECTED_SEARCH_DIM[method])
                self.assertEqual(sub.search_dim, lora_search_dim(method, D, R, B))

    def test_expand_shapes(self) -> None:
        for method in ["lora", *BLOCK_METHODS]:
            with self.subTest(method=method):
                sub = build_subspace(
                    method=method, D=D, d=R, lora_blocks=B, device="cpu"
                )
                z = np.zeros(sub.search_dim)
                self.assertEqual(sub.expand(z).shape, (D,))

                z_batch = np.zeros((4, sub.search_dim))
                self.assertEqual(sub.expand(z_batch).shape, (4, D))

    def test_lora_ib_b1_matches_global_search_dim(self) -> None:
        global_lora = build_subspace(
            method="lora", D=D, d=R, lora_blocks=1, device="cpu"
        )
        ib_lora = build_subspace(
            method="lora_ib", D=D, d=R, lora_blocks=1, device="cpu"
        )
        self.assertEqual(ib_lora.search_dim, global_lora.search_dim)

    def test_additive_assignment_adds_x0_and_clips(self) -> None:
        sub = build_subspace(
            method="lora_shared",
            D=D,
            d=R,
            lora_blocks=B,
            subspace_assignment="additive",
            lb=LB,
            ub=UB,
            x0=X0,
            device="cpu",
        )
        projected = sub.expand(np.zeros(sub.search_dim), x0=np.zeros(D))
        expected = np.clip(X0 + projected, LB, UB)
        self.assertTrue(np.allclose(sub.expand(np.zeros(sub.search_dim)), expected))

    def test_invalid_block_counts_raise(self) -> None:
        for blocks, message in (
            (0, "lora_blocks must be >= 1"),
            (D + 1, "lora_blocks must be <="),
        ):
            with self.subTest(blocks=blocks):
                with self.assertRaisesRegex(ValueError, message):
                    validate_lora_blocks(blocks, D)
                with self.assertRaisesRegex(ValueError, message):
                    build_subspace(
                        method="lora_shared",
                        D=D,
                        d=R,
                        lora_blocks=blocks,
                        device="cpu",
                    )

    def test_reduce_not_implemented(self) -> None:
        sub = build_subspace(
            method="lora_diag", D=D, d=R, lora_blocks=B, device="cpu"
        )
        with self.assertRaises(NotImplementedError):
            sub.reduce(np.zeros(D))


if __name__ == "__main__":
    unittest.main()
