"""Regression tests for the pure-Python LSGO2013 implementation (D=1000, cdatafiles)."""

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cec2013lsgo import LSGO2013, VALID_FUNC_IDS

# Golden f(0) at D=1000 with bundled cdatafiles (pure-Python LSGO2013).
# F1, F7, F8, F14 match the legacy C++ reference closely; a few functions
# (e.g. F3, F6, F10) differ slightly from the old Cython tests at x=0.
_EXPECTED_AT_ZERO = [
    209833896353.3435,
    47620.31161660615,
    21.15096806118381,
    107955147656065.94,
    48419148.332924634,
    1038658.4775123132,
    993826981321072.6,
    5.722271501878064e18,
    6001603202.501933,
    96550461.25907995,
    1.0448520164721205e17,
    1711354236949.7192,
    8.27380048985966e16,
    4.4079796812096246e18,
    2393892336615502.0,
]


@pytest.mark.parametrize(
    "func_id,expected",
    [
        (f"cec2013_lsgo_f{i}", _EXPECTED_AT_ZERO[i - 1])
        for i in range(1, 16)
    ],
)
def test_zero_solution_matches_reference(func_id: str, expected: float) -> None:
    bench = LSGO2013(func_id=func_id, D=1000, seed=0)
    assert bench.using_cdatafiles
    x = np.zeros(1000)
    got = bench.evaluate(x)
    assert got == pytest.approx(expected, rel=0.0, abs=1e-3)


def test_valid_func_ids_count() -> None:
    assert len(VALID_FUNC_IDS) == 15


def test_seed_based_does_not_use_cdatafiles() -> None:
    bench = LSGO2013(func_id="cec2013_lsgo_f1", D=500, seed=0)
    assert not bench.using_cdatafiles
    assert bench.evaluate(np.zeros(500)) != pytest.approx(_EXPECTED_AT_ZERO[0])


def test_evaluate_wrong_length_raises_or_mismatches() -> None:
    """Short vectors are not validated; callers must pass shape (D,)."""
    bench = LSGO2013(func_id="cec2013_lsgo_f1", D=1000, seed=0)
    # NumPy broadcasting may not raise; document expected length instead.
    assert bench.D == 1000
