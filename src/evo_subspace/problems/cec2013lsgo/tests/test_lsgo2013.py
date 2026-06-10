"""Regression tests for the pure-Python LSGO2013 implementation."""

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cec2013lsgo import LSGO2013, VALID_FUNC_IDS

# Golden f(0) at D=1000 with bundled cdatafiles (pure-Python LSGO2013).
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

# Seed-based scalable dimensions (not D=1000 official cdatafiles).
SCALABLE_DIMS = [2, 100, 10_000, 100_000, 1_000_000]

_FUNC_IDS = sorted(VALID_FUNC_IDS)


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


@pytest.mark.parametrize("D", SCALABLE_DIMS)
@pytest.mark.parametrize("func_id", _FUNC_IDS)
def test_scalable_dimension_evaluate_finite(func_id: str, D: int) -> None:
    """F1-F15 at D in {2, 100, 10k, 100k, 1M}: seed-based, finite fitness."""
    bench = LSGO2013(func_id=func_id, D=D, seed=42)
    assert not bench.using_cdatafiles
    assert bench.D == D
    x = np.random.default_rng(0).uniform(bench.lb, bench.ub, size=D)
    fitness = bench.evaluate(x)
    assert np.isfinite(fitness)
    assert fitness >= 0.0


@pytest.mark.parametrize("D", SCALABLE_DIMS)
def test_scalable_dimension_reproducible_with_seed(D: int) -> None:
    """Same seed and x give the same fitness (F1 smoke at each D)."""
    a = LSGO2013("cec2013_lsgo_f1", D=D, seed=99)
    b = LSGO2013("cec2013_lsgo_f1", D=D, seed=99)
    x = np.full(D, 0.5)
    assert a.evaluate(x) == b.evaluate(x)


@pytest.mark.parametrize("D", [2, 100])
def test_small_dimension_bounds_shape(D: int) -> None:
    bench = LSGO2013("cec2013_lsgo_f4", D=D, seed=0)
    assert bench.lb_array.shape == (D,)
    assert bench.ub_array.shape == (D,)
    assert np.all(bench.lb_array == bench.lb)
    assert np.all(bench.ub_array == bench.ub)


def test_dimension_2_separable_shift_length() -> None:
    bench = LSGO2013("cec2013_lsgo_f1", D=2, seed=0)
    z = np.array([1.0, 2.0]) - bench._xopt
    assert z.shape == (2,)
    assert bench.evaluate(np.array([1.0, 2.0])) >= 0.0

