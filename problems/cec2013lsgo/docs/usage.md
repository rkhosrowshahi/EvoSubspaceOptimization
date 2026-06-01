# CEC-2013 LSGO — usage guide

## Package layout

```
cec2013lsgo/                 # repository root (this folder)
├── cec2013lsgo/
│   ├── __init__.py         # exports LSGO2013, VALID_FUNC_IDS
│   ├── benchmarks.py       # pure-Python F1–F15 (primary API)
│   ├── cdatafiles/         # official F1–F15 data for D=1000
│   ├── cec2013.pyx         # legacy Cython binding (optional)
│   └── *.cpp, *.h          # original C++ reference implementation
├── tests/
│   ├── test_lsgo2013.py    # pure-Python regression tests
│   └── test_bench.py       # legacy Cython tests
├── setup.py                # optional install (may build Cython extension)
├── README.md
└── docs/
    └── usage.md            # this guide
```

## Import paths

### A. Path install (no build)

Add the **repository root** (the directory that contains the inner `cec2013lsgo/` package folder) to `PYTHONPATH` or `sys.path`:

```python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent  # repository root
sys.path.insert(0, str(ROOT))

from cec2013lsgo import LSGO2013, VALID_FUNC_IDS
```

Or set the environment once:

`export PYTHONPATH="/path/to/cec2013lsgo:${PYTHONPATH}"` (Linux/macOS)

`$env:PYTHONPATH="C:\path\to\cec2013lsgo;$env:PYTHONPATH"` (PowerShell)

### B. Editable install

From the repository root, run `pip install -e .`, then:

```python
from cec2013lsgo import LSGO2013, VALID_FUNC_IDS
```

## Class `LSGO2013`

### Constructor

```python
LSGO2013(
    func_id: str,           # "cec2013_lsgo_f1" ... "cec2013_lsgo_f15"
    D: int,                 # dimensionality, must be >= 1
    seed: int = 0,          # instance seed (seed-based path only)
    group_size: int = 50,   # sub-component size (seed-based path only)
)
```

| Parameter | Role |
|-----------|------|
| `func_id` | Must be in `VALID_FUNC_IDS`. |
| `D` | Problem dimension. At `D=1000` with bundled `cdatafiles/`, the official instance is loaded. |
| `seed` | Fixes shifts, permutations, rotations, and weights when data are generated. Ignored for structure when `using_cdatafiles` is `True`. |
| `group_size` | Target size of rotated groups in seed-based mode (capped at `D`). |

### Evaluation and bounds

```python
import numpy as np

bench = LSGO2013("cec2013_lsgo_f4", D=1000, seed=0)

x = np.random.uniform(bench.lb, bench.ub, size=bench.D)
fitness = bench.evaluate(x)   # scalar float; x must be shape (D,)

lb = bench.lb_array            # np.ndarray (D,)
ub = bench.ub_array            # np.ndarray (D,)
lo, hi = bench.bounds          # tuple of arrays

print(bench)                   # shows func_id, D, cdatafiles vs seed
print(bench.using_cdatafiles)  # True iff official files were loaded
```

Scalar bounds `bench.lb` and `bench.ub` are the competition box limits for that function (same for every dimension).

### Properties

| Member | Description |
|--------|-------------|
| `func_id`, `D`, `seed`, `group_size` | As constructed |
| `lb`, `ub` | Scalar search bounds |
| `lb_array`, `ub_array` | Per-dimension bound vectors |
| `using_cdatafiles` | Whether official `cdatafiles/` were used |
| `n_groups` | Number of groups (non-separable types); `None` for F1–F3, F12, F15 |

## Seeds and reproducibility

**Official benchmark (D = 1000)**

```python
a = LSGO2013("cec2013_lsgo_f1", D=1000, seed=0)
b = LSGO2013("cec2013_lsgo_f1", D=1000, seed=999)
# a and b use identical shifts/rotations; seed does not affect structure
```

**Scalable instances (D ≠ 1000 or missing data files)**

```python
a = LSGO2013("cec2013_lsgo_f8", D=5000, seed=42)
b = LSGO2013("cec2013_lsgo_f8", D=5000, seed=42)
# same fitness for the same x

c = LSGO2013("cec2013_lsgo_f8", D=5000, seed=43)
# different instance
```

**Tips**

- Use one `seed` per benchmark **instance** (shifts, rotations, weights).
- Use a separate RNG for your algorithm so `seed` only controls the problem, not the searcher.
- Structural data use an isolated `SeedSequence` child stream; passing the same integer to `LSGO2013(..., seed=k)` and to `numpy.random.default_rng(k)` still yields different draws.

## Reference values (sanity check)

At **D = 1000**, **x = 0**, with official `cdatafiles/`, run `pytest tests/test_lsgo2013.py -q` from the repository root.

Golden values are pinned in that test file. Most functions agree with the legacy C++/Cython package within floating-point tolerance; F3, F6, and F10 show small differences at \(\mathbf{0}\) in the pure-Python port.

## Examples

### Loop over all functions

```python
import numpy as np
from cec2013lsgo import LSGO2013, VALID_FUNC_IDS

D, seed = 1000, 0
x = np.zeros(D)

for fid in sorted(VALID_FUNC_IDS):
    bench = LSGO2013(fid, D=D, seed=seed)
    print(fid, bench.evaluate(x))
```

### Arbitrary dimension

```python
bench = LSGO2013("cec2013_lsgo_f11", D=2500, seed=7, group_size=50)
x = np.random.default_rng(0).uniform(bench.lb, bench.ub, size=2500)
print(bench.evaluate(x))
```

### Batch evaluation (your own loop)

```python
def evaluate_population(bench, X):
    """X: (n, D)"""
    return np.array([bench.evaluate(X[i]) for i in range(len(X))])
```

### Minimal optimization loop (illustration)

```python
import numpy as np
from cec2013lsgo import LSGO2013

bench = LSGO2013("cec2013_lsgo_f1", D=1000, seed=0)
rng = np.random.default_rng(1)  # algorithm RNG, separate from benchmark seed

best_x = rng.uniform(bench.lb, bench.ub, bench.D)
best_f = bench.evaluate(best_x)

for _ in range(1000):
    trial = best_x + rng.normal(0, 1, bench.D)
    trial = np.clip(trial, bench.lb, bench.ub)
    f = bench.evaluate(trial)
    if f < best_f:
        best_f, best_x = f, trial

print(best_f)
```

## Legacy Cython API (optional)

[dmolina/cec2013lsgo](https://github.com/dmolina/cec2013lsgo) also ships a Cython binding (F1–F15, **D = 1000** only):

```python
from cec2013lsgo.cec2013 import Benchmark

bench = Benchmark()
info = bench.get_info(1)          # function index 1..15
fn = bench.get_function(1)
fn(np.zeros(1000))
```

Build with `pip install -e .` (g++ and Cython). **New code should use `LSGO2013`** for arbitrary dimension and seed-based instances.

## Troubleshooting

| Issue | Cause / fix |
|-------|-------------|
| `Problem not found` | Use `cec2013_lsgo_fN` strings, not integers `N`. |
| `pip install -e .` fails on Windows | Use path install (section A); only NumPy is required for `benchmarks.py`. |
| Different fitness than papers at D=1000 | Ensure `using_cdatafiles is True`; evaluate in the stated bounds; use `x` shape `(D,)`. |
| Slow at large D | Expected; rotation cost grows with group structure. Consider smaller D for debugging. |

## Further reading

- Competition page: http://goanna.cs.rmit.edu.au/~xiaodong/cec13-lsgo/competition/
- Upstream wrapper: https://github.com/dmolina/cec2013lsgo
