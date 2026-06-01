# CEC-2013 Large-Scale Global Optimization (LSGO)

<p align="center">
  <a href="https://rkhosrowshahi.github.io/cec2013lsgo/">
    <img src="https://img.shields.io/badge/Documentation-GitHub%20Pages-2ea44f?style=for-the-badge&logo=github&logoColor=white" alt="Documentation on GitHub Pages"/>
  </a>
  <a href="https://rkhosrowshahi.github.io/cec2013lsgo/usage/">
    <img src="https://img.shields.io/badge/Usage%20guide-API%20reference-2563eb?style=for-the-badge&logo=readthedocs&logoColor=white" alt="Usage guide"/>
  </a>
  <a href="https://github.com/rkhosrowshahi/cec2013lsgo/blob/main/tests/test_lsgo2013.py">
    <img src="https://img.shields.io/badge/Tests-pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white" alt="pytest test suite"/>
  </a>
</p>

<p align="center">
  <a href="https://github.com/rkhosrowshahi/cec2013lsgo/actions/workflows/tests.yml">
    <img src="https://github.com/rkhosrowshahi/cec2013lsgo/actions/workflows/tests.yml/badge.svg" alt="Tests workflow status"/>
  </a>
  <a href="https://github.com/rkhosrowshahi/cec2013lsgo/actions/workflows/docs.yml">
    <img src="https://github.com/rkhosrowshahi/cec2013lsgo/actions/workflows/docs.yml/badge.svg" alt="Documentation deploy status"/>
  </a>
  <a href="https://github.com/rkhosrowshahi/cec2013lsgo/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-GPLv3-960000?style=flat-square" alt="GPLv3 license"/>
  </a>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+"/>
  </a>
  <a href="https://numpy.org/">
    <img src="https://img.shields.io/badge/NumPy-1.24+-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy 1.24+"/>
  </a>
  <img src="https://img.shields.io/badge/Functions-F1--F15-555?style=flat-square" alt="Functions F1 through F15"/>
  <img src="https://img.shields.io/badge/Dimension-2%20to%201M-555?style=flat-square" alt="Any dimension D from 2 to one million"/>
</p>

<p align="center">
  <a href="https://github.com/rkhosrowshahi/cec2013lsgo">
    <img src="https://img.shields.io/badge/Repository-cec2013lsgo-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub repository"/>
  </a>
  <a href="http://goanna.cs.rmit.edu.au/~xiaodong/cec13-lsgo/competition/">
    <img src="https://img.shields.io/badge/CEC--2013-Competition%20page-0066cc?style=for-the-badge" alt="CEC-2013 LSGO competition"/>
  </a>
  <a href="https://github.com/dmolina/cec2013lsgo">
    <img src="https://img.shields.io/badge/Upstream-dmolina%2Fcec2013lsgo-6e40c9?style=for-the-badge&logo=github&logoColor=white" alt="Upstream dmolina/cec2013lsgo"/>
  </a>
</p>

Pure-Python implementation of the [CEC-2013 LSGO benchmark](http://goanna.cs.rmit.edu.au/~xiaodong/cec13-lsgo/competition/) (F1–F15), forked from [dmolina/cec2013lsgo](https://github.com/dmolina/cec2013lsgo).

Install or add to `PYTHONPATH`, then use **`LSGO2013`** with NumPy only (no C++ build required for the primary API).

## What changed from upstream

| Topic | Original (`dmolina/cec2013lsgo`) | This fork |
|-------|----------------------------------|-----------|
| **API** | Cython wrapper around C++ (`Benchmark`, F1–F15) | **`LSGO2013`** — pure NumPy in `cec2013lsgo/benchmarks.py` |
| **Functions** | F1–F15 (CEC-2013 competition set) | Same **F1–F15**, scalable to any **D** via `seed` |
| **Dimension** | Fixed **D = 1000** | **Any D ≥ 1** via seed-based structural data |
| **D = 1000** | Same as competition | Loads official **`cdatafiles/`** when present (bit-compatible with C++ reference) |
| **Problem IDs** | Integer `1…15` | Strings **`cec2013_lsgo_f1` … `cec2013_lsgo_f15`** |
| **Build** | Requires g++, Cython | **No compile step** for the Python API (only `numpy`) |
| **C++ / Cython** | Primary path | Kept under `cec2013lsgo/*.cpp` for reference; optional legacy install |

Formulas and transformations follow the original C++ sources (Wenxiang Chen, Colorado State University).

## Requirements

- Python 3.10+
- [NumPy](https://numpy.org/) ≥ 1.24

Optional (legacy Cython extension only): Cython, g++.

## Quick start

### Without install

`git clone https://github.com/rkhosrowshahi/cec2013lsgo.git`  
`cd cec2013lsgo`

```python
import sys
sys.path.insert(0, ".")  # repository root (contains the inner cec2013lsgo/ package)

import numpy as np
from cec2013lsgo import LSGO2013, VALID_FUNC_IDS

print(sorted(VALID_FUNC_IDS))  # cec2013_lsgo_f1 ... cec2013_lsgo_f15

bench = LSGO2013(func_id="cec2013_lsgo_f1", D=1000, seed=0)
x = np.random.uniform(bench.lb, bench.ub, size=1000)
print(bench.evaluate(x))
print(bench.lb_array.shape, bench.using_cdatafiles)  # True at D=1000 with bundled data
```

### Editable install (optional)

`pip install numpy` then `pip install -e .` (may compile the legacy Cython module if g++ is available).

If the extension build fails, use the `sys.path` approach above; the pure-Python module does not depend on it.

See the **[usage guide](https://rkhosrowshahi.github.io/cec2013lsgo/usage/)** (source: [`docs/usage.md`](https://github.com/rkhosrowshahi/cec2013lsgo/blob/main/docs/usage.md)).

## Function catalog

| ID | Type | Base function(s) | Bounds \([lb, ub]\) |
|----|------|------------------|---------------------|
| `cec2013_lsgo_f1` | Fully separable | Shifted Elliptic | \([-100, 100]\) |
| `cec2013_lsgo_f2` | Fully separable | Shifted Rastrigin | \([-5, 5]\) |
| `cec2013_lsgo_f3` | Fully separable | Shifted Ackley | \([-32, 32]\) |
| `cec2013_lsgo_f4` | Partially separable | 7-group Elliptic + sep. Elliptic | \([-100, 100]\) |
| `cec2013_lsgo_f5` | Partially separable | 7-group Rastrigin + sep. Rastrigin | \([-5, 5]\) |
| `cec2013_lsgo_f6` | Partially separable | 7-group Ackley + sep. Ackley | \([-32, 32]\) |
| `cec2013_lsgo_f7` | Partially separable | 7-group Schwefel + sep. Sphere | \([-100, 100]\) |
| `cec2013_lsgo_f8` | Fully non-separable | 20-group Elliptic | \([-100, 100]\) |
| `cec2013_lsgo_f9` | Fully non-separable | 20-group Rastrigin | \([-5, 5]\) |
| `cec2013_lsgo_f10` | Fully non-separable | 20-group Ackley | \([-5, 5]\) |
| `cec2013_lsgo_f11` | Fully non-separable | 20-group Schwefel | \([-32, 32]\) |
| `cec2013_lsgo_f12` | Fully non-separable | Shifted Rosenbrock | \([-100, 100]\) |
| `cec2013_lsgo_f13` | Overlapping | 20-group Schwefel (conform) | \([-100, 100]\) |
| `cec2013_lsgo_f14` | Overlapping | 20-group Schwefel (conflict) | \([-100, 100]\) |
| `cec2013_lsgo_f15` | Fully non-separable | Shifted Schwefel | \([-100, 100]\) |

## Data sources

- **`D = 1000`** and `cec2013lsgo/cdatafiles/` present: official competition files (shift, permutation, rotations, group sizes, weights). **`seed` does not change** the instance; it matches the original C++ benchmark.
- **Any other `D`** (or missing data files): all structure is generated from **`seed`** (deterministic). Control group layout with **`group_size`** (default `50`).

Structural RNG uses a child stream of `numpy.random.SeedSequence` so benchmark data stays deterministic and isolated from other RNG streams that share the same integer seed.

## Tests

Pure-Python regression (F1–F15 at \(\mathbf{0}\), D=1000, official data):

`pip install numpy pytest` → `pytest tests/test_lsgo2013.py -q`

Legacy tests in `tests/test_bench.py` target the old Cython `Benchmark` class and require a compiled extension.

## Documentation

| | Link |
|---|------|
| Website | [rkhosrowshahi.github.io/cec2013lsgo](https://rkhosrowshahi.github.io/cec2013lsgo/) |
| Usage | [Usage guide](https://rkhosrowshahi.github.io/cec2013lsgo/usage/) · [`docs/usage.md`](https://github.com/rkhosrowshahi/cec2013lsgo/blob/main/docs/usage.md) |
| Publish locally | [Publishing](https://rkhosrowshahi.github.io/cec2013lsgo/publishing/) |
| Source | [`cec2013lsgo/benchmarks.py`](https://github.com/rkhosrowshahi/cec2013lsgo/blob/main/cec2013lsgo/benchmarks.py) |

## Citation

If you use any part of this code, cite the CEC-2013 benchmark report:

> X. Li, K. Tang, M. N. Omidvar, Z. Yang, and K. Qin, *Benchmark Functions for the CEC'2013 Special Session and Competition on Large Scale Global Optimization*, Technical Report, Evolutionary Computation and Machine Learning Group, RMIT University, Australia, 2013.  
> http://goanna.cs.rmit.edu.au/~xiaodong/cec13-lsgo/competition/

**Original C++ test suite** (formulas and reference implementation):

> Wenxiang Chen, Colorado State University — CEC-2013 LSGO C++ sources in `cec2013lsgo/*.cpp`.

**Original Python/Cython wrapper:**

> D. Molina (2018). *cec2013lsgo*. https://github.com/dmolina/cec2013lsgo

## License

GNU **GPLv3** — see [LICENSE](https://github.com/rkhosrowshahi/cec2013lsgo/blob/main/LICENSE), [NOTICE](https://github.com/rkhosrowshahi/cec2013lsgo/blob/main/NOTICE), and [CHANGELOG.md](https://github.com/rkhosrowshahi/cec2013lsgo/blob/main/CHANGELOG.md).

Original C++ and Python wrapper: copyright Daniel Molina ([dmolina/cec2013lsgo](https://github.com/dmolina/cec2013lsgo)). Pure-Python `LSGO2013` and other modifications: copyright Rasa Khosrowshahli (2026), same license when you redistribute this package.

## Upstream

Daniel Molina — [dmolina/cec2013lsgo](https://github.com/dmolina/cec2013lsgo)
