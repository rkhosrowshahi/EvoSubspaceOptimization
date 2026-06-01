# Changelog

All notable changes to this fork of [dmolina/cec2013lsgo](https://github.com/dmolina/cec2013lsgo) are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [3.0.0] - 2026-06-01

### Added

- Pure-Python **`LSGO2013`** API in `cec2013lsgo/benchmarks.py` (NumPy only; no compile step required).
- Problem IDs `cec2013_lsgo_f1` … `cec2013_lsgo_f15` and `VALID_FUNC_IDS`.
- Seed-based structural data for **any dimension** `D >= 1`; official `cdatafiles/` used at `D = 1000`.
- Regression tests in `tests/test_lsgo2013.py` (reference values at D=1000; scalable dimensions 2 … 1e6).
- MkDocs documentation site and GitHub Pages workflow (`.github/workflows/docs.yml`).
- CI workflow for pytest (`.github/workflows/tests.yml`).
- `NOTICE`, `CHANGELOG.md`, and expanded README / `docs/usage.md`.

### Changed

- Default Python entry point is `LSGO2013`; legacy Cython `Benchmark` remains optional (F1–F15, D=1000).
- Legacy C++ build compiles **F1–F15 only** (`setup.py`, `eval_func.cpp`, `Header.h`, `cec2013.pyx`).
- Packaging uses `README.md` as long description; `MANIFEST.in` ships `LICENSE` and legal notices.

### Removed

- Stale generated `cec2013.cpp` (build uses `cec2013.pyx` + listed sources).
- Root `README.rst` (superseded by `README.md` and the docs site).

## Upstream baseline

[dmolina/cec2013lsgo](https://github.com/dmolina/cec2013lsgo) v2.x: Cython wrapper around the CEC-2013 LSGO C++ benchmark, F1–F15 at D=1000, GPLv3.

[3.0.0]: https://github.com/rkhosrowshahi/cec2013lsgo/compare/v2.1.0...v3.0.0
