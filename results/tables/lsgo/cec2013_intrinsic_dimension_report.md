# CEC-2013 LSGO Intrinsic Dimension Report

**Random projection fitness-variance analysis**

| | |
|---|---|
| **Date** | July 16, 2026 |
| **Ambient dimension** | D = 1000 |
| **Functions** | F1–F15 (CEC-2013 LSGO) |
| **Script** | `experiments/dimensionality/cec2013_intrinsic_dimension.py` |

---

## 1. Summary

We estimated the **intrinsic dimension** of each CEC-2013 LSGO benchmark function using **random projection**, aligned with the subspace optimization pipeline in this repository (`random_projection`, absolute assignment).

Each function is defined on **1000 variables**, but the fitness landscape does not use all subspace directions equally. By sweeping subspace dimension **d = 1, 2, …, 1000** exactly and measuring how quickly fitness variability grows, we obtain a function-specific intrinsic dimension.

### Key findings

| Category | Functions | Intrinsic dim (90% capture) |
|----------|-----------|-------------------------------|
| **Very low** (Ackley-based) | F3, F6, F10 | **8, 13, 17** |
| **Moderate** (overlapping) | F14 | **597** |
| **High** (partial Schwefel) | F7 | **731** |
| **High** (most others) | F1, F2, F4, F5, F8, F11, F12, F13, F15 | **783–941** |
| **Highest** (full Rastrigin / Schwefel) | F9, F11, F15 | **931–941** |

**Ackley-based landscapes** (F3, F6, F10) are highly compressible under random projection: fewer than 20 subspace dimensions capture 90% of the fitness spread available at d = 1000. **Elliptic, Rastrigin, Rosenbrock, and Schwefel** landscapes generally require **700–940** dimensions.

---

## 2. Motivation

Evolutionary subspace optimization searches in a low-dimensional space **z ∈ ℝᵈ** and maps to the full problem space via a projection **P ∈ ℝᵈˣᴰ**:

```
x = clip(z @ P, lb, ub)
```

Choosing subspace dimension **d** is a central design decision. This report quantifies, per benchmark function, how many dimensions are needed for a random projection to express most of the fitness variability available at full rank (d = D).

This is **not** an optimization study (we do not search for the global minimum). It is a **landscape expressiveness** study: how many random-projection degrees of freedom does each function effectively use?

---

## 3. Method

### 3.1 Capture ratio

For a fixed random projection seed, sample **z ~ Uniform([lb, ub]ᵈ)** and evaluate fitness after expansion:

```
capture(d) = std(f(x_subspace at d)) / std(f(x_subspace at d = 1000))
```

- **Numerator:** fitness standard deviation when searching in a **d**-dimensional subspace.
- **Denominator:** same quantity at **d = 1000** (reference for that projection seed).
- **capture(1000) = 1.0** by construction.

### 3.2 Intrinsic dimension definition

```
intrinsic_dim = min { d ∈ {1,…,1000} : mean_capture(d) ≥ τ }
```

where **τ = 0.90** (default threshold) and **mean_capture** is averaged over **5 independent projection seeds** (seeds 0–4). A monotone envelope (`cummax`) is applied to the mean curve before thresholding.

### 3.3 Exact sweep

Unlike an initial coarse grid (1, 2, 5, 10, …), the final analysis evaluates **every integer d from 1 to 1000**, giving exact threshold crossings rather than bracketed approximations.

### 3.4 Nested projections

For each projection seed, a single full-rank matrix **P ∈ ℝ¹⁰⁰⁰ˣ¹⁰⁰⁰** is generated. Subspaces at dimension **d** use the first **d** rows: **P[:d, :]**. This ensures nested subspaces (d = 10 ⊂ d = 11 ⊂ … ⊂ d = 1000) and reduces computation from 1000 QR decompositions per seed to one.

### 3.5 Secondary metric: elbow dimension

The **elbow** is the knee of the capture curve (maximum perpendicular distance from the chord connecting (d=1, capture(1)) to (d=1000, capture(1000)) in log-d space). It is a heuristic complement to the 90% threshold, not a separate optimization criterion.

---

## 4. Experimental setup

| Parameter | Value |
|-----------|-------|
| Ambient dimension D | 1000 |
| Subspace dimensions evaluated | 1, 2, …, 1000 (exact) |
| Samples per (function, seed, d) | 500 |
| Random projection seeds | 0, 1, 2, 3, 4 |
| Capture threshold τ | 0.90 |
| Subspace assignment | absolute (`x = z @ P`) |
| Benchmark seed | 0 |
| Group size | 50 |
| Subspace device | CPU |

---

## 5. Results

### 5.1 Main table

| Function | Base function | Structure | Groups | **Intrinsic dim** | Elbow d |
|----------|---------------|-----------|--------|-------------------|---------|
| F1 | Elliptic | fully separable | — | **807** | 69 |
| F2 | Rastrigin | fully separable | — | **814** | 97 |
| F3 | Ackley | fully separable | — | **8** | 2 |
| F4 | Elliptic | partially separable | 7 | **807** | 77 |
| F5 | Rastrigin | partially separable | 7 | **859** | 152 |
| F6 | Ackley | partially separable | 7 | **13** | 12 |
| F7 | Schwefel + Sphere | partially separable | 7 | **731** | 607 |
| F8 | Elliptic | fully non-separable | 20 | **783** | 129 |
| F9 | Rastrigin | fully non-separable | 20 | **941** | 324 |
| F10 | Ackley | fully non-separable | 20 | **17** | 7 |
| F11 | Schwefel | fully non-separable | 20 | **931** | 225 |
| F12 | Rosenbrock | single group | — | **800** | 105 |
| F13 | Schwefel | overlapping (conform) | 20 | **910** | 739 |
| F14 | Schwefel | overlapping (conflict) | 20 | **597** | 502 |
| F15 | Schwefel | single group | — | **941** | 294 |

### 5.2 Grouped by base function

| Base function | Functions | Intrinsic dim range |
|---------------|-----------|---------------------|
| **Ackley** | F3, F6, F10 | 8 – 17 |
| **Elliptic** | F1, F4, F8 | 783 – 807 |
| **Rastrigin** | F2, F5, F9 | 814 – 941 |
| **Schwefel** | F7, F11, F13, F14, F15 | 597 – 941 |
| **Rosenbrock** | F12 | 800 |

Ackley is consistently low-dimensional under random projection regardless of separability structure. Schwefel spans the widest range (597–941), with overlapping conflict (F14) lowest among Schwefel variants.

### 5.3 Example capture curves

**F3 (Ackley, separable) — intrinsic dim = 8**

| d | capture(d) |
|---|------------|
| 1 | 0.61 |
| 2 | 0.85 |
| 5 | 0.89 |
| **8** | **0.91** |
| 10 | 0.92 |
| 100 | 1.18 |
| 1000 | 1.00 |

**F1 (Elliptic, separable) — intrinsic dim = 807**

| d | capture(d) |
|---|------------|
| 1 | 0.02 |
| 50 | 0.21 |
| 200 | 0.46 |
| 500 | 0.73 |
| **807** | **0.90** |
| 1000 | 1.00 |

**F10 (Ackley, fully non-separable) — intrinsic dim = 17**

| d | capture(d) |
|---|------------|
| 1 | 0.29 |
| 10 | 0.85 |
| **17** | **0.90** |
| 50 | 0.98 |
| 1000 | 1.01 |

---

## 6. Coarse vs exact grid

An initial study used a sparse d grid (22 points). The exact sweep (1000 points) refined several estimates:

| Function | Coarse grid | Exact (1..1000) |
|----------|-------------|-----------------|
| F3 | 15 | **8** |
| F6 | 20 | **13** |
| F10 | 15 | **17** |
| F7 | 1000 | **731** |
| F14 | 900 | **597** |
| F15 | 1000 | **941** |

The exact sweep matters most for low-dimensional Ackley functions, where the 90% crossing falls between coarse grid points.

---

## 7. Interpretation and limitations

### What intrinsic dimension means here

It measures **fitness variance expressiveness under uniform random-projection search**, not:

- the global optimum location or value;
- the number of interacting variable groups in the benchmark definition;
- optimizer performance at a given `--subspace_dim`.

### Limitations

1. **Threshold choice:** τ = 0.90 is arbitrary; other thresholds yield different IDs.
2. **Sampling:** z is uniform in [lb, ub]ᵈ, matching the EA box but not adaptive search.
3. **Absolute assignment:** results may differ under additive assignment (`x = x₀ + z @ P`).
4. **Nested projections:** subspaces at small d are prefixes of the d = 1000 basis, not independent random d-planes.
5. **Stochasticity:** 500 samples and 5 seeds introduce noise; capture curves above 1.0 before cummax reflect sample variance.

Despite these caveats, the method is consistent with the repository's random-projection subspace optimizer and provides a reproducible per-function ranking of compressibility.

---

## 8. Implications for subspace optimization

| Observation | Practical implication |
|-------------|----------------------|
| F3, F6, F10 need d ≈ 8–17 | Random projection with very small `--subspace_dim` may still capture most fitness spread |
| F1, F2, F4, F5, F8, F12 need d ≈ 780–860 | `--subspace_dim` of 100 (common in configs) captures only ~20–30% of full-rank variance |
| F9, F11, F15 need d ≈ 930–941 | Among the hardest to compress; high `--subspace_dim` is critical |
| F14 lowest among Schwefel (597) | Overlapping conflict structure may offer more compressibility than other Schwefel variants |

These are **landscape statistics**, not guarantees of optimizer success. Empirical EA performance should still be validated via W&B sweeps.

---

## 9. Reproducing the analysis

From the repository root:

```bash
pip install -e .
python experiments/dimensionality/cec2013_intrinsic_dimension.py --eval_n_samples 500
```

Options:

| Flag | Effect |
|------|--------|
| `--capture_threshold 0.95` | Stricter intrinsic dimension |
| `--quick` | Coarse d grid instead of 1..1000 |
| `--functions f3,f10` | Subset of functions |
| `--rp_seeds 0,1,2,3,4` | Projection seeds |

Results are written incrementally after each function completes.

---

## 10. Output files

| File | Description |
|------|-------------|
| `cec2013_intrinsic_dimension.csv` | Summary table (one row per function) |
| `cec2013_intrinsic_dimension_summary.txt` | Plain-text summary |
| `cec2013_intrinsic_dimension_curves.json` | Full 1000-point capture curves per function |
| `cec2013_intrinsic_dimension_report.md` | This report |

All files are in `results/tables/lsgo/`.

---

## References

- X. Li et al., "Benchmark Functions for the CEC'2013 Special Session and Competition on Large Scale Global Optimization," RMIT University, 2013.
- CEC-2013 LSGO implementation: `src/evo_subspace/problems/cec2013lsgo/`
- Random projection subspace: `src/evo_subspace/subspaces/random_projection.py`
