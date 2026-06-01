# Evolutionary Subspace Optimization

Evolutionary algorithms (EAs) struggle in very high-dimensional search spaces. This project runs EAs in a **low-dimensional subspace** \(z \in \mathbb{R}^{d_z}\) and maps candidates to the full objective space \(x \in \mathbb{R}^D\) before each fitness evaluation. Experiments target the **CEC-2013 Large Scale Global Optimization (LSGO)** benchmark (F1–F15).

**Documentation:** step-by-step usage, sweep catalog, and full CLI tables → [`docs/USAGE.md`](docs/USAGE.md).

---

## What changed in this repository

This tree is the **active development line** of the evolutionary subspace optimization codebase (fork/evolution of an earlier layout). Notable updates compared to older snapshots:

- **CLI entry points** under `scripts/` (`main.py`, `main_two_phase.py`, `main_dual_ea.py`) instead of a repo-root `main.py`.
- **Bundled CEC-2013 LSGO** in `problems/cec2013lsgo/` with a seed-based Python implementation (F1–F15, arbitrary \(D\)); no runtime dependency on external benchmark packages for LSGO runs.
- **LoRA family**: global `lora` plus block variants (`lora_ib`, `lora_shared`, `lora_gated`, `lora_diag`, `lora_rank1`) with `--lora_blocks`.
- **Optional PyTorch** matmul for random projection and global LoRA (`--subspace_device`).
- **W&B sweep configs** under `configs/cec2013lsgo/` including dual-EA and block-LoRA grids.
- **Results tooling** under `results/` (`generate_table.py`, LaTeX/CSV exports).

---

## Key ideas

| Concept | Description |
|---|---|
| **Objective** | Minimize benchmark fitness \(f(x)\), \(x \in \mathbb{R}^D\) |
| **Objective space** | CEC-2013 LSGO; \(D \in \{10^3,\, 5{\cdot}10^3,\, 10^4,\, 10^5,\, 10^6\}\) |
| **Subspace search** | EA optimizes \(z\); \(\Phi(z)\) maps to \(x\) |
| **Absolute** | \(x = \Phi(z)\) (after bounds handling) |
| **Additive** | \(x = x_0 + \Phi(z)\); anchor \(x_0\) from seed or handoff |

### Subspace methods (summary)

| Method | CLI | Size knob | Search dim |
|---|---|---|---|
| Random projection | `random_projection` | `--subspace_dim` \(d\) | \(d\) |
| Random blocking | `random_blocking` | `--subspace_dim` \(d\) | \(d\) |
| Global LoRA | `lora` | `--lora_rank` \(r\) | \(2\lceil\sqrt{D}\rceil\, r\) |
| Block LoRA variants | `lora_ib`, `lora_shared`, `lora_gated`, `lora_diag`, `lora_rank1` | `--lora_rank`, `--lora_blocks` | See [`docs/USAGE.md`](docs/USAGE.md) |
| Full space | `fullspace` or `none` | — | \(D\) |

Global LoRA treats \(x\) as the leading entries of an \(M \times M\) matrix (\(M=\lceil\sqrt{D}\rceil\)) with \(X = AB\), \(A \in \mathbb{R}^{M\times r}\), \(B \in \mathbb{R}^{r\times M}\). Block variants apply the same idea per contiguous segment of \(x\).

---

## Installation

`pip install -r requirements.txt`

Core dependencies: `numpy`, `scipy`, `scikit-learn`, `pymoo`, `wandb`.

For GPU/CPU matmul in random projection and global LoRA: `pip install torch`

Use `--subspace_device cpu` if PyTorch or CUDA is unavailable.

---

## Quick start

From the repository root:

**Random projection:** `python scripts/main.py --problem cec2013_lsgo_f1 --dim 1000 --subspace_method random_projection --subspace_dim 100 --optimizer de --pop_size 100 --max_nfe 3000000 --seed 0`

**Global LoRA (additive anchor):** `python scripts/main.py --problem cec2013_lsgo_f1 --dim 1000 --subspace_method lora --lora_rank 4 --subspace_assignment additive --subspace_device cpu --max_nfe 3000000 --seed 0`

**Dual-EA:** `python scripts/main_dual_ea.py --problem cec2013_lsgo_f1 --dim 1000 --subspace_method lora --lora_rank 4 --subspace_assignment additive --max_nfe 3000000 --seed 0 --benchmark_seed 0`

**W&B sweep:** `wandb sweep configs/cec2013lsgo/d1000/dual_ea/lora/de_static.yaml` then `wandb agent <entity>/evo-subspace-opt/<sweep_id>`

More examples (two-phase, block LoRA, sweep index): [`docs/USAGE.md`](docs/USAGE.md) (project-wide).  
Benchmark package API: [`problems/cec2013lsgo/README.md`](problems/cec2013lsgo/README.md) and [usage guide](https://rkhosrowshahi.github.io/cec2013lsgo/usage/).

---

## Run modes

| Script | Description |
|---|---|
| [`scripts/main.py`](scripts/main.py) | One subspace (or full space) for the entire `--max_nfe` budget |
| [`scripts/main_two_phase.py`](scripts/main_two_phase.py) | Full-space exploration, then subspace refinement (`--full_nfe` + `--sub_nfe` = `--max_nfe`) |
| [`scripts/main_dual_ea.py`](scripts/main_dual_ea.py) | Coupled full-space and subspace EAs; exchange best each cycle (`--full_iters`, `--sub_iters`) |

Dual-EA and phase 2 require a **reduced** subspace (not `fullspace`). **Additive** subspace assignment is recommended when the subspace EA is anchored to a full-space best (especially LoRA, which has no `reduce()`).

---

## Benchmark (CEC-2013 LSGO)

| Item | Detail |
|---|---|
| Problem IDs | `cec2013_lsgo_f1` … `cec2013_lsgo_f15` |
| Wrapper | [`problems/lsgo.py`](problems/lsgo.py) → [`problems/cec2013lsgo/`](problems/cec2013lsgo/) |
| **D = 1000** | Official competition data in `cdatafiles/` (same instance as the C++ reference) |
| **D ≠ 1000** | Structural data generated from `--benchmark_seed` (scalable instances) |
| Instance | `--benchmark_seed` fixes the LSGO instance; sweep `--seed` for EA/subspace repeats |
| Docs | [`problems/cec2013lsgo/README.md`](problems/cec2013lsgo/README.md), [usage guide](https://rkhosrowshahi.github.io/cec2013lsgo/usage/) |
| License | Bundled benchmark code is **GPLv3** ([`problems/cec2013lsgo/LICENSE`](problems/cec2013lsgo/LICENSE)); other project code is **Apache-2.0** ([`LICENSE`](LICENSE)) |

Forked from [dmolina/cec2013lsgo](https://github.com/dmolina/cec2013lsgo); see the benchmark README for API changes (pure-Python `LSGO2013`, F16–F25 removed).

---

## Project structure

```
.
├── scripts/
│   ├── main.py              # Single-phase
│   ├── main_two_phase.py    # Full-space then subspace
│   └── main_dual_ea.py      # Alternating EAs
├── subspace/                # RP, RB, LoRA variants, fullspace
├── problems/
│   ├── lsgo.py              # LSGO adapter for the pipeline
│   └── cec2013lsgo/         # Vendored benchmark — see cec2013lsgo/README.md
├── optimizers/              # DE, PSO, ES, CMA-ES (PyMOO)
├── utils/                   # SubspaceProblem, logging callback
├── configs/cec2013lsgo/     # W&B sweep YAMLs (d1000, d100000, …)
├── results/                 # Aggregated tables / LaTeX (see results/README.md)
├── docs/USAGE.md            # Project usage guide (if present)
└── experiments/             # Optional landscape / analysis scripts
```

---

## Results

After W&B runs, regenerate summary tables: `cd results` then `python generate_table.py --dim 1000`

See [`results/README.md`](results/README.md) for layout and run-id lookup.

---

## Citing this repository

Use **Cite this repository** on GitHub ([`CITATION.cff`](CITATION.cff)), or:

```bibtex
@misc{khosrowshahi_evo_subspace,
  author       = {Khosrowshahli, Rasa},
  title        = {{Evolutionary Subspace Optimization}},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/rkhosrowshahi/EvoSubspaceOptimization},
  note         = {GitHub repository}
}
```

If you use **block differential evolution** ideas from related work:

```bibtex
@inproceedings{khosrowshahi2023block,
  author    = {Khosrowshahli, Rasa and Rahnamayan, Shahryar},
  title     = {Block Differential Evolution},
  booktitle = {2023 IEEE Congress on Evolutionary Computation ({CEC})},
  pages     = {1--8},
  year      = {2023},
  publisher = {IEEE},
  doi       = {10.1109/CEC53210.2023.10254079}
}
```

---

## References

Khosrowshahli, R., & Rahnamayan, S. (2023). Block differential evolution. In *2023 IEEE Congress on Evolutionary Computation (CEC)* (pp. 1–8). IEEE. https://doi.org/10.1109/CEC53210.2023.10254079

Li, X., Tang, K., Omidvar, M. N., Yang, Z., & Qin, K. (2013). *Benchmark functions for the CEC'2013 special session and competition on large scale global optimization* (Technical Report). RMIT University. http://goanna.cs.rmit.edu.au/~xiaodong/cec13-lsgo/competition/

Molina, D. (2018). *cec2013lsgo* [Computer software]. https://github.com/dmolina/cec2013lsgo
