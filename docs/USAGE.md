# EvoSubspaceOptimization — usage guide

This document covers **running experiments** in the parent repository. For the **CEC-2013 LSGO benchmark package** itself, see:

- [`problems/cec2013lsgo/README.md`](../problems/cec2013lsgo/README.md)
- [CEC-2013 LSGO usage](https://rkhosrowshahi.github.io/cec2013lsgo/usage/) ([`docs/usage.md`](../problems/cec2013lsgo/docs/usage.md))

## Install

`pip install -r requirements.txt`

Optional (GPU/CPU matmul for `--subspace_device`, random projection, LoRA): `pip install torch`

Use `--subspace_device cpu` if PyTorch or CUDA is unavailable.

## Entry points

| Script | Purpose |
|--------|---------|
| `scripts/main.py` | Single subspace (or full space) for full `--max_nfe` |
| `scripts/main_two_phase.py` | Full-space phase, then subspace refinement |
| `scripts/main_dual_ea.py` | Alternating full-space and subspace EAs |

Common flags: `--problem` (`cec2013_lsgo_f1` … `f15`), `--dim`, `--benchmark_seed`, `--seed`, `--subspace_method`, `--max_nfe`.

## Example commands

**Random projection:**

`python scripts/main.py --problem cec2013_lsgo_f1 --dim 1000 --subspace_method random_projection --subspace_dim 100 --optimizer de --pop_size 100 --max_nfe 3000000 --seed 0 --benchmark_seed 0`

**Dual-EA + LoRA:**

`python scripts/main_dual_ea.py --problem cec2013_lsgo_f1 --dim 1000 --subspace_method lora --lora_rank 4 --subspace_assignment additive --max_nfe 3000000 --seed 0 --benchmark_seed 0`

## W&B sweeps

Configs under `configs/cec2013lsgo/` (e.g. `d1000/dual_ea/lora/de_static.yaml`):

`wandb sweep configs/cec2013lsgo/d1000/dual_ea/lora/de_static.yaml`

`wandb agent <entity>/<project>/<sweep_id>`

## Results

See [`results/README.md`](../results/README.md). Regenerate tables: `cd results` then `python generate_table.py --dim 1000`.
