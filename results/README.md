# EvoSubspaceOptimization results

## Layout

| Folder | Contents |
|--------|----------|
| [`cec2013_lsgo/dim{D}/`](cec2013_lsgo/) | CEC-2013 LSGO W&B aggregates at full dimension `D` (CSV, LaTeX, seeds, run ids) |
| [`archive/`](archive/) | Superseded pilot tables (older problem subsets or naming) |
| [`synthetic_lora/tables/`](synthetic_lora/tables/) | Toy LoRA ablation CSVs from early 2D/3D experiments |
| [`synthetic_lora/figures/`](synthetic_lora/figures/) | Plots paired with the synthetic LoRA CSVs |

## Regenerate LSGO tables (dim 1000)

```bash
cd projects/EvoSubspaceOptimization/results
python3 generate_table.py --dim 1000
```

Writes to `cec2013_lsgo/dim1000/`:

- `cec2013_lsgo_all_fs_dim1000_by_group.csv`
- `cec2013_lsgo_all_fs_dim1000_by_group_seeds.json`
- `cec2013_lsgo_all_fs_dim1000_by_group_runs.json`
- `cec2013_lsgo_all_fs_dim1000_by_group.tex`
- `cec2013_lsgo_all_fs_dim1000_by_group_compact.tex` (thesis `\input`)

LaTeX only from existing CSV:

```bash
python3 generate_table.py --dim 1000 --from-local
```

## Run id lookup (no W&B API)

```bash
python3 -c "
from pathlib import Path
from generate_table import load_runs_index, lookup_run_ids_by_wandb_group
p = Path('cec2013_lsgo/dim1000/cec2013_lsgo_all_fs_dim1000_by_group_runs.json')
g = 'cec2013_lsgo_f1-dim1000-dual_ea-lora-lora_rank1-blocks10-additive-reeval-de-F0.5-CR0.9-multiseed'
print(lookup_run_ids_by_wandb_group(load_runs_index(p), g))
"
```
