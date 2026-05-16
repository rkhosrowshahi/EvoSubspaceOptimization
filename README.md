# Evolutionary Subspace Optimisation

Evolutionary Algorithms (EAs) struggle with high-dimensional search spaces. This project applies **dimensionality-reduction subspaces** to large-scale global optimisation problems: the EA operates in a low-dimensional search space `d` while solutions are projected back to the full `D`-dimensional problem space before fitness evaluation.

---

## Key ideas

| Concept | Description |
|---|---|
| **Full space** | D-dimensional problem (CEC-2013 LSGO, D in {1k, 5k, 10k, 100k, 1M}) |
| **Search space** | d-dimensional latent space the EA sees |
| **Subspace** | Mapping z in R^d -> x in R^D |
| **Assignment** | *Absolute*: x = z @ P, or *Additive*: x = x0 + z @ P |

### Subspace methods

| Method | `--subspace_method` | Search dim |
|---|---|---|
| Gaussian Random Projection | `random_projection` | d |
| Random Blocking (grouping) | `random_blocking` | d |
| Low-Rank Adaptation (LoRA) | `lora` | 2*M*r where M=ceil(sqrt(D)), r=`--lora_rank` |

#### LoRA details

The D-dimensional vector is reshaped into an M x M matrix (M = ceil(sqrt(D))). It is then parameterised by two low-rank factors A in R^{M x r} and B in R^{r x M}:

```
x = (A @ B).flatten()[:D]
```

The search vector z in R^{2*M*r} concatenates the flattened A and B. The rank `r` is set via `--lora_rank`. The effective optimiser dimension is 2*M*r.

---

## Installation

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `scipy`, `pymoo`, `opfunu`, `wandb`.

> **Note on D > 1000**: CEC-2013 LSGO was designed for D=1000. For larger D the benchmark is extended by partitioning the vector into non-overlapping 1000-d blocks, evaluating each, and averaging. This is an approximation for ablation studies.

---

## Usage

```bash
python main.py \
    --problem cec2013_lsgo_f1 \
    --dim 1000 \
    --subspace_method random_projection \
    --subspace_dim 100 \
    --assignment absolute \
    --optimizer de \
    --pop_size 100 \
    --init_pop uniform \
    --de_mut_rate 0.8 \
    --de_cr_rate 0.9 \
    --max_nfe 3000000 \
    --seed 0 \
    --benchmark_seed 0
```

### All arguments

| Argument | Default | Description |
|---|---|---|
| `--problem` | `cec2013_lsgo_f1` | Benchmark problem id (must exist in `problems`; CEC-2013 LSGO uses `cec2013_lsgo_f1`-`cec2013_lsgo_f15`) |
| `--dim` | `1000` | Full-space dimensionality D |
| `--subspace_method` | `random_projection` | Subspace method |
| `--subspace_dim` | `100` | d for RP/RB; ignored for LoRA (use `--lora_rank`) |
| `--lora_rank` | (required for LoRA) | LoRA rank r |
| `--assignment` | `absolute` | `absolute` or `additive` |
| `--optimizer` | `de` | `de`, `pso`, `es`, `cmaes` |
| `--pop_size` | `100` | Population size |
| `--init_pop` | `uniform` | `uniform`, `gaussian`, `lhs` |
| `--de_mut_rate` | `0.8` | DE mutation factor F |
| `--de_cr_rate` | `0.9` | DE crossover rate CR |
| `--de_evolving` | off | Enable PyMOO evolutionary adaptation of F and CR |
| `--pso_w` | `0.9` | PSO inertia weight |
| `--pso_c1` | `2.0` | PSO cognitive weight |
| `--pso_c2` | `2.0` | PSO social weight |
| `--pso_evolving` | off | Enable PyMOO adaptive PSO (dynamic w, c1, c2) |
| `--es_sigma` | `0.3` | ES initial step-size sigma |
| `--cmaes_sigma` | `0.5` | CMA-ES initial step-size sigma |
| `--max_nfe` | `3000000` | NFE budget |
| `--seed` | `0` | EA / subspace / NumPy RNG (not the LSGO instance) |
| `--benchmark_seed` | `0` | LSGO structural data seed (shifts, rotations, weights) |
| `--log_every` | `1` | Log every N generations |
| `--wandb` | off | Enable W&B logging |
| `--wandb_entity` | - | W&B entity |
| `--wandb_project` | `evo-subspace-opt` | W&B project |
| `--wandb_group` | - | W&B group; use `{dim}` in the string to inject full-space `--dim` |
| `--wandb_name` | - | Run name; omit, empty, or `__auto__` for deterministic name from problem / D / assignment / subspace / optimiser / seed |

### W&B logging

Pass `--wandb` to enable. Every generation logs:

| Metric | Description |
|---|---|
| `best_fitness` | Minimum fitness in the population |
| `mean_fitness` | Mean fitness in the population |
| `center_fitness` | Fitness at the population centroid |
| `nfe` | Cumulative function evaluations |

---

## Project structure

```
.
+-- main.py               # Entry point + argument parser
+-- requirements.txt
+-- subspace/
|   +-- base.py           # Abstract Subspace class
|   +-- random_projection.py
|   +-- random_blocking.py
|   +-- lora.py
+-- problems/
|   +-- lsgo.py           # CEC-2013 LSGO wrapper (opfunu back-end)
+-- optimizers/
|   +-- builder.py        # Algorithm factory (DE, PSO, ES, CMA-ES)
|   +-- sampling.py       # Custom sampling (Gaussian)
+-- utils/
    +-- callback.py       # Per-generation logging callback
    +-- problem.py        # PyMOO Problem wrapping subspace + LSGO
```

---

## Example ablation sweep

```bash
for method in random_projection random_blocking lora; do
  for d in 50 100 200 500; do
    python main.py \
      --problem cec2013_lsgo_f7 --dim 1000 \
      --subspace_method $method --subspace_dim $d \
      --optimizer de --pop_size 100 \
      --max_nfe 3000000 --seed 0 \
      --wandb --wandb_project evo-subspace-opt \
      --wandb_group "ablation-d" \
      --wandb_name "${method}-d${d}"
  done
done
```

Note: for `lora`, pass `--lora_rank` instead of relying on `--subspace_dim`.
