# Evolutionary Subspace Optimization

Evolutionary Algorithms (EAs) struggle with high-dimensional search spaces. This project applies **dimensionality-reduction subspaces** to large-scale global optimization problems: the EA operates in a low-dimensional search space $d$ while solutions are projected back to the $D$-dimensional objective search space before fitness evaluation.

---

## Key ideas

| Concept | Description |
|---|---|
| **Objective** | Minimize benchmark fitness $f(x)$ with $x \in \mathbb{R}^D$ |
| **Objective search space** | $D$-dimensional problem (CEC-2013 LSGO, $D \in \{10^3,\, 5{\cdot}10^3,\, 10^4,\, 10^5,\, 10^6\}$) |
| **Subspace Search** | $d$-dimensional latent space the EA sees |
| **Subspace to Fullspace** | Mapping $z \in \mathbb{R}^d \xrightarrow{\Phi} x \in \mathbb{R}^D$ |
| **Absolute Assignment** | *Absolute*: $x = z \cdot P$ |
| **Additive Assignment** | *Additive*: $x = x_0 + z \cdot P$ |

### Subspace methods

| Method | `--subspace_method` | $\Phi(z)$ | Search dim |
|---|---|---|---|
| Random Projection | `random_projection` | $z \cdot P$ ($P \in \mathbb{R}^{d \times D}$, row-orthonormal) | $d$ |
| Random Blocking \& Parameter Sharing | `random_blocking` | $[\Phi(z)]_j = z_{g_j}$ (fixed random groups $g_j \in \{1,\ldots,d\}$) | $d$ |
| Low-Rank Adaptation (LoRA) | `lora` | $\Phi(z) = \mathrm{vec}_{1:D}(A B)$ with $A \in \mathbb{R}^{M \times r}$, $B \in \mathbb{R}^{r \times M}$ unpacked from $z$, $M=\lceil\sqrt{D}\rceil$ | $2Mr$ where $M=\lceil\sqrt{D}\rceil$, $r$ from `--lora_rank` |
| Full space (baseline) | `fullspace` or `none` | Identity: $x = z$ (after bounds clipping / additive anchor as for other methods) | $D$ |

The CLI accepts `none` as an alias for `fullspace` (stored internally as `fullspace`).

#### LoRA details

The $D$-dimensional vector is reshaped into an $M \times M$ matrix ($M = \lceil\sqrt{D}\rceil$). It is then parameterized by two low-rank factors $A \in \mathbb{R}^{M \times r}$ and $B \in \mathbb{R}^{r \times M}$:

```
x = (A @ B).flatten()[:D]
```

The search vector $z \in \mathbb{R}^{2Mr}$ concatenates the flattened $A$ and $B$. The rank $r$ is set via `--lora_rank`. The effective optimizer dimension is $2Mr$.

---

## Installation

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `scipy`, `pymoo`, `opfunu`, `wandb`.

## License

Original code in this repository is licensed under the **Apache License 2.0**; see [`LICENSE`](LICENSE). The bundled CEC-2013 LSGO code under `problems/cec2013lsgo/` remains under **GNU GPLv3**; see [`problems/cec2013lsgo/LICENSE`](problems/cec2013lsgo/LICENSE).

> **Note on $D > 10^3$**: CEC-2013 LSGO was designed for $D = 10^3$. For larger $D$ the benchmark is extended by partitioning the vector into non-overlapping $10^3$-dimensional blocks, evaluating each, and averaging. This is an approximation for ablation studies.

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

Full-$D$ baseline (no subspace reduction; EA operates directly in $\mathbb{R}^D$):

```bash
python main.py \
    --problem cec2013_lsgo_f1 \
    --dim 1000 \
    --subspace_method fullspace \
    --assignment absolute \
    --optimizer de \
    --pop_size 100 \
    --max_nfe 3000000 \
    --seed 0
```

### All arguments

| Argument | Default | Description |
|---|---|---|
| `--problem` | `cec2013_lsgo_f1` | Benchmark problem id (must exist in `problems`; CEC-2013 LSGO uses `cec2013_lsgo_f1`-`cec2013_lsgo_f15`) |
| `--dim` | `1000` | Objective search space dimensionality $D$ |
| `--subspace_method` | `random_projection` | `random_projection`, `random_blocking`, `lora`, `fullspace`, or `none` (same as `fullspace`) |
| `--subspace_dim` | (unset) | Required for RP/RB ($d$); ignored for LoRA (use `--lora_rank`), fullspace, and `none` |
| `--lora_rank` | (required for LoRA) | LoRA rank $r$ |
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
| `--wandb_group` | - | W&B group; use `{dim}` in the string to inject objective search space $D$ (`--dim`) |
| `--wandb_name` | - | Run name; omit, empty, or `__auto__` for deterministic name from problem / $D$ / assignment / subspace / optimizer / seed |

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
|   +-- fullspace.py      # Identity map; search_dim = D
+-- configs/              # W&B sweep YAMLs (random_projection/, random_blocking/, lora/, fullspace/, ...)
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

Note: for `lora`, pass `--lora_rank` instead of `--subspace_dim`. For a full-$D$ baseline, use `--subspace_method fullspace` (omit `--subspace_dim`) or run `wandb sweep configs/fullspace/de.yaml`.

---

## Citing this repository

On GitHub, use **Cite this repository** in the right-hand sidebar (generated from [`CITATION.cff`](CITATION.cff)). That file is the [Citation File Format](https://citation-file-format.github.io/) entry for this code.

Sample BibTeX (adjust `year`, `version`, and `note` if you cite a specific release or commit):

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

If you report results with **Block Differential Evolution**, cite the CEC paper as well:

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

Khosrowshahli, R., & Rahnamayan, S. (2023). Block differential evolution. In *2023 IEEE Congress on Evolutionary Computation (CEC)* (pp. 1-8). IEEE. https://doi.org/10.1109/CEC53210.2023.10254079

Li, X., Tang, K., Omidvar, M. N., Yang, Z., & Qin, K. (2013). *Benchmark functions for the CEC'2013 special session and competition on large scale global optimization* (Technical Report). Evolutionary Computation and Machine Learning Group, RMIT University. http://goanna.cs.rmit.edu.au/~xiaodong/cec13-lsgo/competition/

Molina, D. (2018). *cec2013lsgo* [Computer software]. GitHub. https://github.com/dmolina/cec2013lsgo
