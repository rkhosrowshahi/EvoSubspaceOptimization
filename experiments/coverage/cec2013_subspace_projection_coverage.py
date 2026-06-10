r"""Compare full-space and expanded subspace samples in a learned 2D projection.

Workflow:
  1. Draw full-dimensional uniform samples in the CEC-2013 LSGO box.
  2. Fit a 2D projector on those full-space samples.
  3. Draw uniform samples in an optimizer subspace such as LoRA rank 1.
  4. Expand those subspace samples to D dimensions.
  5. Map both sets into the same learned 2D coordinates and plot coverage.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from experiments.landscapes.cec2013_projection_landscape import (
    configure_matplotlib,
    default_ae_hidden_dims,
    fit_autoencoder2,
    fit_pca2,
    parse_functions,
    sample_uniform_in_bounds,
)
from evo_subspace.problems.lsgo import LSGOProblem
from evo_subspace.subspaces import build_subspace

PROJECTION_CHOICES = ("pca", "tsne", "autoencoder")
SUBSPACE_CHOICES = (
    "random_projection",
    "random_blocking",
    "lora",
    "lora_ib",
    "lora_shared",
    "lora_gated",
    "lora_diag",
    "lora_rank1",
)
FIGURES_ROOT = PROJECT_ROOT / "results" / "figures" / "lsgo"
DEFAULT_OUTPUT_ROOT = FIGURES_ROOT / "cec2013_subspace_projection_coverage"
_SAVE_PAD_INCHES = 0.18
_POSITIVE_FLOOR = 1e-30


@dataclass(frozen=True)
class ProjectionResult:
    full_2d: np.ndarray
    sub_2d: np.ndarray
    label: str
    x_label: str
    y_label: str


def _safe_stem(text: str) -> str:
    return (
        text.strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("=", "")
        .replace(",", "")
    )


def _evaluate_many(problem: LSGOProblem, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    values = np.empty(x.shape[0], dtype=float)
    for i, row in enumerate(x):
        values[i] = problem.evaluate(row)
    return values


def _fitness_norm(*values: np.ndarray) -> LogNorm:
    merged = np.concatenate([np.asarray(v, dtype=float).reshape(-1) for v in values])
    finite = merged[np.isfinite(merged)]
    if finite.size == 0:
        return LogNorm(vmin=_POSITIVE_FLOOR, vmax=1.0)
    min_value = float(finite.min())
    if min_value <= 0.0:
        finite = finite - min_value + _POSITIVE_FLOOR
    finite = np.maximum(finite, _POSITIVE_FLOOR)
    vmin = float(finite.min())
    vmax = float(finite.max())
    if vmax <= vmin:
        vmax = vmin * 1.001
    return LogNorm(vmin=vmin, vmax=vmax)


def _positive_fitness(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    min_value = float(np.nanmin(out))
    if min_value <= 0.0:
        out = out - min_value + _POSITIVE_FLOOR
    return np.maximum(out, _POSITIVE_FLOOR)


def _save_figure(fig: plt.Figure, output_base: Path, dpi: int) -> tuple[Path, Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_base.with_suffix(".pdf")
    png_path = output_base.with_suffix(".png")
    fig.savefig(pdf_path, dpi=dpi, pad_inches=_SAVE_PAD_INCHES)
    fig.savefig(png_path, dpi=dpi, pad_inches=_SAVE_PAD_INCHES)
    plt.close(fig)
    return pdf_path, png_path


def _sample_subspace_uniform(
    problem: LSGOProblem,
    *,
    method: str,
    n_samples: int,
    subspace_dim: int,
    lora_rank: int,
    lora_blocks: int,
    assignment: str,
    seed: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    d_param = lora_rank if method.startswith("lora") else subspace_dim
    subspace = build_subspace(
        method=method,
        D=problem.D,
        d=d_param,
        subspace_assignment=assignment,
        seed=seed,
        lb=problem.lb,
        ub=problem.ub,
        device=device,
        lora_blocks=lora_blocks,
    )
    low = float(problem.lb[0])
    high = float(problem.ub[0])
    rng = np.random.default_rng(seed + 1009)
    z = rng.uniform(low, high, size=(n_samples, subspace.search_dim))
    x = subspace.expand(z)
    if method.startswith("lora"):
        label = f"{method}, rank={lora_rank}, search_dim={subspace.search_dim}"
    else:
        label = f"{method}, d={subspace_dim}, search_dim={subspace.search_dim}"
    if method.startswith("lora_"):
        label += f", blocks={lora_blocks}"
    label += f", {assignment}"
    return z, np.asarray(x, dtype=float), label


def _fit_project_pca(full_x: np.ndarray, sub_x: np.ndarray) -> ProjectionResult:
    mean, components, evr = fit_pca2(full_x)
    full_2d = (full_x - mean) @ components.T
    sub_2d = (sub_x - mean) @ components.T
    return ProjectionResult(
        full_2d=full_2d,
        sub_2d=sub_2d,
        label=f"PCA trained on full samples, EVR=({evr[0]:.3f}, {evr[1]:.3f})",
        x_label="PC1",
        y_label="PC2",
    )


def _fit_project_autoencoder(
    full_x: np.ndarray,
    sub_x: np.ndarray,
    problem: LSGOProblem,
    args: argparse.Namespace,
) -> ProjectionResult:
    h1 = args.ae_hidden1 if args.ae_hidden1 > 0 else default_ae_hidden_dims(problem.D)[0]
    h2 = args.ae_hidden2 if args.ae_hidden2 > 0 else default_ae_hidden_dims(problem.D)[1]
    autoencoder = fit_autoencoder2(
        full_x,
        problem.lb,
        problem.ub,
        seed=args.ae_seed,
        hidden1=h1,
        hidden2=h2,
        epochs=args.ae_epochs,
        batch_size=args.ae_batch_size,
        learning_rate=args.ae_learning_rate,
        device=args.ae_device,
    )
    return ProjectionResult(
        full_2d=autoencoder.encode(full_x),
        sub_2d=autoencoder.encode(sub_x),
        label="autoencoder trained on full samples",
        x_label=r"$z_1$",
        y_label=r"$z_2$",
    )


def _fit_project_tsne(
    full_x: np.ndarray,
    sub_x: np.ndarray,
    args: argparse.Namespace,
) -> ProjectionResult:
    try:
        from scipy.spatial import cKDTree
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise ImportError("t-SNE projection requires scikit-learn and scipy") from exc

    kwargs = {
        "n_components": 2,
        "perplexity": args.tsne_perplexity,
        "random_state": args.tsne_seed,
        "init": "pca",
        "learning_rate": "auto",
        "verbose": 1,
    }
    try:
        tsne = TSNE(max_iter=args.tsne_max_iter, **kwargs)
    except TypeError:
        tsne = TSNE(n_iter=args.tsne_max_iter, **kwargs)
    full_2d = tsne.fit_transform(full_x)

    tree = cKDTree(full_x)
    k = min(args.tsne_k_neighbors, full_x.shape[0])
    sub_2d = np.empty((sub_x.shape[0], 2), dtype=float)
    for i, row in enumerate(sub_x):
        dists, idx = tree.query(row, k=k)
        idx = np.atleast_1d(idx)
        dists = np.atleast_1d(dists).astype(float)
        weights = 1.0 / (dists + 1e-12)
        weights /= weights.sum()
        sub_2d[i] = weights @ full_2d[idx]

    return ProjectionResult(
        full_2d=full_2d,
        sub_2d=sub_2d,
        label=f"t-SNE trained on full samples, kNN transform k={k}",
        x_label="t-SNE 1",
        y_label="t-SNE 2",
    )


def fit_and_project(
    full_x: np.ndarray,
    sub_x: np.ndarray,
    problem: LSGOProblem,
    args: argparse.Namespace,
) -> ProjectionResult:
    if args.projection == "pca":
        return _fit_project_pca(full_x, sub_x)
    if args.projection == "autoencoder":
        return _fit_project_autoencoder(full_x, sub_x, problem, args)
    if args.projection == "tsne":
        return _fit_project_tsne(full_x, sub_x, args)
    raise ValueError(f"unknown projection {args.projection!r}")


def save_coverage_plot(
    projection: ProjectionResult,
    *,
    subspace_label: str,
    output_base: Path,
    dpi: int,
) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        projection.full_2d[:, 0],
        projection.full_2d[:, 1],
        s=7,
        alpha=0.25,
        c="#4c72b0",
        label="full uniform",
        rasterized=True,
    )
    ax.scatter(
        projection.sub_2d[:, 0],
        projection.sub_2d[:, 1],
        s=7,
        alpha=0.35,
        c="#dd8452",
        label="expanded subspace uniform",
        rasterized=True,
    )
    ax.set_xlabel(projection.x_label)
    ax.set_ylabel(projection.y_label)
    ax.set_title(f"{projection.label}\n{subspace_label}", fontsize=10)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return _save_figure(fig, output_base.with_name(f"{output_base.name}_coverage"), dpi)


def save_fitness_plot(
    projection: ProjectionResult,
    *,
    full_f: np.ndarray,
    sub_f: np.ndarray,
    subspace_label: str,
    output_base: Path,
    dpi: int,
) -> tuple[Path, Path]:
    norm = _fitness_norm(full_f, sub_f)
    full_pos = _positive_fitness(full_f)
    sub_pos = _positive_fitness(sub_f)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True, sharey=True)
    sc0 = axes[0].scatter(
        projection.full_2d[:, 0],
        projection.full_2d[:, 1],
        c=full_pos,
        cmap="viridis",
        norm=norm,
        s=7,
        alpha=0.6,
        rasterized=True,
    )
    axes[1].scatter(
        projection.sub_2d[:, 0],
        projection.sub_2d[:, 1],
        c=sub_pos,
        cmap="viridis",
        norm=norm,
        s=7,
        alpha=0.6,
        rasterized=True,
    )
    axes[0].set_title("full uniform")
    axes[1].set_title("expanded subspace uniform")
    for ax in axes:
        ax.set_xlabel(projection.x_label)
    axes[0].set_ylabel(projection.y_label)
    fig.suptitle(f"{projection.label}\n{subspace_label}", fontsize=10)
    cbar = fig.colorbar(sc0, ax=axes, shrink=0.88, pad=0.02)
    cbar.ax.set_title(r"$f(x)$", fontsize=10, pad=6)
    fig.subplots_adjust(left=0.07, right=0.88, bottom=0.12, top=0.82, wspace=0.08)
    return _save_figure(fig, output_base.with_name(f"{output_base.name}_fitness"), dpi)


def save_npz(
    *,
    output_base: Path,
    full_x: np.ndarray,
    sub_z: np.ndarray,
    sub_x: np.ndarray,
    full_f: np.ndarray,
    sub_f: np.ndarray,
    projection: ProjectionResult,
    subspace_label: str,
) -> Path:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    path = output_base.with_name(f"{output_base.name}_data.npz")
    np.savez_compressed(
        path,
        full_x=full_x,
        sub_z=sub_z,
        sub_x=sub_x,
        full_fitness=full_f,
        sub_fitness=sub_f,
        full_2d=projection.full_2d,
        sub_2d=projection.sub_2d,
        projection_label=np.asarray([projection.label]),
        subspace_label=np.asarray([subspace_label]),
    )
    return path


def output_base_for(func_id: str, args: argparse.Namespace) -> Path:
    short = func_id.replace("cec2013_lsgo_", "")
    method = _safe_stem(args.subspace_method)
    assignment = _safe_stem(args.subspace_assignment)
    if args.subspace_method.startswith("lora"):
        method = f"{method}_{assignment}_rank{args.lora_rank}"
    else:
        method = f"{method}_{assignment}_d{args.subspace_dim}"
    stem = f"{short}_{args.projection}_{method}_n{args.n_full_samples}_d{args.dim}"
    return args.output_dir / args.projection / args.subspace_method / stem


def run_one_function(func_id: str, args: argparse.Namespace) -> list[Path]:
    problem = LSGOProblem(
        func_id=func_id,
        D=args.dim,
        seed=args.benchmark_seed,
        group_size=args.group_size,
    )
    rng = np.random.default_rng(args.sample_seed)
    full_x = sample_uniform_in_bounds(args.n_full_samples, problem.lb, problem.ub, rng)
    full_f = _evaluate_many(problem, full_x)
    sub_z, sub_x, subspace_label = _sample_subspace_uniform(
        problem,
        method=args.subspace_method,
        n_samples=args.n_sub_samples,
        subspace_dim=args.subspace_dim,
        lora_rank=args.lora_rank,
        lora_blocks=args.lora_blocks,
        assignment=args.subspace_assignment,
        seed=args.subspace_seed,
        device=args.subspace_device,
    )
    sub_f = _evaluate_many(problem, sub_x)
    projection = fit_and_project(full_x, sub_x, problem, args)
    output_base = output_base_for(func_id, args)
    paths: list[Path] = []
    paths.extend(
        save_coverage_plot(
            projection,
            subspace_label=subspace_label,
            output_base=output_base,
            dpi=args.dpi,
        )
    )
    paths.extend(
        save_fitness_plot(
            projection,
            full_f=full_f,
            sub_f=sub_f,
            subspace_label=subspace_label,
            output_base=output_base,
            dpi=args.dpi,
        )
    )
    paths.append(
        save_npz(
            output_base=output_base,
            full_x=full_x,
            sub_z=sub_z,
            sub_x=sub_x,
            full_f=full_f,
            sub_f=sub_f,
            projection=projection,
            subspace_label=subspace_label,
        )
    )
    return paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Project full uniform and expanded subspace samples into a shared 2D view.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--projection", choices=PROJECTION_CHOICES, default="pca")
    parser.add_argument("--subspace_method", choices=SUBSPACE_CHOICES, default="lora")
    parser.add_argument("--functions", type=str, default="f1")
    parser.add_argument("--dim", type=int, default=1000)
    parser.add_argument("--benchmark_seed", type=int, default=0)
    parser.add_argument("--group_size", type=int, default=50)
    parser.add_argument("--n_full_samples", type=int, default=5000)
    parser.add_argument("--n_sub_samples", type=int, default=5000)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--subspace_seed", type=int, default=42)
    parser.add_argument("--subspace_dim", type=int, default=10)
    parser.add_argument("--lora_rank", type=int, default=1)
    parser.add_argument("--lora_blocks", type=int, default=10)
    parser.add_argument(
        "--subspace_assignment",
        choices=("absolute", "additive"),
        default="additive",
    )
    parser.add_argument("--subspace_device", type=str, default="cpu")
    parser.add_argument("--tsne_seed", type=int, default=42)
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--tsne_max_iter", type=int, default=100000)
    parser.add_argument("--tsne_k_neighbors", type=int, default=8)
    parser.add_argument("--ae_seed", type=int, default=42)
    parser.add_argument("--ae_hidden1", type=int, default=0)
    parser.add_argument("--ae_hidden2", type=int, default=0)
    parser.add_argument("--ae_epochs", type=int, default=80)
    parser.add_argument("--ae_batch_size", type=int, default=512)
    parser.add_argument("--ae_learning_rate", type=float, default=1e-3)
    parser.add_argument("--ae_device", type=str, default="cpu")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dpi", type=int, default=220)
    return parser


def main() -> None:
    configure_matplotlib()
    args = build_arg_parser().parse_args()
    if args.dim < 1:
        raise ValueError("--dim must be >= 1")
    if args.n_full_samples < 2:
        raise ValueError("--n_full_samples must be >= 2")
    if args.n_sub_samples < 1:
        raise ValueError("--n_sub_samples must be >= 1")
    if args.subspace_dim < 1:
        raise ValueError("--subspace_dim must be >= 1")
    if args.lora_rank < 1:
        raise ValueError("--lora_rank must be >= 1")
    if args.lora_blocks < 1:
        raise ValueError("--lora_blocks must be >= 1")
    if args.tsne_k_neighbors < 1:
        raise ValueError("--tsne_k_neighbors must be >= 1")
    if args.tsne_perplexity >= args.n_full_samples:
        raise ValueError("--tsne_perplexity must be < n_full_samples")
    if args.tsne_max_iter < 250:
        raise ValueError("--tsne_max_iter must be >= 250")
    if args.ae_epochs < 1:
        raise ValueError("--ae_epochs must be >= 1")
    if args.ae_batch_size < 1:
        raise ValueError("--ae_batch_size must be >= 1")
    if args.ae_learning_rate <= 0.0:
        raise ValueError("--ae_learning_rate must be > 0")

    func_ids = parse_functions(args.functions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Subspace projection coverage ({args.projection}): "
        f"D={args.dim}, full={args.n_full_samples}, sub={args.n_sub_samples}"
    )
    print(
        f"Subspace: {args.subspace_method}, assignment={args.subspace_assignment}, "
        f"rank={args.lora_rank}, d={args.subspace_dim}"
    )
    print(f"Functions: {', '.join(func_ids)}")
    print(f"Output root: {args.output_dir}")

    saved: list[tuple[str, list[Path]]] = []
    for fid in func_ids:
        print(f"  plotting {fid} ...", flush=True)
        paths = run_one_function(fid, args)
        saved.append((fid, paths))
        for path in paths:
            print(f"    -> {path.relative_to(args.output_dir)}")

    print("\nDone.")
    for fid, paths in saved:
        print(f"  {fid}: {len(paths)} files")


if __name__ == "__main__":
    main()
