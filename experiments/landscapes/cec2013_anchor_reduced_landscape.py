r"""Anchored reduced landscape views for CEC 2013 LSGO benchmarks.

This script answers the local landscape question by fixing an anchor point x0,
choosing two reduced directions or coordinates, and evaluating

    f(x0 + alpha u + beta v)

for PCA, or the analogous local reduced coordinates for t-SNE and an
autoencoder. The output includes a 3D surface, a 2D heat map with contours, and
an NPZ file containing the evaluated grid.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from experiments.landscapes.cec2013_projection_landscape import (
    configure_matplotlib,
    default_ae_hidden_dims,
    fit_autoencoder2,
    fit_pca2,
    parse_functions,
    prepare_surface,
    sample_uniform_in_bounds,
)
from evo_subspace.problems.lsgo import LSGOProblem
from evo_subspace.subspaces import build_subspace

PROJECTION_CHOICES = ("pca", "tsne", "autoencoder")
SUBSPACE_METHOD_CHOICES = (
    "random_projection",
    "random_blocking",
    "lora",
    "lora_ib",
    "lora_shared",
    "lora_gated",
    "lora_diag",
    "lora_rank1",
)
LANDSCAPE_METHOD_CHOICES = (*PROJECTION_CHOICES, *SUBSPACE_METHOD_CHOICES)
ANCHOR_CHOICES = ("best_sample", "center", "random")
FIGURES_ROOT = PROJECT_ROOT / "results" / "figures" / "lsgo"
DEFAULT_OUTPUT_ROOT = FIGURES_ROOT / "cec2013_anchor_reduced_landscapes"
_Z_LABEL = r"$f(x)$"
_SAVE_PAD_INCHES = 0.18


@dataclass(frozen=True)
class Anchor:
    x: np.ndarray
    fitness: float
    label: str


@dataclass(frozen=True)
class LandscapeGrid:
    alpha: np.ndarray
    beta: np.ndarray
    fitness: np.ndarray
    anchor: Anchor
    projection_label: str
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


def _load_anchor(path: Path, key: str) -> np.ndarray:
    data = np.load(path)
    if isinstance(data, np.lib.npyio.NpzFile):
        if key not in data:
            keys = ", ".join(data.files)
            raise KeyError(f"anchor key {key!r} not found in {path}; available keys: {keys}")
        x = data[key]
    else:
        x = data
    return np.asarray(x, dtype=float).reshape(-1)


def choose_anchor(
    problem: LSGOProblem,
    samples: np.ndarray,
    sample_fitness: np.ndarray,
    *,
    anchor_mode: str,
    anchor_path: Path | None,
    anchor_key: str,
    rng: np.random.Generator,
) -> Anchor:
    if anchor_path is not None:
        x = np.clip(_load_anchor(anchor_path, anchor_key), problem.lb, problem.ub)
        if x.size != problem.D:
            raise ValueError(f"anchor has dimension {x.size}, expected {problem.D}")
        f = float(problem.evaluate(x))
        return Anchor(x=x, fitness=f, label=f"file_{anchor_path.stem}")

    if anchor_mode == "center":
        x = 0.5 * (problem.lb + problem.ub)
        f = float(problem.evaluate(x))
        return Anchor(x=x, fitness=f, label="center")

    if anchor_mode == "random":
        x = sample_uniform_in_bounds(1, problem.lb, problem.ub, rng)[0]
        f = float(problem.evaluate(x))
        return Anchor(x=x, fitness=f, label="random")

    if anchor_mode == "best_sample":
        idx = int(np.argmin(sample_fitness))
        return Anchor(
            x=np.asarray(samples[idx], dtype=float).copy(),
            fitness=float(sample_fitness[idx]),
            label="best_sample",
        )

    raise ValueError(f"unknown anchor mode {anchor_mode!r}")


def _axis_offsets(values: np.ndarray, grid: int, span_scale: float) -> np.ndarray:
    spread = float(np.std(values))
    if not np.isfinite(spread) or spread <= 0.0:
        spread = max(float(np.max(values) - np.min(values)), 1.0)
    extent = max(span_scale * spread, 1e-12)
    return np.linspace(-extent, extent, grid)


def _evaluate_pca_anchor_grid(
    problem: LSGOProblem,
    samples: np.ndarray,
    anchor: Anchor,
    *,
    grid: int,
    span_scale: float,
) -> LandscapeGrid:
    mean, components, evr = fit_pca2(samples)
    projected = (samples - mean) @ components.T
    alpha_1d = _axis_offsets(projected[:, 0], grid, span_scale)
    beta_1d = _axis_offsets(projected[:, 1], grid, span_scale)
    alpha, beta = np.meshgrid(alpha_1d, beta_1d)

    flat = np.column_stack([alpha.ravel(), beta.ravel()])
    x = anchor.x + flat @ components
    x = np.clip(x, problem.lb, problem.ub)
    fitness = _evaluate_many(problem, x).reshape(alpha.shape)
    label = f"PCA EVR=({evr[0]:.3f}, {evr[1]:.3f})"
    return LandscapeGrid(
        alpha=alpha,
        beta=beta,
        fitness=fitness,
        anchor=anchor,
        projection_label=label,
        x_label=r"$\alpha$ along PC1",
        y_label=r"$\beta$ along PC2",
    )


def _fit_tsne2(
    samples: np.ndarray,
    *,
    seed: int,
    perplexity: float,
    max_iter: int,
) -> np.ndarray:
    try:
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise ImportError("t-SNE requires scikit-learn. Install with pip install scikit-learn") from exc

    if perplexity >= samples.shape[0]:
        raise ValueError("--tsne_perplexity must be smaller than the number of embedded samples")

    kwargs = {
        "n_components": 2,
        "perplexity": perplexity,
        "random_state": seed,
        "init": "pca",
        "learning_rate": "auto",
        "verbose": 1,
    }
    try:
        tsne = TSNE(max_iter=max_iter, **kwargs)
    except TypeError:
        tsne = TSNE(n_iter=max_iter, **kwargs)
    return tsne.fit_transform(samples)


def _knn_lift(
    embedding: np.ndarray,
    full_points: np.ndarray,
    query: np.ndarray,
    *,
    k_neighbors: int,
) -> np.ndarray:
    from scipy.spatial import cKDTree

    tree = cKDTree(embedding)
    k = min(k_neighbors, embedding.shape[0])
    lifted = np.empty((query.shape[0], full_points.shape[1]), dtype=float)
    for i, point in enumerate(query):
        dists, idx = tree.query(point, k=k)
        idx = np.atleast_1d(idx)
        dists = np.atleast_1d(dists).astype(float)
        weights = 1.0 / (dists + 1e-12)
        weights /= weights.sum()
        lifted[i] = weights @ full_points[idx]
    return lifted


def _evaluate_tsne_anchor_grid(
    problem: LSGOProblem,
    samples: np.ndarray,
    anchor: Anchor,
    *,
    grid: int,
    span_scale: float,
    seed: int,
    perplexity: float,
    max_iter: int,
    k_neighbors: int,
) -> LandscapeGrid:
    points = np.vstack([samples, anchor.x.reshape(1, -1)])
    embedding = _fit_tsne2(points, seed=seed, perplexity=perplexity, max_iter=max_iter)
    anchor_z = embedding[-1]
    alpha_1d = _axis_offsets(embedding[:, 0], grid, span_scale)
    beta_1d = _axis_offsets(embedding[:, 1], grid, span_scale)
    alpha, beta = np.meshgrid(alpha_1d, beta_1d)

    query = anchor_z + np.column_stack([alpha.ravel(), beta.ravel()])
    x = _knn_lift(embedding, points, query, k_neighbors=k_neighbors)
    x = np.clip(x, problem.lb, problem.ub)
    fitness = _evaluate_many(problem, x).reshape(alpha.shape)
    return LandscapeGrid(
        alpha=alpha,
        beta=beta,
        fitness=fitness,
        anchor=anchor,
        projection_label=f"t-SNE kNN lift, max_iter={max_iter}",
        x_label=r"$\alpha$ around t-SNE anchor",
        y_label=r"$\beta$ around t-SNE anchor",
    )


def _evaluate_autoencoder_anchor_grid(
    problem: LSGOProblem,
    samples: np.ndarray,
    anchor: Anchor,
    *,
    grid: int,
    span_scale: float,
    seed: int,
    hidden1: int,
    hidden2: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    decode_batch_size: int,
) -> LandscapeGrid:
    autoencoder = fit_autoencoder2(
        samples,
        problem.lb,
        problem.ub,
        seed=seed,
        hidden1=hidden1,
        hidden2=hidden2,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
    )
    embedding = autoencoder.encode(samples)
    anchor_z = autoencoder.encode(anchor.x.reshape(1, -1))[0]
    alpha_1d = _axis_offsets(embedding[:, 0], grid, span_scale)
    beta_1d = _axis_offsets(embedding[:, 1], grid, span_scale)
    alpha, beta = np.meshgrid(alpha_1d, beta_1d)

    flat_z = anchor_z + np.column_stack([alpha.ravel(), beta.ravel()])
    fitness = np.empty(flat_z.shape[0], dtype=float)
    for start in range(0, flat_z.shape[0], decode_batch_size):
        batch_z = flat_z[start : start + decode_batch_size]
        x = autoencoder.decode(batch_z)
        x = np.clip(x, problem.lb, problem.ub)
        fitness[start : start + x.shape[0]] = _evaluate_many(problem, x)
    return LandscapeGrid(
        alpha=alpha,
        beta=beta,
        fitness=fitness.reshape(alpha.shape),
        anchor=anchor,
        projection_label="autoencoder latent plane",
        x_label=r"$\alpha$ around latent $z_1$",
        y_label=r"$\beta$ around latent $z_2$",
    )


def _subspace_d_parameter(method: str, *, subspace_dim: int, lora_rank: int) -> int:
    if method.startswith("lora"):
        return lora_rank
    return subspace_dim


def _random_search_plane(search_dim: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if search_dim < 2:
        raise ValueError("subspace search dimension must be at least 2 for a 2D plane")
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(search_dim, 2))
    q, _ = np.linalg.qr(raw)
    return q[:, 0], q[:, 1]


def _evaluate_optimizer_subspace_grid(
    problem: LSGOProblem,
    anchor: Anchor,
    *,
    method: str,
    subspace_dim: int,
    lora_rank: int,
    lora_blocks: int,
    subspace_assignment: str,
    subspace_seed: int,
    subspace_device: str,
    plane_seed: int,
    grid: int,
    z_span: float,
) -> LandscapeGrid:
    d_param = _subspace_d_parameter(
        method,
        subspace_dim=subspace_dim,
        lora_rank=lora_rank,
    )
    subspace = build_subspace(
        method=method,
        D=problem.D,
        d=d_param,
        subspace_assignment=subspace_assignment,
        seed=subspace_seed,
        lb=problem.lb,
        ub=problem.ub,
        x0=anchor.x if subspace_assignment == "additive" else None,
        device=subspace_device,
        lora_blocks=lora_blocks,
    )
    u, v = _random_search_plane(subspace.search_dim, plane_seed)
    alpha_1d = np.linspace(-z_span, z_span, grid)
    beta_1d = np.linspace(-z_span, z_span, grid)
    alpha, beta = np.meshgrid(alpha_1d, beta_1d)

    flat = np.column_stack([alpha.ravel(), beta.ravel()])
    z = flat[:, :1] * u.reshape(1, -1) + flat[:, 1:2] * v.reshape(1, -1)
    x = subspace.expand(z)
    fitness = _evaluate_many(problem, x).reshape(alpha.shape)
    if method.startswith("lora"):
        param_label = f"rank={lora_rank}, search_dim={subspace.search_dim}"
    else:
        param_label = f"d={subspace_dim}, search_dim={subspace.search_dim}"
    if method.startswith("lora_"):
        param_label += f", blocks={lora_blocks}"
    return LandscapeGrid(
        alpha=alpha,
        beta=beta,
        fitness=fitness,
        anchor=anchor,
        projection_label=f"{method}, {subspace_assignment}, {param_label}",
        x_label=r"$\alpha$ along search direction $u$",
        y_label=r"$\beta$ along search direction $v$",
    )


def _log_contour_levels(z_plot: np.ndarray, n_levels: int = 12) -> np.ndarray:
    finite = z_plot[np.isfinite(z_plot)]
    if finite.size == 0:
        return np.array([], dtype=float)
    vmin = float(finite.min())
    vmax = float(finite.max())
    if vmax <= vmin:
        return np.array([], dtype=float)
    return np.geomspace(vmin, vmax, num=n_levels + 2)[1:-1]


def _save_figure(fig: plt.Figure, output_base: Path, dpi: int) -> tuple[Path, Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_base.with_suffix(".pdf")
    png_path = output_base.with_suffix(".png")
    fig.savefig(pdf_path, dpi=dpi, pad_inches=_SAVE_PAD_INCHES)
    fig.savefig(png_path, dpi=dpi, pad_inches=_SAVE_PAD_INCHES)
    plt.close(fig)
    return pdf_path, png_path


def save_surface(grid: LandscapeGrid, *, output_base: Path, dpi: int, z_clip_percentile: float | None) -> tuple[Path, Path]:
    z_plot, norm = prepare_surface(grid.fitness, z_clip_percentile)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        grid.alpha,
        grid.beta,
        z_plot,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
        alpha=0.92,
        norm=norm,
    )
    ax.set_xlabel(grid.x_label, labelpad=6)
    ax.set_ylabel(grid.y_label, labelpad=6)
    ax.set_zlabel(_Z_LABEL, labelpad=14)
    ax.set_zscale("log")
    ax.view_init(elev=25.0, azim=45.0)
    ax.set_title(grid.projection_label, fontsize=11, pad=10)
    cbar = fig.colorbar(surf, ax=ax, shrink=0.72, pad=0.12)
    cbar.ax.set_title(_Z_LABEL, fontsize=10, pad=6)
    fig.tight_layout()
    return _save_figure(fig, output_base.with_name(f"{output_base.name}_surface"), dpi)


def save_heatmap(grid: LandscapeGrid, *, output_base: Path, dpi: int, z_clip_percentile: float | None) -> tuple[Path, Path]:
    z_plot, norm = prepare_surface(grid.fitness, z_clip_percentile)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    heatmap = ax.pcolormesh(
        grid.alpha,
        grid.beta,
        z_plot,
        cmap="viridis",
        norm=norm,
        shading="auto",
    )
    levels = _log_contour_levels(z_plot)
    if levels.size > 0:
        contours = ax.contour(
            grid.alpha,
            grid.beta,
            z_plot,
            levels=levels,
            colors="white",
            linewidths=0.6,
            alpha=0.85,
        )
        ax.clabel(contours, inline=True, fontsize=7, fmt="%.1e")
    ax.scatter([0.0], [0.0], c="red", s=28, marker="x", label=r"$x_0$")
    ax.set_xlabel(grid.x_label)
    ax.set_ylabel(grid.y_label)
    ax.set_title(grid.projection_label, fontsize=11)
    ax.legend(loc="best", fontsize=8)
    cbar = fig.colorbar(heatmap, ax=ax, pad=0.03)
    cbar.ax.set_title(_Z_LABEL, fontsize=10, pad=6)
    fig.tight_layout()
    return _save_figure(fig, output_base.with_name(f"{output_base.name}_heatmap"), dpi)


def save_grid_npz(grid: LandscapeGrid, *, output_base: Path) -> Path:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    path = output_base.with_name(f"{output_base.name}_grid.npz")
    np.savez_compressed(
        path,
        alpha=grid.alpha,
        beta=grid.beta,
        fitness=grid.fitness,
        anchor_x=grid.anchor.x,
        anchor_fitness=np.asarray([grid.anchor.fitness], dtype=float),
        anchor_label=np.asarray([grid.anchor.label]),
        projection_label=np.asarray([grid.projection_label]),
    )
    return path


def build_reduced_landscape(
    problem: LSGOProblem,
    samples: np.ndarray,
    sample_fitness: np.ndarray,
    *,
    projection: str,
    anchor_mode: str,
    anchor_path: Path | None,
    anchor_key: str,
    rng: np.random.Generator,
    grid: int,
    span_scale: float,
    tsne_seed: int,
    tsne_perplexity: float,
    tsne_max_iter: int,
    tsne_k_neighbors: int,
    ae_seed: int,
    ae_hidden1: int,
    ae_hidden2: int,
    ae_epochs: int,
    ae_batch_size: int,
    ae_learning_rate: float,
    ae_device: str,
    ae_decode_batch_size: int,
    subspace_dim: int,
    lora_rank: int,
    lora_blocks: int,
    subspace_assignment: str,
    subspace_seed: int,
    subspace_device: str,
    plane_seed: int,
    z_span: float,
) -> LandscapeGrid:
    anchor = choose_anchor(
        problem,
        samples,
        sample_fitness,
        anchor_mode=anchor_mode,
        anchor_path=anchor_path,
        anchor_key=anchor_key,
        rng=rng,
    )
    print(f"      anchor={anchor.label}, f(x0)={anchor.fitness:.6e}", flush=True)

    if projection in SUBSPACE_METHOD_CHOICES:
        return _evaluate_optimizer_subspace_grid(
            problem,
            anchor,
            method=projection,
            subspace_dim=subspace_dim,
            lora_rank=lora_rank,
            lora_blocks=lora_blocks,
            subspace_assignment=subspace_assignment,
            subspace_seed=subspace_seed,
            subspace_device=subspace_device,
            plane_seed=plane_seed,
            grid=grid,
            z_span=z_span,
        )

    if projection == "pca":
        return _evaluate_pca_anchor_grid(
            problem,
            samples,
            anchor,
            grid=grid,
            span_scale=span_scale,
        )

    if projection == "tsne":
        return _evaluate_tsne_anchor_grid(
            problem,
            samples,
            anchor,
            grid=grid,
            span_scale=span_scale,
            seed=tsne_seed,
            perplexity=tsne_perplexity,
            max_iter=tsne_max_iter,
            k_neighbors=tsne_k_neighbors,
        )

    if projection == "autoencoder":
        return _evaluate_autoencoder_anchor_grid(
            problem,
            samples,
            anchor,
            grid=grid,
            span_scale=span_scale,
            seed=ae_seed,
            hidden1=ae_hidden1,
            hidden2=ae_hidden2,
            epochs=ae_epochs,
            batch_size=ae_batch_size,
            learning_rate=ae_learning_rate,
            device=ae_device,
            decode_batch_size=ae_decode_batch_size,
        )

    raise ValueError(f"unknown projection {projection!r}")


def output_base_for(
    *,
    output_dir: Path,
    projection: str,
    func_short: str,
    dim: int,
    anchor_label: str,
) -> Path:
    stem = f"{func_short}_{projection}_anchor_{_safe_stem(anchor_label)}_d{dim}"
    return output_dir / projection / stem


def plot_one_function(func_id: str, args: argparse.Namespace) -> list[Path]:
    problem = LSGOProblem(
        func_id=func_id,
        D=args.dim,
        seed=args.benchmark_seed,
        group_size=args.group_size,
    )
    rng = np.random.default_rng(args.sample_seed)
    samples = sample_uniform_in_bounds(args.n_samples, problem.lb, problem.ub, rng)
    sample_fitness = _evaluate_many(problem, samples)
    h1 = args.ae_hidden1 if args.ae_hidden1 > 0 else default_ae_hidden_dims(args.dim)[0]
    h2 = args.ae_hidden2 if args.ae_hidden2 > 0 else default_ae_hidden_dims(args.dim)[1]

    grid = build_reduced_landscape(
        problem,
        samples,
        sample_fitness,
        projection=args.projection,
        anchor_mode=args.anchor,
        anchor_path=args.anchor_path,
        anchor_key=args.anchor_key,
        rng=rng,
        grid=args.grid,
        span_scale=args.span_scale,
        tsne_seed=args.tsne_seed,
        tsne_perplexity=args.tsne_perplexity,
        tsne_max_iter=args.tsne_max_iter,
        tsne_k_neighbors=args.tsne_k_neighbors,
        ae_seed=args.ae_seed,
        ae_hidden1=h1,
        ae_hidden2=h2,
        ae_epochs=args.ae_epochs,
        ae_batch_size=args.ae_batch_size,
        ae_learning_rate=args.ae_learning_rate,
        ae_device=args.ae_device,
        ae_decode_batch_size=args.ae_decode_batch_size,
        subspace_dim=args.subspace_dim,
        lora_rank=args.lora_rank,
        lora_blocks=args.lora_blocks,
        subspace_assignment=args.subspace_assignment,
        subspace_seed=args.subspace_seed,
        subspace_device=args.subspace_device,
        plane_seed=args.plane_seed,
        z_span=args.z_span,
    )

    short = func_id.replace("cec2013_lsgo_", "")
    output_base = output_base_for(
        output_dir=args.output_dir,
        projection=args.projection,
        func_short=short,
        dim=args.dim,
        anchor_label=grid.anchor.label,
    )
    paths: list[Path] = []
    paths.extend(
        save_surface(
            grid,
            output_base=output_base,
            dpi=args.dpi,
            z_clip_percentile=args.z_clip_percentile,
        )
    )
    paths.extend(
        save_heatmap(
            grid,
            output_base=output_base,
            dpi=args.dpi,
            z_clip_percentile=args.z_clip_percentile,
        )
    )
    paths.append(save_grid_npz(grid, output_base=output_base))
    return paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Anchored CEC 2013 reduced landscape surface and contour plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--projection", choices=LANDSCAPE_METHOD_CHOICES, default="pca")
    parser.add_argument("--functions", type=str, default="f1")
    parser.add_argument("--dim", type=int, default=1000)
    parser.add_argument("--benchmark_seed", type=int, default=0)
    parser.add_argument("--group_size", type=int, default=50)
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--grid", type=int, default=41)
    parser.add_argument(
        "--span_scale",
        type=float,
        default=1.0,
        help="Grid half width as a multiple of reduced coordinate standard deviation.",
    )
    parser.add_argument("--anchor", choices=ANCHOR_CHOICES, default="best_sample")
    parser.add_argument("--anchor_path", type=Path, default=None)
    parser.add_argument("--anchor_key", type=str, default="global_best_x_full")
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
    parser.add_argument("--ae_decode_batch_size", type=int, default=512)
    parser.add_argument("--subspace_dim", type=int, default=10)
    parser.add_argument("--lora_rank", type=int, default=1)
    parser.add_argument("--lora_blocks", type=int, default=10)
    parser.add_argument(
        "--subspace_assignment",
        choices=("absolute", "additive"),
        default="additive",
        help="Assignment mode for optimizer subspace methods.",
    )
    parser.add_argument("--subspace_seed", type=int, default=42)
    parser.add_argument("--subspace_device", type=str, default="cpu")
    parser.add_argument(
        "--plane_seed",
        type=int,
        default=123,
        help="Seed for the two search-space directions used by optimizer subspaces.",
    )
    parser.add_argument(
        "--z_span",
        type=float,
        default=10.0,
        help="Half width for alpha and beta when plotting optimizer subspace planes.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root. Files are grouped under a projection subfolder.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--z_clip_percentile", type=float, default=99.0)
    return parser


def main() -> None:
    configure_matplotlib()
    args = build_arg_parser().parse_args()
    if args.dim < 1:
        raise ValueError("--dim must be >= 1")
    if args.grid < 5:
        raise ValueError("--grid must be >= 5")
    if args.n_samples < 10:
        raise ValueError("--n_samples must be >= 10")
    if args.span_scale <= 0.0:
        raise ValueError("--span_scale must be > 0")
    if args.tsne_k_neighbors < 1:
        raise ValueError("--tsne_k_neighbors must be >= 1")
    if args.tsne_max_iter < 250:
        raise ValueError("--tsne_max_iter must be >= 250")
    if args.ae_epochs < 1:
        raise ValueError("--ae_epochs must be >= 1")
    if args.ae_batch_size < 1:
        raise ValueError("--ae_batch_size must be >= 1")
    if args.ae_decode_batch_size < 1:
        raise ValueError("--ae_decode_batch_size must be >= 1")
    if args.ae_learning_rate <= 0.0:
        raise ValueError("--ae_learning_rate must be > 0")
    if args.subspace_dim < 2:
        raise ValueError("--subspace_dim must be >= 2")
    if args.lora_rank < 1:
        raise ValueError("--lora_rank must be >= 1")
    if args.lora_blocks < 1:
        raise ValueError("--lora_blocks must be >= 1")
    if args.z_span <= 0.0:
        raise ValueError("--z_span must be > 0")
    if args.z_clip_percentile is not None and not (0.0 < args.z_clip_percentile <= 100.0):
        raise ValueError("--z_clip_percentile must be in (0, 100] or disabled")

    func_ids = parse_functions(args.functions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Anchored reduced landscapes ({args.projection}): D={args.dim}, "
        f"grid={args.grid}x{args.grid}, n_samples={args.n_samples}"
    )
    print(f"Functions: {', '.join(func_ids)}")
    print(f"Output root: {args.output_dir}")

    saved: list[tuple[str, list[Path]]] = []
    for fid in func_ids:
        print(f"  plotting {fid} ...", flush=True)
        paths = plot_one_function(fid, args)
        saved.append((fid, paths))
        for path in paths:
            print(f"    -> {path.relative_to(args.output_dir)}")

    print("\nDone.")
    for fid, paths in saved:
        print(f"  {fid}: {len(paths)} files")


if __name__ == "__main__":
    main()
