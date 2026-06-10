r"""Overlay expanded subspace samples on a 3D projected CEC-2013 landscape.

This keeps the same landscape style as ``cec2013_projection_landscape.py``:
the surface is a regular grid in a learned 2D projection, with height f(x).
Expanded subspace samples, such as LoRA rank 1 samples, are projected into the
same 2D coordinates and drawn as points at their actual objective value.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from experiments.landscapes.cec2013_projection_landscape import (
    _add_colorbar_right,
    _plot_surface,
    configure_matplotlib,
    evaluate_grid_pca,
    fit_pca2,
    parse_functions,
    pca_grid_bounds,
    prepare_surface,
    sample_uniform_in_bounds,
)
from evo_subspace.problems.lsgo import LSGOProblem
from evo_subspace.subspaces import build_subspace

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
DEFAULT_OUTPUT_ROOT = FIGURES_ROOT / "cec2013_subspace_landscape_overlay"
_Z_LABEL = r"$f(x)$"
_POSITIVE_FLOOR = 1e-30
_SAVE_PAD_INCHES = 0.18


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


def _match_surface_log_scale(values: np.ndarray, surface_min: float) -> np.ndarray:
    z = np.asarray(values, dtype=float).copy()
    if surface_min <= 0.0:
        z = z - surface_min + _POSITIVE_FLOOR
    return np.maximum(z, _POSITIVE_FLOOR)


def _sample_subspace(
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
    anchor_x: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, str]:
    d_param = lora_rank if method.startswith("lora") else subspace_dim
    subspace = build_subspace(
        method=method,
        D=problem.D,
        d=d_param,
        subspace_assignment=assignment,
        seed=seed,
        lb=problem.lb,
        ub=problem.ub,
        x0=anchor_x if assignment == "additive" else None,
        device=device,
        lora_blocks=lora_blocks,
    )
    rng = np.random.default_rng(seed + 1009)
    low = float(problem.lb[0])
    high = float(problem.ub[0])
    z = rng.uniform(low, high, size=(n_samples, subspace.search_dim))
    x = np.asarray(subspace.expand(z), dtype=float)
    anchor = subspace.x0 if assignment == "additive" else None
    if method.startswith("lora"):
        label = f"{method}, rank={lora_rank}, search dim={subspace.search_dim}"
    else:
        label = f"{method}, d={subspace_dim}, search dim={subspace.search_dim}"
    if method.startswith("lora_"):
        label += f", blocks={lora_blocks}"
    label += f", {assignment}"
    if assignment == "additive" and anchor_x is not None:
        label += ", x0=best full sample"
    return z, x, anchor, label


def _save_figure(fig: plt.Figure, output_base: Path, dpi: int) -> tuple[Path, Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_base.with_suffix(".pdf")
    png_path = output_base.with_suffix(".png")
    fig.savefig(pdf_path, dpi=dpi, pad_inches=_SAVE_PAD_INCHES)
    fig.savefig(png_path, dpi=dpi, pad_inches=_SAVE_PAD_INCHES)
    plt.close(fig)
    return pdf_path, png_path


def save_3d_overlay(
    *,
    p1: np.ndarray,
    p2: np.ndarray,
    surface_f: np.ndarray,
    sub_2d: np.ndarray,
    sub_f: np.ndarray,
    anchor_2d: np.ndarray | None,
    anchor_f: float | None,
    projection_label: str,
    subspace_label: str,
    output_base: Path,
    dpi: int,
    z_clip_percentile: float | None,
    overlay_z_lift: float,
) -> tuple[Path, Path]:
    surface_z, norm = prepare_surface(surface_f, z_clip_percentile)
    sub_z = _match_surface_log_scale(sub_f, float(surface_f.min()))
    if z_clip_percentile is not None:
        sub_z = np.minimum(sub_z, float(surface_z.max()))
    sub_z = sub_z * overlay_z_lift
    anchor_z = None
    if anchor_f is not None:
        anchor_z = _match_surface_log_scale(np.asarray([anchor_f]), float(surface_f.min()))
        if z_clip_percentile is not None:
            anchor_z = np.minimum(anchor_z, float(surface_z.max()))
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = _plot_surface(
        ax,
        p1,
        p2,
        surface_z,
        norm=norm,
        azim=45.0,
        x_label="PC1",
        y_label="PC2",
        z_tick_pad=14.0,
        z_labelpad=18.0,
    )
    ax.scatter(
        sub_2d[:, 0],
        sub_2d[:, 1],
        sub_z,
        c="#d62728",
        s=7,
        alpha=0.45,
        depthshade=False,
        label="expanded subspace samples",
    )
    if anchor_2d is not None and anchor_z is not None:
        ax.scatter(
            [anchor_2d[0]],
            [anchor_2d[1]],
            anchor_z,
            c="black",
            s=72,
            marker="*",
            edgecolors="white",
            linewidths=0.7,
            depthshade=False,
            label="additive anchor $x_0$",
        )
    ax.legend(loc="upper left", fontsize=8)
    fig.subplots_adjust(left=0.10, right=0.66, bottom=0.10, top=0.94)
    _add_colorbar_right(fig, surf, cbar_left=0.87)
    return _save_figure(fig, output_base.with_name(f"{output_base.name}_3d_overlay"), dpi)


def save_2d_overlay(
    *,
    p1: np.ndarray,
    p2: np.ndarray,
    surface_f: np.ndarray,
    sub_2d: np.ndarray,
    anchor_2d: np.ndarray | None,
    projection_label: str,
    subspace_label: str,
    output_base: Path,
    dpi: int,
    z_clip_percentile: float | None,
) -> tuple[Path, Path]:
    surface_z, norm = prepare_surface(surface_f, z_clip_percentile)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    heat = ax.pcolormesh(p1, p2, surface_z, cmap="viridis", norm=norm, shading="auto")
    ax.scatter(
        sub_2d[:, 0],
        sub_2d[:, 1],
        c="#d62728",
        s=6,
        alpha=0.35,
        label="expanded subspace samples",
        rasterized=True,
    )
    if anchor_2d is not None:
        ax.scatter(
            [anchor_2d[0]],
            [anchor_2d[1]],
            c="black",
            s=72,
            marker="*",
            edgecolors="white",
            linewidths=0.7,
            label="additive anchor $x_0$",
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"{projection_label}\n{subspace_label}", fontsize=10)
    ax.legend(loc="best", fontsize=8)
    cbar = fig.colorbar(heat, ax=ax, pad=0.03)
    cbar.ax.set_title(_Z_LABEL, fontsize=10, pad=6)
    fig.tight_layout()
    return _save_figure(fig, output_base.with_name(f"{output_base.name}_2d_overlay"), dpi)


def save_data(
    *,
    output_base: Path,
    p1: np.ndarray,
    p2: np.ndarray,
    surface_f: np.ndarray,
    full_samples: np.ndarray,
    full_2d: np.ndarray,
    sub_z: np.ndarray,
    sub_x: np.ndarray,
    sub_2d: np.ndarray,
    sub_f: np.ndarray,
    anchor_x: np.ndarray | None,
    anchor_2d: np.ndarray | None,
    anchor_f: float | None,
    projection_label: str,
    subspace_label: str,
) -> Path:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    path = output_base.with_name(f"{output_base.name}_data.npz")
    np.savez_compressed(
        path,
        p1=p1,
        p2=p2,
        surface_fitness=surface_f,
        full_samples=full_samples,
        full_2d=full_2d,
        sub_z=sub_z,
        sub_x=sub_x,
        sub_2d=sub_2d,
        sub_fitness=sub_f,
        anchor_x=np.asarray([] if anchor_x is None else anchor_x),
        anchor_2d=np.asarray([] if anchor_2d is None else anchor_2d),
        anchor_fitness=np.asarray([] if anchor_f is None else [anchor_f], dtype=float),
        projection_label=np.asarray([projection_label]),
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
    stem = f"{short}_pca_{method}_n{args.n_full_samples}_d{args.dim}"
    return args.output_dir / "pca" / args.subspace_method / stem


def run_one_function(func_id: str, args: argparse.Namespace) -> list[Path]:
    problem = LSGOProblem(
        func_id=func_id,
        D=args.dim,
        seed=args.benchmark_seed,
        group_size=args.group_size,
    )
    rng = np.random.default_rng(args.sample_seed)
    full_samples = sample_uniform_in_bounds(args.n_full_samples, problem.lb, problem.ub, rng)
    full_f = _evaluate_many(problem, full_samples)
    best_full_idx = int(np.argmin(full_f))
    additive_anchor = full_samples[best_full_idx].copy()
    additive_anchor_f = float(full_f[best_full_idx])
    if args.subspace_assignment == "additive":
        print(
            f"      additive x0=best full sample, f(x0)={additive_anchor_f:.6e}",
            flush=True,
        )
    mean, components, evr = fit_pca2(full_samples)
    lo, hi = pca_grid_bounds(full_samples, mean, components, args.margin)
    p1_1d = np.linspace(lo[0], hi[0], args.grid)
    p2_1d = np.linspace(lo[1], hi[1], args.grid)
    p1, p2 = np.meshgrid(p1_1d, p2_1d)
    surface_f = evaluate_grid_pca(problem, mean, components, p1, p2)

    sub_z, sub_x, anchor_x, subspace_label = _sample_subspace(
        problem,
        method=args.subspace_method,
        n_samples=args.n_sub_samples,
        subspace_dim=args.subspace_dim,
        lora_rank=args.lora_rank,
        lora_blocks=args.lora_blocks,
        assignment=args.subspace_assignment,
        seed=args.subspace_seed,
        device=args.subspace_device,
        anchor_x=additive_anchor,
    )
    sub_2d = (sub_x - mean) @ components.T
    sub_f = _evaluate_many(problem, sub_x)
    anchor_2d = None
    anchor_f = None
    if anchor_x is not None:
        anchor_2d = ((anchor_x.reshape(1, -1) - mean) @ components.T).reshape(-1)
        anchor_f = float(problem.evaluate(anchor_x))
    full_2d = (full_samples - mean) @ components.T
    projection_label = f"PCA landscape trained on full samples, EVR=({evr[0]:.3f}, {evr[1]:.3f})"
    output_base = output_base_for(func_id, args)

    paths: list[Path] = []
    paths.extend(
        save_3d_overlay(
            p1=p1,
            p2=p2,
            surface_f=surface_f,
            sub_2d=sub_2d,
            sub_f=sub_f,
            anchor_2d=anchor_2d,
            anchor_f=anchor_f,
            projection_label=projection_label,
            subspace_label=subspace_label,
            output_base=output_base,
            dpi=args.dpi,
            z_clip_percentile=args.z_clip_percentile,
            overlay_z_lift=args.overlay_z_lift,
        )
    )
    paths.extend(
        save_2d_overlay(
            p1=p1,
            p2=p2,
            surface_f=surface_f,
            sub_2d=sub_2d,
            anchor_2d=anchor_2d,
            projection_label=projection_label,
            subspace_label=subspace_label,
            output_base=output_base,
            dpi=args.dpi,
            z_clip_percentile=args.z_clip_percentile,
        )
    )
    paths.append(
        save_data(
            output_base=output_base,
            p1=p1,
            p2=p2,
            surface_f=surface_f,
            full_samples=full_samples,
            full_2d=full_2d,
            sub_z=sub_z,
            sub_x=sub_x,
            sub_2d=sub_2d,
            sub_f=sub_f,
            anchor_x=anchor_x,
            anchor_2d=anchor_2d,
            anchor_f=anchor_f,
            projection_label=projection_label,
            subspace_label=subspace_label,
        )
    )
    return paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Overlay expanded subspace samples on a PCA 3D landscape.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--subspace_method", choices=SUBSPACE_CHOICES, default="lora")
    parser.add_argument("--functions", type=str, default="f1")
    parser.add_argument("--dim", type=int, default=1000)
    parser.add_argument("--benchmark_seed", type=int, default=0)
    parser.add_argument("--group_size", type=int, default=50)
    parser.add_argument("--n_full_samples", type=int, default=5000)
    parser.add_argument("--n_sub_samples", type=int, default=5000)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--subspace_seed", type=int, default=42)
    parser.add_argument("--grid", type=int, default=50)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--subspace_dim", type=int, default=10)
    parser.add_argument("--lora_rank", type=int, default=1)
    parser.add_argument("--lora_blocks", type=int, default=10)
    parser.add_argument(
        "--subspace_assignment",
        choices=("absolute", "additive"),
        default="additive",
    )
    parser.add_argument("--subspace_device", type=str, default="cpu")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--z_clip_percentile", type=float, default=99.0)
    parser.add_argument(
        "--overlay_z_lift",
        type=float,
        default=1.08,
        help="Multiplicative z lift for overlay points on the log-scale surface.",
    )
    return parser


def main() -> None:
    configure_matplotlib()
    args = build_arg_parser().parse_args()
    if args.dim < 1:
        raise ValueError("--dim must be >= 1")
    if args.grid < 5:
        raise ValueError("--grid must be >= 5")
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
    if args.z_clip_percentile is not None and not (0.0 < args.z_clip_percentile <= 100.0):
        raise ValueError("--z_clip_percentile must be in (0, 100] or disabled")
    if args.overlay_z_lift <= 0.0:
        raise ValueError("--overlay_z_lift must be > 0")

    func_ids = parse_functions(args.functions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Subspace landscape overlay: D={args.dim}, grid={args.grid}x{args.grid}, "
        f"full={args.n_full_samples}, sub={args.n_sub_samples}"
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
