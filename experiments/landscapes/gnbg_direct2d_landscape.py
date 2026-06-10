r"""Direct 2D GNBG landscape slices without projection.

The provided GNBG instances are 30 dimensional. This script visualizes a direct
2D slice by varying two selected coordinates and holding all remaining
coordinates fixed at an anchor, using the known optimum position by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from experiments.landscapes.cec2013_projection_landscape import configure_matplotlib
from experiments.landscapes.gnbg_projection_landscape import (
    SCALE_CHOICES,
    _linear_contour_levels,
    _log_contour_levels,
    _save_figure,
    load_gnbg_problem,
    parse_functions,
    prepare_plot_values,
    save_data,
    save_heatmap,
)

FIGURES_ROOT = PROJECT_ROOT / "results" / "figures" / "gnbg"
DEFAULT_OUTPUT_ROOT = FIGURES_ROOT / "gnbg_direct2d_landscapes"


def _finite_min_max(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("landscape has no finite values")
    return float(finite.min()), float(finite.max())


def parse_axes(spec: str) -> tuple[int, int]:
    parts = [part.strip() for part in spec.split(",")]
    if len(parts) != 2:
        raise ValueError("--axes must contain exactly two comma-separated coordinates")
    axes = tuple(int(part) - 1 for part in parts)
    if axes[0] == axes[1]:
        raise ValueError("--axes must select two distinct coordinates")
    if min(axes) < 0:
        raise ValueError("--axes is one-indexed and must be >= 1")
    return axes


def build_direct2d_grid(problem, axes: tuple[int, int], grid: int, anchor: str):
    if max(axes) >= problem.dimension:
        raise ValueError(f"axes exceed GNBG dimension {problem.dimension}")
    if anchor == "optimum":
        base = problem.optimum_position.copy()
    elif anchor == "center":
        base = 0.5 * (problem.lb + problem.ub)
    else:
        raise ValueError(f"unknown anchor {anchor!r}")

    x1 = np.linspace(problem.min_coordinate, problem.max_coordinate, grid)
    x2 = np.linspace(problem.min_coordinate, problem.max_coordinate, grid)
    p1, p2 = np.meshgrid(x1, x2)
    points = np.repeat(base.reshape(1, -1), p1.size, axis=0)
    points[:, axes[0]] = p1.ravel()
    points[:, axes[1]] = p2.ravel()
    z = problem.evaluate_batch(points).reshape(p1.shape)
    x_label = rf"$x_{{{axes[0] + 1}}}$"
    y_label = rf"$x_{{{axes[1] + 1}}}$"
    return p1, p2, z, x_label, y_label


def save_direct2d_surface(
    *,
    p1: np.ndarray,
    p2: np.ndarray,
    z: np.ndarray,
    x_label: str,
    y_label: str,
    z_label: str,
    output_base: Path,
    dpi: int,
    z_clip_percentile: float | None,
    scale: str,
    log_dynamic_range: float,
) -> tuple[Path, Path]:
    z_plot, norm, scale_used = prepare_plot_values(
        z,
        z_clip_percentile=z_clip_percentile,
        scale=scale,
        log_dynamic_range=log_dynamic_range,
    )
    vmin, vmax = _finite_min_max(z_plot)
    z_range = max(vmax - vmin, np.finfo(float).eps)
    floor = 0.0 if vmin >= 0.0 else vmin - 0.04 * z_range

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        p1,
        p2,
        z_plot,
        cmap="jet",
        linewidth=0,
        antialiased=True,
        norm=norm,
    )

    levels = _log_contour_levels(z_plot) if scale_used == "log" else _linear_contour_levels(z_plot)
    if levels.size > 0:
        ax.contour(
            p1,
            p2,
            z_plot,
            zdir="z",
            offset=floor,
            levels=levels,
            cmap="jet",
            linewidths=0.7,
            norm=norm,
        )

    ax.set_xlabel(x_label, labelpad=7)
    ax.set_ylabel(y_label, labelpad=7)
    ax.set_zlabel(z_label, labelpad=9)
    ax.set_zlim(floor, vmax)
    ax.view_init(elev=25.0, azim=-135.0)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
    return _save_figure(fig, output_base.with_name(f"{output_base.name}_surface"), dpi)


def output_base_for(func_id: str, args: argparse.Namespace) -> Path:
    axes_label = args.axes.replace(",", "_")
    stem = f"{func_id}_direct2d_x{axes_label}_anchor_{args.anchor}_landscape"
    return args.output_dir / f"x{axes_label}" / args.anchor / stem


def plot_one_function(func_id: str, args: argparse.Namespace) -> list[Path]:
    problem = load_gnbg_problem(func_id)
    axes = parse_axes(args.axes)
    p1, p2, z, x_label, y_label = build_direct2d_grid(
        problem,
        axes=axes,
        grid=args.grid,
        anchor=args.anchor,
    )
    output_base = output_base_for(func_id, args)
    z_label = rf"$f(x_{{{axes[0] + 1}}},x_{{{axes[1] + 1}}})$"
    paths: list[Path] = []
    paths.extend(
        save_direct2d_surface(
            p1=p1,
            p2=p2,
            z=z,
            x_label=x_label,
            y_label=y_label,
            z_label=z_label,
            output_base=output_base,
            dpi=args.dpi,
            z_clip_percentile=args.z_clip_percentile,
            scale=args.scale,
            log_dynamic_range=args.log_dynamic_range,
        )
    )
    paths.extend(
        save_heatmap(
            p1=p1,
            p2=p2,
            z=z,
            x_label=x_label,
            y_label=y_label,
            output_base=output_base,
            dpi=args.dpi,
            z_clip_percentile=args.z_clip_percentile,
            scale=args.scale,
            log_dynamic_range=args.log_dynamic_range,
        )
    )
    paths.append(save_data(output_base, p1, p2, z))
    return paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Direct 2D GNBG landscape slices without projection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--functions", type=str, default="all")
    parser.add_argument("--axes", type=str, default="1,2")
    parser.add_argument("--anchor", choices=("optimum", "center"), default="optimum")
    parser.add_argument("--grid", type=int, default=500)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--z_clip_percentile", type=float, default=99.0)
    parser.add_argument("--scale", choices=SCALE_CHOICES, default="auto")
    parser.add_argument("--log_dynamic_range", type=float, default=1e3)
    return parser


def main() -> None:
    configure_matplotlib()
    args = build_arg_parser().parse_args()
    if args.grid < 5:
        raise ValueError("--grid must be >= 5")
    if args.z_clip_percentile is not None and not (0.0 < args.z_clip_percentile <= 100.0):
        raise ValueError("--z_clip_percentile must be in (0, 100] or disabled")
    if args.log_dynamic_range <= 1.0:
        raise ValueError("--log_dynamic_range must be > 1")

    func_ids = parse_functions(args.functions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"GNBG direct 2D landscapes: functions={len(func_ids)}, "
        f"axes={args.axes}, grid={args.grid}x{args.grid}, anchor={args.anchor}"
    )
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
