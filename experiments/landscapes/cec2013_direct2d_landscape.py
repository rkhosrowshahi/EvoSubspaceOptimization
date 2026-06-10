r"""Direct 2D CEC-2013 LSGO landscape slices without projection.

This script varies two selected coordinates and holds all remaining coordinates
fixed at an anchor. The default anchor is the center of the CEC search box.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from experiments.landscapes.cec2013_projection_landscape import (
    SCALE_CHOICES,
    configure_matplotlib,
    parse_functions,
    save_data,
    save_heatmap,
    save_surface,
)
from evo_subspace.problems.lsgo import LSGOProblem

FIGURES_ROOT = PROJECT_ROOT / "results" / "figures" / "lsgo"
DEFAULT_OUTPUT_ROOT = FIGURES_ROOT / "cec2013_direct2d_landscapes"


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


def shift_anchor(problem: LSGOProblem) -> np.ndarray:
    func = problem._func
    base = 0.5 * (problem.lb + problem.ub)
    if func.func_type == "conflict":
        raise ValueError(
            f"{problem.func_id} has per-group conflict shifts, so --anchor shift is not defined"
        )
    if not hasattr(func, "_xopt"):
        raise ValueError(f"{problem.func_id} does not expose a shift vector")
    xopt = np.asarray(func._xopt, dtype=float).ravel()
    if xopt.size == problem.D:
        return xopt.copy()
    if getattr(func, "func_type", None) == "conform" and xopt.size <= problem.D:
        base[: xopt.size] = xopt
        return base
    raise ValueError(
        f"{problem.func_id} shift vector has length {xopt.size}, expected {problem.D}"
    )


def build_anchor(problem: LSGOProblem, anchor: str) -> np.ndarray:
    if anchor == "center":
        return 0.5 * (problem.lb + problem.ub)
    if anchor == "shift":
        return shift_anchor(problem)
    raise ValueError(f"unknown anchor {anchor!r}")


def build_direct2d_grid(
    problem: LSGOProblem,
    axes: tuple[int, int],
    grid: int,
    anchor: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    if max(axes) >= problem.D:
        raise ValueError(f"axes exceed CEC dimension {problem.D}")
    base = build_anchor(problem, anchor)
    x1 = np.linspace(problem.lb[axes[0]], problem.ub[axes[0]], grid)
    x2 = np.linspace(problem.lb[axes[1]], problem.ub[axes[1]], grid)
    p1, p2 = np.meshgrid(x1, x2)
    points = np.repeat(base.reshape(1, -1), p1.size, axis=0)
    points[:, axes[0]] = p1.ravel()
    points[:, axes[1]] = p2.ravel()
    z = np.empty(points.shape[0], dtype=float)
    for i, row in enumerate(points):
        z[i] = problem.evaluate(row)
    x_label = rf"$x_{{{axes[0] + 1}}}$"
    y_label = rf"$x_{{{axes[1] + 1}}}$"
    return p1, p2, z.reshape(p1.shape), x_label, y_label


def output_base_for(func_id: str, args: argparse.Namespace) -> Path:
    axes_label = args.axes.replace(",", "_")
    short = func_id.replace("cec2013_lsgo_", "")
    stem = f"{short}_direct2d_x{axes_label}_anchor_{args.anchor}_landscape"
    return args.output_dir / f"x{axes_label}" / args.anchor / stem


def plot_one_function(func_id: str, args: argparse.Namespace) -> list[Path]:
    problem = LSGOProblem(
        func_id=func_id,
        D=args.dim,
        seed=args.benchmark_seed,
        group_size=args.group_size,
    )
    axes = parse_axes(args.axes)
    p1, p2, z, x_label, y_label = build_direct2d_grid(
        problem,
        axes=axes,
        grid=args.grid,
        anchor=args.anchor,
    )
    output_base = output_base_for(func_id, args)
    paths: list[Path] = []
    paths.extend(
        save_surface(
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
        description="Direct 2D CEC-2013 LSGO landscape slices without projection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--functions", type=str, default="all")
    parser.add_argument("--dim", type=int, default=1000)
    parser.add_argument("--benchmark_seed", type=int, default=0)
    parser.add_argument("--group_size", type=int, default=50)
    parser.add_argument("--axes", type=str, default="1,2")
    parser.add_argument("--anchor", choices=("center", "shift"), default="center")
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
    if args.dim < 1:
        raise ValueError("--dim must be >= 1")
    if args.grid < 5:
        raise ValueError("--grid must be >= 5")
    if args.z_clip_percentile is not None and not (0.0 < args.z_clip_percentile <= 100.0):
        raise ValueError("--z_clip_percentile must be in (0, 100] or disabled")
    if args.log_dynamic_range <= 1.0:
        raise ValueError("--log_dynamic_range must be > 1")

    func_ids = parse_functions(args.functions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"CEC-2013 direct 2D landscapes: functions={len(func_ids)}, "
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
