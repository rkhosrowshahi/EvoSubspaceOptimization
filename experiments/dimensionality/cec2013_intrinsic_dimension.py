"""Estimate intrinsic dimension of CEC2013 LSGO functions via random projection.

Each function is defined on D=1000 variables. We estimate how many random-
projection subspace dimensions d are needed to reproduce the fitness variability
that uniform search in the full box explores.

Method (aligned with ``evo-subspace`` random projection optimization):
  1. Sample z uniformly in [lb, ub]^d (the optimizer search box).
  2. Expand x = z @ P with an orthonormal Gaussian random projection P.
  3. Clip x to the full-space bounds and evaluate fitness.
  4. Compare std(f(x_subspace)) / std(f(x_subspace at d=D)) over held-out samples.
     Normalizing by d=D makes the capture ratio 1.0 at full projection rank and
     isolates how quickly fitness variability grows with subspace dimension.
  5. Intrinsic dimension = smallest d where the capture ratio reaches a
     threshold (default 0.90), averaged over multiple projection seeds.

By default every subspace dimension d in {1, ..., D} is evaluated exactly.
Use ``--quick`` for a coarse d grid.

Run from repo root:
  python experiments/dimensionality/cec2013_intrinsic_dimension.py
  python experiments/dimensionality/cec2013_intrinsic_dimension.py --quick
  python experiments/dimensionality/cec2013_intrinsic_dimension.py --exact
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments.landscapes.cec2013_projection_landscape import (
    parse_functions,
    sample_uniform_in_bounds,
)
from evo_subspace.problems.lsgo import LSGOProblem
from evo_subspace.subspaces import build_subspace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "tables" / "lsgo"


def evaluate_batch(
    problem: LSGOProblem,
    x: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    values = np.empty(x.shape[0], dtype=float)
    for start in range(0, x.shape[0], batch_size):
        batch = x[start : start + batch_size]
        for offset, row in enumerate(batch):
            values[start + offset] = problem.evaluate(row)
    return values


def d_grid(dim: int, *, quick: bool) -> list[int]:
    if quick:
        return default_d_grid(dim)
    return list(range(1, dim + 1))


def default_d_grid(dim: int) -> list[int]:
    candidates = [1, 5, 10, 25, 50, 100, 200, 350, 500, 750, dim]
    return sorted({d for d in candidates if 1 <= d <= dim})


def estimate_threshold_dimension(
    d_values: list[int],
    capture_values: list[float],
    *,
    threshold: float,
) -> int | None:
    for d, capture in zip(d_values, capture_values):
        if capture >= threshold:
            return d
    return None


def estimate_elbow_dimension(
    d_values: list[int],
    capture_values: list[float],
) -> int:
    if len(d_values) < 3:
        return d_values[-1]
    x = np.log(np.asarray(d_values, dtype=float))
    y = np.asarray(capture_values, dtype=float)
    x0, y0 = x[0], y[0]
    x1, y1 = x[-1], y[-1]
    denom = np.hypot(x1 - x0, y1 - y0)
    if denom <= 0.0:
        return d_values[-1]
    distances = np.abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0) / denom
    return int(d_values[int(np.argmax(distances))])


def benchmark_structure(problem: LSGOProblem) -> dict:
    bench = problem._func
    info = {
        "func_type": bench.func_type,
        "n_groups": bench.n_groups,
        "nonsep_D": getattr(bench, "_nonsep_D", None),
        "sep_D": getattr(bench, "_sep_D", None),
        "eff_D": getattr(bench, "_eff_D", None),
    }
    return info


def analyze_function(
    func_id: str,
    *,
    dim: int,
    benchmark_seed: int,
    group_size: int,
    eval_n_samples: int,
    sample_seed: int,
    rp_seeds: list[int],
    d_values: list[int],
    capture_threshold: float,
    eval_batch_size: int,
    subspace_device: str,
) -> dict:
    problem = LSGOProblem(
        func_id=func_id,
        D=dim,
        seed=benchmark_seed,
        group_size=group_size,
    )
    rng = np.random.default_rng(sample_seed)

    per_seed_curves: list[dict[int, float]] = []
    z_lb = np.full(1, problem.lb[0])
    z_ub = np.full(1, problem.ub[0])

    for rp_seed in rp_seeds:
        curve: dict[int, float] = {}
        std_by_d: dict[int, float] = {}
        subspace_full = build_subspace(
            "random_projection",
            D=dim,
            d=dim,
            subspace_assignment="absolute",
            seed=rp_seed,
            lb=problem.lb,
            ub=problem.ub,
            device=subspace_device,
        )
        projection_full = subspace_full.P
        z_full = sample_uniform_in_bounds(
            eval_n_samples,
            np.full(dim, z_lb[0]),
            np.full(dim, z_ub[0]),
            rng,
        )
        for d in d_values:
            x_sub = np.clip(z_full[:, :d] @ projection_full[:d, :], problem.lb, problem.ub)
            y_sub = evaluate_batch(problem, x_sub, eval_batch_size)
            std_by_d[d] = float(np.std(y_sub))
            if d % 100 == 0 or d == d_values[-1]:
                print(f"    seed={rp_seed} d={d}/{dim}", flush=True)
        full_d = d_values[-1]
        baseline_std = std_by_d[full_d]
        if baseline_std <= 0.0:
            raise ValueError(f"{func_id}: random projection at d=D has zero fitness variance")
        for d in d_values:
            curve[d] = std_by_d[d] / baseline_std
        per_seed_curves.append(curve)

    mean_capture = {
        d: float(np.mean([curve[d] for curve in per_seed_curves]))
        for d in d_values
    }
    std_capture = {
        d: float(np.std([curve[d] for curve in per_seed_curves]))
        for d in d_values
    }
    ordered_d = sorted(d_values)
    mean_curve = [mean_capture[d] for d in ordered_d]
    # Adding projection dimensions should not reduce expressiveness in expectation;
    # enforce a monotone envelope before threshold / elbow estimates.
    mean_curve = list(np.maximum.accumulate(mean_curve))

    return {
        "func_id": func_id,
        "dim": dim,
        "capture_threshold": capture_threshold,
        "intrinsic_dim": estimate_threshold_dimension(
            ordered_d,
            mean_curve,
            threshold=capture_threshold,
        ),
        "intrinsic_dim_elbow": estimate_elbow_dimension(ordered_d, mean_curve),
        "d_values": ordered_d,
        "mean_capture": mean_curve,
        "std_capture": [std_capture[d] for d in ordered_d],
        "capture_at_full_d": 1.0,
        "n_rp_seeds": len(rp_seeds),
        "eval_n_samples": eval_n_samples,
        "structure": benchmark_structure(problem),
    }


def render_summary_table(rows: list[dict]) -> str:
    lines = [
        "CEC2013 intrinsic dimension via random projection (fitness variance capture)",
        "",
        (
            f"{'Function':<8} {'Type':<10} {'Groups':>6} {'ID@thr':>8} "
            f"{'Elbow d':>8} {'Cap@D':>8}"
        ),
        "-" * 56,
    ]
    for row in rows:
        func_label = row["func_id"].replace("cec2013_lsgo_", "").upper()
        structure = row["structure"]
        func_type = structure["func_type"]
        n_groups = structure["n_groups"]
        groups_str = str(n_groups) if n_groups is not None else "-"
        intrinsic = row["intrinsic_dim"]
        intrinsic_str = (
            str(intrinsic) if intrinsic is not None else f">{row['dim']}"
        )
        lines.append(
            f"{func_label:<8} {func_type:<10} {groups_str:>6} "
            f"{intrinsic_str:>8} {row['intrinsic_dim_elbow']:>8} "
            f"{row['capture_at_full_d']:>8.3f}"
        )
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "func_id",
        "func_type",
        "n_groups",
        "nonsep_D",
        "sep_D",
        "eff_D",
        "ambient_dim",
        "intrinsic_dim",
        "intrinsic_dim_elbow",
        "capture_threshold",
        "capture_at_full_d",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            structure = row["structure"]
            writer.writerow(
                {
                    "func_id": row["func_id"],
                    "func_type": structure["func_type"],
                    "n_groups": structure["n_groups"],
                    "nonsep_D": structure["nonsep_D"],
                    "sep_D": structure["sep_D"],
                    "eff_D": structure["eff_D"],
                    "ambient_dim": row["dim"],
                    "intrinsic_dim": row["intrinsic_dim"],
                    "intrinsic_dim_elbow": row["intrinsic_dim_elbow"],
                    "capture_threshold": row["capture_threshold"],
                    "capture_at_full_d": row["capture_at_full_d"],
                }
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate CEC2013 intrinsic dimension via random projection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--functions", type=str, default="all")
    parser.add_argument("--dim", type=int, default=1000)
    parser.add_argument("--benchmark_seed", type=int, default=0)
    parser.add_argument("--group_size", type=int, default=50)
    parser.add_argument("--eval_n_samples", type=int, default=1000)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument(
        "--rp_seeds",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated seeds for independent random projections.",
    )
    parser.add_argument(
        "--capture_threshold",
        type=float,
        default=0.90,
        help="Fitness std capture ratio threshold for intrinsic dimension.",
    )
    parser.add_argument("--eval_batch_size", type=int, default=5000)
    parser.add_argument("--subspace_device", type=str, default="cpu")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a coarse d grid (default sweeps every d from 1 to D).",
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def _write_results(output_dir: Path, results: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "cec2013_intrinsic_dimension_summary.txt"
    csv_path = output_dir / "cec2013_intrinsic_dimension.csv"
    json_path = output_dir / "cec2013_intrinsic_dimension_curves.json"
    summary_path.write_text(render_summary_table(results), encoding="utf-8")
    write_csv(csv_path, results)
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


def main() -> None:
    args = build_arg_parser().parse_args()
    func_ids = parse_functions(args.functions)
    rp_seeds = [int(token.strip()) for token in args.rp_seeds.split(",") if token.strip()]
    if not rp_seeds:
        raise ValueError("at least one random projection seed is required")

    eval_n_samples = 300 if args.quick else args.eval_n_samples
    d_values = d_grid(args.dim, quick=args.quick)

    print(
        f"Estimating intrinsic dimension for {len(func_ids)} functions "
        f"(D={args.dim}, eval_n={eval_n_samples}, "
        f"d sweep={'1..'+str(args.dim) if not args.quick else f'{len(d_values)} points'}, "
        f"threshold={args.capture_threshold})..."
    )

    results: list[dict] = []
    for func_id in func_ids:
        print(f"  analyzing {func_id}...")
        result = analyze_function(
            func_id,
            dim=args.dim,
            benchmark_seed=args.benchmark_seed,
            group_size=args.group_size,
            eval_n_samples=eval_n_samples,
            sample_seed=args.sample_seed,
            rp_seeds=rp_seeds,
            d_values=d_values,
            capture_threshold=args.capture_threshold,
            eval_batch_size=args.eval_batch_size,
            subspace_device=args.subspace_device,
        )
        results.append(result)
        _write_results(args.output_dir, results)

    print()
    print(render_summary_table(results))
    print(f"Wrote {args.output_dir / 'cec2013_intrinsic_dimension_summary.txt'}")
    print(f"Wrote {args.output_dir / 'cec2013_intrinsic_dimension.csv'}")
    print(f"Wrote {args.output_dir / 'cec2013_intrinsic_dimension_curves.json'}")


if __name__ == "__main__":
    main()
