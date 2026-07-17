"""Run optimizers at per-function intrinsic subspace dimensions and compare to full-space DE.

Uses the intrinsic dimensions from ``cec2013_intrinsic_dimension.csv`` with
random projection (absolute assignment). Supports DE (pymoo) and CMA-ES (evosax).

Run from repo root:
  python experiments/dimensionality/cec2013_intrinsic_dim_opt.py
  python experiments/dimensionality/cec2013_intrinsic_dim_opt.py --optimizer cmaes --seeds 0,1,2
  python experiments/dimensionality/cec2013_intrinsic_dim_opt.py --compare-only
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from evo_subspace.optimizers.builder import build_algorithm
from evo_subspace.runtime.evosax_runner import run_evosax_optimization, setup_evosax_optimizer
from evo_subspace.problems import LSGOProblem
from evo_subspace.runtime import SubspaceProblem
from evo_subspace.subspaces import build_subspace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTRINSIC_DIM_CSV = (
    PROJECT_ROOT / "results" / "tables" / "lsgo" / "cec2013_intrinsic_dimension.csv"
)
FULLSPACE_CSV = (
    PROJECT_ROOT
    / "results"
    / "cec2013_lsgo"
    / "dim1000"
    / "cec2013_lsgo_all_fs_dim1000_by_group.csv"
)
FULLSPACE_SEEDS_JSON = (
    PROJECT_ROOT
    / "results"
    / "cec2013_lsgo"
    / "dim1000"
    / "cec2013_lsgo_all_fs_dim1000_by_group_seeds.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "cec2013_lsgo" / "dim1000"
DEFAULT_RUNS_JSON = DEFAULT_OUTPUT_DIR / "intrinsic_dim_de_runs.json"
DEFAULT_CMAES_RUNS_JSON = DEFAULT_OUTPUT_DIR / "intrinsic_dim_cmaes_runs.json"
DEFAULT_COMPARISON_CSV = DEFAULT_OUTPUT_DIR / "intrinsic_dim_de_vs_fullspace.csv"
DEFAULT_COMPARISON_MD = DEFAULT_OUTPUT_DIR / "intrinsic_dim_de_vs_fullspace.md"

FULLSPACE_METHOD = "Full space (F=0.5, CR=0.9)"


from evo_subspace.analysis.intrinsic_dim import load_intrinsic_dims


def load_fullspace_means(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        means: dict[str, float] = {}
        for row in reader:
            fn = row["function"].strip().upper()
            raw = row.get(FULLSPACE_METHOD, "").strip()
            if raw and raw != "--":
                means[f"cec2013_lsgo_{fn.lower()}"] = float(raw)
        return means


def load_fullspace_seed_values(path: Path) -> dict[str, list[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, list[float]] = {}
    for entry in payload.get("fitness", []):
        if entry.get("method") != FULLSPACE_METHOD:
            continue
        values = entry.get("values") or []
        if not values:
            continue
        if isinstance(values[0], (int, float)) and values[0] > 1e6:
            continue
        out[entry["problem"]] = [float(v) for v in values]
    return out


def parse_functions(spec: str, intrinsic_dims: dict[str, int]) -> list[str]:
    if spec.strip().lower() in ("all", "*"):
        return sorted(intrinsic_dims, key=lambda fid: int(fid.split("_f")[-1]))
    out: list[str] = []
    for part in spec.split(","):
        token = part.strip().lower()
        if not token:
            continue
        if token.startswith("f") and token[1:].isdigit():
            fid = f"cec2013_lsgo_{token}"
        elif token.startswith("cec2013_lsgo_f"):
            fid = token
        else:
            raise ValueError(f"Unknown function token {part!r}")
        if fid not in intrinsic_dims:
            raise ValueError(f"No intrinsic dimension for {fid!r}")
        if fid not in out:
            out.append(fid)
    return out


def parse_seeds(spec: str) -> list[int]:
    seeds: list[int] = []
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            seeds.extend(range(int(start_s), int(end_s) + 1))
        else:
            seeds.append(int(token))
    return sorted(set(seeds))


def build_algorithm_args(
    *,
    optimizer: str,
    pop_size: int,
    de_mut_rate: float,
    de_cr_rate: float,
    cmaes_sigma: float,
    seed: int,
):
    from types import SimpleNamespace

    args = SimpleNamespace(
        optimizer=optimizer,
        pop_size=pop_size,
        init_pop="uniform",
        pop_sigma=0.5,
        cmaes_sigma=cmaes_sigma,
        de_mut_rate=de_mut_rate,
        de_cr_rate=de_cr_rate,
        de_selection="rand",
        de_n_diffs=1,
        de_crossover="bin",
        de_jitter=False,
        de_evolving=False,
        seed=seed,
    )
    return args


def run_optimization(
    *,
    func_id: str,
    subspace_dim: int,
    dim: int,
    seed: int,
    benchmark_seed: int,
    max_nfe: int,
    optimizer: str,
    pop_size: int,
    de_mut_rate: float,
    de_cr_rate: float,
    cmaes_sigma: float,
    subspace_device: str,
) -> dict:
    np.random.seed(seed)
    lsgo = LSGOProblem(func_id=func_id, D=dim, seed=benchmark_seed)
    subspace = build_subspace(
        method="random_projection",
        D=dim,
        d=subspace_dim,
        subspace_assignment="absolute",
        seed=seed,
        lb=lsgo.lb,
        ub=lsgo.ub,
        device=subspace_device,
    )
    problem = SubspaceProblem(lsgo=lsgo, subspace=subspace)

    t0 = time.perf_counter()
    if optimizer == "cmaes":
        algo_args = build_algorithm_args(
            optimizer=optimizer,
            pop_size=pop_size,
            de_mut_rate=de_mut_rate,
            de_cr_rate=de_cr_rate,
            cmaes_sigma=cmaes_sigma,
            seed=seed,
        )
        algorithm = setup_evosax_optimizer(
            algo_args,
            problem,
            search_dim=subspace.search_dim,
            max_nfe=max_nfe,
            seed=seed,
        )
        best_fitness, total_nfe = run_evosax_optimization(
            algorithm,
            problem,
            max_nfe=max_nfe,
        )
    else:
        algo_args = build_algorithm_args(
            optimizer=optimizer,
            pop_size=pop_size,
            de_mut_rate=de_mut_rate,
            de_cr_rate=de_cr_rate,
            cmaes_sigma=cmaes_sigma,
            seed=seed,
        )
        algorithm = build_algorithm(algo_args)
        termination = get_termination("n_eval", max_nfe)
        result = minimize(
            problem,
            algorithm,
            termination,
            seed=seed,
            verbose=False,
            save_history=False,
        )
        best_fitness = float(result.F.flatten()[0])
        total_nfe = int(result.algorithm.evaluator.n_eval)
    elapsed = time.perf_counter() - t0

    return {
        "func_id": func_id,
        "function": func_id.replace("cec2013_lsgo_", "").upper(),
        "subspace_dim": subspace_dim,
        "search_dim": subspace.search_dim,
        "seed": seed,
        "best_fitness": best_fitness,
        "total_nfe": total_nfe,
        "elapsed_seconds": elapsed,
        "subspace_assignment": "absolute",
        "subspace_method": "random_projection",
        "optimizer": optimizer,
        "optimizer_backend": "evosax" if optimizer == "cmaes" else "pymoo",
        "benchmark_seed": benchmark_seed,
        "dim": dim,
        "pop_size": pop_size,
        "de_mut_rate": de_mut_rate,
        "de_cr_rate": de_cr_rate,
        "cmaes_sigma": cmaes_sigma,
        "max_nfe": max_nfe,
    }


def load_runs(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def save_runs(path: Path, runs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(runs, indent=2), encoding="utf-8")


def run_key(run: dict) -> tuple[str, int, int, str, str]:
    return (
        run["func_id"],
        int(run["seed"]),
        int(run.get("max_nfe", 0)),
        str(run.get("optimizer", "de")),
        str(run.get("optimizer_backend", "pymoo")),
    )


def merge_run(existing: list[dict], new_run: dict) -> list[dict]:
    merged = {run_key(r): r for r in existing}
    merged[run_key(new_run)] = new_run
    return sorted(
        merged.values(),
        key=lambda r: (int(r["func_id"].split("_f")[-1]), int(r["seed"])),
    )


def summarize_runs(runs: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = {}
    for run in runs:
        grouped.setdefault(run["func_id"], []).append(run)
    summary: dict[str, dict] = {}
    for func_id, group in grouped.items():
        values = [float(r["best_fitness"]) for r in group]
        summary[func_id] = {
            "function": group[0]["function"],
            "subspace_dim": int(group[0]["subspace_dim"]),
            "n_runs": len(values),
            "mean_best_fitness": float(np.mean(values)),
            "std_best_fitness": float(np.std(values)),
            "min_best_fitness": float(np.min(values)),
            "max_best_fitness": float(np.max(values)),
            "seeds": sorted(int(r["seed"]) for r in group),
        }
    return summary


def format_sci(value: float) -> str:
    return f"{value:.6e}"


def write_comparison(
    runs: list[dict],
    *,
    csv_path: Path,
    md_path: Path,
    fullspace_means: dict[str, float],
    fullspace_seeds: dict[str, list[float]],
) -> None:
    summary = summarize_runs(runs)
    rows: list[dict] = []
    optimizer_label = str(runs[0].get("optimizer", "de")).upper()
    backend = runs[0].get("optimizer_backend", "pymoo")

    for func_id in sorted(summary, key=lambda fid: int(fid.split("_f")[-1])):
        item = summary[func_id]
        fs_mean = fullspace_means.get(func_id)
        rp_mean = item["mean_best_fitness"]
        ratio = rp_mean / fs_mean if fs_mean and fs_mean > 0 else None
        matched = [
            (seed, fullspace_seeds[func_id][seed])
            for seed in item["seeds"]
            if func_id in fullspace_seeds and seed < len(fullspace_seeds[func_id])
        ]
        per_seed_wins = sum(
            1
            for run in runs
            if run["func_id"] == func_id
            and run["seed"] < len(fullspace_seeds.get(func_id, []))
            and run["best_fitness"] < fullspace_seeds[func_id][run["seed"]]
        )
        rows.append(
            {
                "function": item["function"],
                "func_id": func_id,
                "intrinsic_dim": item["subspace_dim"],
                "n_runs": item["n_runs"],
                "rp_mean_best_fitness": rp_mean,
                "rp_std_best_fitness": item["std_best_fitness"],
                "fullspace_mean_best_fitness": fs_mean,
                "ratio_rp_over_fullspace": ratio,
                "matched_seed_wins": per_seed_wins,
                "matched_seed_total": len(matched),
            }
        )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"# Intrinsic-dimension random projection {optimizer_label} vs full-space DE",
        "",
        f"Baseline: `{FULLSPACE_METHOD}` from past W&B runs (3M NFE, D=1000).",
        f"Treatment: random projection at the per-function intrinsic dimension "
        f"(90% fitness-variance capture), absolute assignment, {optimizer_label} "
        f"via {backend} (pop={runs[0].get('pop_size', '--')}).",
        "",
        "| Function | Intrinsic d | RP mean best | Full-space mean | Ratio (RP/FS) | Seed wins |",
        "|----------|-------------|--------------|-----------------|---------------|-----------|",
    ]
    for row in rows:
        fs = row["fullspace_mean_best_fitness"]
        fs_str = format_sci(fs) if fs is not None else "--"
        ratio = row["ratio_rp_over_fullspace"]
        ratio_str = f"{ratio:.3f}" if ratio is not None else "--"
        wins = row["matched_seed_wins"]
        total = row["matched_seed_total"]
        win_str = f"{wins}/{total}" if total else "--"
        lines.append(
            f"| {row['function']} | {row['intrinsic_dim']} | "
            f"{format_sci(row['rp_mean_best_fitness'])} | {fs_str} | "
            f"{ratio_str} | {win_str} |"
        )
    lines.extend(
        [
            "",
            "Ratio < 1 means intrinsic-dimension RP beat the full-space mean on average.",
            "Seed wins counts matched-seed runs where RP best fitness < full-space best fitness.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run intrinsic-dimension optimization experiments and compare to full-space baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--functions", type=str, default="all")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--dim", type=int, default=1000)
    parser.add_argument("--benchmark_seed", type=int, default=0)
    parser.add_argument("--max_nfe", type=int, default=3_000_000)
    parser.add_argument(
        "--optimizer",
        type=str,
        default="de",
        choices=["de", "cmaes"],
    )
    parser.add_argument("--pop_size", type=int, default=100)
    parser.add_argument("--de_mut_rate", type=float, default=0.5)
    parser.add_argument("--de_cr_rate", type=float, default=0.9)
    parser.add_argument("--cmaes_sigma", type=float, default=0.5)
    parser.add_argument("--subspace_device", type=str, default="cuda:0")
    parser.add_argument(
        "--intrinsic_dim_csv",
        type=Path,
        default=INTRINSIC_DIM_CSV,
    )
    parser.add_argument(
        "--output_runs",
        type=Path,
        default=None,
        help="Defaults to intrinsic_dim_{optimizer}_runs.json under results/cec2013_lsgo/dim1000/.",
    )
    parser.add_argument("--output_csv", type=Path, default=None)
    parser.add_argument("--output_md", type=Path, default=None)
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Skip optimization; regenerate comparison tables from saved runs.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.output_runs is None:
        args.output_runs = (
            DEFAULT_CMAES_RUNS_JSON
            if args.optimizer == "cmaes"
            else DEFAULT_RUNS_JSON
        )
    if args.output_csv is None:
        args.output_csv = DEFAULT_OUTPUT_DIR / f"intrinsic_dim_{args.optimizer}_vs_fullspace.csv"
    if args.output_md is None:
        args.output_md = DEFAULT_OUTPUT_DIR / f"intrinsic_dim_{args.optimizer}_vs_fullspace.md"

    intrinsic_dims = load_intrinsic_dims(args.intrinsic_dim_csv)
    fullspace_means = load_fullspace_means(FULLSPACE_CSV)
    fullspace_seeds = load_fullspace_seed_values(FULLSPACE_SEEDS_JSON)

    runs = load_runs(args.output_runs)

    if not args.compare_only:
        func_ids = parse_functions(args.functions, intrinsic_dims)
        seeds = parse_seeds(args.seeds)
        print(
            f"Running intrinsic-dimension {args.optimizer.upper()} for {len(func_ids)} functions, "
            f"seeds={seeds}, max_nfe={args.max_nfe}..."
        )
        for func_id in func_ids:
            subspace_dim = intrinsic_dims[func_id]
            for seed in seeds:
                if any(
                    r["func_id"] == func_id
                    and r["seed"] == seed
                    and int(r.get("max_nfe", 0)) == args.max_nfe
                    and str(r.get("optimizer", "de")) == args.optimizer
                    and str(r.get("optimizer_backend", "pymoo"))
                    == ("evosax" if args.optimizer == "cmaes" else "pymoo")
                    for r in runs
                ):
                    print(f"  skip {func_id} seed={seed} (already in {args.output_runs})")
                    continue
                print(
                    f"  running {func_id} d={subspace_dim} seed={seed}...",
                    flush=True,
                )
                result = run_optimization(
                    func_id=func_id,
                    subspace_dim=subspace_dim,
                    dim=args.dim,
                    seed=seed,
                    benchmark_seed=args.benchmark_seed,
                    max_nfe=args.max_nfe,
                    optimizer=args.optimizer,
                    pop_size=args.pop_size,
                    de_mut_rate=args.de_mut_rate,
                    de_cr_rate=args.de_cr_rate,
                    cmaes_sigma=args.cmaes_sigma,
                    subspace_device=args.subspace_device,
                )
                runs = merge_run(runs, result)
                save_runs(args.output_runs, runs)
                print(
                    f"    best={result['best_fitness']:.6e} "
                    f"nfe={result['total_nfe']} "
                    f"time={result['elapsed_seconds']:.1f}s",
                    flush=True,
                )

    if not runs:
        raise SystemExit(f"No runs found at {args.output_runs}")

    write_comparison(
        runs,
        csv_path=args.output_csv,
        md_path=args.output_md,
        fullspace_means=fullspace_means,
        fullspace_seeds=fullspace_seeds,
    )
    print(f"Wrote {args.output_runs}")
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
