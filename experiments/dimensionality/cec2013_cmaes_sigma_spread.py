"""One-shot CMA-ES offspring spread: intrinsic RP vs full space.

Samples one generation (ask + evaluate) per sigma and compares z / x / fitness spread.

Run from repo root:
  python experiments/dimensionality/cec2013_cmaes_sigma_spread.py
  python experiments/dimensionality/cec2013_cmaes_sigma_spread.py --modes intrinsic,fullspace
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from evo_subspace.analysis.intrinsic_dim import default_intrinsic_dim_csv, load_intrinsic_dims
from evo_subspace.problems import LSGOProblem
from evo_subspace.runtime import SubspaceProblem
from evo_subspace.runtime.evosax_runner import _evaluate_batch, setup_evosax_optimizer
from evo_subspace.subspaces import build_subspace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "cec2013_lsgo"
    / "dim1000"
    / "f1_cmaes_sigma_spread_compare.json"
)


def _pairwise_mean_distance(X: np.ndarray) -> float:
    n = X.shape[0]
    if n < 2:
        return 0.0
    dists = [
        float(np.linalg.norm(X[i] - X[j]))
        for i in range(n)
        for j in range(i + 1, n)
    ]
    return float(np.mean(dists))


def _spread_stats(Z: np.ndarray, F: np.ndarray, *, label: str, mode: str) -> dict:
    norms = np.linalg.norm(Z, axis=1)
    per_dim_std = np.std(Z, axis=0)
    return {
        "label": label,
        "mode": mode,
        "n_points": int(Z.shape[0]),
        "search_dim": int(Z.shape[1]),
        "z_norm_mean": float(norms.mean()),
        "z_norm_std": float(norms.std()),
        "z_norm_min": float(norms.min()),
        "z_norm_max": float(norms.max()),
        "z_per_dim_std_mean": float(per_dim_std.mean()),
        "z_per_dim_std_max": float(per_dim_std.max()),
        "z_pairwise_dist_mean": _pairwise_mean_distance(Z),
        "fitness_min": float(F.min()),
        "fitness_max": float(F.max()),
        "fitness_mean": float(F.mean()),
        "fitness_std": float(F.std()),
        "fitness_median": float(np.median(F)),
    }


def sample_cmaes_once(
    *,
    problem: SubspaceProblem,
    search_dim: int,
    seed: int,
    pop_size: int,
    cmaes_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    args = SimpleNamespace(
        optimizer="cmaes",
        pop_size=pop_size,
        init_pop="uniform",
        pop_sigma=0.5,
        cmaes_sigma=cmaes_sigma,
        seed=seed,
    )
    optim = setup_evosax_optimizer(
        args,
        problem,
        search_dim=search_dim,
        max_nfe=pop_size,
        seed=seed,
    )
    Z = optim.ask()
    F = _evaluate_batch(problem, Z).reshape(-1)
    return np.asarray(Z, dtype=float), F


def sample_uniform_once(
    *,
    problem: SubspaceProblem,
    lb: float,
    ub: float,
    search_dim: int,
    seed: int,
    pop_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    Z = rng.uniform(lb, ub, size=(pop_size, search_dim))
    F = _evaluate_batch(problem, Z).reshape(-1)
    return Z, F


def expand_batch(subspace, Z: np.ndarray) -> np.ndarray:
    return np.stack([subspace.expand(z) for z in Z], axis=0)


def build_problem(
    *,
    func_id: str,
    dim: int,
    benchmark_seed: int,
    mode: str,
    subspace_dim: int,
    subspace_assignment: str,
    subspace_seed: int,
    subspace_device: str,
) -> tuple[SubspaceProblem, object, int, str]:
    lsgo = LSGOProblem(func_id=func_id, D=dim, seed=benchmark_seed)
    if mode == "fullspace":
        subspace = build_subspace(
            method="fullspace",
            D=dim,
            d=dim,
            subspace_assignment=subspace_assignment,
            seed=subspace_seed,
            lb=lsgo.lb,
            ub=lsgo.ub,
            device=subspace_device,
        )
        mode_label = f"fullspace (D={dim})"
    elif mode == "intrinsic":
        subspace = build_subspace(
            method="random_projection",
            D=dim,
            d=subspace_dim,
            subspace_assignment=subspace_assignment,
            seed=subspace_seed,
            lb=lsgo.lb,
            ub=lsgo.ub,
            device=subspace_device,
        )
        mode_label = f"intrinsic {subspace_assignment} (d={subspace_dim})"
    else:
        raise ValueError(f"Unknown mode {mode!r}")
    problem = SubspaceProblem(lsgo=lsgo, subspace=subspace)
    return problem, subspace, subspace.search_dim, mode_label


def run_mode(
    *,
    mode: str,
    func_id: str,
    dim: int,
    subspace_dim: int,
    benchmark_seed: int,
    seed: int,
    pop_size: int,
    sigmas: list[float],
    subspace_assignment: str,
    subspace_device: str,
    include_uniform: bool,
) -> dict:
    problem, subspace, search_dim, mode_label = build_problem(
        func_id=func_id,
        dim=dim,
        benchmark_seed=benchmark_seed,
        mode=mode,
        subspace_dim=subspace_dim,
        subspace_assignment=subspace_assignment,
        subspace_seed=seed,
        subspace_device=subspace_device,
    )
    lb = float(np.asarray(problem.lsgo.lb).reshape(-1)[0])
    ub = float(np.asarray(problem.lsgo.ub).reshape(-1)[0])

    out: dict = {
        "mode": mode,
        "mode_label": mode_label,
        "search_dim": search_dim,
        "subspace_assignment": subspace_assignment,
        "anchor": None,
        "sigmas": {},
        "uniform_baseline": None,
    }

    if subspace_assignment == "additive" and subspace.x0 is not None:
        x0 = subspace.x0
        f_x0 = float(problem.lsgo.evaluate(x0))
        out["anchor"] = {
            "x0_norm": float(np.linalg.norm(x0)),
            "fitness_x0": f_x0,
        }

    if include_uniform:
        Z_u, F_u = sample_uniform_once(
            problem=problem,
            lb=lb,
            ub=ub,
            search_dim=search_dim,
            seed=seed + 1000,
            pop_size=pop_size,
        )
        X_u = expand_batch(subspace, Z_u)
        u_stats = _spread_stats(Z_u, F_u, label="uniform", mode=mode)
        u_stats["x_norm_mean"] = float(np.linalg.norm(X_u, axis=1).mean())
        u_stats["x_pairwise_dist_mean"] = _pairwise_mean_distance(X_u)
        out["uniform_baseline"] = u_stats

    for sigma in sigmas:
        Z, F = sample_cmaes_once(
            problem=problem,
            search_dim=search_dim,
            seed=seed,
            pop_size=pop_size,
            cmaes_sigma=sigma,
        )
        X = expand_batch(subspace, Z)
        stats = _spread_stats(Z, F, label=f"cmaes_sigma={sigma}", mode=mode)
        stats["cmaes_sigma"] = sigma
        stats["x_norm_mean"] = float(np.linalg.norm(X, axis=1).mean())
        stats["x_pairwise_dist_mean"] = _pairwise_mean_distance(X)
        out["sigmas"][str(sigma)] = stats

    return out


def format_comparison(results: dict) -> str:
    lines = [
        f"{results['problem']} CMA-ES one-shot spread "
        f"({results['subspace_assignment']}, pop={results['pop_size']}, seed={results['seed']})",
    ]
    for mode_key in results["modes"]:
        anchor = results["modes"][mode_key].get("anchor")
        if anchor:
            lines.append(
                f"  anchor x0: ||x0||={anchor['x0_norm']:.2f}, f(x0)={anchor['fitness_x0']:.6e}"
            )
    lines.extend(
        [
        "",
        f"{'mode':<28} {'sigma':>5} {'dim':>5} {'|z| mean':>10} {'z pair d':>10} "
        f"{'|x| mean':>10} {'x pair d':>10} {'f min':>12} {'f mean':>12} {'f max':>12}",
        "-" * 126,
        ]
    )

    for mode_key in results["modes"]:
        block = results["modes"][mode_key]
        mode_label = block["mode_label"]
        search_dim = block["search_dim"]

        if block.get("uniform_baseline"):
            row = block["uniform_baseline"]
            lines.append(
                f"{mode_label:<28} {'--':>5} {search_dim:>5} "
                f"{row['z_norm_mean']:>10.3f} {row['z_pairwise_dist_mean']:>10.3f} "
                f"{row['x_norm_mean']:>10.1f} {row['x_pairwise_dist_mean']:>10.1f} "
                f"{row['fitness_min']:>12.3e} {row['fitness_mean']:>12.3e} {row['fitness_max']:>12.3e}"
            )

        for sigma in sorted(block["sigmas"], key=float):
            row = block["sigmas"][sigma]
            lines.append(
                f"{mode_label:<28} {sigma:>5} {search_dim:>5} "
                f"{row['z_norm_mean']:>10.3f} {row['z_pairwise_dist_mean']:>10.3f} "
                f"{row['x_norm_mean']:>10.1f} {row['x_pairwise_dist_mean']:>10.1f} "
                f"{row['fitness_min']:>12.3e} {row['fitness_mean']:>12.3e} {row['fitness_max']:>12.3e}"
            )

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="One-shot CMA-ES sigma spread: intrinsic RP vs full space."
    )
    p.add_argument("--problem", default="cec2013_lsgo_f1")
    p.add_argument("--dim", type=int, default=1000)
    p.add_argument("--benchmark_seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pop_size", type=int, default=24)
    p.add_argument("--sigmas", type=str, default="1,0.5,0.1")
    p.add_argument(
        "--modes",
        type=str,
        default="intrinsic,fullspace",
        help="Comma-separated: intrinsic, fullspace",
    )
    p.add_argument("--subspace_assignment", default="absolute", choices=["absolute", "additive"])
    p.add_argument("--subspace_device", default="cuda:0")
    p.add_argument("--intrinsic_dim_csv", type=Path, default=default_intrinsic_dim_csv())
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--no-uniform", action="store_true", help="Skip uniform random baseline.")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.sigmas = [float(x.strip()) for x in args.sigmas.split(",") if x.strip()]
    args.mode_list = [m.strip() for m in args.modes.split(",") if m.strip()]
    args.include_uniform = not args.no_uniform

    np.random.seed(args.seed)
    intrinsic_dims = load_intrinsic_dims(args.intrinsic_dim_csv)
    subspace_dim = intrinsic_dims[args.problem]
    lb = -100.0
    ub = 100.0

    results: dict = {
        "problem": args.problem,
        "dim": args.dim,
        "intrinsic_dim": subspace_dim,
        "subspace_assignment": args.subspace_assignment,
        "pop_size": args.pop_size,
        "seed": args.seed,
        "benchmark_seed": args.benchmark_seed,
        "z_bounds": [lb, ub],
        "sigmas": args.sigmas,
        "modes": {},
    }

    for mode in args.mode_list:
        results["modes"][mode] = run_mode(
            mode=mode,
            func_id=args.problem,
            dim=args.dim,
            subspace_dim=subspace_dim,
            benchmark_seed=args.benchmark_seed,
            seed=args.seed,
            pop_size=args.pop_size,
            sigmas=args.sigmas,
            subspace_assignment=args.subspace_assignment,
            subspace_device=args.subspace_device,
            include_uniform=args.include_uniform,
        )

    table = format_comparison(results)
    print(table)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
