"""Demonstrate when LoRA is structurally superior to fixed subspaces.

Compares LoRA at multiple ranks (default r=1,2,4) against random projection and
random blocking with matched search dimension d = 2*M*r at each rank.

For low_rank targets X_* = A_* B_* (rank target_rank), LoRA with r >= target_rank
can represent the optimum exactly; fixed RP/RB generally cannot.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class Reconstruction:
    name: str
    x_hat: np.ndarray
    rel_mse: float
    seed: int | None = None


@dataclass
class LoraOptimizationResult:
    A: np.ndarray
    B: np.ndarray
    rel_mse: float
    history: np.ndarray
    restart: int
    success: bool
    message: str


@dataclass
class RankSweepResult:
    rank: int
    search_dim: int
    lora_svd: Reconstruction
    lora_opt: LoraOptimizationResult
    rp_best: Reconstruction
    rb_best: Reconstruction
    rp_results: list[Reconstruction]
    rb_results: list[Reconstruction]


BENCHMARK_CHOICES = ("sphere", "rastrigin", "ackley")


def parse_benchmarks(benchmarks_str: str) -> list[str]:
    benchmarks = []
    for part in benchmarks_str.split(","):
        name = part.strip().lower()
        if not name:
            continue
        if name not in BENCHMARK_CHOICES:
            raise ValueError(f"Unknown benchmark {name!r}; choose from {BENCHMARK_CHOICES}")
        if name not in benchmarks:
            benchmarks.append(name)
    if not benchmarks:
        raise ValueError("At least one benchmark is required")
    return benchmarks


def parse_ranks(ranks_str: str) -> list[int]:
    ranks = sorted({int(part.strip()) for part in ranks_str.split(",") if part.strip()})
    if not ranks:
        raise ValueError("At least one rank is required")
    return ranks


def relative_mse(x_hat: np.ndarray, x_target: np.ndarray) -> float:
    denom = float(np.sum(x_target * x_target))
    if denom == 0.0:
        return float(np.mean((x_hat - x_target) ** 2))
    return float(np.sum((x_hat - x_target) ** 2) / denom)


def shifted_benchmark_value(x_hat: np.ndarray, x_target: np.ndarray, benchmark: str) -> float:
    """Evaluate a standard shifted benchmark with global optimum at x_target."""
    y = np.asarray(x_hat, dtype=float) - np.asarray(x_target, dtype=float)
    benchmark = benchmark.lower()
    if benchmark == "sphere":
        return float(np.mean(y * y))
    if benchmark == "rastrigin":
        return float(10.0 + np.mean(y * y - 10.0 * np.cos(2.0 * np.pi * y)))
    if benchmark == "ackley":
        mean_sq = float(np.mean(y * y))
        mean_cos = float(np.mean(np.cos(2.0 * np.pi * y)))
        return float(-20.0 * np.exp(-0.2 * np.sqrt(mean_sq)) - np.exp(mean_cos) + 20.0 + np.e)
    raise ValueError(f"Unknown benchmark {benchmark!r}; choose sphere, rastrigin, or ackley")


def make_low_rank_target(M: int, rank: int, seed: int, normalize: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    A_star = rng.normal(size=(M, rank))
    B_star = rng.normal(size=(rank, M))
    X_star = A_star @ B_star
    if normalize:
        scale = np.std(X_star)
        if scale > 0:
            X_star = X_star / scale
            A_star = A_star / np.sqrt(scale)
            B_star = B_star / np.sqrt(scale)
    return X_star, A_star, B_star


def make_random_target(M: int, seed: int, normalize: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X_star = rng.normal(size=(M, M))
    if normalize:
        scale = np.std(X_star)
        if scale > 0:
            X_star = X_star / scale
    return X_star


def sample_random_projection(D: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    G = rng.normal(size=(D, d))
    Q, _ = np.linalg.qr(G, mode="reduced")
    return Q.T


def best_random_projection_reconstruction(x_target: np.ndarray, d: int, seed: int) -> Reconstruction:
    P = sample_random_projection(D=x_target.size, d=d, seed=seed)
    z_best = x_target @ P.T
    x_hat = z_best @ P
    return Reconstruction(
        name="Random projection",
        x_hat=x_hat,
        rel_mse=relative_mse(x_hat, x_target),
        seed=seed,
    )


def sample_random_blocking_groups(D: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, d, size=D)


def best_random_blocking_reconstruction(x_target: np.ndarray, d: int, seed: int) -> Reconstruction:
    groups = sample_random_blocking_groups(D=x_target.size, d=d, seed=seed)
    counts = np.bincount(groups, minlength=d).astype(float)
    z_best = np.zeros(d, dtype=float)
    np.add.at(z_best, groups, x_target)
    z_best = z_best / np.maximum(counts, 1.0)
    x_hat = z_best[groups]
    return Reconstruction(
        name="Random blocking",
        x_hat=x_hat,
        rel_mse=relative_mse(x_hat, x_target),
        seed=seed,
    )


def svd_lora_reconstruction(X_target: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray, Reconstruction]:
    U, s, Vt = np.linalg.svd(X_target, full_matrices=False)
    sqrt_s = np.sqrt(s[:rank])
    A = U[:, :rank] * sqrt_s[None, :]
    B = sqrt_s[:, None] * Vt[:rank, :]
    X_hat = A @ B
    return A, B, Reconstruction(
        name="LoRA/SVD rank-r",
        x_hat=X_hat.reshape(-1),
        rel_mse=relative_mse(X_hat.reshape(-1), X_target.reshape(-1)),
        seed=None,
    )


def pack_lora(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.concatenate([A.reshape(-1), B.reshape(-1)])


def unpack_lora(z: np.ndarray, M: int, rank: int) -> tuple[np.ndarray, np.ndarray]:
    split = M * rank
    A = z[:split].reshape(M, rank)
    B = z[split:].reshape(rank, M)
    return A, B


def lora_objective_and_grad(z: np.ndarray, X_target: np.ndarray, rank: int) -> tuple[float, np.ndarray]:
    M = X_target.shape[0]
    A, B = unpack_lora(z, M=M, rank=rank)
    residual = A @ B - X_target
    norm = float(np.sum(X_target * X_target))
    if norm == 0.0:
        norm = 1.0
    value = float(np.sum(residual * residual) / norm)
    grad_A = (2.0 / norm) * residual @ B.T
    grad_B = (2.0 / norm) * A.T @ residual
    grad = pack_lora(grad_A, grad_B)
    return value, grad


def optimize_lora(
    X_target: np.ndarray,
    rank: int,
    n_restarts: int,
    max_iter: int,
    seed: int,
    init_scale: float,
) -> LoraOptimizationResult:
    rng = np.random.default_rng(seed)
    M = X_target.shape[0]
    best: LoraOptimizationResult | None = None

    for restart in range(n_restarts):
        A0 = rng.normal(scale=init_scale, size=(M, rank))
        B0 = rng.normal(scale=init_scale, size=(rank, M))
        z0 = pack_lora(A0, B0)
        history: list[float] = []

        def fun(z: np.ndarray) -> tuple[float, np.ndarray]:
            return lora_objective_and_grad(z, X_target=X_target, rank=rank)

        def callback(z: np.ndarray) -> None:
            value, _ = lora_objective_and_grad(z, X_target=X_target, rank=rank)
            history.append(value)

        result = minimize(
            fun,
            z0,
            method="L-BFGS-B",
            jac=True,
            callback=callback,
            options={"maxiter": max_iter, "ftol": 1e-14, "gtol": 1e-10, "maxls": 50},
        )
        A, B = unpack_lora(result.x, M=M, rank=rank)
        rel = relative_mse((A @ B).reshape(-1), X_target.reshape(-1))
        hist = np.array(history if history else [rel], dtype=float)
        current = LoraOptimizationResult(
            A=A,
            B=B,
            rel_mse=rel,
            history=hist,
            restart=restart,
            success=bool(result.success),
            message=str(result.message),
        )
        if best is None or current.rel_mse < best.rel_mse:
            best = current

    assert best is not None
    return best


def run_fixed_subspace_trials(
    x_target: np.ndarray,
    d: int,
    n_seeds: int,
    seed: int,
) -> tuple[list[Reconstruction], list[Reconstruction]]:
    rp_results = []
    rb_results = []
    for offset in range(n_seeds):
        method_seed = seed + offset
        rp_results.append(best_random_projection_reconstruction(x_target, d=d, seed=method_seed))
        rb_results.append(best_random_blocking_reconstruction(x_target, d=d, seed=method_seed))
    return rp_results, rb_results


def best_result(results: list[Reconstruction]) -> Reconstruction:
    return min(results, key=lambda r: r.rel_mse)


def entry_benchmark_values(
    entry: RankSweepResult,
    x_target: np.ndarray,
    benchmark: str,
) -> dict[str, float]:
    x_lora_opt = (entry.lora_opt.A @ entry.lora_opt.B).reshape(-1)
    return {
        "LoRA SVD": shifted_benchmark_value(entry.lora_svd.x_hat, x_target, benchmark),
        "LoRA optimized": shifted_benchmark_value(x_lora_opt, x_target, benchmark),
        "Best RP": shifted_benchmark_value(entry.rp_best.x_hat, x_target, benchmark),
        "Best RB": shifted_benchmark_value(entry.rb_best.x_hat, x_target, benchmark),
    }


def run_rank_sweep(
    X_target: np.ndarray,
    x_target: np.ndarray,
    ranks: list[int],
    M: int,
    n_seeds: int,
    subspace_seed: int,
    lora_restarts: int,
    max_iter: int,
    seed: int,
    init_scale: float,
) -> list[RankSweepResult]:
    results: list[RankSweepResult] = []
    for rank in ranks:
        search_dim = 2 * M * rank
        _, _, lora_svd = svd_lora_reconstruction(X_target, rank=rank)
        lora_opt = optimize_lora(
            X_target,
            rank=rank,
            n_restarts=lora_restarts,
            max_iter=max_iter,
            seed=seed + 1000 * rank,
            init_scale=init_scale,
        )
        rp_results, rb_results = run_fixed_subspace_trials(
            x_target,
            d=search_dim,
            n_seeds=n_seeds,
            seed=subspace_seed + 10000 * rank,
        )
        rp_best = best_result(rp_results)
        rb_best = best_result(rb_results)
        results.append(
            RankSweepResult(
                rank=rank,
                search_dim=search_dim,
                lora_svd=lora_svd,
                lora_opt=lora_opt,
                rp_best=rp_best,
                rb_best=rb_best,
                rp_results=rp_results,
                rb_results=rb_results,
            )
        )
    return results


def benchmark_table_path(table_dir: Path, target_mode: str, benchmark: str) -> Path:
    return table_dir / f"lora_{target_mode}_{benchmark}.csv"


def _csv_row(
    target_mode: str,
    rank: int,
    search_dim: int,
    method: str,
    seed: int | None,
    is_best: bool,
    relative_mse_value: float,
    benchmark_value: float,
) -> str:
    seed_str = "" if seed is None else str(seed)
    return (
        f"{target_mode},{rank},{search_dim},{method},{seed_str},{int(is_best)},"
        f"{relative_mse_value:.16e},{benchmark_value:.16e}"
    )


def save_benchmark_csv(
    path: Path,
    sweep: list[RankSweepResult],
    x_target: np.ndarray,
    target_mode: str,
    benchmark: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        f"# target_mode={target_mode}",
        f"# benchmark={benchmark}",
        "target_mode,lora_rank,search_dim,method,seed,is_best,relative_mse,benchmark_value",
    ]
    for entry in sweep:
        x_lora_opt = (entry.lora_opt.A @ entry.lora_opt.B).reshape(-1)
        rows.append(
            _csv_row(
                target_mode,
                entry.rank,
                entry.search_dim,
                "LoRA_SVD",
                None,
                True,
                entry.lora_svd.rel_mse,
                shifted_benchmark_value(entry.lora_svd.x_hat, x_target, benchmark),
            )
        )
        rows.append(
            _csv_row(
                target_mode,
                entry.rank,
                entry.search_dim,
                "LoRA_LBFGS",
                entry.lora_opt.restart,
                True,
                entry.lora_opt.rel_mse,
                shifted_benchmark_value(x_lora_opt, x_target, benchmark),
            )
        )
        for rp in entry.rp_results:
            rows.append(
                _csv_row(
                    target_mode,
                    entry.rank,
                    entry.search_dim,
                    "Random_projection",
                    rp.seed,
                    rp.seed == entry.rp_best.seed,
                    rp.rel_mse,
                    shifted_benchmark_value(rp.x_hat, x_target, benchmark),
                )
            )
        for rb in entry.rb_results:
            rows.append(
                _csv_row(
                    target_mode,
                    entry.rank,
                    entry.search_dim,
                    "Random_blocking",
                    rb.seed,
                    rb.seed == entry.rb_best.seed,
                    rb.rel_mse,
                    shifted_benchmark_value(rb.x_hat, x_target, benchmark),
                )
            )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def plot_matrix(ax: plt.Axes, matrix: np.ndarray, title: str, vmin: float, vmax: float):
    image = ax.imshow(matrix, cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return image


def format_method_dimensions(M: int, rank: int) -> dict[str, str]:
    """Human-readable search dimensions for each method at LoRA rank r."""
    d = 2 * M * rank
    D = M * M
    return {
        "LoRA": rf"$d=2Mr={d}$ ($A\in\mathbb{{R}}^{{{M}\times{rank}}}$, $B\in\mathbb{{R}}^{{{rank}\times{M}}}$)",
        "RP": rf"$d={d}$ ($P\in\mathbb{{R}}^{{{d}\times{D}}}$)",
        "RB": rf"$d={d}$ ($D={D}\to\{{0,\ldots,{d - 1}\}}$)",
    }


def dimensions_setup_text(M: int, ranks: list[int]) -> str:
    D = M * M
    lines = [
        rf"Full space: $D=M^2={D}$.",
        "",
        "At each LoRA rank $r$, all methods search in",
        rf"the same dimension $d=2Mr$:",
        "",
        "LoRA: rank-$r$ factors $(A,B)$, $d=2Mr$",
        "RP:   random projection $P$, $d=2Mr$",
        "RB:   random blocking groups, $d=2Mr$",
        "",
        "Per-rank search dimensions:",
    ]
    for rank in ranks:
        d = 2 * M * rank
        lines.append(f"  r={rank}  ->  d={d}")
    return "\n".join(lines)


def _grouped_bars(
    ax: plt.Axes,
    ranks: list[int],
    search_dims: list[int],
    series: dict[str, list[float]],
    ylabel: str,
    title: str,
    log_y: bool,
) -> None:
    x = np.arange(len(ranks))
    n_series = len(series)
    width = 0.8 / n_series
    colors = {"LoRA SVD": "C0", "LoRA optimized": "C0", "Best RP": "C1", "Best RB": "C2"}
    for i, (name, values) in enumerate(series.items()):
        offset = (i - (n_series - 1) / 2) * width
        plot_vals = [max(v, 1e-16) for v in values]
        ax.bar(
            x + offset,
            plot_vals,
            width,
            label=name,
            color=colors.get(name, None),
            alpha=0.85 if "LoRA optimized" in name else 0.95,
            hatch="//" if name == "LoRA optimized" else None,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([rf"$r={r}$" + "\n" + rf"$d={d}$" for r, d in zip(ranks, search_dims)])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)


def save_figure(
    output_base: Path,
    X_target: np.ndarray,
    sweep: list[RankSweepResult],
    M: int,
    target_mode: str,
    target_rank: int | None,
    benchmark_name: str,
    benchmark_values_by_rank: dict[int, dict[str, float]],
) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_base.with_suffix(".pdf")
    png_path = output_base.with_suffix(".png")

    ranks = [entry.rank for entry in sweep]
    search_dims = [entry.search_dim for entry in sweep]
    D = M * M
    low_rank_mode = target_mode == "low_rank"
    headline = (
        "LoRA rank sweep on low-rank shifted benchmark"
        if low_rank_mode
        else "LoRA rank sweep on random full-rank shift"
    )

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), layout="constrained")
    dim_summary = ", ".join(f"r={r}: d={d}" for r, d in zip(ranks, search_dims))
    if low_rank_mode:
        subtitle = (
            rf"$M={M}$, $D=M^2={D}$, target rank={target_rank}, "
            rf"benchmark={benchmark_name}"
        )
    else:
        subtitle = rf"$M={M}$, $D=M^2={D}$, benchmark={benchmark_name}"
    fig.suptitle(f"{headline}\n{subtitle}\nmatched search dims: {dim_summary}", fontsize=13)

    vmin = float(X_target.min())
    vmax = float(X_target.max())
    target_title = r"Target $X_* = A_*B_*$" if low_rank_mode else r"Random shifted optimum $X_*$"
    plot_matrix(axes[0, 0], X_target, target_title, vmin, vmax)

    mid_rank = ranks[len(ranks) // 2]
    mid = next(entry for entry in sweep if entry.rank == mid_rank)
    X_lora = mid.lora_svd.x_hat.reshape(M, M)
    vmin2 = float(min(vmin, X_lora.min(), mid.rp_best.x_hat.min(), mid.rb_best.x_hat.min()))
    vmax2 = float(max(vmax, X_lora.max(), mid.rp_best.x_hat.max(), mid.rb_best.x_hat.max()))
    mid_dims = format_method_dimensions(M, mid_rank)
    plot_matrix(
        axes[0, 1],
        X_lora,
        rf"LoRA SVD at r={mid_rank}, {mid_dims['LoRA']}"
        + "\n"
        + rf"rel. MSE={mid.lora_svd.rel_mse:.2e}",
        vmin2,
        vmax2,
    )
    plot_matrix(
        axes[0, 2],
        mid.rp_best.x_hat.reshape(M, M),
        rf"Best RP at r={mid_rank}, {mid_dims['RP']}"
        + "\n"
        + rf"rel. MSE={mid.rp_best.rel_mse:.2e}",
        vmin2,
        vmax2,
    )

    axes[1, 0].axis("off")
    dim_text = dimensions_setup_text(M, ranks)
    if low_rank_mode:
        explanation = (
            f"Target rank = {target_rank}.\n\n"
            "LoRA with r >= target rank can represent\n"
            "the optimum exactly (SVD / factor tuning).\n\n"
            "RP/RB use fixed random maps; at each rank\n"
            "they only tune z in the matched d.\n\n"
            f"At r={mid_rank}, RB uses {mid_dims['RB']}."
        )
    else:
        explanation = (
            "Random full-rank optimum.\n\n"
            "Higher LoRA rank improves the best\n"
            "rank-r approximation (SVD lower bound).\n\n"
            "RP/RB baselines are recomputed at each\n"
            "matched search dimension.\n\n"
            f"At r={mid_rank}, RB uses {mid_dims['RB']}."
        )
    axes[1, 0].text(0.0, 0.98, "Setup", fontsize=13, fontweight="bold", transform=axes[1, 0].transAxes)
    axes[1, 0].text(
        0.0,
        0.94,
        dim_text,
        fontsize=10,
        va="top",
        transform=axes[1, 0].transAxes,
        family="monospace",
    )
    axes[1, 0].text(0.0, 0.38, explanation, fontsize=10, va="top", transform=axes[1, 0].transAxes)

    mse_series = {
        "LoRA SVD": [entry.lora_svd.rel_mse for entry in sweep],
        "LoRA optimized": [entry.lora_opt.rel_mse for entry in sweep],
        "Best RP": [entry.rp_best.rel_mse for entry in sweep],
        "Best RB": [entry.rb_best.rel_mse for entry in sweep],
    }
    bench_series = {
        "LoRA SVD": [benchmark_values_by_rank[entry.rank]["LoRA SVD"] for entry in sweep],
        "LoRA optimized": [benchmark_values_by_rank[entry.rank]["LoRA optimized"] for entry in sweep],
        "Best RP": [benchmark_values_by_rank[entry.rank]["Best RP"] for entry in sweep],
        "Best RB": [benchmark_values_by_rank[entry.rank]["Best RB"] for entry in sweep],
    }
    _grouped_bars(
        axes[1, 1],
        ranks,
        search_dims,
        mse_series,
        ylabel="relative MSE",
        title="Reconstruction error vs rank (matched d per method)",
        log_y=True,
    )
    _grouped_bars(
        axes[1, 2],
        ranks,
        search_dims,
        bench_series,
        ylabel="benchmark value",
        title=f"Shifted {benchmark_name} vs rank (matched d per method)",
        log_y=True,
    )

    fig.savefig(pdf_path, dpi=220)
    fig.savefig(png_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA rank sweep vs fixed subspaces.")
    parser.add_argument("--M", type=int, default=20, help="Target matrix side length; D=M*M.")
    parser.add_argument(
        "--ranks",
        type=str,
        default="1,2,4",
        help="Comma-separated LoRA ranks to compare (each with matched RP/RB dimension).",
    )
    parser.add_argument(
        "--target_rank",
        type=int,
        default=2,
        help="Rank of low-rank target when --target_mode=low_rank.",
    )
    parser.add_argument(
        "--target_mode",
        type=str,
        default="low_rank",
        choices=("low_rank", "random"),
        help="Shift optimum structure: exact low-rank target or full random Gaussian target.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Target generation seed.")
    parser.add_argument("--subspace_seed", type=int, default=1000, help="First random seed for RP/RB trials.")
    parser.add_argument("--num_random_seeds", type=int, default=30, help="Number of RP/RB random subspaces per rank.")
    parser.add_argument("--lora_restarts", type=int, default=5, help="L-BFGS random restarts for LoRA.")
    parser.add_argument("--max_iter", type=int, default=500, help="Max L-BFGS iterations per LoRA restart.")
    parser.add_argument("--init_scale", type=float, default=0.1, help="LoRA random factor initialization scale.")
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="sphere,rastrigin,ackley",
        help="Comma-separated shifted benchmarks; writes one CSV per benchmark.",
    )
    parser.add_argument("--no_normalize", action="store_true", help="Do not normalize target matrix standard deviation.")
    parser.add_argument(
        "--plot_path",
        type=str,
        default="results/synthetic_lora/figures/lora_low_rank_superiority",
        help="Output path base without extension; appends _{benchmark} when multiple benchmarks.",
    )
    parser.add_argument(
        "--table_dir",
        type=str,
        default="results/synthetic_lora/tables",
        help="Directory for benchmark CSV tables (one file per benchmark).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ranks = parse_ranks(args.ranks)
    benchmarks = parse_benchmarks(args.benchmarks)
    if args.M <= 1:
        raise ValueError("--M must be > 1")
    if any(r <= 0 or r > args.M for r in ranks):
        raise ValueError(f"Each rank must satisfy 0 < rank <= M={args.M}")
    if args.target_mode == "low_rank" and (args.target_rank <= 0 or args.target_rank > args.M):
        raise ValueError(f"--target_rank must satisfy 0 < target_rank <= M={args.M}")
    if args.num_random_seeds <= 0:
        raise ValueError("--num_random_seeds must be positive")
    if args.lora_restarts <= 0:
        raise ValueError("--lora_restarts must be positive")

    M = args.M
    D = M * M
    target_rank = args.target_rank if args.target_mode == "low_rank" else None

    if args.target_mode == "low_rank":
        X_target, _, _ = make_low_rank_target(
            M=M,
            rank=args.target_rank,
            seed=args.seed,
            normalize=not args.no_normalize,
        )
    else:
        X_target = make_random_target(
            M=M,
            seed=args.seed,
            normalize=not args.no_normalize,
        )
    x_target = X_target.reshape(-1)

    sweep = run_rank_sweep(
        X_target=X_target,
        x_target=x_target,
        ranks=ranks,
        M=M,
        n_seeds=args.num_random_seeds,
        subspace_seed=args.subspace_seed,
        lora_restarts=args.lora_restarts,
        max_iter=args.max_iter,
        seed=args.seed + 999,
        init_scale=args.init_scale,
    )

    table_dir = REPO_ROOT / args.table_dir
    plot_base = REPO_ROOT / args.plot_path
    saved_tables: list[Path] = []
    saved_figures: list[Path] = []

    print("LoRA rank-sweep experiment complete")
    print(f"D={D}, M={M}, target_mode={args.target_mode}, benchmarks={benchmarks}")
    if target_rank is not None:
        print(f"target_rank={target_rank}")
    print(f"LoRA ranks={ranks}")

    for benchmark in benchmarks:
        benchmark_values_by_rank = {
            entry.rank: entry_benchmark_values(entry, x_target, benchmark) for entry in sweep
        }
        table_path = benchmark_table_path(table_dir, args.target_mode, benchmark)
        save_benchmark_csv(
            table_path,
            sweep=sweep,
            x_target=x_target,
            target_mode=args.target_mode,
            benchmark=benchmark,
        )
        saved_tables.append(table_path)

        if len(benchmarks) == 1:
            output_base = plot_base
        else:
            output_base = plot_base.parent / f"{plot_base.name}_{benchmark}"
        save_figure(
            output_base=output_base,
            X_target=X_target,
            sweep=sweep,
            M=M,
            target_mode=args.target_mode,
            target_rank=target_rank,
            benchmark_name=benchmark,
            benchmark_values_by_rank=benchmark_values_by_rank,
        )
        saved_figures.extend([output_base.with_suffix(".pdf"), output_base.with_suffix(".png")])

        print(f"\n=== Benchmark: {benchmark} ===")
        for entry in sweep:
            dims = format_method_dimensions(M, entry.rank)
            bench_vals = benchmark_values_by_rank[entry.rank]
            print(f"\n--- LoRA rank r={entry.rank} ---")
            print(f"  LoRA: {dims['LoRA']}")
            print(f"  RP:   {dims['RP']}")
            print(f"  RB:   {dims['RB']}")
            print(f"  LoRA SVD rel. MSE: {entry.lora_svd.rel_mse:.3e}")
            print(f"  LoRA optimized rel. MSE: {entry.lora_opt.rel_mse:.3e} (restart={entry.lora_opt.restart})")
            print(f"  Best RP rel. MSE: {entry.rp_best.rel_mse:.3e} (seed={entry.rp_best.seed})")
            print(f"  Best RB rel. MSE: {entry.rb_best.rel_mse:.3e} (seed={entry.rb_best.seed})")
            print(f"  Shifted {benchmark}:")
            for name, value in bench_vals.items():
                print(f"    {name}: {value:.3e}")

    print("\nSaved tables:")
    for path in saved_tables:
        print(f"  {path}")
    print("Saved figures:")
    for path in saved_figures:
        print(f"  {path}")


if __name__ == "__main__":
    main()
