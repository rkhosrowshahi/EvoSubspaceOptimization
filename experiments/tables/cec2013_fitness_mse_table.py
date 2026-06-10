"""Generate the CEC2013 LSGO fitness preservation MSE table."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.landscapes.cec2013_projection_landscape import (
    AE_LOG_FITNESS_MOO_VARIANT,
    AE_LOG_FITNESS_VARIANT,
    DEFAULT_OUTPUT_ROOT,
    PROJECT_ROOT,
    autoencoder_checkpoint_path,
    fit_pca2,
    fit_random_projection2,
    load_autoencoder_checkpoint,
    output_base_for,
    parse_functions,
    sample_uniform_in_bounds,
)
from evo_subspace.problems.lsgo import LSGOProblem

DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results" / "tables" / "lsgo" / "reconstruction_mse_values.tex"
)
METHOD_LABELS = {
    "pca": "PCA",
    "random_projection": "Random projection",
    "raw_mse": r"AE raw $\LMSE$",
    AE_LOG_FITNESS_VARIANT: r"AE raw $\LSOFAAE$",
    AE_LOG_FITNESS_MOO_VARIANT: r"AE raw $\AUPGrad(\LMSE, \Lfit)$",
    "norm_sigmoid_bce": r"AE normalized data, sigmoid output, $\LBCE$",
    "norm_sigmoid_mse": r"AE normalized data, sigmoid output, $\LMSE$",
}
REPORT_METHODS = (
    "pca",
    "random_projection",
    "raw_mse",
    AE_LOG_FITNESS_VARIANT,
    AE_LOG_FITNESS_MOO_VARIANT,
    "norm_sigmoid_bce",
    "norm_sigmoid_mse",
)


def evaluate_batch(problem: LSGOProblem, x: np.ndarray, batch_size: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    values = np.empty(x.shape[0], dtype=float)
    for start in range(0, x.shape[0], batch_size):
        batch = x[start : start + batch_size]
        for offset, row in enumerate(batch):
            values[start + offset] = problem.evaluate(row)
    return values


def reconstruct_with_pca(
    train_samples: np.ndarray,
    eval_samples: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
) -> np.ndarray:
    mean, components, _ = fit_pca2(train_samples)
    embedding = (eval_samples - mean) @ components.T
    reconstructed = mean + embedding @ components
    return np.clip(reconstructed, lb, ub)


def reconstruct_with_random_projection(
    train_samples: np.ndarray,
    eval_samples: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    mean, projection, _ = fit_random_projection2(train_samples, seed=seed)
    embedding = (eval_samples - mean) @ projection.T
    reconstructed = mean + embedding @ np.linalg.pinv(projection.T)
    return np.clip(reconstructed, lb, ub)


def reconstruct_with_autoencoder(
    samples: np.ndarray,
    *,
    variant: str,
    func_id: str,
    dim: int,
    benchmark_seed: int,
    group_size: int,
    checkpoint_dir: Path,
    device: str,
) -> np.ndarray:
    output_base = output_base_for(
        func_id,
        argparse.Namespace(
            projection="autoencoder",
            ae_variant=variant,
            output_dir=checkpoint_dir,
            dim=dim,
        ),
    )
    checkpoint_path = autoencoder_checkpoint_path(output_base)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"missing checkpoint for {func_id} {variant}: {checkpoint_path}. "
            "Regenerate the corresponding CEC2013 landscape first."
        )
    autoencoder = load_autoencoder_checkpoint(checkpoint_path, device=device)
    reconstructed = autoencoder.decode(autoencoder.encode(samples))
    problem = LSGOProblem(
        func_id=func_id,
        D=dim,
        seed=benchmark_seed,
        group_size=group_size,
    )
    return np.clip(reconstructed, problem.lb, problem.ub)


def latex_function_label(func_id: str) -> str:
    index = int(func_id.split("_f")[-1])
    return rf"$F_{{{index}}}$"


def format_value(value: float) -> str:
    return f"{value:.2e}"


def format_row_values(values: list[float]) -> list[str]:
    min_value = min(values)
    formatted_values = []
    for value in values:
        formatted = format_value(value)
        if value == min_value:
            formatted = rf"\textbf{{{formatted}}}"
        formatted_values.append(formatted)
    return formatted_values


def render_table(func_ids: list[str], metrics: dict[str, list[float]]) -> str:
    methods = REPORT_METHODS
    columns = "l" + "c" * len(methods)
    lines = [
        r"\begin{table}[!htbp]",
        r"  \centering",
        r"  \caption{Fitness preservation MSE by CEC2013 function and dimensionality reduction method.}",
        r"  \label{tab:cec-reconstruction-mse}",
        r"  \small",
        r"  \resizebox{\textwidth}{!}{%",
        rf"  \begin{{tabular}}{{{columns}}}",
        r"    \hline",
        "    Function & "
        + " & ".join(METHOD_LABELS[method] for method in methods)
        + r" \\ ",
        r"    \hline",
    ]
    for row_idx, func_id in enumerate(func_ids):
        row_values = [metrics[method][row_idx] for method in methods]
        values = " & ".join(format_row_values(row_values))
        lines.append(f"    {latex_function_label(func_id)} & {values} \\\\ ")
    average_values = [float(np.mean(metrics[method])) for method in methods]
    std_values = [float(np.std(metrics[method])) for method in methods]
    lines.extend(
        [
            r"    \hline",
            "    Average & "
            + " & ".join(format_row_values(average_values))
            + r" \\ ",
            "    Std. & "
            + " & ".join(format_row_values(std_values))
            + r" \\ ",
            r"    \hline",
            r"  \end{tabular}",
            r"  }",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate CEC2013 LSGO fitness preservation MSE table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--functions", type=str, default="all")
    parser.add_argument("--dim", type=int, default=1000)
    parser.add_argument("--benchmark_seed", type=int, default=0)
    parser.add_argument("--group_size", type=int, default=50)
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5000,
        help="Number of evaluation samples used to compute the table.",
    )
    parser.add_argument(
        "--train_n_samples",
        type=int,
        default=100000,
        help="Number of samples used to fit PCA and random projection baselines.",
    )
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--rp_seed", type=int, default=42)
    parser.add_argument(
        "--eval_seed",
        type=int,
        default=12345,
        help="Random seed for held out table evaluation samples.",
    )
    parser.add_argument("--ae_device", type=str, default="cpu")
    parser.add_argument("--eval_batch_size", type=int, default=20000)
    parser.add_argument("--checkpoint_dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    func_ids = parse_functions(args.functions)
    methods = REPORT_METHODS
    metrics = {method: [] for method in methods}

    print(
        "Evaluating dimensionality reduction maps on "
        f"{args.n_samples} CEC2013 samples..."
    )
    for func_id in func_ids:
        print(f"  evaluating {func_id}...")
        problem = LSGOProblem(
            func_id=func_id,
            D=args.dim,
            seed=args.benchmark_seed,
            group_size=args.group_size,
        )
        train_rng = np.random.default_rng(args.sample_seed)
        train_samples = sample_uniform_in_bounds(
            args.train_n_samples,
            problem.lb,
            problem.ub,
            train_rng,
        )
        eval_rng = np.random.default_rng(args.eval_seed)
        eval_samples = sample_uniform_in_bounds(
            args.n_samples,
            problem.lb,
            problem.ub,
            eval_rng,
        )
        y = evaluate_batch(problem, eval_samples, args.eval_batch_size)

        reconstructions = {
            "pca": reconstruct_with_pca(
                train_samples,
                eval_samples,
                problem.lb,
                problem.ub,
            ),
            "random_projection": reconstruct_with_random_projection(
                train_samples,
                eval_samples,
                problem.lb,
                problem.ub,
                seed=args.rp_seed,
            ),
        }
        for variant in methods:
            if variant in reconstructions:
                continue
            print(f"    loading {variant}...")
            reconstructions[variant] = reconstruct_with_autoencoder(
                eval_samples,
                variant=variant,
                func_id=func_id,
                dim=args.dim,
                benchmark_seed=args.benchmark_seed,
                group_size=args.group_size,
                checkpoint_dir=args.checkpoint_dir,
                device=args.ae_device,
            )
        for method in methods:
            y_hat = evaluate_batch(problem, reconstructions[method], args.eval_batch_size)
            metrics[method].append(float(np.mean((y - y_hat) ** 2)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_table(func_ids, metrics))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
