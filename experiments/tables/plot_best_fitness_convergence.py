"""Plot W&B best_fitness convergence over function evaluations.

The script reads the ``*_runs.json`` sidecar written by ``generate_table.py``,
fetches sampled W&B histories by run id, averages runs that share the same
simplified method label and benchmark function, and writes convergence curves.

Typical usage from the project root::

    python3 -m experiments.tables.plot_best_fitness_convergence --dim 1000

To keep a preliminary figure quick to refresh, limit the number of histories per
group::

    python3 -m experiments.tables.plot_best_fitness_convergence \
        --dim 1000 --max-runs-per-group 3
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from experiments.tables.generate_table import (
    DEFAULT_PROJECT,
    DEFAULT_PROBLEMS,
    load_runs_index,
)
from experiments.tables.plot_best_fitness_by_group import (
    DEFAULT_BASENAME_TEMPLATE,
    infer_entity_from_index,
    parse_csv_list,
    problem_sort_key,
    run_ids_from_index,
    selected_methods_from_index,
    short_method_label,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results"
DEFAULT_FIGURE = (
    RESULTS_ROOT
    / "figures"
    / "lsgo"
    / "cec2013_lsgo_best_fitness_convergence"
    / "cec2013_lsgo_best_fitness_convergence.pdf"
)
DEFAULT_METHODS = (
    "Full space (F=0.5, CR=0.9, abs.)",
    "Dual EA LoRA r=1 (add.)",
    "Dual EA LoRA r=1 (add. v2)",
    "Dual EA LoRA r=2 (add.)",
    "Dual EA LoRA r=4 (add.)",
    "Dual EA LoRA r=8 (add.)",
)


@dataclass(frozen=True)
class RunMeta:
    """Metadata needed to group one W&B run history."""

    run_id: str
    method: str
    problem: str
    function: str
    seed: object | None
    url: str


@dataclass(frozen=True)
class RunHistory:
    """One run trajectory after basic cleaning."""

    meta: RunMeta
    fe: np.ndarray
    best_fitness: np.ndarray


@dataclass(frozen=True)
class Curve:
    """Averaged convergence curve for one method on one function."""

    method: str
    problem: str
    function: str
    fe_grid: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    n_runs: np.ndarray
    run_ids: tuple[str, ...]


def default_runs_index_path(dim: int, basename: str) -> Path:
    """Default run index path produced by generate_table.py."""
    return RESULTS_ROOT / "tables" / "cec2013_lsgo" / f"dim{dim}" / f"{basename}_runs.json"


def default_curve_csv_path(out_figure: Path) -> Path:
    """CSV curve data paired with the figure."""
    return out_figure.with_suffix(".csv")


def default_cache_path(out_figure: Path) -> Path:
    """JSONL cache path paired with the figure."""
    try:
        rel_figure = out_figure.relative_to(RESULTS_ROOT / "figures")
    except ValueError:
        return out_figure.with_name(f"{out_figure.stem}_history_cache.jsonl")
    return (
        RESULTS_ROOT
        / "caches"
        / rel_figure.parent
        / f"{out_figure.stem}_history_cache.jsonl"
    )


def function_sort_key(function: str) -> tuple[int, str]:
    """Sort labels like F1, F2, ..., F15."""
    if function.startswith("F") and function[1:].isdigit():
        return int(function[1:]), function
    return 10_000, function


def load_history_cache(path: Path) -> dict[str, dict[str, object]]:
    """Load cached sampled histories keyed by run id."""
    records: dict[str, dict[str, object]] = {}
    if not path.is_file():
        return records
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            run_id = str(record.get("id") or "")
            if run_id:
                records[run_id] = record
    return records


def append_history_cache(path: Path, record: dict[str, object]) -> None:
    """Append one sampled history record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def selected_run_metadata(
    index: dict[str, object],
    *,
    methods: tuple[str, ...],
    problems: tuple[str, ...],
    max_runs_per_group: int | None,
) -> list[RunMeta]:
    """Select run ids and keep at most ``max_runs_per_group`` per method/function."""
    raw = run_ids_from_index(index, methods=methods, problems=problems)
    by_group: defaultdict[tuple[str, str], list[RunMeta]] = defaultdict(list)
    for run_id, meta in raw.items():
        method = str(meta["method"])
        problem = str(meta["problem"])
        by_group[(method, problem)].append(
            RunMeta(
                run_id=run_id,
                method=method,
                problem=problem,
                function=str(meta["function"]),
                seed=meta.get("seed"),
                url=str(meta.get("url") or ""),
            )
        )

    selected: list[RunMeta] = []
    for key in sorted(by_group, key=lambda item: (problem_sort_key(item[1]), item[0])):
        runs = sorted(by_group[key], key=lambda item: str(item.seed))
        if max_runs_per_group is not None:
            runs = runs[:max_runs_per_group]
        selected.extend(runs)
    return selected


def _json_number_list(values: Iterable[float]) -> list[float]:
    """Convert numpy values to plain JSON floats."""
    out: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(val):
            out.append(val)
    return out


def fetch_one_history(
    api,
    *,
    entity: str,
    project: str,
    meta: RunMeta,
    fitness_key: str,
    samples: int,
) -> dict[str, object]:
    """Fetch one sampled W&B history and return a cacheable record."""
    run = api.run(f"{entity}/{project}/{meta.run_id}")
    history = run.history(keys=[fitness_key, "nfe"], samples=samples, pandas=True)
    if "nfe" in history:
        fe_values = history["nfe"].to_numpy(dtype=float)
    else:
        fe_values = history["_step"].to_numpy(dtype=float)
    fitness_values = history[fitness_key].to_numpy(dtype=float)

    summary = getattr(run, "summary", {}) or {}
    summary_fe = summary.get("nfe") or summary.get("total_nfe") or summary.get("_step")
    summary_fitness = summary.get(fitness_key)
    if summary_fe is not None and summary_fitness is not None:
        fe_values = np.append(fe_values, float(summary_fe))
        fitness_values = np.append(fitness_values, float(summary_fitness))

    return {
        "id": meta.run_id,
        "method": meta.method,
        "problem": meta.problem,
        "function": meta.function,
        "seed": meta.seed,
        "url": meta.url,
        "fitness_key": fitness_key,
        "samples": int(samples),
        "fe": _json_number_list(fe_values),
        "best_fitness": _json_number_list(fitness_values),
    }


def record_to_history(record: dict[str, object], meta: RunMeta) -> RunHistory | None:
    """Convert a cache record to sorted finite arrays."""
    fe_raw = record.get("fe")
    fitness_raw = record.get("best_fitness")
    if not isinstance(fe_raw, list) or not isinstance(fitness_raw, list):
        return None
    n = min(len(fe_raw), len(fitness_raw))
    if n == 0:
        return None

    fe = np.asarray(fe_raw[:n], dtype=float)
    fitness = np.asarray(fitness_raw[:n], dtype=float)
    mask = np.isfinite(fe) & np.isfinite(fitness) & (fe >= 0) & (fitness > 0)
    if not np.any(mask):
        return None
    fe = fe[mask]
    fitness = fitness[mask]
    order = np.argsort(fe)
    fe = fe[order]
    fitness = fitness[order]

    unique_fe, unique_idx = np.unique(fe, return_index=True)
    unique_fit = fitness[unique_idx]
    return RunHistory(meta=meta, fe=unique_fe, best_fitness=unique_fit)


def load_or_fetch_histories(
    index: dict[str, object],
    *,
    metas: list[RunMeta],
    project: str,
    entity: str | None,
    fitness_key: str,
    samples: int,
    cache_path: Path,
    refresh_cache: bool,
) -> list[RunHistory]:
    """Load sampled histories from cache or W&B."""
    import wandb

    resolved_entity = entity or infer_entity_from_index(index, project)
    if resolved_entity is None:
        raise ValueError("Could not infer W&B entity. Pass --wandb_entity.")

    cached = load_history_cache(cache_path)
    api = wandb.Api(timeout=600)
    histories: list[RunHistory] = []

    for idx, meta in enumerate(metas, start=1):
        record = None if refresh_cache else cached.get(meta.run_id)
        if record is None or record.get("fitness_key") != fitness_key:
            record = fetch_one_history(
                api,
                entity=resolved_entity,
                project=project,
                meta=meta,
                fitness_key=fitness_key,
                samples=samples,
            )
            append_history_cache(cache_path, record)
            cached[meta.run_id] = record

        history = record_to_history(record, meta)
        if history is not None:
            histories.append(history)

        if idx % 25 == 0 or idx == len(metas):
            print(f"Prepared {idx}/{len(metas)} sampled histories...", flush=True)

    return histories


def step_values_on_grid(history: RunHistory, fe_grid: np.ndarray) -> np.ndarray:
    """Evaluate a logged best-fitness step curve on a fixed FE grid."""
    idx = np.searchsorted(history.fe, fe_grid, side="right") - 1
    values = np.full_like(fe_grid, np.nan, dtype=float)
    valid = idx >= 0
    values[valid] = history.best_fitness[idx[valid]]
    if np.any(~valid) and len(history.best_fitness):
        values[~valid] = history.best_fitness[0]
    return values


def aggregate_histories(
    histories: list[RunHistory],
    *,
    fe_grid: np.ndarray,
) -> list[Curve]:
    """Average histories by method and benchmark function."""
    buckets: defaultdict[tuple[str, str, str], list[RunHistory]] = defaultdict(list)
    for history in histories:
        buckets[(history.meta.method, history.meta.problem, history.meta.function)].append(
            history
        )

    curves: list[Curve] = []
    for (method, problem, function), runs in sorted(
        buckets.items(), key=lambda item: (problem_sort_key(item[0][1]), item[0][0])
    ):
        values = np.vstack([step_values_on_grid(run, fe_grid) for run in runs])
        n_runs = np.sum(np.isfinite(values), axis=0)
        curves.append(
            Curve(
                method=method,
                problem=problem,
                function=function,
                fe_grid=fe_grid,
                mean=np.nanmean(values, axis=0),
                std=np.nanstd(values, axis=0),
                n_runs=n_runs,
                run_ids=tuple(run.meta.run_id for run in runs),
            )
        )
    return curves


def write_curves_csv(path: Path, curves: list[Curve]) -> None:
    """Write mean convergence curves to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "function",
                "problem",
                "method",
                "fe",
                "mean_best_fitness",
                "std_best_fitness",
                "n_runs",
                "run_ids",
            ),
        )
        writer.writeheader()
        for curve in curves:
            for fe, mean, std, n_runs in zip(
                curve.fe_grid, curve.mean, curve.std, curve.n_runs
            ):
                writer.writerow(
                    {
                        "function": curve.function,
                        "problem": curve.problem,
                        "method": curve.method,
                        "fe": int(round(float(fe))),
                        "mean_best_fitness": f"{float(mean):.12g}",
                        "std_best_fitness": f"{float(std):.12g}",
                        "n_runs": int(n_runs),
                        "run_ids": " ".join(curve.run_ids),
                    }
                )
    print(f"Wrote {path}")


def save_legend_figure(
    handles: list[Line2D],
    labels: list[str],
    out_path: Path,
    *,
    also_png: bool,
) -> None:
    """Save a standalone legend."""
    n_cols = min(3, max(1, math.ceil(len(labels) / 2)))
    height = 0.45 * math.ceil(len(labels) / n_cols) + 0.25
    fig = plt.figure(figsize=(7.0, height))
    fig.legend(
        handles,
        labels,
        loc="center",
        ncol=n_cols,
        fontsize=8,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.0,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    if also_png:
        fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.05, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_path}")
    if also_png:
        print(f"Wrote {out_path.with_suffix('.png')}")


def plot_curves(
    curves: list[Curve],
    *,
    methods: tuple[str, ...],
    out_path: Path,
    legend_path: Path | None,
    also_png: bool,
) -> None:
    """Plot one small convergence panel per benchmark function."""
    functions = tuple(sorted({curve.function for curve in curves}, key=function_sort_key))
    if not functions:
        raise ValueError("No curves to plot.")

    by_key = {(curve.method, curve.function): curve for curve in curves}
    n_panels = len(functions)
    n_cols = min(3, n_panels)
    n_rows = math.ceil(n_panels / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.4 * n_cols, 2.3 * n_rows),
        squeeze=False,
        sharex=True,
    )

    colors = plt.cm.tab10(np.linspace(0, 0.95, max(1, len(methods))))
    linestyles = ("-", "-", "--", ":", "-.", (0, (3, 1, 1, 1)))
    plotted_handles: list[Line2D] = []
    plotted_labels: list[str] = []

    for panel_idx, function in enumerate(functions):
        ax = axes[panel_idx // n_cols][panel_idx % n_cols]
        ax.set_title(f"${function}$", fontsize=9)
        ax.set_yscale("log")
        for method_idx, method in enumerate(methods):
            curve = by_key.get((method, function))
            if curve is None:
                continue
            (line,) = ax.plot(
                curve.fe_grid,
                curve.mean,
                lw=1.2,
                color=colors[method_idx],
                ls=linestyles[method_idx % len(linestyles)],
                label=short_method_label(method),
            )
            if function == functions[0]:
                plotted_handles.append(line)
                plotted_labels.append(short_method_label(method))
        ax.grid(True, which="both", alpha=0.25, linewidth=0.6)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(6, 6))

    for panel_idx in range(n_panels, n_rows * n_cols):
        axes[panel_idx // n_cols][panel_idx % n_cols].set_visible(False)

    fig.supxlabel("Function evaluations")
    fig.supylabel("Mean best fitness")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    if also_png:
        fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Wrote {out_path}")
    if also_png:
        print(f"Wrote {out_path.with_suffix('.png')}")

    leg_path = legend_path or out_path.with_name(f"{out_path.stem}_legend{out_path.suffix}")
    save_legend_figure(plotted_handles, plotted_labels, leg_path, also_png=also_png)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=1000, help="Problem dimension.")
    parser.add_argument("--basename", type=str, default=None, help="Run index basename.")
    parser.add_argument("--runs-index", type=Path, default=None, help="Run index JSON path.")
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help=(
            "Comma-separated method labels. Method labels containing commas are "
            "easier to select by leaving this unset and using the default."
        ),
    )
    parser.add_argument(
        "--problems",
        type=str,
        default=None,
        help="Comma-separated problem ids. Defaults to all CEC-2013 LSGO functions.",
    )
    parser.add_argument("--fitness-key", type=str, default="best_fitness")
    parser.add_argument("--wandb_project", type=str, default=DEFAULT_PROJECT)
    parser.add_argument("--wandb_entity", type=str, default=os.environ.get("WANDB_ENTITY") or None)
    parser.add_argument("--samples", type=int, default=220, help="Sampled history rows per run.")
    parser.add_argument("--grid-points", type=int, default=120, help="FE grid points per curve.")
    parser.add_argument("--max-fe", type=float, default=3_000_000, help="Maximum FE for plots.")
    parser.add_argument(
        "--max-runs-per-group",
        type=int,
        default=None,
        help="Optional cap on run histories per method/function group.",
    )
    parser.add_argument("--cache", type=Path, default=None, help="History cache JSONL path.")
    parser.add_argument("--refresh-cache", action="store_true", help="Ignore cached histories.")
    parser.add_argument("--out", type=Path, default=DEFAULT_FIGURE, help="Output PDF path.")
    parser.add_argument("--csv-out", type=Path, default=None, help="Output curve CSV path.")
    parser.add_argument("--legend-out", type=Path, default=None, help="Output legend PDF path.")
    parser.add_argument("--no-png", action="store_true", help="Skip companion PNG files.")
    args = parser.parse_args()

    if args.dim <= 0:
        parser.error("--dim must be positive")
    if args.samples <= 1:
        parser.error("--samples must be greater than one")
    if args.grid_points <= 1:
        parser.error("--grid-points must be greater than one")
    if args.max_runs_per_group is not None and args.max_runs_per_group <= 0:
        parser.error("--max-runs-per-group must be positive when provided")

    basename = args.basename or DEFAULT_BASENAME_TEMPLATE.format(dim=args.dim)
    runs_index_path = args.runs_index or default_runs_index_path(args.dim, basename)
    index = load_runs_index(runs_index_path)

    requested_methods = parse_csv_list(args.methods) if args.methods is not None else DEFAULT_METHODS
    methods = selected_methods_from_index(index, requested_methods)
    if not methods:
        parser.error("No selected methods were found in the run index.")

    problems = parse_csv_list(args.problems) or DEFAULT_PROBLEMS
    out_path = args.out
    csv_path = args.csv_out or default_curve_csv_path(out_path)
    cache_path = args.cache or default_cache_path(out_path)

    metas = selected_run_metadata(
        index,
        methods=methods,
        problems=problems,
        max_runs_per_group=args.max_runs_per_group,
    )
    if not metas:
        parser.error("No run ids matched the selected methods and problems.")

    histories = load_or_fetch_histories(
        index,
        metas=metas,
        project=args.wandb_project,
        entity=args.wandb_entity,
        fitness_key=args.fitness_key,
        samples=args.samples,
        cache_path=cache_path,
        refresh_cache=args.refresh_cache,
    )
    fe_grid = np.linspace(0, float(args.max_fe), int(args.grid_points))
    curves = aggregate_histories(histories, fe_grid=fe_grid)
    write_curves_csv(csv_path, curves)
    plot_curves(
        curves,
        methods=methods,
        out_path=out_path,
        legend_path=args.legend_out,
        also_png=not args.no_png,
    )


if __name__ == "__main__":
    main()
