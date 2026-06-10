"""Plot mean W&B best_fitness for LSGO run groups.

This companion to ``generate_table.py`` reads the ``*_runs.json`` sidecar that
stores W&B run ids for each benchmark function and simplified method label. It
can either reuse the best_fitness values already stored in that sidecar, or
refresh run summaries from W&B by run id before plotting.

Typical usage from ``projects/EvoSubspaceOptimization``::

    python3 -m experiments.tables.plot_best_fitness_by_group --dim 1000 --from-local-index
    python3 -m experiments.tables.plot_best_fitness_by_group --dim 1000 --refresh-wandb

The default figure is written under ``results/figures/lsgo``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from experiments.tables.generate_table import (
    COMPACT_TABLE_FALLBACKS,
    DEFAULT_PROJECT,
    DEFAULT_PROBLEMS,
    _compact_table_columns,
    _problem_row_label,
    default_out_dir_for_dim,
    load_runs_index,
    runs_sidecar_path,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results"
DEFAULT_BASENAME_TEMPLATE = "cec2013_lsgo_all_fs_dim{dim}_by_group"
DEFAULT_FIGURE = RESULTS_ROOT / "figures" / "lsgo" / "cec2013_lsgo_best_fitness_by_group.pdf"


@dataclass(frozen=True)
class RunSummary:
    """One run-level best_fitness value with enough metadata for grouping."""

    run_id: str
    method: str
    problem: str
    function: str
    wandb_group: str
    best_fitness: float
    seed: object | None = None
    url: str = ""


@dataclass(frozen=True)
class GroupSummary:
    """Mean best_fitness for one method on one benchmark function."""

    method: str
    problem: str
    function: str
    wandb_group: str
    run_ids: tuple[str, ...]
    mean_best_fitness: float
    std_best_fitness: float | None
    n_runs: int


def infer_entity_from_index(index: dict[str, object], project: str) -> str | None:
    """Infer W&B entity from the saved URL when the sidecar omits it."""
    entity = index.get("wandb_entity")
    if entity:
        return str(entity)

    pattern = re.compile(rf"https://wandb\.ai/([^/]+)/{re.escape(project)}/runs/")
    for entry in index.get("groups", []):
        if not isinstance(entry, dict):
            continue
        for run in entry.get("runs", []):
            if not isinstance(run, dict):
                continue
            url = str(run.get("url") or "")
            match = pattern.search(url)
            if match:
                return match.group(1)
    return os.environ.get("WANDB_ENTITY") or None


def default_runs_index_path(dim: int, basename: str) -> Path:
    """Default run id index path produced by generate_table.py."""
    return runs_sidecar_path(default_out_dir_for_dim(dim), basename)


def default_csv_path(out_figure: Path) -> Path:
    """CSV data file paired with the requested figure path."""
    return out_figure.with_suffix(".csv")


def default_cache_path(out_figure: Path) -> Path:
    """JSONL cache of W&B summaries paired with the requested figure path."""
    try:
        rel_figure = out_figure.relative_to(RESULTS_ROOT / "figures")
    except ValueError:
        return out_figure.with_name(f"{out_figure.stem}_wandb_summary_cache.jsonl")
    return (
        RESULTS_ROOT
        / "caches"
        / rel_figure.parent
        / f"{out_figure.stem}_wandb_summary_cache.jsonl"
    )


def parse_csv_list(value: str | None) -> tuple[str, ...] | None:
    """Parse a comma-separated CLI list while preserving inner spaces."""
    if value is None:
        return None
    parts = tuple(part.strip() for part in value.split(",") if part.strip())
    return parts or None


def problem_sort_key(problem: str) -> tuple[int, str]:
    """Sort CEC function ids in numeric order."""
    match = re.search(r"_f(\d+)$", problem)
    if match:
        return int(match.group(1)), problem
    return 10_000, problem


def function_sort_key(function: str) -> tuple[int, str]:
    """Sort labels like F1, F2, ..., F15 in numeric order."""
    match = re.fullmatch(r"F(\d+)", function)
    if match:
        return int(match.group(1)), function
    return 10_000, function


def selected_methods_from_index(
    index: dict[str, object],
    requested_methods: tuple[str, ...] | None,
) -> tuple[str, ...]:
    """Choose plotting methods, defaulting to the compact chapter columns."""
    available = {
        str(entry.get("method"))
        for entry in index.get("groups", [])
        if isinstance(entry, dict) and entry.get("method")
    }
    if requested_methods is not None:
        selected_methods: list[str] = []
        missing: list[str] = []
        for method in requested_methods:
            candidates = (method,) + COMPACT_TABLE_FALLBACKS.get(method, ())
            match = next((candidate for candidate in candidates if candidate in available), None)
            if match is None:
                missing.append(method)
            elif match not in selected_methods:
                selected_methods.append(match)
        if missing:
            print("Warning: requested methods not present in the run index:")
            for method in missing:
                print(f"  {method}")
        return tuple(selected_methods)

    compact_methods: list[str] = []
    for method in _compact_table_columns():
        candidates = (method,) + COMPACT_TABLE_FALLBACKS.get(method, ())
        match = next((candidate for candidate in candidates if candidate in available), None)
        if match is not None and match not in compact_methods:
            compact_methods.append(match)
    compact = tuple(compact_methods)
    if compact:
        return compact
    return tuple(sorted(available))


def iter_index_run_summaries(
    index: dict[str, object],
    *,
    methods: tuple[str, ...],
    problems: tuple[str, ...] | None,
) -> Iterable[RunSummary]:
    """Yield run summaries from the saved run id index."""
    wanted_methods = set(methods)
    wanted_problems = set(problems) if problems else None

    for entry in index.get("groups", []):
        if not isinstance(entry, dict):
            continue
        method = str(entry.get("method") or "")
        problem = str(entry.get("problem") or "")
        if method not in wanted_methods:
            continue
        if wanted_problems is not None and problem not in wanted_problems:
            continue
        function = str(entry.get("function") or _problem_row_label(problem))
        wandb_group = str(entry.get("wandb_group") or "")

        for run in entry.get("runs", []):
            if not isinstance(run, dict):
                continue
            best_fitness = run.get("best_fitness")
            if best_fitness is None:
                continue
            try:
                value = float(best_fitness)
            except (TypeError, ValueError):
                continue
            yield RunSummary(
                run_id=str(run.get("id") or ""),
                method=method,
                problem=problem,
                function=function,
                wandb_group=wandb_group,
                best_fitness=value,
                seed=run.get("seed"),
                url=str(run.get("url") or ""),
            )


def load_wandb_summary_cache(path: Path) -> dict[str, dict[str, object]]:
    """Load cached W&B summary records keyed by run id."""
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


def append_wandb_summary_cache(path: Path, record: dict[str, object]) -> None:
    """Append one W&B summary record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def run_ids_from_index(
    index: dict[str, object],
    *,
    methods: tuple[str, ...],
    problems: tuple[str, ...] | None,
) -> dict[str, dict[str, object]]:
    """Return run id metadata selected from the sidecar."""
    wanted_methods = set(methods)
    wanted_problems = set(problems) if problems else None
    selected: dict[str, dict[str, object]] = {}

    for entry in index.get("groups", []):
        if not isinstance(entry, dict):
            continue
        method = str(entry.get("method") or "")
        problem = str(entry.get("problem") or "")
        if method not in wanted_methods:
            continue
        if wanted_problems is not None and problem not in wanted_problems:
            continue
        function = str(entry.get("function") or _problem_row_label(problem))
        wandb_group = str(entry.get("wandb_group") or "")
        for run in entry.get("runs", []):
            if not isinstance(run, dict):
                continue
            run_id = str(run.get("id") or "")
            if not run_id:
                continue
            selected[run_id] = {
                "method": method,
                "problem": problem,
                "function": function,
                "wandb_group": wandb_group,
                "seed": run.get("seed"),
                "url": str(run.get("url") or ""),
            }
    return selected


def fetch_run_summaries_from_wandb(
    index: dict[str, object],
    *,
    methods: tuple[str, ...],
    problems: tuple[str, ...] | None,
    project: str,
    entity: str | None,
    fitness_key: str,
    cache_path: Path,
    refresh_cache: bool,
) -> list[RunSummary]:
    """Fetch W&B runs by id and return selected run summaries."""
    import wandb

    selected = run_ids_from_index(index, methods=methods, problems=problems)
    if not selected:
        return []

    resolved_entity = entity or infer_entity_from_index(index, project)
    if resolved_entity is None:
        raise ValueError(
            "Could not infer W&B entity. Pass --wandb_entity or set WANDB_ENTITY."
        )

    api = wandb.Api(timeout=600)
    cached = load_wandb_summary_cache(cache_path)
    summaries: list[RunSummary] = []

    for idx, (run_id, meta) in enumerate(selected.items(), start=1):
        record = None if refresh_cache else cached.get(run_id)
        if record is None:
            run = api.run(f"{resolved_entity}/{project}/{run_id}")
            summary = getattr(run, "summary", {}) or {}
            config = getattr(run, "config", {}) or {}
            group = str(getattr(run, "group", "") or meta["wandb_group"])
            record = {
                "id": str(run.id),
                "name": str(getattr(run, "name", "") or ""),
                "state": str(getattr(run, "state", "") or ""),
                "group": group,
                "url": str(getattr(run, "url", "") or meta["url"]),
                "problem": str(config.get("problem") or meta["problem"]),
                "seed": config.get("seed", meta["seed"]),
                "best_fitness": summary.get(fitness_key),
            }
            append_wandb_summary_cache(cache_path, record)

        best_fitness = record.get("best_fitness")
        if best_fitness is None:
            continue
        try:
            value = float(best_fitness)
        except (TypeError, ValueError):
            continue
        summaries.append(
            RunSummary(
                run_id=run_id,
                method=str(meta["method"]),
                problem=str(record.get("problem") or meta["problem"]),
                function=str(meta["function"]),
                wandb_group=str(record.get("group") or meta["wandb_group"]),
                best_fitness=value,
                seed=record.get("seed", meta["seed"]),
                url=str(record.get("url") or meta["url"]),
            )
        )
        if idx % 25 == 0 or idx == len(selected):
            print(f"Prepared {idx}/{len(selected)} W&B summaries...", flush=True)

    return summaries


def aggregate_runs(runs: Iterable[RunSummary]) -> list[GroupSummary]:
    """Average best_fitness over run ids that share a method and problem."""
    buckets: defaultdict[tuple[str, str, str, str], list[RunSummary]] = defaultdict(list)
    for run in runs:
        buckets[(run.method, run.problem, run.function, run.wandb_group)].append(run)

    summaries: list[GroupSummary] = []
    for (method, problem, function, wandb_group), values in sorted(
        buckets.items(), key=lambda item: (problem_sort_key(item[0][1]), item[0][0])
    ):
        fitness = [run.best_fitness for run in values]
        summaries.append(
            GroupSummary(
                method=method,
                problem=problem,
                function=function,
                wandb_group=wandb_group,
                run_ids=tuple(run.run_id for run in values),
                mean_best_fitness=float(statistics.fmean(fitness)),
                std_best_fitness=(
                    float(statistics.stdev(fitness)) if len(fitness) > 1 else None
                ),
                n_runs=len(fitness),
            )
        )
    return summaries


def write_group_csv(path: Path, summaries: list[GroupSummary]) -> None:
    """Write the plotted means and run ids."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "function",
                "problem",
                "method",
                "wandb_group",
                "n_runs",
                "mean_best_fitness",
                "std_best_fitness",
                "run_ids",
            ),
        )
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "function": summary.function,
                    "problem": summary.problem,
                    "method": summary.method,
                    "wandb_group": summary.wandb_group,
                    "n_runs": summary.n_runs,
                    "mean_best_fitness": f"{summary.mean_best_fitness:.12g}",
                    "std_best_fitness": (
                        "" if summary.std_best_fitness is None else f"{summary.std_best_fitness:.12g}"
                    ),
                    "run_ids": " ".join(summary.run_ids),
                }
            )
    print(f"Wrote {path}")


def short_method_label(method: str) -> str:
    """Compact legend labels for the LSGO chapter figure."""
    replacements = {
        "Full space (F=0.5, CR=0.9)": "Full Space EA (DE)",
        "Full space (F=0.5, CR=0.9, abs.)": "Full Space EA (DE)",
        "Dual EA LoRA r=1 (add.)": r"Dual EA LoRA $r=1$",
        "Dual EA LoRA r=1 (add. v2)": r"Dual EA LoRA $r=1$ v2",
        "Dual EA LoRA r=2 (add.)": r"Dual EA LoRA $r=2$",
        "Dual EA LoRA r=4 (add.)": r"Dual EA LoRA $r=4$",
        "Dual EA LoRA r=8 (add.)": r"Dual EA LoRA $r=8$",
        "Dual EA IB-LoRA B=10 r=1 (add.)": "IB-LoRA",
        "Dual EA S-LoRA B=10 r=1 (add.)": "S-LoRA",
        "Dual EA GS-LoRA B=10 r=1 (add.)": "GS-LoRA",
        "Dual EA Diag-LoRA B=10 r=1 (add.)": "Diag-LoRA",
        "Dual EA R1-LoRA B=10 r=1 (add.)": "R1-LoRA",
    }
    return replacements.get(method, method)


def save_legend_figure(
    handles: list[Line2D],
    labels: list[str],
    out_path: Path,
    *,
    also_png: bool,
) -> None:
    """Save a standalone legend so LaTeX can stack it above the plot."""
    n_cols = min(4, max(1, math.ceil(len(labels) / 2)))
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


def plot_group_summaries(
    summaries: list[GroupSummary],
    *,
    methods: tuple[str, ...],
    out_path: Path,
    legend_path: Path | None,
    also_png: bool,
) -> None:
    """Plot mean best_fitness by benchmark function for each selected method."""
    functions = tuple(
        sorted({summary.function for summary in summaries}, key=function_sort_key)
    )
    if not functions:
        raise ValueError("No group summaries to plot.")

    values_by_key = {
        (summary.method, summary.function): summary.mean_best_fitness
        for summary in summaries
    }

    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(methods))))
    markers = ("o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "p")
    x = np.arange(len(functions), dtype=float)
    plotted_handles: list[Line2D] = []
    plotted_labels: list[str] = []

    for idx, method in enumerate(methods):
        y_values = []
        for function in functions:
            value = values_by_key.get((method, function))
            if value is None or value <= 0:
                y_values.append(np.nan)
            else:
                y_values.append(value)
        if all(np.isnan(y_values)):
            continue
        (line,) = ax.plot(
            x,
            y_values,
            marker=markers[idx % len(markers)],
            ms=4.0,
            lw=1.4,
            color=colors[idx],
            label=short_method_label(method),
        )
        plotted_handles.append(line)
        plotted_labels.append(short_method_label(method))

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"${function}$" for function in functions])
    ax.set_xlabel("CEC-2013 LSGO function")
    ax.set_ylabel(r"Mean best fitness")
    ax.grid(True, which="both", alpha=0.25, linewidth=0.6)
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
    parser = argparse.ArgumentParser(
        description=(
            "Fetch W&B run summaries by run id, average best_fitness by group, "
            "and plot the selected CEC-2013 LSGO methods."
        )
    )
    parser.add_argument("--dim", type=int, default=1000, help="Problem dimension.")
    parser.add_argument(
        "--basename",
        type=str,
        default=None,
        help="Base name of the generate_table.py run index.",
    )
    parser.add_argument(
        "--runs-index",
        type=Path,
        default=None,
        help="Path to the generate_table.py *_runs.json sidecar.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        metavar="LIST",
        help="Comma-separated simplified method labels to plot. Defaults to compact table columns.",
    )
    parser.add_argument(
        "--problems",
        type=str,
        default=None,
        metavar="LIST",
        help=(
            "Comma-separated problem ids. Defaults to all problems found in the "
            "run index after method filtering."
        ),
    )
    parser.add_argument(
        "--fitness-key",
        type=str,
        default="best_fitness",
        help="W&B summary key to fetch.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=DEFAULT_PROJECT,
        help=f"W&B project name. Default is {DEFAULT_PROJECT!r}.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=os.environ.get("WANDB_ENTITY") or None,
        help="W&B entity. Inferred from the run index URLs when omitted.",
    )
    parser.add_argument(
        "--refresh-wandb",
        action="store_true",
        help="Fetch W&B summaries by run id instead of using sidecar values.",
    )
    parser.add_argument(
        "--from-local-index",
        action="store_true",
        help="Use best_fitness values already saved in the run id index.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="JSONL cache for W&B summary fetches.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore existing summary cache records when --refresh-wandb is used.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_FIGURE,
        help="Output PDF path.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to the figure path with .csv suffix.",
    )
    parser.add_argument(
        "--legend-out",
        type=Path,
        default=None,
        help="Output legend PDF path. Defaults to <out stem>_legend.pdf.",
    )
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="Skip companion PNG files.",
    )
    args = parser.parse_args()

    if args.dim <= 0:
        parser.error("--dim must be positive")
    if args.refresh_wandb and args.from_local_index:
        parser.error("Choose either --refresh-wandb or --from-local-index, not both.")

    basename = args.basename or DEFAULT_BASENAME_TEMPLATE.format(dim=args.dim)
    runs_index_path = args.runs_index or default_runs_index_path(args.dim, basename)
    index = load_runs_index(runs_index_path)
    requested_methods = parse_csv_list(args.methods)
    methods = selected_methods_from_index(index, requested_methods)
    if not methods:
        parser.error("No selected methods were found in the run index.")

    requested_problems = parse_csv_list(args.problems)
    if requested_problems is None:
        available_problems = {
            str(entry.get("problem"))
            for entry in index.get("groups", [])
            if isinstance(entry, dict) and entry.get("problem")
        }
        requested_problems = tuple(sorted(available_problems, key=problem_sort_key))
        if not requested_problems:
            requested_problems = DEFAULT_PROBLEMS

    out_path = args.out
    csv_path = args.csv_out or default_csv_path(out_path)
    cache_path = args.cache or default_cache_path(out_path)

    if args.refresh_wandb:
        runs = fetch_run_summaries_from_wandb(
            index,
            methods=methods,
            problems=requested_problems,
            project=args.wandb_project,
            entity=args.wandb_entity,
            fitness_key=args.fitness_key,
            cache_path=cache_path,
            refresh_cache=args.refresh_cache,
        )
    else:
        if not args.from_local_index:
            print(
                "Using saved run index values. Pass --refresh-wandb to refresh summaries by run id."
            )
        runs = list(
            iter_index_run_summaries(
                index,
                methods=methods,
                problems=requested_problems,
            )
        )

    summaries = aggregate_runs(runs)
    write_group_csv(csv_path, summaries)
    plot_group_summaries(
        summaries,
        methods=methods,
        out_path=out_path,
        legend_path=args.legend_out,
        also_png=not args.no_png,
    )


if __name__ == "__main__":
    main()
