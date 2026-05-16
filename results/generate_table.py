"""Aggregate W&B run summaries into CSV tables.

Fetches finished runs for configurable problem ids (``--problems``) and
``config.dim`` (``--dim``). It averages ``wandb.summary['best_fitness']`` within each
W&B ``group``, after stripping ``cec2013_lsgo_f*-dim{D}-`` from group strings, and writes
a CSV with **benchmark functions as rows** and **experiment groups / methods as
columns**, under ``results/tables/`` (see ``--out_dir``).
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict
from pathlib import Path

DEFAULT_PROJECT = os.environ.get("WANDB_PROJECT", "evo-subspace-opt")

DEFAULT_PROBLEMS: tuple[str, ...] = tuple(
    f"cec2013_lsgo_f{i}" for i in range(1, 6)
)


def _problem_slug(problem_id: str) -> str:
    """Short token for default output filenames."""
    p = problem_id.strip()
    if p.startswith("cec2013_lsgo_"):
        return p[len("cec2013_lsgo_") :]
    return p.replace("/", "_")


def _api_path(entity: str | None, project: str) -> str:
    return f"{entity}/{project}" if entity else project


def _parse_dim(val: object) -> int | None:
    """Normalize ``config.dim`` whether stored as int or str."""
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _strip_lsgo_problem_group_prefix(group: str, dim: int) -> str:
    """Strip leading ``cec2013_lsgo_f*-dim{D}-`` when ``D`` equals ``dim``.

    Occasionally ``wandb.group`` embeds the wrong function id while
    ``config.problem`` is correct; dropping this prefix aligns rows that share the
    same experimental suffix (e.g. ``fullspace-absolute-de-multiseed``).
    """
    if not isinstance(group, str):
        return ""
    g = group.strip()
    if not g:
        return ""
    stripped = re.sub(
        rf"^cec2013_lsgo_f\d+-dim{int(dim)}-",
        "",
        g,
        count=1,
    )
    return stripped


def collect_runs(
    api,
    entity: str | None,
    project: str,
    dim: int,
    problems: tuple[str, ...],
) -> list:
    """Runs that match ``problem`` subset and dimension (client-filtered fallback)."""

    path = _api_path(entity, project)
    plist = list(problems)
    want_dim = {"$or": [{"config.dim": dim}, {"config.dim": str(dim)}]}

    def match_run(run) -> bool:
        if getattr(run, "state", "") not in ("finished", "crashed"):
            return False
        cfg = getattr(run, "config", {}) or {}
        if cfg.get("problem") not in problems:
            return False
        if _parse_dim(cfg.get("dim")) != dim:
            return False
        return True

    filters = {
        "$and": [
            {"state": {"$in": ["finished", "crashed"]}},
            want_dim,
            {"config.problem": {"$in": plist}},
        ]
    }

    try:
        out = list(api.runs(path, filters=filters))
        if out:
            return [r for r in out if match_run(r)]
    except Exception:
        pass

    return [r for r in api.runs(path) if match_run(r)]


def main() -> None:
    prob_help = (
        "Comma-separated benchmark ids "
        "(e.g. ``cec2013_lsgo_f1,cec2013_lsgo_f2``). "
        f"Default: ``{DEFAULT_PROBLEMS[0]}`` through ``{DEFAULT_PROBLEMS[-1]}``."
    )
    parser = argparse.ArgumentParser(
        description=(
            "Average W&B summary best_fitness by group across selected "
            "problems and ``config.dim``."
        )
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=os.environ.get("WANDB_ENTITY") or None,
        help=(
            "W&B entity/username. Resolved from ``WANDB_ENTITY`` when omitted; "
            "required if your workspace is not inferred."
        ),
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=DEFAULT_PROJECT,
        help=f"W&B project (default: {DEFAULT_PROJECT!r}).",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=1000,
        help="Match runs where ``wandb.run.config.dim`` equals this integer.",
    )
    parser.add_argument(
        "--problems",
        type=str,
        default=None,
        metavar="LIST",
        help=prob_help,
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path(__file__).resolve().parent / "tables",
        help="Directory for output CSV files (created if missing).",
    )
    parser.add_argument(
        "--basename",
        type=str,
        default=None,
        help=(
            "Base filename without ``.csv`` (default derived from "
            "``--dim`` and ``--problems``)."
        ),
    )
    args = parser.parse_args()

    if args.dim <= 0:
        parser.error("--dim must be positive")

    if args.problems is None:
        problems: tuple[str, ...] = DEFAULT_PROBLEMS
    else:
        problems = tuple(
            p.strip() for p in args.problems.split(",") if p.strip()
        )
        if not problems:
            parser.error("--problems yielded an empty list after parsing")

    if args.basename is None:
        slug = "_".join(_problem_slug(p) for p in problems)
        args.basename = f"cec2013_lsgo_{slug}_dim{args.dim}_by_group"

    import wandb  # defer so ``--help`` works without importing wandb.api

    api = wandb.Api(timeout=180)
    runs = collect_runs(
        api, args.wandb_entity, args.wandb_project, args.dim, problems
    )

    # (group, problem) -> list of best_fitness
    buckets: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    skipped: list[str] = []

    for run in runs:
        g_raw = getattr(run, "group", None)
        g = g_raw.strip() if isinstance(g_raw, str) else ""
        g = _strip_lsgo_problem_group_prefix(g, args.dim)
        cfg = getattr(run, "config", {}) or {}
        problem = cfg.get("problem")
        if problem not in problems:
            continue

        summary = getattr(run, "summary", {}) or {}
        bf = summary.get("best_fitness")
        if bf is None:
            skipped.append(f"{run.id} ({run.name}) missing summary best_fitness")
            continue
        try:
            val = float(bf)
        except (TypeError, ValueError):
            skipped.append(f"{run.id} ({run.name}) non-numeric best_fitness={bf!r}")
            continue

        buckets[(g, str(problem))].append(val)

    groups_sorted = sorted({g for (g, _) in buckets.keys()}, key=lambda s: (s == "", s))

    prob_display = ", ".join(problems)
    print(
        f"Runs matched (dim={args.dim}, problems [{prob_display}]): "
        f"{len(runs)}"
    )
    print(f"Distinct W&B groups with usable summary data: {len(groups_sorted)}")
    for gn in groups_sorted:
        disp = gn if gn else "(empty / no group)"
        print(f"  - {disp}")

    if skipped:
        print(f"\nSkipped {len(skipped)} run(s); example reasons:")
        for line in skipped[:20]:
            print(f"  {line}")
        if len(skipped) > 20:
            print(f"  ... ({len(skipped) - 20} more)")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = args.out_dir / f"{args.basename}.csv"

    labeled_groups = [(gn if gn else "(no_group)", gn) for gn in groups_sorted]
    group_labels = tuple(lbl for lbl, _ in labeled_groups)
    fieldnames = ("problem",) + group_labels
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        ww = csv.DictWriter(f, fieldnames=fieldnames)
        ww.writeheader()
        for p in problems:
            row: dict[str, str] = {"problem": str(p)}
            for lbl, g in labeled_groups:
                vals = buckets.get((g, p), [])
                row[lbl] = f"{sum(vals) / len(vals):.14e}" if vals else ""
            ww.writerow(row)

    print(
        f"\nWrote CSV (functions x methods; problem rows, mean best_fitness "
        f"in scientific notation):\n  {out_csv}"
    )


if __name__ == "__main__":
    main()
