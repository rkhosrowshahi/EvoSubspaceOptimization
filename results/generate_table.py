"""Aggregate W&B run summaries into CSV tables.

Fetches finished runs for configurable problem ids (``--problems``, default
``cec2013_lsgo_f1``--``f15``) and ``config.dim`` (default 1000). It averages
``wandb.summary['best_fitness']`` over all runs that share the same simplified method
label (derived from each run's W&B ``group``), and writes a CSV with benchmark
functions as rows and methods as columns under ``results/tables/``. Missing
(group, function) pairs are written as ``--``.

Use ``--from-local`` to skip the W&B API and regenerate LaTeX from an existing
CSV in ``--out_dir`` (see ``--input-csv``). Pair the CSV with a ``{basename}_seeds.json``
sidecar (written on full W&B runs) so Wilcoxon markers and the w/t/l row are preserved.
A ``{basename}_runs.json`` sidecar lists W&B run ids per ``wandb.group`` so you can
inspect or cite runs without querying the API again.

Outputs are written under ``results/cec2013_lsgo/dim{D}/`` by default (see
``default_out_dir_for_dim``).
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
from pathlib import Path

from scipy.stats import ranksums

DEFAULT_PROJECT = os.environ.get("WANDB_PROJECT", "evo-subspace-opt")

RESULTS_ROOT = Path(__file__).resolve().parent


def default_out_dir_for_dim(dim: int) -> Path:
    """Default output folder for CEC-2013 LSGO tables at problem dimension ``dim``."""
    return RESULTS_ROOT / "cec2013_lsgo" / f"dim{dim}"

DEFAULT_PROBLEMS: tuple[str, ...] = tuple(
    f"cec2013_lsgo_f{i}" for i in range(1, 16)
)

MISSING_CELL = "--"

# Significance level for compact-table win/tie/loss (Wilcoxon rank-sum per function).
WTL_RANKSUM_ALPHA = 0.05

# Superscript markers vs the leftmost baseline column (per function, Wilcoxon rank-sum).
STAT_MARKER_WIN = r"\textsuperscript{\dag}"
STAT_MARKER_LOSS = r"\textsuperscript{\ddag}"
STAT_MARKER_TIE = r"\textsuperscript{$\approx$}"

# Compact chapter table: full single-phase subspace grid plus fixed Dual EA LoRA ranks.
COMPACT_BASELINE_LABEL = "Full space (F=0.5, CR=0.9, abs.)"

COMPACT_TABLE_SINGLE_PHASE_COLUMNS: tuple[str, ...] = (
    COMPACT_BASELINE_LABEL,
    "Random projection d=50 (abs.)",
    "Random projection d=50 (add.)",
    "Random projection d=10 (abs.)",
    "Random projection d=10 (add.)",
    "Random blocking d=50 (abs.)",
    "Random blocking d=50 (add.)",
    "Random blocking d=10 (abs.)",
    "Random blocking d=10 (add.)",
    "LoRA r=8 (abs.)",
)

# Dual EA LoRA columns always listed (empty cells when a rank has no runs yet).
DEFAULT_DUAL_EA_LORA_RANKS: tuple[int, ...] = (1, 2, 4, 8)

# Block-wise Dual EA LoRA ablations (``lora_ib``, ``lora_shared``, etc. in W&B group).
# The suffix ``blocks10`` on ``dual_ea-lora-lora_rank*`` groups marks dual EA v2 (elite
# handoff), not $B = 10$ block LoRA.
DEFAULT_DUAL_EA_BLOCK_COUNT = 10
DEFAULT_DUAL_EA_BLOCK_LORA_RANK = 1
DUAL_EA_V2_ASSIGNMENT_SUFFIX = "add. v2"
# Legacy CSV / sidecar column name before the v2 label was standardized.
DUAL_EA_V2_LEGACY_LABEL_SUFFIX = ", revised"
DUAL_EA_BLOCK_LORA_VARIANTS: tuple[tuple[str, str], ...] = (
    ("I", "IB-LoRA"),
    ("S", "S-LoRA"),
    ("GS", "GS-LoRA"),
    ("Diag", "Diag-LoRA"),
    ("R1", "R1-LoRA"),
)
_WANDB_SUBSPACE_TO_BLOCK_VARIANT: dict[str, str] = {
    "lora_ib": "I",
    "lora_shared": "S",
    "lora_gated": "GS",
    "lora_diag": "Diag",
    "lora_rank1": "R1",
}

COMPACT_TABLE_FALLBACKS: dict[str, tuple[str, ...]] = {
    COMPACT_BASELINE_LABEL: (
        "Full space (F=0.5, CR=0.9)",
    ),
}

COMPACT_HEADER_OVERRIDES: dict[str, str] = {
    COMPACT_BASELINE_LABEL: r"\shortstack{Full space\\$F{=}0.5$, $CR{=}0.9$}",
    "Random projection d=50 (abs.)": r"\shortstack{RandProj\\abs.}",
    "Random projection d=50 (add.)": r"\shortstack{RandProj\\add.}",
    "Random projection d=10 (abs.)": r"\shortstack{RandProj\\abs.}",
    "Random projection d=10 (add.)": r"\shortstack{RandProj\\add.}",
    "Random blocking d=50 (abs.)": r"\shortstack{RandBlock\\abs.}",
    "Random blocking d=50 (add.)": r"\shortstack{RandBlock\\add.}",
    "Random blocking d=10 (abs.)": r"\shortstack{RandBlock\\abs.}",
    "Random blocking d=10 (add.)": r"\shortstack{RandBlock\\add.}",
    "LoRA r=8 (abs.)": r"\shortstack{LoRA\\$r{=}8$, abs.}",
    "Dual EA LoRA r=1 (add. v2)": r"\shortstack{Dual EA LoRA\\$r{=}1$, add., v2}",
}


def _dual_ea_lora_v2_label(rank: int, mode: str = "additive") -> str:
    abbrev = _assignment_abbrev(mode)
    return f"Dual EA LoRA r={rank} ({abbrev} v2)"


def _canonical_dual_ea_v2_label(label: str) -> str:
    """Map legacy ``(add., revised)`` column names to ``(add. v2)``."""
    m = re.match(r"Dual EA LoRA r=(\d+) \(add\., revised\)", label)
    if m:
        return _dual_ea_lora_v2_label(int(m.group(1)))
    return label


def _default_compact_dual_ea_block_columns() -> tuple[str, ...]:
    """Blockwise Dual EA LoRA columns shown in the chapter compact table."""
    return tuple(
        _dual_ea_block_lora_label(
            key,
            blocks=DEFAULT_DUAL_EA_BLOCK_COUNT,
            rank=DEFAULT_DUAL_EA_BLOCK_LORA_RANK,
            mode="additive",
        )
        for key, _ in DUAL_EA_BLOCK_LORA_VARIANTS
    )


def _dual_ea_lora_rank(label: str) -> int | None:
    """Rank for global Dual EA LoRA v1 (``add.`` only, not v2)."""
    m = re.match(r"Dual EA LoRA r=(\d+) \(add\.\)$", label)
    return int(m.group(1)) if m else None


def _dual_ea_global_lora_rank(label: str) -> int | None:
    """Rank for global Dual EA LoRA columns (v1 or v2)."""
    m = re.match(r"Dual EA LoRA r=(\d+) \(add(?:\. v2)?\.\)", label)
    return int(m.group(1)) if m else None


def _dual_ea_lora_is_v2(label: str) -> bool:
    return bool(re.match(r"Dual EA LoRA r=\d+ \(add\. v2\)", label))


def _dual_ea_block_lora_label(
    variant_key: str,
    *,
    blocks: int,
    rank: int,
    mode: str,
) -> str:
    display = dict(DUAL_EA_BLOCK_LORA_VARIANTS).get(variant_key, variant_key)
    return f"Dual EA {display} B={blocks} r={rank} ({_assignment_abbrev(mode)})"


def _parse_dual_ea_block_lora_label(label: str) -> tuple[str, int, int] | None:
    """Return (variant key, B, r) for ``Dual EA *-LoRA B=… r=…`` labels."""
    m = re.match(
        r"Dual EA ([A-Za-z0-9]+-LoRA) B=(\d+) r=(\d+)",
        label,
    )
    if not m:
        return None
    display = m.group(1)
    by_display = {name: key for key, name in DUAL_EA_BLOCK_LORA_VARIANTS}
    variant = by_display.get(display)
    if variant is None:
        return None
    return variant, int(m.group(2)), int(m.group(3))


def _discovered_dual_ea_lora_additive_labels(
    buckets: dict[tuple[str, str], list[float]],
) -> tuple[str, ...]:
    """Dual EA LoRA additive columns present in W&B (e.g. ranks 1, 2, 4, 8)."""
    labels = {lbl for (lbl, _) in buckets if _dual_ea_lora_rank(lbl) is not None}
    return tuple(sorted(labels, key=lambda s: _dual_ea_lora_rank(s) or 0))


def _dual_ea_lora_labels_in_columns(group_labels: tuple[str, ...]) -> tuple[str, ...]:
    """Dual EA LoRA additive columns present in a CSV header."""
    labels = [lbl for lbl in group_labels if _dual_ea_lora_rank(lbl) is not None]
    return tuple(sorted(labels, key=lambda s: _dual_ea_lora_rank(s) or 0))


def _compact_table_columns(
    buckets: dict[tuple[str, str], list[float]] | None = None,
    *,
    group_labels: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    """Baseline, then Dual EA (latest protocol), then single-phase subspace grid."""
    del buckets, group_labels  # fixed layout; discovery helpers used only for logging
    dual_cols: list[str] = []
    for r in DEFAULT_DUAL_EA_LORA_RANKS:
        dual_cols.append(f"Dual EA LoRA r={r} (add.)")
        if r == 1:
            dual_cols.append(_dual_ea_lora_v2_label(1))
    block = _default_compact_dual_ea_block_columns()
    return (
        (COMPACT_BASELINE_LABEL,)
        + tuple(dual_cols)
        + block
        + COMPACT_TABLE_SINGLE_PHASE_COLUMNS[1:]
    )


def _compact_column_header(label: str) -> str:
    if label in COMPACT_HEADER_OVERRIDES:
        return COMPACT_HEADER_OVERRIDES[label]
    if _dual_ea_lora_is_v2(label):
        rank = _dual_ea_global_lora_rank(label)
        if rank is not None:
            return rf"\shortstack{{Dual EA LoRA\\$r{{=}}{rank}$, add., v2}}"
    rank = _dual_ea_lora_rank(label)
    if rank is not None:
        return rf"\shortstack{{Dual EA LoRA\\$r{{=}}{rank}$, add.}}"
    parsed = _parse_dual_ea_block_lora_label(label)
    if parsed is not None:
        variant, blocks, r = parsed
        display = dict(DUAL_EA_BLOCK_LORA_VARIANTS)[variant]
        return (
            rf"\shortstack{{Dual EA {display}\\"
            rf"$B{{=}}{blocks}$, $r{{=}}{r}$, add.}}"
        )
    return _latex_column_header(label)


def _global_lora_matrix_side(dim: int) -> int:
    r"""Return $M = \lceil \sqrt{D} \rceil$ for the global LoRA reshape."""
    return math.ceil(math.sqrt(dim))


def _global_lora_search_dim(dim: int, rank: int) -> int:
    r"""Return $d_{\mathrm{search}} = 2Mr$ for global LoRA."""
    return 2 * _global_lora_matrix_side(dim) * rank


def _balanced_block_symbols(dim: int, blocks: int) -> tuple[int, int]:
    """Balanced block size upper bound ``s`` and ``M_s = ceil(sqrt(s))``."""
    s = math.ceil(dim / blocks)
    m_s = math.ceil(math.sqrt(s))
    return s, m_s


def _block_lora_search_dim(variant: str, dim: int, blocks: int, rank: int) -> int:
    """Search dimension for block LoRA variants (matches chapter equations)."""
    _, m_s = _balanced_block_symbols(dim, blocks)
    if variant == "G":
        return _global_lora_search_dim(dim, rank)
    if variant == "I":
        return 2 * blocks * rank * m_s
    if variant == "S":
        return 2 * rank * m_s
    if variant == "GS":
        return 2 * rank * m_s + blocks
    if variant == "Diag":
        return 2 * m_s * rank + blocks * m_s
    if variant == "R1":
        return 2 * m_s * rank + 2 * blocks * m_s
    raise ValueError(f"unknown block LoRA variant {variant!r}")


def _parse_config_int(val: object) -> int | None:
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _lora_rank_from_config_or_label(cfg: dict, label: str) -> int | None:
    rank = _parse_config_int(cfg.get("lora_rank"))
    if rank is not None:
        return rank
    m = re.match(r"LoRA r=(\d+)", label)
    if m:
        return int(m.group(1))
    return _dual_ea_global_lora_rank(label)


def _search_dim_from_wandb_config(
    cfg: dict, label: str, problem_dim: int
) -> int | None:
    r"""Read $d_{\mathrm{search}}$ from W&B run config (``dim``, ``subspace_dim``, ``search_dim*``)."""
    problem_d = _parse_config_int(cfg.get("dim")) or problem_dim
    method = str(cfg.get("subspace_method") or "").strip().lower()

    if _dual_ea_lora_rank(label) is not None or label.startswith("Dual EA"):
        for key in ("search_dim_sub", "search_dim"):
            v = _parse_config_int(cfg.get(key))
            if v is not None:
                return v
        rank = _lora_rank_from_config_or_label(cfg, label)
        if rank is not None:
            return _global_lora_search_dim(problem_d, rank)
        return None

    if label.startswith("Full space") or method == "fullspace":
        for key in ("search_dim", "dim", "search_dim_full"):
            v = _parse_config_int(cfg.get(key))
            if v is not None:
                return v
        return problem_d

    if method == "lora" or label.startswith("LoRA"):
        v = _parse_config_int(cfg.get("search_dim"))
        if v is not None:
            return v
        rank = _lora_rank_from_config_or_label(cfg, label)
        if rank is not None:
            return _global_lora_search_dim(problem_d, rank)
        return None

    for key in ("search_dim", "subspace_dim"):
        v = _parse_config_int(cfg.get(key))
        if v is not None:
            return v
    return None


def _compact_search_dimension(label: str, dim: int) -> int | None:
    r"""Fallback $d_{\mathrm{search}}$ when no W&B config values were collected."""
    if label.startswith("Full space"):
        return dim
    m = re.match(r"Random projection d=(\d+)", label)
    if m:
        return int(m.group(1))
    m = re.match(r"Random blocking d=(\d+)", label)
    if m:
        return int(m.group(1))
    m = re.match(r"LoRA r=(\d+)", label)
    if m:
        return _global_lora_search_dim(dim, int(m.group(1)))
    rank = _dual_ea_global_lora_rank(label)
    if rank is not None:
        return _global_lora_search_dim(dim, rank)
    parsed = _parse_dual_ea_block_lora_label(label)
    if parsed is not None:
        variant, blocks, r = parsed
        return _block_lora_search_dim(variant, dim, blocks, r)
    return None


def _resolved_compact_search_dim(
    search_dim_buckets: dict[tuple[str, str], list[int]],
    label: str,
    problems: tuple[str, ...],
    problem_dim: int,
) -> int | None:
    """Most common logged search dimension for a compact column (constant across functions)."""
    per_problem: list[int] = []
    for problem in problems:
        vals = search_dim_buckets.get((label, problem), [])
        if not vals:
            continue
        per_problem.append(int(statistics.mode(vals)))
    if per_problem:
        return int(statistics.mode(per_problem))
    return _compact_search_dimension(label, problem_dim)


def _fmt_search_dim_cell(value: int | None) -> str:
    if value is None:
        return r"\multicolumn{1}{c}{--}"
    return rf"\multicolumn{{1}}{{c}}{{{int(value)}}}"


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


def _normalize_raw_group(group: str, dim: int) -> str:
    """Canonicalize W&B group string before simplification."""
    g = _strip_lsgo_problem_group_prefix(group, dim)
    if not g:
        return g
    # Legacy runs used ``cec2013_lsgo_fN dim=1000 ...`` (spaces, not hyphens).
    m = re.match(
        r"^cec2013_lsgo_f\d+\s+dim=\d+\s+(.+)$",
        g,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    return g


def _assignment_abbrev(token: str) -> str:
    if token == "absolute":
        return "abs."
    if token == "additive":
        return "add."
    return token


_DUAL_EA_LORA_GROUP_RE = re.compile(
    r"^dual_ea-"
    r"(?P<method>lora(?:_(?:ib|diag|gated|shared|rank1))?)"
    r"-lora_rank(?P<rank>\d+)"
    r"(?:-blocks(?P<blocks>\d+))?"
    r"-(?P<mode>absolute|additive)-"
)


def _simplify_dual_ea_lora_group(g: str) -> str | None:
    m = _DUAL_EA_LORA_GROUP_RE.match(g)
    if not m:
        return None
    method = m.group("method")
    rank = int(m.group("rank"))
    mode = m.group("mode")
    blocks_s = m.group("blocks")
    abbrev = _assignment_abbrev(mode)
    if method == "lora":
        if blocks_s is not None:
            return _dual_ea_lora_v2_label(rank, mode)
        return f"Dual EA LoRA r={rank} ({abbrev})"
    if blocks_s is None:
        return None
    variant = _WANDB_SUBSPACE_TO_BLOCK_VARIANT.get(method)
    if variant is None:
        return None
    return _dual_ea_block_lora_label(
        variant, blocks=int(blocks_s), rank=rank, mode=mode
    )


def _simplify_group_name(raw_group: str) -> str:
    """Map stripped W&B group suffix to a short column label."""
    g = raw_group.strip()
    if not g:
        return "(no group)"

    # Legacy two-phase full-subspace sweeps (problem prefix already removed).
    if g.startswith("2_phase"):
        np_m = re.search(r"\bnp=(\d+)\b", g)
        np_s = f", np={np_m.group(1)}" if np_m else ""
        sub_m = re.search(r"\bsub_dim=(\d+)\b", g)
        d_s = sub_m.group(1) if sub_m else "?"
        if "mut_f=" in g:
            mf = re.search(r"\bmut_f=([\d.]+)\b", g)
            mut_s = f", F={mf.group(1)}" if mf else ""
        elif re.search(r"\bF=([\d.]+)\s+CR=([\d.]+)\b", g):
            fm = re.search(r"\bF=([\d.]+)\s+CR=([\d.]+)\b", g)
            mut_s = f", F={fm.group(1)}, CR={fm.group(2)}"
        else:
            mut_s = ""
        return f"2-phase full-sub add. d={d_s}{np_s}{mut_s}"

    dual_lora = _simplify_dual_ea_lora_group(g)
    if dual_lora is not None:
        return dual_lora

    m = re.match(r"lora-lora_rank(\d+)-(absolute|additive)-", g)
    if m:
        r, mode = m.group(1), m.group(2)
        return f"LoRA r={r} ({_assignment_abbrev(mode)})"

    if g.startswith("fullspace-"):
        if "evolving_controls" in g or "evolving" in g:
            return "Full space (self evolving)"
        if re.search(r"rand1bin-F0\.5-CR0\.9", g):
            if "absolute" in g:
                return "Full space (F=0.5, CR=0.9, abs.)"
            return "Full space (F=0.5, CR=0.9)"
        if "absolute-de-multiseed" in g or g.endswith("-de-multiseed"):
            return "Full space (evolving DE)"
        return "Full space"

    m = re.match(
        r"random_projection-subdim(\d+)-(absolute|additive)-",
        g,
    )
    if m:
        d, mode = m.group(1), m.group(2)
        de = " F=0.5, CR=0.9" if "rand1bin-F0.5-CR0.9" in g else ""
        return f"Random projection d={d} ({_assignment_abbrev(mode)}{de})"

    m = re.match(
        r"random_blocking-subdim(\d+)-(absolute|additive)-",
        g,
    )
    if m:
        d, mode = m.group(1), m.group(2)
        de = " F=0.5, CR=0.9" if "rand1bin-F0.5-CR0.9" in g else ""
        return f"Random blocking d={d} ({_assignment_abbrev(mode)}{de})"

    return g


def _group_column_order(label: str) -> tuple:
    """Stable column order aligned with thesis-style baselines first."""
    prefix_order = (
        "Full space",
        "Random projection",
        "Random blocking",
        "LoRA",
        "Dual EA",
        "2-phase",
        "(no group)",
    )
    for i, prefix in enumerate(prefix_order):
        if label.startswith(prefix) or label == prefix:
            return (i, label)
    return (len(prefix_order), label)


def _problem_row_label(problem_id: str) -> str:
    m = re.search(r"_f(\d+)$", problem_id)
    if m:
        return f"F{m.group(1)}"
    return problem_id


def _latex_function_row_label(fn_label: str) -> str:
    m = re.match(r"F(\d+)$", fn_label.strip())
    if m:
        return rf"$F_{{{m.group(1)}}}$"
    return fn_label


def _parse_numeric_cell(cell: str) -> float | None:
    if cell in ("", MISSING_CELL, "--"):
        return None
    try:
        return float(cell)
    except (TypeError, ValueError):
        return None


def _stat_marker_latex(outcome: str | None) -> str:
    if outcome == "w":
        return STAT_MARKER_WIN
    if outcome == "l":
        return STAT_MARKER_LOSS
    if outcome == "t":
        return STAT_MARKER_TIE
    return ""


def _fmt_siunitx_cell(
    cell: str,
    *,
    bold: bool = False,
    vs_baseline: str | None = None,
) -> str:
    if cell in ("", MISSING_CELL, "--"):
        return r"\multicolumn{1}{c}{--}"
    val = float(cell)
    s = format(val, ".3g")
    s = re.sub(r"e\+0*", "e", s, flags=re.IGNORECASE)
    marker = _stat_marker_latex(vs_baseline)
    # Use the same \num formatting for every cell in text columns. \textbf{\num}
    # with mode=text (set locally in each table) keeps bold and non-bold consistent.
    if bold:
        return rf"\multicolumn{{1}}{{c}}{{\textbf{{\num{{{s}}}{marker}}}}}"
    return rf"\multicolumn{{1}}{{c}}{{\num{{{s}}}{marker}}}"


def _compact_bucket_vals(
    buckets: dict[tuple[str, str], list[float]],
    problem: str,
    label: str,
) -> list[float]:
    """Per-seed best fitness values for a compact column (with label fallbacks)."""
    for candidate in (label,) + COMPACT_TABLE_FALLBACKS.get(label, ()):
        vals = buckets.get((candidate, problem), [])
        if vals:
            return list(vals)
        legacy_v2 = _canonical_dual_ea_v2_label(candidate)
        if legacy_v2 != candidate:
            vals = buckets.get((legacy_v2, problem), [])
            if vals:
                return list(vals)
    return []


def _ranksum_wtl_outcome(
    method_vals: list[float],
    baseline_vals: list[float],
    *,
    alpha: float = WTL_RANKSUM_ALPHA,
) -> str:
    """Win/tie/loss for method vs baseline via two-sided Wilcoxon rank-sum."""
    _, p_value = ranksums(method_vals, baseline_vals)
    if p_value >= alpha:
        return "t"
    method_med = statistics.median(method_vals)
    base_med = statistics.median(baseline_vals)
    if method_med < base_med:
        return "w"
    if method_med > base_med:
        return "l"
    return "t"


def _wtl_ranksum_vs_baseline(
    buckets: dict[tuple[str, str], list[float]],
    problems: tuple[str, ...],
    baseline_label: str,
    method_label: str,
    *,
    alpha: float = WTL_RANKSUM_ALPHA,
) -> tuple[int, int, int] | None:
    """Aggregate win/tie/loss over functions using Wilcoxon rank-sum on per-seed fitness."""
    w = t = l = 0
    compared = 0
    for problem in problems:
        base_vals = _compact_bucket_vals(buckets, problem, baseline_label)
        method_vals = _compact_bucket_vals(buckets, problem, method_label)
        if not base_vals or not method_vals:
            continue
        compared += 1
        outcome = _ranksum_wtl_outcome(method_vals, base_vals, alpha=alpha)
        if outcome == "w":
            w += 1
        elif outcome == "t":
            t += 1
        else:
            l += 1
    if compared == 0:
        return None
    return (w, t, l)


def _fmt_wtl_cell(counts: tuple[int, int, int] | None) -> str:
    if counts is None:
        return r"\multicolumn{1}{c}{--}"
    w, t, l = counts
    return rf"\multicolumn{{1}}{{c}}{{{w}/{t}/{l}}}"


def _best_labels_in_row(
    raw_cells: dict[str, str],
    labels: tuple[str, ...],
) -> set[str]:
    """Return labels tied for minimum numeric fitness in a table row."""
    numeric: list[tuple[str, float]] = []
    for lbl in labels:
        val = _parse_numeric_cell(raw_cells.get(lbl, MISSING_CELL))
        if val is not None:
            numeric.append((lbl, val))
    if not numeric:
        return set()
    min_val = min(v for _, v in numeric)
    tol = max(1e-12, 1e-9 * abs(min_val))
    return {lbl for lbl, v in numeric if abs(v - min_val) <= tol}


def _format_row_cells(
    cells: dict[str, str],
    labels: tuple[str, ...],
    *,
    compact: bool = False,
    buckets: dict[tuple[str, str], list[float]] | None = None,
    problem: str | None = None,
    baseline_label: str | None = None,
) -> str:
    raw: dict[str, str] = {}
    for lbl in labels:
        if compact:
            raw[lbl] = _compact_cell(cells, lbl)
        else:
            raw[lbl] = cells.get(lbl, MISSING_CELL)
    best = _best_labels_in_row(raw, labels)
    base_lbl = baseline_label or (COMPACT_BASELINE_LABEL if compact else "")
    parts: list[str] = []
    for lbl in labels:
        vs: str | None = None
        if (
            compact
            and buckets is not None
            and problem
            and lbl != base_lbl
            and raw[lbl] not in ("", MISSING_CELL, "--")
        ):
            base_vals = _compact_bucket_vals(buckets, problem, base_lbl)
            method_vals = _compact_bucket_vals(buckets, problem, lbl)
            if base_vals and method_vals:
                vs = _ranksum_wtl_outcome(method_vals, base_vals)
        parts.append(_fmt_siunitx_cell(raw[lbl], bold=lbl in best, vs_baseline=vs))
    return " & ".join(parts)


def _compact_cell(cells: dict[str, str], label: str) -> str:
    for candidate in (label,) + COMPACT_TABLE_FALLBACKS.get(label, ()):
        for key in (candidate, _canonical_dual_ea_v2_label(candidate)):
            value = cells.get(key, MISSING_CELL)
            if value not in ("", MISSING_CELL, "--"):
                return value
    return MISSING_CELL


def _latex_column_header(label: str) -> str:
    """Compact \\shortstack header for each simplified method label."""
    headers: dict[str, str] = {
        "Full space (F=0.5, CR=0.9)": r"\shortstack{Full space\\$F{=}0.5$, $CR{=}0.9$}",
        "Full space (F=0.5, CR=0.9, abs.)": r"\shortstack{Full space\\$F{=}0.5$, $CR{=}0.9$, abs.}",
        "Full space (evolving DE)": r"\shortstack{Full space\\evolving DE}",
        "Full space (fixed DE)": r"\shortstack{Full space\\evolving DE}",
        "Random projection d=10 (abs. F=0.5, CR=0.9)": r"\shortstack{RandProj\\abs., $F{=}0.5$}",
        "Random projection d=10 (abs.)": r"\shortstack{RandProj\\abs.}",
        "Random projection d=10 (add.)": r"\shortstack{RandProj\\add.}",
        "Random projection d=50 (abs. F=0.5, CR=0.9)": r"\shortstack{RandProj\\abs., $F{=}0.5$}",
        "Random projection d=50 (abs.)": r"\shortstack{RandProj\\abs.}",
        "Random projection d=50 (add.)": r"\shortstack{RandProj\\add.}",
        "Random blocking d=10 (abs.)": r"\shortstack{RandBlock\\abs.}",
        "Random blocking d=10 (add. F=0.5, CR=0.9)": r"\shortstack{RandBlock\\add., $F{=}0.5$}",
        "Random blocking d=10 (add.)": r"\shortstack{RandBlock\\add.}",
        "Random blocking d=50 (abs.)": r"\shortstack{RandBlock\\abs.}",
        "Random blocking d=50 (add. F=0.5, CR=0.9)": r"\shortstack{RandBlock\\add., $F{=}0.5$}",
        "Random blocking d=50 (add.)": r"\shortstack{RandBlock\\add.}",
        "LoRA r=8 (abs.)": r"\shortstack{LoRA\\$r{=}8$, abs.}",
        "Dual EA LoRA r=1 (add.)": r"\shortstack{Dual EA LoRA\\$r{=}1$, add.}",
        "Dual EA LoRA r=2 (add.)": r"\shortstack{Dual EA LoRA\\$r{=}2$, add.}",
        "Dual EA LoRA r=4 (add.)": r"\shortstack{Dual EA LoRA\\$r{=}4$, add.}",
        "Dual EA LoRA r=8 (add.)": r"\shortstack{Dual EA LoRA\\$r{=}8$, add.}",
        "2-phase full-sub add. d=50, np=128, F=0.5": r"\shortstack{2-phase full-sub\\$n_p{=}128$}",
        "2-phase full-sub add. d=50, np=128, F=0.5, CR=0.9": (
            r"\shortstack{2-phase full-sub\\$F{=}0.5$, $CR{=}0.9$}"
        ),
        "2-phase full-sub add. d=50, np=64": r"\shortstack{2-phase full-sub\\$n_p{=}64$}",
    }
    if label in headers:
        return headers[label]
    escaped = label.replace("_", r"\_")
    return rf"\shortstack{{{escaped}}}"


def _latex_header_family(label: str) -> str:
    for prefix, family in (
        ("Full space", "Full space"),
        ("Random projection", "RandProj"),
        ("Random blocking", "RandBlock"),
        ("LoRA", "LoRA"),
        ("Dual EA", "Dual EA"),
        ("2-phase", "2-phase"),
    ):
        if label.startswith(prefix):
            return family
    return label


def write_latex_table(
    out_path: Path,
    *,
    dim: int,
    problems: tuple[str, ...],
    group_labels: tuple[str, ...],
    rows: list[tuple[str, dict[str, str]]],
) -> None:
    ncols = len(group_labels)
    col_spec = rf"@{{}}l *{{{ncols}}}{{c}} @{{}}"
    families: list[tuple[str, int]] = []
    for lbl in group_labels:
        fam = _latex_header_family(lbl)
        if families and families[-1][0] == fam:
            families[-1] = (fam, families[-1][1] + 1)
        else:
            families.append((fam, 1))

    fam_line = " & ".join(
        rf"\multicolumn{{{span}}}{{c}}{{\textit{{{fam}}}}}"
        for fam, span in families
    )
    col_line = " & ".join(_latex_column_header(lbl) for lbl in group_labels)

    body_lines: list[str] = []
    for fn_label, cells in rows:
        cells_fmt = _format_row_cells(cells, group_labels)
        body_lines.append(f"{_latex_function_row_label(fn_label)} & {cells_fmt} \\\\")

    nfe = r"3 \times 10^{6}"
    dim_tex = r"10^{3}" if dim == 1000 else str(dim)
    caption = (
        f"Mean best fitness on the CEC~2013 LSGO benchmark at $D = {dim_tex}$ after "
        f"${nfe}$ function evaluations with differential evolution (DE). "
        f"Rows are benchmark functions $F_1$--$F_{{15}}$; columns are subspace maps "
        f"and baselines (assignment mode and DE settings as indicated). "
        f"Each entry is the mean of final best fitness over replicate runs for that "
        f"method. Lower is better. Bold entries are the best mean in each row. "
        f"Missing cells indicate no finished runs for that function and method."
    )

    tex = "\n".join(
        [
            "% Auto-generated by generate_table.py. Regenerate with:",
            "%   python3 generate_table.py --dim 1000",
            r"\begin{table}[p]",
            r"\centering",
            rf"\caption{{{caption}}}",
            rf"\label{{tab:lsgo-all-fs-dim{dim}}}",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{2pt}",
            r"\resizebox{\textwidth}{!}{%",
            r"\sisetup{mode=text, reset-text-series=false, detect-weight=true}%",
            rf"\begin{{tabular}}{{{col_spec}}}",
            r"\toprule",
            rf" & {fam_line} \\",
            r"\cmidrule(lr){2-" + str(ncols + 1) + "}",
            "Function & " + col_line + r" \\",
            r"\midrule",
            *body_lines,
            r"\bottomrule",
            r"\end{tabular}%",
            "}",
            r"\end{table}",
            "",
        ]
    )
    out_path.write_text(tex, encoding="utf-8")


def write_latex_compact_table(
    out_path: Path,
    *,
    dim: int,
    rows: list[tuple[str, dict[str, str]]],
    buckets: dict[tuple[str, str], list[float]],
    search_dim_buckets: dict[tuple[str, str], list[int]],
    problems: tuple[str, ...],
) -> None:
    """Pilot-style table for the LSGO chapter (readable width)."""
    row_group_labels = tuple(rows[0][1].keys()) if rows else ()
    compact_columns = _compact_table_columns(buckets, group_labels=row_group_labels)
    compact_headers = tuple(_compact_column_header(lbl) for lbl in compact_columns)
    ncols = len(compact_columns)
    col_spec = rf"@{{}}l *{{{ncols}}}{{c}} @{{}}"
    col_line = " & ".join(
        rf"\multicolumn{{1}}{{c}}{{{hdr}}}" for hdr in compact_headers
    )
    search_dim_cells = [
        _fmt_search_dim_cell(
            _resolved_compact_search_dim(search_dim_buckets, lbl, problems, dim)
        )
        for lbl in compact_columns
    ]
    search_dim_row = r"$d_{\mathrm{search}}$ & " + " & ".join(search_dim_cells) + r" \\"

    baseline_label = COMPACT_BASELINE_LABEL
    problem_by_fn = {_problem_row_label(str(p)): str(p) for p in problems}

    body_lines: list[str] = []
    for fn_label, cells in rows:
        problem = problem_by_fn.get(fn_label, fn_label)
        vals = _format_row_cells(
            cells,
            compact_columns,
            compact=True,
            buckets=buckets,
            problem=problem,
            baseline_label=baseline_label,
        )
        body_lines.append(f"{_latex_function_row_label(fn_label)} & {vals} \\\\")

    wtl_cells: list[str] = []
    for col_idx, lbl in enumerate(compact_columns):
        if col_idx == 0:
            wtl_cells.append(r"\multicolumn{1}{c}{---}")
        else:
            wtl_cells.append(
                _fmt_wtl_cell(
                    _wtl_ranksum_vs_baseline(
                        buckets, problems, baseline_label, lbl
                    )
                )
            )
    wtl_row = r"$w/t/l$ & " + " & ".join(wtl_cells) + r" \\"

    nfe = r"3 \times 10^{6}"
    dim_tex = r"10^{3}" if dim == 1000 else str(dim)
    fn_range = (
        f"{_latex_function_row_label(rows[0][0])}--{_latex_function_row_label(rows[-1][0])}"
        if rows
        else r"$F_1$--$F_{15}$"
    )
    caption = (
        f"Mean best fitness on the CEC~2013 LSGO benchmark at $D = {dim_tex}$ after "
        f"${nfe}$ function evaluations with differential evolution (DE). "
        f"Rows are benchmark functions {fn_range}; columns are subspace maps with assignment "
        f"mode; the leftmost column is the full space baseline with fixed $F$ and $CR$. "
        f"Dual EA LoRA columns with additive assignment at ranks "
        f"$r \\in \\{{1, 2, 4, 8\\}}$ follow immediately after the baseline. "
        f"Column \\emph{{v2}} at $r = 1$ uses the elite handoff in "
        f"\\cref{{alg:lsgo-dual-ea}} (lines 279--281). Earlier $r = 1$ runs without "
        f"\\emph{{v2}} used unconditional replacement of the worst full space member. "
        f"Blockwise Dual EA LoRA variants at $B = 10$ and $r = 1$ follow "
        f"(IB, S, GS, Diag, and R1 in \\cref{{tab:lsgo-lora-compression-comparison}}). "
        f"Single phase columns then list RandProj, RandBlock, "
        f"and LoRA at $d \\in \\{{10, 50\\}}$ or $r {{=}} 8$ in both assignment modes "
        f"where applicable, even when runs are not yet complete. "
        f"The $d_{{\\mathrm{{search}}}}$ row lists the optimizer search "
        f"dimension logged for each configuration (problem dimension $D$ for full "
        f"space, subspace dimension otherwise). Each entry is the mean of final best fitness "
        f"averaged over three algorithmic seed runs. Lower is better. Bold entries are the "
        f"best mean in each row. Superscript $\\dag$, $\\ddag$, and $\\approx$ on an "
        f"entry indicate that the method is significantly better than, significantly "
        f"worse than, or not significantly different from the leftmost full space "
        f"baseline (fixed $F$ and $CR$) on that function (Wilcoxon rank-sum, "
        f"$\\alpha = {WTL_RANKSUM_ALPHA}$). The bottom row gives aggregate "
        f"win/tie/loss counts from the same tests. Missing cells indicate missing runs."
    )

    tex = "\n".join(
        [
            "% Auto-generated by generate_table.py (compact pilot layout). Regenerate:",
            "%   python3 generate_table.py --dim 1000",
            r"\begin{table}[t]",
            r"\centering",
            rf"\caption{{{caption}}}",
            r"\label{tab:lsgo-preliminary}",
            r"\resizebox{\textwidth}{!}{%",
            r"\sisetup{mode=text, reset-text-series=false, detect-weight=true}%",
            rf"\begin{{tabular}}{{{col_spec}}}",
            r"\toprule",
            rf" & \multicolumn{{{ncols}}}{{c}}{{DE}} \\",
            rf"\cmidrule(lr){{2-{ncols + 1}}}",
            " & " + col_line + r" \\",
            r"\midrule",
            search_dim_row,
            r"\midrule",
            *body_lines,
            r"\midrule",
            wtl_row,
            r"\bottomrule",
            r"\end{tabular}%",
            "}",
            r"\end{table}",
            "",
        ]
    )
    out_path.write_text(tex, encoding="utf-8")


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


def load_table_from_csv(
    csv_path: Path,
    problems: tuple[str, ...],
) -> tuple[list[tuple[str, dict[str, str]]], tuple[str, ...]]:
    """Load mean-fitness table rows and method columns from a local CSV file."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Local table CSV not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "function" not in reader.fieldnames:
            raise ValueError(f"CSV must have a 'function' column: {csv_path}")
        group_labels = tuple(h for h in reader.fieldnames if h != "function")
        rows_by_fn: dict[str, dict[str, str]] = {}
        for row in reader:
            fn_label = row["function"].strip()
            cells = {
                lbl: (row.get(lbl) or MISSING_CELL).strip() or MISSING_CELL
                for lbl in group_labels
            }
            rows_by_fn[fn_label] = cells

    table_rows: list[tuple[str, dict[str, str]]] = []
    for problem in problems:
        fn_label = _problem_row_label(str(problem))
        cells = rows_by_fn.get(
            fn_label,
            {lbl: MISSING_CELL for lbl in group_labels},
        )
        table_rows.append((fn_label, cells))

    return table_rows, group_labels


def seeds_sidecar_path(out_dir: Path, basename: str) -> Path:
    """Per-seed fitness sidecar written alongside the mean CSV."""
    return out_dir / f"{basename}_seeds.json"


def runs_sidecar_path(out_dir: Path, basename: str) -> Path:
    """W&B run id index written alongside the mean CSV."""
    return out_dir / f"{basename}_runs.json"


def _wandb_run_record(run, best_fitness: float) -> dict[str, object]:
    cfg = getattr(run, "config", {}) or {}
    seed = cfg.get("seed")
    if seed is not None:
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            pass
    url = getattr(run, "url", None) or ""
    return {
        "id": str(run.id),
        "name": str(getattr(run, "name", "") or ""),
        "state": str(getattr(run, "state", "") or ""),
        "seed": seed,
        "best_fitness": float(best_fitness),
        "url": url,
    }


def save_runs_index(
    path: Path,
    *,
    dim: int,
    wandb_project: str,
    wandb_entity: str | None,
    group_entries: list[dict[str, object]],
) -> None:
    """Persist W&B run ids keyed by full ``wandb.group`` strings."""
    payload = {
        "dim": dim,
        "wandb_project": wandb_project,
        "wandb_entity": wandb_entity,
        "groups": group_entries,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_runs_index(path: Path) -> dict[str, object]:
    """Load a runs sidecar produced by ``save_runs_index``."""
    return json.loads(path.read_text(encoding="utf-8"))


def lookup_run_ids_by_wandb_group(
    index: dict[str, object], wandb_group: str
) -> list[str]:
    """Return run ids for an exact W&B group string from a runs sidecar."""
    for entry in index.get("groups", []):
        if entry.get("wandb_group") == wandb_group:
            return [str(r["id"]) for r in entry.get("runs", [])]
    return []


def lookup_run_ids_by_method(
    index: dict[str, object],
    *,
    method: str,
    problem: str | None = None,
    function: str | None = None,
) -> list[str]:
    """Return run ids for a simplified table method label (optional problem filter)."""
    ids: list[str] = []
    for entry in index.get("groups", []):
        if entry.get("method") != method:
            continue
        if problem is not None and entry.get("problem") != problem:
            continue
        if function is not None and entry.get("function") != function:
            continue
        ids.extend(str(r["id"]) for r in entry.get("runs", []))
    return ids


def save_seed_buckets(
    path: Path,
    buckets: dict[tuple[str, str], list[float]],
    search_dim_buckets: dict[tuple[str, str], list[int]],
) -> None:
    """Persist per-run best_fitness (and search dims) for offline Wilcoxon tests."""
    payload = {
        "fitness": [
            {"method": lbl, "problem": prob, "values": vals}
            for (lbl, prob), vals in sorted(buckets.items())
            if vals
        ],
        "search_dim": [
            {"method": lbl, "problem": prob, "values": vals}
            for (lbl, prob), vals in sorted(search_dim_buckets.items())
            if vals
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_seed_buckets(
    path: Path,
) -> tuple[dict[tuple[str, str], list[float]], dict[tuple[str, str], list[int]]]:
    """Load per-seed buckets saved by a prior full W&B aggregation."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    buckets: dict[tuple[str, str], list[float]] = {}
    for rec in payload.get("fitness", []):
        buckets[(rec["method"], rec["problem"])] = [float(v) for v in rec["values"]]
    search_dim_buckets: dict[tuple[str, str], list[int]] = {}
    for rec in payload.get("search_dim", []):
        search_dim_buckets[(rec["method"], rec["problem"])] = [
            int(v) for v in rec["values"]
        ]
    return buckets, search_dim_buckets


def write_table_outputs(
    *,
    out_dir: Path,
    basename: str,
    dim: int,
    problems: tuple[str, ...],
    group_labels: tuple[str, ...],
    table_rows: list[tuple[str, dict[str, str]]],
    buckets: dict[tuple[str, str], list[float]],
    search_dim_buckets: dict[tuple[str, str], list[int]],
    run_group_entries: list[dict[str, object]] | None = None,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    write_csv: bool = True,
    write_seeds: bool = True,
    write_runs: bool = True,
) -> None:
    """Write CSV (optional), seeds sidecar (optional), and LaTeX tables."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{basename}.csv"
    out_seeds = seeds_sidecar_path(out_dir, basename)
    out_runs = runs_sidecar_path(out_dir, basename)
    out_tex = out_dir / f"{basename}.tex"
    out_tex_compact = out_dir / f"{basename}_compact.tex"

    if write_seeds and buckets:
        save_seed_buckets(out_seeds, buckets, search_dim_buckets)
    if write_runs and run_group_entries is not None and wandb_project:
        save_runs_index(
            out_runs,
            dim=dim,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            group_entries=run_group_entries,
        )

    if write_csv:
        fieldnames = ("function",) + tuple(group_labels)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            ww = csv.DictWriter(f, fieldnames=fieldnames)
            ww.writeheader()
            for fn_label, cells in table_rows:
                row = {"function": fn_label, **cells}
                ww.writerow(row)

    write_latex_table(
        out_tex,
        dim=dim,
        problems=problems,
        group_labels=group_labels,
        rows=table_rows,
    )
    write_latex_compact_table(
        out_tex_compact,
        dim=dim,
        rows=table_rows,
        buckets=buckets,
        search_dim_buckets=search_dim_buckets,
        problems=problems,
    )

    if write_csv:
        print(
            f"\nWrote CSV (functions x methods; problem rows, mean best_fitness "
            f"in scientific notation):\n  {out_csv}"
        )
    if write_seeds and buckets:
        print(f"Wrote per-seed sidecar (Wilcoxon / w/t/l):\n  {out_seeds}")
    if write_runs and run_group_entries is not None and wandb_project:
        print(f"Wrote W&B run id index:\n  {out_runs}")
    print(f"Wrote LaTeX table (all methods):\n  {out_tex}")
    print(f"Wrote LaTeX table (compact, for chapter):\n  {out_tex_compact}")


def main() -> None:
    prob_help = (
        "Comma-separated benchmark ids "
        "(e.g. ``cec2013_lsgo_f1,cec2013_lsgo_f2``). "
        f"Default: ``{DEFAULT_PROBLEMS[0]}`` through ``{DEFAULT_PROBLEMS[-1]}`` "
        f"(all fifteen CEC-2013 LSGO functions)."
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
        default=None,
        help=(
            "Directory for output files (created if missing). Defaults to "
            "``results/cec2013_lsgo/dim{D}/`` for the chosen ``--dim``."
        ),
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
    parser.add_argument(
        "--from-local",
        action="store_true",
        help=(
            "Skip the W&B API and regenerate LaTeX from an existing CSV in "
            "``--out_dir`` (default ``{basename}.csv``, overridable with "
            "``--input-csv``). Loads ``{basename}_seeds.json`` when present "
            "for Wilcoxon superscripts and the w/t/l row."
        ),
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help=(
            "Local CSV to load when ``--from-local`` is set. Defaults to "
            "``--out_dir/{basename}.csv``."
        ),
    )
    args = parser.parse_args()

    if args.dim <= 0:
        parser.error("--dim must be positive")

    if args.out_dir is None:
        args.out_dir = default_out_dir_for_dim(args.dim)

    if args.problems is None:
        problems: tuple[str, ...] = DEFAULT_PROBLEMS
    else:
        problems = tuple(
            p.strip() for p in args.problems.split(",") if p.strip()
        )
        if not problems:
            parser.error("--problems yielded an empty list after parsing")

    if args.basename is None:
        if problems == DEFAULT_PROBLEMS:
            slug = "all_fs"
        else:
            slug = "_".join(_problem_slug(p) for p in problems)
        args.basename = f"cec2013_lsgo_{slug}_dim{args.dim}_by_group"

    if args.from_local:
        csv_path = args.input_csv or (args.out_dir / f"{args.basename}.csv")
        seeds_path = seeds_sidecar_path(args.out_dir, args.basename)
        runs_path = runs_sidecar_path(args.out_dir, args.basename)
        table_rows, group_labels = load_table_from_csv(csv_path, problems)
        buckets: dict[tuple[str, str], list[float]] = {}
        search_dim_buckets: dict[tuple[str, str], list[int]] = {}
        if seeds_path.is_file():
            buckets, search_dim_buckets = load_seed_buckets(seeds_path)
            print(f"Loaded per-seed sidecar: {seeds_path}")
        else:
            print(
                f"Warning: no seeds sidecar at {seeds_path}; "
                "Wilcoxon superscripts and w/t/l row will be empty. "
                "Run once without --from-local to create it."
            )
        print(f"Loaded local table: {csv_path}")
        if runs_path.is_file():
            print(f"W&B run id index (no API): {runs_path}")
        print(f"Functions: {len(table_rows)}, methods: {len(group_labels)}")
        dual_ranks = _dual_ea_lora_labels_in_columns(group_labels)
        if dual_ranks:
            ranks = ", ".join(f"r={_dual_ea_lora_rank(lbl)}" for lbl in dual_ranks)
            print(f"Compact table Dual EA LoRA columns: {ranks}")
        write_table_outputs(
            out_dir=args.out_dir,
            basename=args.basename,
            dim=args.dim,
            problems=problems,
            group_labels=group_labels,
            table_rows=table_rows,
            buckets=buckets,
            search_dim_buckets=search_dim_buckets,
            write_csv=False,
            write_seeds=False,
        )
        return

    import wandb  # defer so ``--help`` works without importing wandb.api

    api = wandb.Api(timeout=180)
    runs = collect_runs(
        api, args.wandb_entity, args.wandb_project, args.dim, problems
    )

    # (simplified group label, problem) -> list of best_fitness / search_dim
    buckets: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    search_dim_buckets: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
    raw_by_label: defaultdict[str, set[str]] = defaultdict(set)
    runs_by_wandb_group: dict[str, list[dict[str, object]]] = {}
    group_meta: dict[str, dict[str, str]] = {}
    skipped: list[str] = []

    for run in runs:
        g_raw = getattr(run, "group", None)
        g_raw_s = g_raw.strip() if isinstance(g_raw, str) else ""
        g_norm = _normalize_raw_group(g_raw_s, args.dim)
        label = _simplify_group_name(g_norm)
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

        buckets[(label, str(problem))].append(val)
        sd = _search_dim_from_wandb_config(cfg, label, args.dim)
        if sd is not None:
            search_dim_buckets[(label, str(problem))].append(sd)
        if g_norm:
            raw_by_label[label].add(g_norm)
        if g_raw_s:
            runs_by_wandb_group.setdefault(g_raw_s, []).append(
                _wandb_run_record(run, val)
            )
            group_meta[g_raw_s] = {
                "wandb_group_normalized": g_norm,
                "method": label,
                "problem": str(problem),
                "function": _problem_row_label(str(problem)),
            }

    group_labels = sorted({lbl for (lbl, _) in buckets.keys()}, key=_group_column_order)
    compact_columns = _compact_table_columns(buckets)
    compact_only = set(compact_columns)
    full_only = [lbl for lbl in group_labels if lbl not in compact_only]

    prob_display = ", ".join(problems)
    print(
        f"Runs matched (dim={args.dim}, problems [{prob_display}]): "
        f"{len(runs)}"
    )
    print(f"Distinct methods (simplified group labels): {len(group_labels)}")
    dual_ranks = _discovered_dual_ea_lora_additive_labels(buckets)
    if dual_ranks:
        ranks = ", ".join(f"r={_dual_ea_lora_rank(lbl)}" for lbl in dual_ranks)
        print(f"Compact table Dual EA LoRA columns: {ranks}")
    if full_only:
        print(
            f"Methods in full table only ({len(full_only)}); see "
            f"*_by_group.tex / .csv:"
        )
        for lbl in full_only:
            print(f"  - {lbl}")
    for lbl in group_labels:
        raws = sorted(raw_by_label.get(lbl, ()))
        if len(raws) == 1:
            print(f"  - {lbl}  <-  {raws[0]}")
        elif raws:
            print(f"  - {lbl}")
            for r in raws:
                print(f"      <-  {r}")
        else:
            print(f"  - {lbl}")

    if skipped:
        print(f"\nSkipped {len(skipped)} run(s); example reasons:")
        for line in skipped[:20]:
            print(f"  {line}")
        if len(skipped) > 20:
            print(f"  ... ({len(skipped) - 20} more)")

    table_rows: list[tuple[str, dict[str, str]]] = []
    for p in problems:
        fn_label = _problem_row_label(str(p))
        cells = {
            lbl: (
                f"{sum(buckets.get((lbl, p), [])) / len(buckets.get((lbl, p), [])):.6e}"
                if buckets.get((lbl, p), [])
                else MISSING_CELL
            )
            for lbl in group_labels
        }
        table_rows.append((fn_label, cells))

    run_group_entries: list[dict[str, object]] = []
    for wandb_group in sorted(runs_by_wandb_group):
        meta = group_meta[wandb_group]
        run_group_entries.append(
            {
                "wandb_group": wandb_group,
                "wandb_group_normalized": meta["wandb_group_normalized"],
                "method": meta["method"],
                "problem": meta["problem"],
                "function": meta["function"],
                "run_ids": [str(r["id"]) for r in runs_by_wandb_group[wandb_group]],
                "runs": sorted(
                    runs_by_wandb_group[wandb_group],
                    key=lambda r: (
                        r.get("seed") is None,
                        r.get("seed") if r.get("seed") is not None else 0,
                        str(r.get("id", "")),
                    ),
                ),
            }
        )

    write_table_outputs(
        out_dir=args.out_dir,
        basename=args.basename,
        dim=args.dim,
        problems=problems,
        group_labels=tuple(group_labels),
        table_rows=table_rows,
        buckets=dict(buckets),
        search_dim_buckets=dict(search_dim_buckets),
        run_group_entries=run_group_entries,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        write_csv=True,
        write_seeds=True,
        write_runs=True,
    )


if __name__ == "__main__":
    main()
