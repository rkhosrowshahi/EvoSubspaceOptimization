"""Load per-function intrinsic dimensions from analysis CSV outputs."""

from __future__ import annotations

import csv
from pathlib import Path

INTRINSIC_DIM_CSV_NAME = "cec2013_intrinsic_dimension.csv"
INTRINSIC_DIM_CSV_REL = Path("results") / "tables" / "lsgo" / INTRINSIC_DIM_CSV_NAME


def default_intrinsic_dim_csv() -> Path:
    """Return the default intrinsic-dimension CSV path (repo root or cwd)."""
    cwd_candidate = Path.cwd() / INTRINSIC_DIM_CSV_REL
    if cwd_candidate.exists():
        return cwd_candidate
    repo_candidate = Path(__file__).resolve().parents[3] / INTRINSIC_DIM_CSV_REL
    if repo_candidate.exists():
        return repo_candidate
    return cwd_candidate


def load_intrinsic_dims(csv_path: Path | None = None) -> dict[str, int]:
    path = csv_path or default_intrinsic_dim_csv()
    if not path.exists():
        raise FileNotFoundError(
            f"Intrinsic dimension CSV not found: {path}. "
            "Run experiments/dimensionality/cec2013_intrinsic_dimension.py first."
        )
    dims: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            dims[row["func_id"]] = int(row["intrinsic_dim"])
    return dims


def resolve_intrinsic_subspace_dim(
    func_id: str,
    csv_path: Path | None = None,
) -> int:
    dims = load_intrinsic_dims(csv_path)
    if func_id not in dims:
        raise KeyError(f"No intrinsic dimension for {func_id!r} in {csv_path or default_intrinsic_dim_csv()}")
    return dims[func_id]
