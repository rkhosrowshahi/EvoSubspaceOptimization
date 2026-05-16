"""Pure-Python CEC-2013 Large-Scale Global Optimisation (LSGO) benchmark
functions F1-F15.

Data-source policy
------------------
* **D = 1000 and ``cdatafiles/`` present**: the original fixed data files
  (shift vectors, permutation, rotation matrices, group sizes, weights) are
  loaded exactly as specified by the CEC-2013 competition.  This reproduces
  the authoritative benchmark.
* **Any other D** (or when ``cdatafiles/`` is absent): all structural data are
  generated from a user-supplied integer *seed* via NumPy's default RNG.
  This makes every function fully scalable to arbitrary D.

Mathematical formulas are ported verbatim from Benchmarks.cpp / F*.cpp in
the original C++ implementation by Wenxiang Chen (Colorado State University).

Function catalogue
------------------
F1   Shifted Elliptic                    fully separable
F2   Shifted Rastrigin                   fully separable
F3   Shifted Ackley                      fully separable
F4   7-group Elliptic + sep Elliptic     partially separable
F5   7-group Rastrigin + sep Rastrigin   partially separable
F6   7-group Ackley + sep Ackley         partially separable
F7   7-group Schwefel + sep Sphere       partially separable
F8   20-group Elliptic                   fully non-separable
F9   20-group Rastrigin                  fully non-separable
F10  20-group Ackley                     fully non-separable
F11  20-group Schwefel                   fully non-separable
F12  Shifted Rosenbrock                  fully non-separable (no rotation)
F13  20-group Schwefel (overlap conform) overlapping
F14  20-group Schwefel (overlap conflict)overlapping
F15  Shifted Schwefel                    fully non-separable (no rotation)

References
----------
X. Li et al., "Benchmark Functions for the CEC'2013 Special Session and
Competition on Large Scale Global Optimization," RMIT University, 2013.
"""

from __future__ import annotations

import math
import pathlib
from typing import Optional

import numpy as np

PI = math.pi
E = math.e

_DATA_DIR = pathlib.Path(__file__).parent / "cdatafiles"

# Prefix for CEC-2013 LSGO problem ids (``cec2013_lsgo_f1`` ... ``cec2013_lsgo_f15``).
FUNC_ID_PREFIX = "cec2013_lsgo_"


def _func_file_index(canonical_id: str) -> int:
    """Benchmark index 1..15 used in ``F{n}-*.txt`` data filenames."""
    if not canonical_id.startswith(FUNC_ID_PREFIX):
        raise ValueError(f"unexpected func_id for file index: {canonical_id!r}")
    short = canonical_id[len(FUNC_ID_PREFIX) :]
    if len(short) < 2 or short[0] != "f" or not short[1:].isdigit():
        raise ValueError(f"unexpected func_id suffix in {canonical_id!r}")
    return int(short[1:])


# ---------------------------------------------------------------------------
# Default structural parameters (mirror original CEC-2013 spec for D=1000)
# ---------------------------------------------------------------------------
_GROUP_SIZE = 50      # default sub-component size for seed-based generation
_OVERLAP = 5          # overlap between consecutive groups in F13/F14


# ===========================================================================
# Transformation helpers  (ported from Benchmarks.cpp)
# ===========================================================================

def _t_osz(z: np.ndarray) -> np.ndarray:
    """Irregular oscillation mapping T_osz (vectorised)."""
    z = np.asarray(z, dtype=float).copy()
    nonzero = z != 0.0
    safe_abs = np.where(nonzero, np.abs(z), 1.0)
    h = np.where(nonzero, np.log(safe_abs), 0.0)
    c1 = np.where(z > 0.0, 10.0, 5.5)
    c2 = np.where(z > 0.0, 7.9, 3.1)
    osz = np.sign(z) * np.exp(h + 0.049 * (np.sin(c1 * h) + np.sin(c2 * h)))
    osz[~nonzero] = 0.0
    return osz


def _t_asy(z: np.ndarray, beta: float) -> np.ndarray:
    """Asymmetric transformation T_asy^beta."""
    z = np.asarray(z, dtype=float).copy()
    dim = len(z)
    if dim <= 1:
        return z
    pos = z > 0.0
    if not np.any(pos):
        return z
    idx = np.arange(dim, dtype=float)
    exp = 1.0 + beta * idx / (dim - 1) * np.sqrt(np.maximum(z, 0.0))
    z[pos] = np.power(z[pos], exp[pos])
    return z


def _t_lambda(z: np.ndarray, alpha: float) -> np.ndarray:
    """Diagonal scaling transformation Lambda_alpha."""
    z = np.asarray(z, dtype=float).copy()
    dim = len(z)
    if dim <= 1:
        return z
    idx = np.arange(dim, dtype=float)
    return z * np.power(float(alpha), 0.5 * idx / (dim - 1))


# ===========================================================================
# Base objective functions  (ported from Benchmarks.cpp)
# ===========================================================================

def _elliptic(z: np.ndarray) -> float:
    z = _t_osz(z)
    dim = len(z)
    if dim == 1:
        return float(1e6 * z[0] ** 2)
    exps = np.arange(dim, dtype=float) / (dim - 1)
    return float(np.dot(np.power(1e6, exps), z ** 2))


def _rastrigin(z: np.ndarray) -> float:
    z = _t_osz(z)
    z = _t_asy(z, 0.2)
    z = _t_lambda(z, 10.0)
    return float(np.sum(z ** 2 - 10.0 * np.cos(2.0 * PI * z) + 10.0))


def _ackley(z: np.ndarray) -> float:
    z = _t_osz(z)
    dim = len(z)
    s1 = np.sum(z ** 2)
    s2 = np.sum(np.cos(2.0 * PI * z))
    return float(
        -20.0 * np.exp(-0.2 * math.sqrt(s1 / dim)) - np.exp(s2 / dim) + 20.0 + E
    )


def _schwefel(z: np.ndarray) -> float:
    z = _t_osz(z)
    z = _t_asy(z, 0.2)
    return float(np.sum(np.cumsum(z) ** 2))


def _sphere(z: np.ndarray) -> float:
    return float(np.dot(z, z))


def _rosenbrock(z: np.ndarray) -> float:
    return float(
        np.sum(100.0 * (z[1:] - z[:-1] ** 2) ** 2 + (z[:-1] - 1.0) ** 2)
    )


_FUNC_MAP: dict[str, object] = {
    "elliptic": _elliptic,
    "rastrigin": _rastrigin,
    "ackley": _ackley,
    "schwefel": _schwefel,
    "sphere": _sphere,
    "rosenbrock": _rosenbrock,
}


# ===========================================================================
# Orthogonal matrix generation (seed mode)
# ===========================================================================

def _random_orthogonal(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-distributed random ``dim x dim`` orthogonal matrix."""
    G = rng.standard_normal((dim, dim))
    Q, R = np.linalg.qr(G)
    return Q * np.sign(np.diag(R))


# ===========================================================================
# cdatafile readers
# ===========================================================================

def _read_flat(path: pathlib.Path) -> list[float]:
    """Read a flat comma-/newline-delimited file of floats."""
    text = path.read_text()
    return [float(v) for v in text.replace("\n", ",").split(",") if v.strip()]


def _read_matrix(path: pathlib.Path, dim: int) -> np.ndarray:
    """Read a square ``dim x dim`` matrix stored as CSV rows."""
    rows = []
    for line in path.read_text().strip().splitlines():
        line = line.strip()
        if line:
            rows.append([float(v) for v in line.split(",") if v.strip()])
    return np.array(rows, dtype=float)


def _read_ints(path: pathlib.Path) -> list[int]:
    """Read comma-separated integers (supports 1 int per line too)."""
    text = path.read_text()
    return [int(float(v)) for v in text.replace("\n", ",").split(",") if v.strip()]


def _read_lines_float(path: pathlib.Path) -> list[float]:
    """Read one float per line."""
    return [float(l.strip()) for l in path.read_text().strip().splitlines() if l.strip()]


def _cdatafiles_available(func_id: str, data_dir: pathlib.Path) -> bool:
    """Return True if the required cdatafiles exist for *func_id*."""
    fnum = _func_file_index(func_id)
    required = [f"F{fnum}-xopt.txt"]
    func_type = _CONFIGS[func_id][0]
    if func_type not in ("sep", "single"):
        for suf in ("p", "R25", "R50", "R100", "s", "w"):
            required.append(f"F{fnum}-{suf}.txt")
    return all((data_dir / f).exists() for f in required)


# ===========================================================================
# Function configuration table
# ===========================================================================
# (func_type, nonsep_base, sep_base, lb, ub)
_CONFIGS: dict[str, tuple] = {
    "cec2013_lsgo_f1":  ("sep",      "elliptic",   None,        -100, 100),
    "cec2013_lsgo_f2":  ("sep",      "rastrigin",  None,           -5,   5),
    "cec2013_lsgo_f3":  ("sep",      "ackley",     None,          -32,  32),
    "cec2013_lsgo_f4":  ("partial",  "elliptic",   "elliptic",  -100, 100),
    "cec2013_lsgo_f5":  ("partial",  "rastrigin",  "rastrigin",   -5,   5),
    "cec2013_lsgo_f6":  ("partial",  "ackley",     "ackley",     -32,  32),
    "cec2013_lsgo_f7":  ("partial",  "schwefel",   "sphere",    -100, 100),
    "cec2013_lsgo_f8":  ("full",     "elliptic",   None,        -100, 100),
    "cec2013_lsgo_f9":  ("full",     "rastrigin",  None,          -5,   5),
    "cec2013_lsgo_f10": ("full",     "ackley",     None,          -5,   5),
    "cec2013_lsgo_f11": ("full",     "schwefel",   None,         -32,  32),
    "cec2013_lsgo_f12": ("single",   "rosenbrock", None,        -100, 100),
    "cec2013_lsgo_f13": ("conform",  "schwefel",   None,        -100, 100),
    "cec2013_lsgo_f14": ("conflict", "schwefel",   None,        -100, 100),
    "cec2013_lsgo_f15": ("single",   "schwefel",   None,        -100, 100),
}

VALID_FUNC_IDS = set(_CONFIGS.keys())


# ===========================================================================
# Main benchmark class
# ===========================================================================

class LSGO2013:
    """CEC-2013 LSGO benchmark function, scalable to any D.

    When **D = 1000** the canonical data files from the ``cdatafiles/``
    directory are used, reproducing the authoritative benchmark exactly.

    For all other D values (or if the data files are absent) all structural
    data are generated from *seed* via NumPy's default RNG.

    Parameters
    ----------
    func_id : str
        ``"cec2013_lsgo_f1"`` ... ``"cec2013_lsgo_f15"``.
    D : int
        Problem dimensionality.
    seed : int
        RNG seed (used only when the seed path is taken).
    group_size : int
        Sub-component size for seed-based generation (default 50).
    """

    OVERLAP = _OVERLAP

    def __init__(
        self,
        func_id: str,
        D: int,
        seed: int = 0,
        group_size: int = _GROUP_SIZE,
    ) -> None:
        fid = func_id
        if fid not in _CONFIGS:
            raise ValueError(f"Problem not found: {func_id!r}.")
        if D < 1:
            raise ValueError(f"D must be >= 1, got {D}.")

        self.func_id = fid
        self.D = D
        self.seed = seed
        self.group_size = min(group_size, D)

        cfg = _CONFIGS[fid]
        self.func_type: str = cfg[0]
        self._nonsep_fn = _FUNC_MAP[cfg[1]]
        self._sep_fn = _FUNC_MAP[cfg[2]] if cfg[2] is not None else None
        self.lb: float = float(cfg[3])
        self.ub: float = float(cfg[4])

        # --- choose data source ---
        use_files = (D == 1000 and _cdatafiles_available(fid, _DATA_DIR))
        if use_files:
            self._load_cdatafiles(_DATA_DIR)
            self.using_cdatafiles = True
        else:
            self._generate_data_seed(np.random.default_rng(seed))
            self.using_cdatafiles = False

    # ------------------------------------------------------------------
    # Internal unified data structures
    # ------------------------------------------------------------------
    # All eval methods share:
    #   _xopt          : shift vector (D,) - or (eff_D,) for conform
    #   _perm          : permutation 0-indexed (D,) - or (eff_D,)
    #   _R_dict        : {size: orthogonal_matrix}
    #   _group_sizes_s : int array (n_groups,) - size of each sub-component
    #   _weights       : float array (n_groups,)
    #   _n_groups      : int
    #   _nonsep_D      : dims covered by groups (partial only)
    #   _sep_D         : remaining separable dims (partial only)
    #   _eff_D         : effective input length (conform/conflict only)
    #   _xopt_groups   : list of per-group xopt arrays (conflict only)

    # ------------------------------------------------------------------
    # cdatafiles loader (D = 1000)
    # ------------------------------------------------------------------

    def _load_cdatafiles(self, data_dir: pathlib.Path) -> None:
        fnum = _func_file_index(self.func_id)
        D = self.D

        xopt_vals = _read_flat(data_dir / f"F{fnum}-xopt.txt")

        if self.func_type in ("sep", "single"):
            self._xopt = np.array(xopt_vals[:D], dtype=float)
            self._n_groups = None
            return

        # Rotation matrices (fixed sizes 25, 50, 100)
        R_dict = {
            25:  _read_matrix(data_dir / f"F{fnum}-R25.txt",  25),
            50:  _read_matrix(data_dir / f"F{fnum}-R50.txt",  50),
            100: _read_matrix(data_dir / f"F{fnum}-R100.txt", 100),
        }

        # Permutation (1-indexed in file -> 0-indexed here)
        p_raw = _read_ints(data_dir / f"F{fnum}-p.txt")
        perm = np.array(p_raw, dtype=int) - 1

        # Group sizes and weights
        s = np.array(_read_lines_float(data_dir / f"F{fnum}-s.txt"), dtype=int)
        w = np.array(_read_lines_float(data_dir / f"F{fnum}-w.txt"), dtype=float)
        n_g = len(s)

        self._R_dict = R_dict
        self._perm = perm
        self._group_sizes_s = s
        self._weights = w
        self._n_groups = n_g

        if self.func_type == "partial":
            nonsep_D = int(s.sum())
            self._xopt = np.array(xopt_vals[:D], dtype=float)
            self._nonsep_D = nonsep_D
            self._sep_D = D - nonsep_D

        elif self.func_type == "full":
            self._xopt = np.array(xopt_vals[:D], dtype=float)

        elif self.func_type == "conform":
            # xopt length = eff_D (905 for original D=1000)
            self._eff_D = len(xopt_vals)
            self._xopt = np.array(xopt_vals, dtype=float)

        elif self.func_type == "conflict":
            # perm covers eff_D dims; xopt split into per-group chunks
            self._eff_D = len(perm)
            # xopt_vals has sum(s) = 1000 values; split by group sizes
            self._xopt_groups = []
            offset = 0
            for si in s:
                self._xopt_groups.append(
                    np.array(xopt_vals[offset : offset + si], dtype=float)
                )
                offset += si

    # ------------------------------------------------------------------
    # Seed-based data generator (arbitrary D)
    # ------------------------------------------------------------------

    def _generate_data_seed(self, rng: np.random.Generator) -> None:
        D, gs, lb, ub = self.D, self.group_size, self.lb, self.ub
        ov = self.OVERLAP

        if self.func_type in ("sep", "single"):
            self._xopt = rng.uniform(lb, ub, D)
            self._n_groups = None
            return

        # Generate rotation matrix for the standard group size
        R = _random_orthogonal(gs, rng)
        R_dict: dict[int, np.ndarray] = {gs: R}

        if self.func_type == "partial":
            # Scale group count proportionally: ~7 groups at D=1000
            n_g = max(1, round(7 * D / 1000))
            while n_g * gs > D and n_g > 1:
                n_g -= 1
            nonsep_D = n_g * gs
            sep_D = D - nonsep_D

            self._xopt = rng.uniform(lb, ub, D)
            self._perm = rng.permutation(D)
            self._R_dict = R_dict
            self._group_sizes_s = np.full(n_g, gs, dtype=int)
            self._weights = np.exp(rng.uniform(math.log(1e-3), math.log(1e3), n_g))
            self._n_groups = n_g
            self._nonsep_D = nonsep_D
            self._sep_D = sep_D

        elif self.func_type == "full":
            # Scale: ~20 groups at D=1000 -> ceil(D / group_size)
            n_g = max(1, math.ceil(D / gs))
            last_gs = D - (n_g - 1) * gs  # remainder absorbed by last group
            sizes = np.full(n_g, gs, dtype=int)
            sizes[-1] = last_gs
            if last_gs != gs:
                R_dict[last_gs] = _random_orthogonal(last_gs, rng)

            self._xopt = rng.uniform(lb, ub, D)
            self._perm = rng.permutation(D)
            self._R_dict = R_dict
            self._group_sizes_s = sizes
            self._weights = np.exp(rng.uniform(math.log(1e-3), math.log(1e3), n_g))
            self._n_groups = n_g

        elif self.func_type == "conform":
            step = gs - ov
            n_g = max(1, (D - ov) // step)
            eff_D = n_g * step + ov

            self._eff_D = eff_D
            self._xopt = rng.uniform(lb, ub, eff_D)
            self._perm = rng.permutation(eff_D)
            self._R_dict = R_dict
            self._group_sizes_s = np.full(n_g, gs, dtype=int)
            self._weights = np.exp(rng.uniform(math.log(1e-3), math.log(1e3), n_g))
            self._n_groups = n_g

        elif self.func_type == "conflict":
            step = gs - ov
            n_g = max(1, (D - ov) // step)
            eff_D = n_g * step + ov

            self._eff_D = eff_D
            self._xopt_groups = [rng.uniform(lb, ub, gs) for _ in range(n_g)]
            self._perm = rng.permutation(eff_D)
            self._R_dict = R_dict
            self._group_sizes_s = np.full(n_g, gs, dtype=int)
            self._weights = np.exp(rng.uniform(math.log(1e-3), math.log(1e3), n_g))
            self._n_groups = n_g

        else:
            raise RuntimeError(f"Unknown func_type {self.func_type!r}")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the benchmark at *x* (shape ``(D,)``)."""
        x = np.asarray(x, dtype=float).ravel()
        ft = self.func_type

        if ft in ("sep", "single"):
            return self._nonsep_fn(x - self._xopt)
        if ft == "partial":
            return self._eval_partial(x)
        if ft == "full":
            return self._eval_full(x)
        if ft == "conform":
            return self._eval_conform(x)
        if ft == "conflict":
            return self._eval_conflict(x)
        raise RuntimeError(f"Unknown func_type {ft!r}")

    def _eval_partial(self, x: np.ndarray) -> float:
        """F4-F7: k rotated groups + separable remainder."""
        z = x - self._xopt
        P = self._perm
        s = self._group_sizes_s
        result = 0.0
        c = 0
        for i in range(self._n_groups):
            si = int(s[i])
            grp_z = self._R_dict[si] @ z[P[c : c + si]]
            result += float(self._weights[i]) * self._nonsep_fn(grp_z)
            c += si
        if self._sep_D > 0:
            result += self._sep_fn(z[P[c : c + self._sep_D]])  # type: ignore[misc]
        return result

    def _eval_full(self, x: np.ndarray) -> float:
        """F8-F11: all dims covered by rotated groups."""
        z = x - self._xopt
        P = self._perm
        s = self._group_sizes_s
        result = 0.0
        c = 0
        for i in range(self._n_groups):
            si = int(s[i])
            grp_z = self._R_dict[si] @ z[P[c : c + si]]
            result += float(self._weights[i]) * self._nonsep_fn(grp_z)
            c += si
        return result

    def _eval_conform(self, x: np.ndarray) -> float:
        """F13: overlapping groups, single global shift (conform)."""
        xe = x[: self._eff_D]
        z = xe - self._xopt
        P = self._perm
        s = self._group_sizes_s
        ov = self.OVERLAP
        result = 0.0
        c = 0
        for i in range(self._n_groups):
            si = int(s[i])
            start = c - i * ov
            grp_z = self._R_dict[si] @ z[P[start : start + si]]
            result += float(self._weights[i]) * self._nonsep_fn(grp_z)
            c += si
        return result

    def _eval_conflict(self, x: np.ndarray) -> float:
        """F14: overlapping groups, per-group shift vectors (conflict)."""
        xe = x[: self._eff_D]
        P = self._perm
        s = self._group_sizes_s
        ov = self.OVERLAP
        result = 0.0
        c = 0
        for i in range(self._n_groups):
            si = int(s[i])
            start = c - i * ov
            grp_z = self._R_dict[si] @ (xe[P[start : start + si]] - self._xopt_groups[i])
            result += float(self._weights[i]) * self._nonsep_fn(grp_z)
            c += si
        return result

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def lb_array(self) -> np.ndarray:
        return np.full(self.D, self.lb)

    @property
    def ub_array(self) -> np.ndarray:
        return np.full(self.D, self.ub)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.lb_array, self.ub_array

    @property
    def n_groups(self) -> Optional[int]:
        return self._n_groups

    def __repr__(self) -> str:
        src = "cdatafiles" if self.using_cdatafiles else f"seed={self.seed}"
        info = f"func_id={self.func_id!r}, D={self.D}, source={src!r}"
        if self._n_groups is not None:
            info += f", n_groups={self._n_groups}"
        return f"LSGO2013({info})"
