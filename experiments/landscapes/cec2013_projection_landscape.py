r"""Approximate CEC-2013 LSGO landscapes via 2D projections (D=1000 -> 2D).

For each benchmark F1-F15:
  1. Sample points uniformly in the benchmark box [lb, ub]^D.
  2. Reduce to 2D with PCA or t-SNE (--projection).
  3. Build a grid in the 2D embedding and estimate f(x) on the grid.
  4. Plot (dim-1, dim-2, f(x)) as a 3D surface and a 2D heat map.

Outputs per function include one 2D heat map, three single views, and one combined
three panel figure in both PNG and PDF.
Log-scaled f(x) via matplotlib (set_zscale + LogNorm), Times font.

Run from repo root:
  python experiments/cec2013_pca_landscape.py --projection pca
  python experiments/cec2013_pca_landscape.py --projection tsne --functions f1,f5,f15
  python experiments/cec2013_pca_landscape.py --projection autoencoder --functions f1
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FixedLocator, FuncFormatter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - registers 3D projection

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from evo_subspace.problems.lsgo import LSGOProblem
from evo_subspace.problems.cec2013lsgo.cec2013lsgo.benchmarks import VALID_FUNC_IDS

# Fixed elevation keeps f(x) as the vertical axis; azimuth rotates around it.
_VIEW_ELEV = 25.0
_VIEW_AZIMS = (45.0, 135.0, 225.0)
_Z_LABEL = r"$f(x)$"
_POSITIVE_Z_FLOOR = 1e-30
PROJECTION_CHOICES = ("pca", "random_projection", "tsne", "autoencoder")
SCALE_CHOICES = ("auto", "linear", "log")
AE_LOG_FITNESS_VARIANT = "raw_mse_logfit"
AE_LOG_FITNESS_MOO_VARIANT = "raw_mse_logfit_moo"
AE_FITNESS_AWARE_VARIANTS = (AE_LOG_FITNESS_VARIANT, AE_LOG_FITNESS_MOO_VARIANT)
AE_VARIANT_CHOICES = (
    "raw_mse",
    AE_LOG_FITNESS_VARIANT,
    AE_LOG_FITNESS_MOO_VARIANT,
    "norm_sigmoid_bce",
    "norm_sigmoid_mse",
)
AE_VARIANT_CONFIG = {
    "raw_mse": {
        "normalize_inputs": False,
        "decoder_activation": "linear",
        "reconstruction_loss": "mse",
    },
    AE_LOG_FITNESS_VARIANT: {
        "normalize_inputs": False,
        "decoder_activation": "linear",
        "reconstruction_loss": "mse",
    },
    AE_LOG_FITNESS_MOO_VARIANT: {
        "normalize_inputs": False,
        "decoder_activation": "linear",
        "reconstruction_loss": "mse",
    },
    "norm_sigmoid_bce": {
        "normalize_inputs": True,
        "decoder_activation": "sigmoid",
        "reconstruction_loss": "bce",
    },
    "norm_sigmoid_mse": {
        "normalize_inputs": True,
        "decoder_activation": "sigmoid",
        "reconstruction_loss": "mse",
    },
}
FIGURES_ROOT = PROJECT_ROOT / "results" / "figures" / "lsgo"
DEFAULT_OUTPUT_ROOT = FIGURES_ROOT / "cec2013_projection_landscapes"
FIGURE_OUTPUT_DIRS = {
    "combined": "cec2013_projection_landscapes",
    "single": "cec2013_projection_landscapes_single_view",
    "heatmap": "cec2013_projection_landscapes_2d_heatmap",
}


def parse_functions(spec: str) -> list[str]:
    if spec.strip().lower() in ("all", "*"):
        return sorted(VALID_FUNC_IDS, key=lambda fid: int(fid.split("_f")[-1]))
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
        if fid not in VALID_FUNC_IDS:
            raise ValueError(f"Invalid CEC-2013 LSGO id {fid!r}")
        if fid not in out:
            out.append(fid)
    if not out:
        raise ValueError("At least one function is required")
    return out


def fit_pca2(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean, components[2,D], explained_variance_ratio[2])."""
    mean = samples.mean(axis=0)
    centered = samples - mean
    n = centered.shape[0]
    if n < 2:
        raise ValueError("need at least two samples for PCA")
    # Covariance eigendecomposition (efficient when n >> D).
    cov = (centered.T @ centered) / (n - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    components = eigvecs[:, order[:2]].T
    total = float(eigvals.sum())
    if total <= 0.0:
        evr = np.zeros(2, dtype=float)
    else:
        evr = eigvals[order[:2]] / total
    return mean, components, evr


def sample_uniform_in_bounds(
    n: int,
    lb: np.ndarray,
    ub: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw *n* IID points uniformly in the benchmark hyperrectangle.

    Each coordinate ``x_i`` is independent and uniform on ``[lb_i, ub_i]``,
    matching the CEC-2013 LSGO search domain for that function.
    """
    lb = np.asarray(lb, dtype=float).ravel()
    ub = np.asarray(ub, dtype=float).ravel()
    if lb.size != ub.size:
        raise ValueError("lb and ub must have the same dimension")
    if np.any(ub <= lb):
        raise ValueError("each upper bound must be strictly greater than the lower bound")
    u = rng.random((n, lb.size))
    return lb + u * (ub - lb)


def embedding_grid_bounds(embedding: np.ndarray, margin: float) -> tuple[np.ndarray, np.ndarray]:
    lo = embedding.min(axis=0)
    hi = embedding.max(axis=0)
    span = np.maximum(hi - lo, 1e-12)
    pad = margin * span
    return lo - pad, hi + pad


def pca_grid_bounds(
    samples: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
    margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    proj = (samples - mean) @ components.T
    return embedding_grid_bounds(proj, margin)


def fit_tsne2(
    samples: np.ndarray,
    *,
    seed: int,
    perplexity: float,
) -> np.ndarray:
    """Embed samples in R^D to 2D with t-SNE."""
    try:
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise ImportError(
            "t-SNE requires scikit-learn. Install with: pip install scikit-learn"
        ) from exc

    n = samples.shape[0]
    if perplexity >= n:
        raise ValueError(f"--tsne_perplexity ({perplexity}) must be < n_samples ({n})")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        init="pca",
        learning_rate="auto",
        max_iter=10000,
        verbose=1,
    )
    return tsne.fit_transform(samples)


def evaluate_grid_pca(
    problem: LSGOProblem,
    mean: np.ndarray,
    components: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
) -> np.ndarray:
    """Evaluate f on a PCA-plane grid; clip to the problem box."""
    flat_p = np.column_stack([p1.ravel(), p2.ravel()])
    x = mean + flat_p @ components
    x = np.clip(x, problem.lb, problem.ub)
    z = np.empty(flat_p.shape[0], dtype=float)
    for i in range(flat_p.shape[0]):
        z[i] = problem.evaluate(x[i])
    return z.reshape(p1.shape)


def sample_orthonormal_random_projection(
    dimension: int,
    target_dim: int,
    *,
    seed: int,
) -> np.ndarray:
    """Return P in R^{target_dim x dimension} with orthonormal rows."""
    if target_dim < 1:
        raise ValueError("target_dim must be >= 1")
    if target_dim > dimension:
        raise ValueError("target_dim must be <= dimension")
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(dimension, target_dim))
    q, _ = np.linalg.qr(raw, mode="reduced")
    return q.T


def fit_random_projection2(samples: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(samples, axis=0)
    projection = sample_orthonormal_random_projection(samples.shape[1], 2, seed=seed)
    embedding = (samples - mean) @ projection.T
    return mean, projection, embedding


def evaluate_grid_random_projection(
    problem: LSGOProblem,
    mean: np.ndarray,
    projection: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
) -> np.ndarray:
    """Evaluate f on an orthonormal random projection grid."""
    flat_p = np.column_stack([p1.ravel(), p2.ravel()])
    x = mean + flat_p @ np.linalg.pinv(projection.T)
    x = np.clip(x, problem.lb, problem.ub)
    z = np.empty(flat_p.shape[0], dtype=float)
    for i in range(flat_p.shape[0]):
        z[i] = problem.evaluate(x[i])
    return z.reshape(p1.shape)


def evaluate_grid_tsne(
    problem: LSGOProblem,
    samples: np.ndarray,
    embedding: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    *,
    k_neighbors: int,
) -> np.ndarray:
    """Evaluate f on a t-SNE grid via inverse-distance k-NN lift to R^D."""
    from scipy.spatial import cKDTree

    flat_p = np.column_stack([p1.ravel(), p2.ravel()])
    tree = cKDTree(embedding)
    k = min(k_neighbors, embedding.shape[0])
    z = np.empty(flat_p.shape[0], dtype=float)

    for i, point in enumerate(flat_p):
        dists, idx = tree.query(point, k=k)
        idx = np.atleast_1d(idx)
        dists = np.atleast_1d(dists).astype(float)
        weights = 1.0 / (dists + 1e-12)
        weights /= weights.sum()
        x = np.clip(weights @ samples[idx], problem.lb, problem.ub)
        z[i] = problem.evaluate(x)
    return z.reshape(p1.shape)


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError(
            "Autoencoder projection requires PyTorch. Install with: pip install torch"
        ) from exc
    return torch, nn


def default_ae_hidden_dims(d: int) -> tuple[int, int]:
    return min(512, d), min(128, max(32, d // 8))


@dataclass
class TrainedAutoencoder:
    """MLP autoencoder trained on uniform benchmark samples."""

    lb: np.ndarray
    ub: np.ndarray
    model: object
    device: object
    normalize_inputs: bool = True

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        if not self.normalize_inputs:
            return np.asarray(x, dtype=np.float32)
        span = np.maximum(self.ub - self.lb, 1e-12)
        return (x - self.lb) / span

    def _denormalize(self, x_norm: np.ndarray) -> np.ndarray:
        if not self.normalize_inputs:
            return np.asarray(x_norm, dtype=np.float32)
        span = self.ub - self.lb
        return self.lb + x_norm * span

    def encode(self, x: np.ndarray) -> np.ndarray:
        torch, _ = _require_torch()
        self.model.eval()
        with torch.no_grad():
            x_norm = self._normalize(np.asarray(x, dtype=np.float32))
            t = torch.as_tensor(x_norm, dtype=torch.float32, device=self.device)
            z = self.model.encode(t).cpu().numpy()
        return z

    def decode(self, z: np.ndarray) -> np.ndarray:
        torch, _ = _require_torch()
        self.model.eval()
        with torch.no_grad():
            t = torch.as_tensor(np.asarray(z, dtype=np.float32), device=self.device)
            x_norm = self.model.decode(t).cpu().numpy()
        return self._denormalize(x_norm)


def _build_autoencoder_model(
    d: int,
    hidden1: int,
    hidden2: int,
    *,
    decoder_activation: str,
):
    torch, nn = _require_torch()
    if decoder_activation not in {"sigmoid", "linear"}:
        raise ValueError("decoder_activation must be 'sigmoid' or 'linear'")

    class MLPAutoencoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(d, hidden1),
                nn.ReLU(),
                nn.Linear(hidden1, hidden2),
                nn.ReLU(),
                nn.Linear(hidden2, 2),
            )
            decoder_layers = [
                nn.Linear(2, hidden2),
                nn.ReLU(),
                nn.Linear(hidden2, hidden1),
                nn.ReLU(),
                nn.Linear(hidden1, d),
            ]
            if decoder_activation == "sigmoid":
                decoder_layers.append(nn.Sigmoid())
            self.decoder = nn.Sequential(*decoder_layers)

        def encode(self, x):
            return self.encoder(x)

        def decode(self, z):
            return self.decoder(z)

        def forward(self, x):
            return self.decode(self.encode(x))

    return MLPAutoencoder()


def fit_autoencoder2(
    samples: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    *,
    seed: int,
    hidden1: int,
    hidden2: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    normalize_inputs: bool = True,
    decoder_activation: str = "sigmoid",
    reconstruction_loss: str = "mse",
) -> TrainedAutoencoder:
    """Train a 2D-latent MLP autoencoder on samples in [lb, ub]^D."""
    torch, nn = _require_torch()
    if decoder_activation not in {"sigmoid", "linear"}:
        raise ValueError("decoder_activation must be 'sigmoid' or 'linear'")
    if reconstruction_loss not in {"mse", "bce"}:
        raise ValueError("reconstruction_loss must be 'mse' or 'bce'")
    if reconstruction_loss == "bce" and (not normalize_inputs or decoder_activation != "sigmoid"):
        raise ValueError("BCE reconstruction requires normalized inputs and sigmoid decoder output")

    lb = np.asarray(lb, dtype=np.float32).ravel()
    ub = np.asarray(ub, dtype=np.float32).ravel()
    span = np.maximum(ub - lb, 1e-12)
    if normalize_inputs:
        x_train = ((samples - lb) / span).astype(np.float32)
    else:
        x_train = np.asarray(samples, dtype=np.float32)
    d = x_train.shape[1]

    torch_device = torch.device(device)
    torch.manual_seed(seed)
    model = _build_autoencoder_model(
        d,
        hidden1,
        hidden2,
        decoder_activation=decoder_activation,
    ).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss() if reconstruction_loss == "mse" else nn.BCELoss()

    n = x_train.shape[0]
    batch_size = min(batch_size, n)
    rng = np.random.default_rng(seed)
    loss_label = reconstruction_loss.upper()

    def eval_loss() -> float:
        model.eval()
        total = 0.0
        n_batches = 0
        with torch.no_grad():
            for start in range(0, n, batch_size):
                batch = torch.as_tensor(x_train[start : start + batch_size], device=torch_device)
                recon = model(batch)
                total += float(loss_fn(recon, batch).item())
                n_batches += 1
        return total / max(n_batches, 1)

    print(f"      AE epoch 0/{epochs}, {loss_label}={eval_loss():.6e}", flush=True)

    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(n)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            batch = torch.as_tensor(x_train[idx], device=torch_device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        if epoch == epochs - 1 or (epoch + 1) % max(epochs // 5, 1) == 0:
            mean_loss = epoch_loss / max(n_batches, 1)
            print(f"      AE epoch {epoch + 1}/{epochs}, {loss_label}={mean_loss:.6e}", flush=True)

    return TrainedAutoencoder(
        lb=lb,
        ub=ub,
        model=model,
        device=torch_device,
        normalize_inputs=normalize_inputs,
    )


def autoencoder_checkpoint_path(output_base: Path) -> Path:
    return output_base.with_name(f"{output_base.name}_autoencoder.pt")


def save_autoencoder_checkpoint(
    autoencoder: TrainedAutoencoder,
    output_base: Path,
    *,
    problem: LSGOProblem,
    args: argparse.Namespace,
    hidden1: int,
    hidden2: int,
) -> Path:
    torch, _ = _require_torch()
    ae_config = AE_VARIANT_CONFIG[args.ae_variant]
    path = autoencoder_checkpoint_path(output_base)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": autoencoder.model.state_dict(),
        "metadata": {
            "variant": args.ae_variant,
            "func_id": problem.func_id,
            "dimension": problem.D,
            "hidden1": hidden1,
            "hidden2": hidden2,
            "normalize_inputs": bool(ae_config["normalize_inputs"]),
            "decoder_activation": str(ae_config["decoder_activation"]),
            "reconstruction_loss": str(ae_config["reconstruction_loss"]),
            "benchmark_seed": args.benchmark_seed,
            "group_size": args.group_size,
            "sample_seed": args.sample_seed,
            "n_samples": args.n_samples,
            "ae_seed": args.ae_seed,
            "ae_epochs": args.ae_epochs,
            "ae_batch_size": args.ae_batch_size,
            "ae_learning_rate": args.ae_learning_rate,
            "ae_fitness_lambda": args.ae_fitness_lambda,
            "lb": np.asarray(problem.lb, dtype=float).tolist(),
            "ub": np.asarray(problem.ub, dtype=float).tolist(),
        },
    }
    torch.save(checkpoint, path)
    return path


def load_autoencoder_checkpoint(path: Path, *, device: str = "cpu") -> TrainedAutoencoder:
    torch, _ = _require_torch()
    torch_device = torch.device(device)
    try:
        checkpoint = torch.load(path, map_location=torch_device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=torch_device)
    metadata = checkpoint["metadata"]
    model = _build_autoencoder_model(
        int(metadata["dimension"]),
        int(metadata["hidden1"]),
        int(metadata["hidden2"]),
        decoder_activation=str(metadata["decoder_activation"]),
    ).to(torch_device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return TrainedAutoencoder(
        lb=np.asarray(metadata["lb"], dtype=np.float32),
        ub=np.asarray(metadata["ub"], dtype=np.float32),
        model=model,
        device=torch_device,
        normalize_inputs=bool(metadata["normalize_inputs"]),
    )


_CEC_NONSEP_BASE = {
    "cec2013_lsgo_f1": "elliptic",
    "cec2013_lsgo_f2": "rastrigin",
    "cec2013_lsgo_f3": "ackley",
    "cec2013_lsgo_f4": "elliptic",
    "cec2013_lsgo_f5": "rastrigin",
    "cec2013_lsgo_f6": "ackley",
    "cec2013_lsgo_f7": "schwefel",
    "cec2013_lsgo_f8": "elliptic",
    "cec2013_lsgo_f9": "rastrigin",
    "cec2013_lsgo_f10": "ackley",
    "cec2013_lsgo_f11": "schwefel",
    "cec2013_lsgo_f12": "rosenbrock",
    "cec2013_lsgo_f13": "schwefel",
    "cec2013_lsgo_f14": "schwefel",
    "cec2013_lsgo_f15": "schwefel",
}
_CEC_SEP_BASE = {
    "cec2013_lsgo_f4": "elliptic",
    "cec2013_lsgo_f5": "rastrigin",
    "cec2013_lsgo_f6": "ackley",
    "cec2013_lsgo_f7": "sphere",
}


def _torch_t_osz(z):
    torch, _ = _require_torch()
    nonzero = z != 0.0
    safe_abs = torch.where(nonzero, torch.abs(z), torch.ones_like(z))
    h = torch.where(nonzero, torch.log(safe_abs), torch.zeros_like(z))
    c1 = torch.where(z > 0.0, torch.full_like(z, 10.0), torch.full_like(z, 5.5))
    c2 = torch.where(z > 0.0, torch.full_like(z, 7.9), torch.full_like(z, 3.1))
    osz = torch.sign(z) * torch.exp(h + 0.049 * (torch.sin(c1 * h) + torch.sin(c2 * h)))
    return torch.where(nonzero, osz, torch.zeros_like(z))


def _torch_t_asy(z, beta: float):
    torch, _ = _require_torch()
    dim = z.shape[1]
    if dim <= 1:
        return z
    idx = torch.arange(dim, dtype=z.dtype, device=z.device)
    exponent = 1.0 + beta * idx / (dim - 1) * torch.sqrt(torch.clamp(z, min=0.0))
    return torch.where(z > 0.0, torch.pow(z, exponent), z)


def _torch_t_lambda(z, alpha: float):
    torch, _ = _require_torch()
    dim = z.shape[1]
    if dim <= 1:
        return z
    idx = torch.arange(dim, dtype=z.dtype, device=z.device)
    scale = torch.pow(torch.as_tensor(float(alpha), dtype=z.dtype, device=z.device), 0.5 * idx / (dim - 1))
    return z * scale


def _torch_cec_base(name: str, z):
    torch, _ = _require_torch()
    if name == "elliptic":
        z = _torch_t_osz(z)
        dim = z.shape[1]
        if dim == 1:
            return 1e6 * z[:, 0] ** 2
        idx = torch.arange(dim, dtype=z.dtype, device=z.device)
        weights = torch.pow(torch.as_tensor(1e6, dtype=z.dtype, device=z.device), idx / (dim - 1))
        return torch.sum(weights * z**2, dim=1)
    if name == "rastrigin":
        z = _torch_t_lambda(_torch_t_asy(_torch_t_osz(z), 0.2), 10.0)
        return torch.sum(z**2 - 10.0 * torch.cos(2.0 * np.pi * z) + 10.0, dim=1)
    if name == "ackley":
        z = _torch_t_osz(z)
        dim = z.shape[1]
        s1 = torch.sum(z**2, dim=1)
        s2 = torch.sum(torch.cos(2.0 * np.pi * z), dim=1)
        return -20.0 * torch.exp(-0.2 * torch.sqrt(s1 / dim)) - torch.exp(s2 / dim) + 20.0 + np.e
    if name == "schwefel":
        z = _torch_t_asy(_torch_t_osz(z), 0.2)
        return torch.sum(torch.cumsum(z, dim=1) ** 2, dim=1)
    if name == "sphere":
        return torch.sum(z**2, dim=1)
    if name == "rosenbrock":
        if z.shape[1] < 2:
            return torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        return torch.sum(100.0 * (z[:, 1:] - z[:, :-1] ** 2) ** 2 + (z[:, :-1] - 1.0) ** 2, dim=1)
    raise ValueError(f"unknown CEC base function {name!r}")


def _torch_array(value, *, dtype, device):
    torch, _ = _require_torch()
    return torch.as_tensor(value, dtype=dtype, device=device)


def _torch_cec_evaluate(problem: LSGOProblem, x):
    """Differentiable CEC-2013 LSGO evaluation for AE fitness preservation."""
    torch, _ = _require_torch()
    func = problem._func
    dtype = x.dtype
    device = x.device
    ft = func.func_type
    nonsep_name = _CEC_NONSEP_BASE[func.func_id]

    if ft in {"sep", "single"}:
        xopt = _torch_array(func._xopt, dtype=dtype, device=device)
        return _torch_cec_base(nonsep_name, x - xopt)

    perm = np.asarray(func._perm, dtype=int)
    sizes = np.asarray(func._group_sizes_s, dtype=int)
    weights = _torch_array(func._weights, dtype=dtype, device=device)

    if ft == "partial":
        xopt = _torch_array(func._xopt, dtype=dtype, device=device)
        z = x - xopt
        result = torch.zeros(x.shape[0], dtype=dtype, device=device)
        c = 0
        for i, si_raw in enumerate(sizes):
            si = int(si_raw)
            rotation = _torch_array(func._R_dict[si], dtype=dtype, device=device)
            grp_z = z[:, perm[c : c + si]] @ rotation.T
            result = result + weights[i] * _torch_cec_base(nonsep_name, grp_z)
            c += si
        if func._sep_D > 0:
            sep_name = _CEC_SEP_BASE[func.func_id]
            result = result + _torch_cec_base(sep_name, z[:, perm[c : c + func._sep_D]])
        return result

    if ft == "full":
        xopt = _torch_array(func._xopt, dtype=dtype, device=device)
        z = x - xopt
        result = torch.zeros(x.shape[0], dtype=dtype, device=device)
        c = 0
        for i, si_raw in enumerate(sizes):
            si = int(si_raw)
            rotation = _torch_array(func._R_dict[si], dtype=dtype, device=device)
            grp_z = z[:, perm[c : c + si]] @ rotation.T
            result = result + weights[i] * _torch_cec_base(nonsep_name, grp_z)
            c += si
        return result

    if ft == "conform":
        xe = x[:, : func._eff_D]
        xopt = _torch_array(func._xopt, dtype=dtype, device=device)
        z = xe - xopt
        result = torch.zeros(x.shape[0], dtype=dtype, device=device)
        c = 0
        for i, si_raw in enumerate(sizes):
            si = int(si_raw)
            start = c - i * func.OVERLAP
            rotation = _torch_array(func._R_dict[si], dtype=dtype, device=device)
            grp_z = z[:, perm[start : start + si]] @ rotation.T
            result = result + weights[i] * _torch_cec_base(nonsep_name, grp_z)
            c += si
        return result

    if ft == "conflict":
        xe = x[:, : func._eff_D]
        result = torch.zeros(x.shape[0], dtype=dtype, device=device)
        c = 0
        for i, si_raw in enumerate(sizes):
            si = int(si_raw)
            start = c - i * func.OVERLAP
            rotation = _torch_array(func._R_dict[si], dtype=dtype, device=device)
            xopt_group = _torch_array(func._xopt_groups[i], dtype=dtype, device=device)
            grp_z = (xe[:, perm[start : start + si]] - xopt_group) @ rotation.T
            result = result + weights[i] * _torch_cec_base(nonsep_name, grp_z)
            c += si
        return result

    raise RuntimeError(f"unknown CEC function type {ft!r}")


def _log_fitness_stats(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    shift = 0.0
    min_value = float(np.min(values))
    if min_value <= 0.0:
        shift = min_value - 1.0
    log_values = np.log(np.maximum(values - shift, np.finfo(float).tiny))
    return shift, float(np.mean(log_values)), float(np.std(log_values) + 1e-12)


def _evaluate_batch_numpy(problem: LSGOProblem, x: np.ndarray) -> np.ndarray:
    values = np.empty(x.shape[0], dtype=float)
    for i, row in enumerate(x):
        values[i] = problem.evaluate(row)
    return values


def fit_autoencoder2_with_log_fitness(
    problem: LSGOProblem,
    samples: np.ndarray,
    *,
    seed: int,
    hidden1: int,
    hidden2: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    fitness_lambda: float,
) -> TrainedAutoencoder:
    """Train a raw MSE AE with a standardized log fitness preservation term."""
    torch, nn = _require_torch()
    lb = np.asarray(problem.lb, dtype=np.float32).ravel()
    ub = np.asarray(problem.ub, dtype=np.float32).ravel()
    x_train = np.asarray(samples, dtype=np.float32)
    d = x_train.shape[1]
    target_fitness = _evaluate_batch_numpy(problem, samples)
    fitness_shift, fitness_mean, fitness_std = _log_fitness_stats(target_fitness)

    torch_device = torch.device(device)
    torch.manual_seed(seed)
    model = _build_autoencoder_model(
        d,
        hidden1,
        hidden2,
        decoder_activation="linear",
    ).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rec_loss_fn = nn.MSELoss()

    n = x_train.shape[0]
    batch_size = min(batch_size, n)
    rng = np.random.default_rng(seed)
    target_log = np.log(np.maximum(target_fitness - fitness_shift, np.finfo(float).tiny))
    target_log = ((target_log - fitness_mean) / fitness_std).astype(np.float32)
    lb_t = torch.as_tensor(lb, dtype=torch.float32, device=torch_device)
    ub_t = torch.as_tensor(ub, dtype=torch.float32, device=torch_device)

    def transformed_fitness_loss(recon, target):
        recon_clipped = torch.clamp(recon, min=lb_t, max=ub_t)
        recon_fitness = _torch_cec_evaluate(problem, recon_clipped)
        log_fitness = torch.log(
            torch.clamp(recon_fitness - fitness_shift, min=torch.finfo(recon.dtype).tiny)
        )
        scaled = (log_fitness - fitness_mean) / fitness_std
        return rec_loss_fn(scaled, target)

    def eval_losses() -> tuple[float, float, float]:
        model.eval()
        rec_total = 0.0
        fit_total = 0.0
        n_batches = 0
        with torch.no_grad():
            for start in range(0, n, batch_size):
                batch = torch.as_tensor(x_train[start : start + batch_size], device=torch_device)
                target = torch.as_tensor(target_log[start : start + batch_size], device=torch_device)
                recon = model(batch)
                rec = rec_loss_fn(recon, batch)
                fit = transformed_fitness_loss(recon, target)
                rec_total += float(rec.item())
                fit_total += float(fit.item())
                n_batches += 1
        rec_mean = rec_total / max(n_batches, 1)
        fit_mean = fit_total / max(n_batches, 1)
        return rec_mean, fit_mean, rec_mean + fitness_lambda * fit_mean

    rec0, fit0, total0 = eval_losses()
    print(
        f"      AE epoch 0/{epochs}, MSE={rec0:.6e}, logfit={fit0:.6e}, total={total0:.6e}",
        flush=True,
    )

    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(n)
        rec_total = 0.0
        fit_total = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            batch = torch.as_tensor(x_train[idx], device=torch_device)
            target = torch.as_tensor(target_log[idx], device=torch_device)
            optimizer.zero_grad()
            recon = model(batch)
            rec_loss = rec_loss_fn(recon, batch)
            fit_loss = transformed_fitness_loss(recon, target)
            loss = rec_loss + fitness_lambda * fit_loss
            loss.backward()
            optimizer.step()
            rec_total += float(rec_loss.item())
            fit_total += float(fit_loss.item())
            n_batches += 1
        if epoch == epochs - 1 or (epoch + 1) % max(epochs // 5, 1) == 0:
            rec_mean = rec_total / max(n_batches, 1)
            fit_mean = fit_total / max(n_batches, 1)
            total_mean = rec_mean + fitness_lambda * fit_mean
            print(
                f"      AE epoch {epoch + 1}/{epochs}, MSE={rec_mean:.6e}, "
                f"logfit={fit_mean:.6e}, total={total_mean:.6e}",
                flush=True,
            )

    return TrainedAutoencoder(
        lb=lb,
        ub=ub,
        model=model,
        device=torch_device,
        normalize_inputs=False,
    )


def fit_autoencoder2_with_log_fitness_moo(
    problem: LSGOProblem,
    samples: np.ndarray,
    *,
    seed: int,
    hidden1: int,
    hidden2: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
) -> TrainedAutoencoder:
    """Train a raw MSE AE by aggregating MSE and log fitness objectives with UPGrad."""
    torch, nn = _require_torch()
    try:
        from torchjd.autojac import jac_to_grad, mtl_backward
        from torchjd.aggregation import UPGrad
    except ImportError as exc:
        raise ImportError(
            "raw_mse_logfit_moo requires TorchJD. Install with "
            'pip install "torchjd[quadprog_projector]"'
        ) from exc

    lb = np.asarray(problem.lb, dtype=np.float32).ravel()
    ub = np.asarray(problem.ub, dtype=np.float32).ravel()
    x_train = np.asarray(samples, dtype=np.float32)
    d = x_train.shape[1]
    target_fitness = _evaluate_batch_numpy(problem, samples)
    fitness_shift, fitness_mean, fitness_std = _log_fitness_stats(target_fitness)

    torch_device = torch.device(device)
    torch.manual_seed(seed)
    model = _build_autoencoder_model(
        d,
        hidden1,
        hidden2,
        decoder_activation="linear",
    ).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    aggregator = UPGrad()
    shared_params = list(model.parameters())
    rec_loss_fn = nn.MSELoss()

    n = x_train.shape[0]
    batch_size = min(batch_size, n)
    rng = np.random.default_rng(seed)
    target_log = np.log(np.maximum(target_fitness - fitness_shift, np.finfo(float).tiny))
    target_log = ((target_log - fitness_mean) / fitness_std).astype(np.float32)
    lb_t = torch.as_tensor(lb, dtype=torch.float32, device=torch_device)
    ub_t = torch.as_tensor(ub, dtype=torch.float32, device=torch_device)

    def transformed_fitness_loss(recon, target):
        recon_clipped = torch.clamp(recon, min=lb_t, max=ub_t)
        recon_fitness = _torch_cec_evaluate(problem, recon_clipped)
        log_fitness = torch.log(
            torch.clamp(recon_fitness - fitness_shift, min=torch.finfo(recon.dtype).tiny)
        )
        scaled = (log_fitness - fitness_mean) / fitness_std
        return rec_loss_fn(scaled, target)

    def eval_losses() -> tuple[float, float]:
        model.eval()
        rec_total = 0.0
        fit_total = 0.0
        n_batches = 0
        with torch.no_grad():
            for start in range(0, n, batch_size):
                batch = torch.as_tensor(x_train[start : start + batch_size], device=torch_device)
                target = torch.as_tensor(target_log[start : start + batch_size], device=torch_device)
                recon = model(batch)
                rec = rec_loss_fn(recon, batch)
                fit = transformed_fitness_loss(recon, target)
                rec_total += float(rec.item())
                fit_total += float(fit.item())
                n_batches += 1
        return rec_total / max(n_batches, 1), fit_total / max(n_batches, 1)

    rec0, fit0 = eval_losses()
    print(
        f"      AE epoch 0/{epochs}, MSE={rec0:.6e}, logfit={fit0:.6e}, aggregator=UPGrad",
        flush=True,
    )

    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(n)
        rec_total = 0.0
        fit_total = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            batch = torch.as_tensor(x_train[idx], device=torch_device)
            target = torch.as_tensor(target_log[idx], device=torch_device)
            optimizer.zero_grad()
            recon = model(batch)
            rec_loss = rec_loss_fn(recon, batch)
            fit_loss = transformed_fitness_loss(recon, target)
            mtl_backward([rec_loss, fit_loss], features=recon)
            jac_to_grad(shared_params, aggregator)
            optimizer.step()
            rec_total += float(rec_loss.item())
            fit_total += float(fit_loss.item())
            n_batches += 1
        if epoch == epochs - 1 or (epoch + 1) % max(epochs // 5, 1) == 0:
            rec_mean = rec_total / max(n_batches, 1)
            fit_mean = fit_total / max(n_batches, 1)
            print(
                f"      AE epoch {epoch + 1}/{epochs}, MSE={rec_mean:.6e}, "
                f"logfit={fit_mean:.6e}, aggregator=UPGrad",
                flush=True,
            )

    return TrainedAutoencoder(
        lb=lb,
        ub=ub,
        model=model,
        device=torch_device,
        normalize_inputs=False,
    )


def evaluate_grid_autoencoder(
    problem: LSGOProblem,
    autoencoder: TrainedAutoencoder,
    p1: np.ndarray,
    p2: np.ndarray,
    *,
    decode_batch_size: int,
) -> np.ndarray:
    """Decode latent grid points and evaluate f(x)."""
    flat_z = np.column_stack([p1.ravel(), p2.ravel()])
    z = np.empty(flat_z.shape[0], dtype=float)
    for start in range(0, flat_z.shape[0], decode_batch_size):
        batch_z = flat_z[start : start + decode_batch_size]
        x = autoencoder.decode(batch_z)
        x = np.clip(x, problem.lb, problem.ub)
        for i, row in enumerate(x):
            z[start + i] = problem.evaluate(row)
    return z.reshape(p1.shape)


def configure_matplotlib() -> None:
    """Use Times (New Roman) for all plot text."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def _clip_z(z: np.ndarray, z_clip_percentile: float | None) -> np.ndarray:
    z_plot = z.astype(float, copy=True)
    if z_clip_percentile is None:
        return z_plot
    cap = float(np.percentile(z_plot, z_clip_percentile))
    if np.any(z_plot > cap):
        z_plot = np.minimum(z_plot, cap)
    return z_plot


def ensure_positive_for_log_scale(z: np.ndarray) -> np.ndarray:
    """Shift f(x) only when needed so matplotlib log scaling is valid."""
    z_pos = z.astype(float, copy=True)
    z_min = float(z_pos.min())
    if z_min <= 0.0:
        z_pos = z_pos - z_min + _POSITIVE_Z_FLOOR
    return np.maximum(z_pos, _POSITIVE_Z_FLOOR)


def prepare_surface(
    z: np.ndarray,
    z_clip_percentile: float | None,
) -> tuple[np.ndarray, LogNorm]:
    z_plot = ensure_positive_for_log_scale(_clip_z(z, z_clip_percentile))
    vmin = float(z_plot.min())
    vmax = float(z_plot.max())
    if vmax <= vmin:
        vmax = vmin * 1.001
    return z_plot, LogNorm(vmin=vmin, vmax=vmax)


def _log_contour_levels(z_plot: np.ndarray, n_levels: int = 12) -> np.ndarray:
    finite = z_plot[np.isfinite(z_plot)]
    if finite.size == 0:
        return np.array([], dtype=float)
    vmin = float(finite.min())
    vmax = float(finite.max())
    if vmax <= vmin:
        return np.array([], dtype=float)
    return np.geomspace(vmin, vmax, num=n_levels + 2)[1:-1]


def _style_3d_axes(
    ax,
    *,
    x_label: str,
    y_label: str,
    z_tick_pad: float,
    z_labelpad: float,
) -> None:
    """Pad tick labels and axis titles away from the 3D box."""
    ax.set_xlabel(x_label, labelpad=6)
    ax.set_ylabel(y_label, labelpad=6)
    ax.set_zlabel(_Z_LABEL, labelpad=z_labelpad)
    ax.tick_params(axis="x", pad=4)
    ax.tick_params(axis="y", pad=4)
    ax.tick_params(axis="z", pad=z_tick_pad)
    # 3D tick pad is more reliable when set on each tick artist.
    for tick in ax.zaxis.get_major_ticks():
        tick.set_pad(z_tick_pad)


def _plot_surface(
    ax,
    p1: np.ndarray,
    p2: np.ndarray,
    z_plot: np.ndarray,
    *,
    norm: LogNorm,
    azim: float,
    x_label: str,
    y_label: str,
    z_tick_pad: float = 8.0,
    z_labelpad: float = 12.0,
):
    surf = ax.plot_surface(
        p1,
        p2,
        z_plot,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
        alpha=0.92,
        norm=norm,
    )
    _style_3d_axes(
        ax,
        x_label=x_label,
        y_label=y_label,
        z_tick_pad=z_tick_pad,
        z_labelpad=z_labelpad,
    )
    ax.set_zscale("log")
    ax.view_init(elev=_VIEW_ELEV, azim=azim)
    return surf


# Dedicated colorbar axes (figure coords) — keeps a clear gap from the 3D plots
# and margin from the right figure edge.
_CBAR_WIDTH = 0.022
_CBAR_BOTTOM = 0.14
_CBAR_HEIGHT = 0.72
_SAVE_PAD_INCHES = 0.18


def _add_colorbar_right(fig: plt.Figure, mappable, *, cbar_left: float) -> None:
    """Place colorbar in a fixed right slot with padding from plot and figure edge."""
    cax = fig.add_axes(
        [cbar_left, _CBAR_BOTTOM, _CBAR_WIDTH, _CBAR_HEIGHT],
        label="colorbar_axes",
    )
    cbar = fig.colorbar(mappable, cax=cax)
    # Label above the bar so it does not extend past the right figure edge.
    cbar.ax.set_title(_Z_LABEL, fontsize=11, pad=6)
    cbar.ax.tick_params(labelsize=9)


def _save_figure(fig: plt.Figure, output_base: Path, dpi: int) -> tuple[Path, Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_base.with_suffix(".pdf")
    png_path = output_base.with_suffix(".png")
    fig.savefig(pdf_path, dpi=dpi, pad_inches=_SAVE_PAD_INCHES)
    fig.savefig(png_path, dpi=dpi, pad_inches=_SAVE_PAD_INCHES)
    plt.close(fig)
    return pdf_path, png_path


def save_single_view(
    *,
    p1: np.ndarray,
    p2: np.ndarray,
    z_plot: np.ndarray,
    norm: LogNorm,
    azim: float,
    x_label: str,
    y_label: str,
    view_index: int,
    output_base: Path,
    dpi: int,
) -> tuple[Path, Path]:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    mappable = _plot_surface(
        ax,
        p1,
        p2,
        z_plot,
        norm=norm,
        azim=azim,
        x_label=x_label,
        y_label=y_label,
        z_tick_pad=14.0,
        z_labelpad=18.0,
    )
    # Single-view only: extra gap before colorbar; bar nudged slightly right.
    fig.subplots_adjust(left=0.10, right=0.66, bottom=0.10, top=0.94)
    _add_colorbar_right(fig, mappable, cbar_left=0.87)
    stem = f"{output_base.name}_view{view_index}"
    return _save_figure(fig, output_base.parent / stem, dpi)


def save_combined_views(
    *,
    p1: np.ndarray,
    p2: np.ndarray,
    z_plot: np.ndarray,
    norm: LogNorm,
    x_label: str,
    y_label: str,
    output_base: Path,
    dpi: int,
) -> tuple[Path, Path]:
    """Save one figure with three views (fixed elev, rotated azimuth around z)."""
    fig = plt.figure(figsize=(22, 5.5))
    gs = GridSpec(
        1,
        3,
        figure=fig,
        left=0.04,
        right=0.72,
        bottom=0.10,
        top=0.94,
        wspace=0.38,
    )
    mappable = None
    for idx, azim in enumerate(_VIEW_AZIMS):
        ax = fig.add_subplot(gs[0, idx], projection="3d")
        mappable = _plot_surface(
            ax,
            p1,
            p2,
            z_plot,
            norm=norm,
            azim=azim,
            x_label=x_label,
            y_label=y_label,
            z_tick_pad=6.0,
            z_labelpad=10.0,
        )
    if mappable is not None:
        _add_colorbar_right(fig, mappable, cbar_left=0.88)
    return _save_figure(fig, output_base, dpi)


def save_2d_heatmap(
    *,
    p1: np.ndarray,
    p2: np.ndarray,
    z_plot: np.ndarray,
    norm: LogNorm,
    x_label: str,
    y_label: str,
    output_base: Path,
    dpi: int,
) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    heatmap = ax.pcolormesh(
        p1,
        p2,
        z_plot,
        cmap="viridis",
        norm=norm,
        shading="auto",
    )
    levels = _log_contour_levels(z_plot)
    if levels.size > 0:
        contours = ax.contour(
            p1,
            p2,
            z_plot,
            levels=levels,
            colors="white",
            linewidths=0.6,
            alpha=0.85,
        )
        ax.clabel(contours, inline=True, fontsize=7, fmt="%.1e")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(float(p1.min()), float(p1.max()))
    ax.set_ylim(float(p2.min()), float(p2.max()))
    cbar = fig.colorbar(heatmap, ax=ax, pad=0.03)
    cbar.ax.set_title(_Z_LABEL, fontsize=11, pad=6)
    cbar.ax.tick_params(labelsize=9)
    fig.tight_layout()
    stem = f"{output_base.name}_2d_heatmap"
    return _save_figure(fig, output_base.parent / stem, dpi)


def save_all_landscapes(
    *,
    p1: np.ndarray,
    p2: np.ndarray,
    z: np.ndarray,
    x_label: str,
    y_label: str,
    combined_output_base: Path,
    single_output_base: Path,
    heatmap_output_base: Path,
    dpi: int,
    z_clip_percentile: float | None,
    save_kind: str = "all",
    single_view_index: int = 1,
) -> list[Path]:
    z_plot, norm = prepare_surface(z, z_clip_percentile)
    paths: list[Path] = []

    if save_kind not in {"all", "single"}:
        raise ValueError(f"unknown save_kind {save_kind!r}")
    if not 1 <= single_view_index <= len(_VIEW_AZIMS):
        raise ValueError(f"single_view_index must be in [1, {len(_VIEW_AZIMS)}]")

    if save_kind == "all":
        heatmap = save_2d_heatmap(
            p1=p1,
            p2=p2,
            z_plot=z_plot,
            norm=norm,
            x_label=x_label,
            y_label=y_label,
            output_base=heatmap_output_base,
            dpi=dpi,
        )
        paths.extend(heatmap)

        combined = save_combined_views(
            p1=p1,
            p2=p2,
            z_plot=z_plot,
            norm=norm,
            x_label=x_label,
            y_label=y_label,
            output_base=combined_output_base,
            dpi=dpi,
        )
        paths.extend(combined)

    view_indices = range(1, len(_VIEW_AZIMS) + 1) if save_kind == "all" else (single_view_index,)
    for view_index in view_indices:
        azim = _VIEW_AZIMS[view_index - 1]
        single = save_single_view(
            p1=p1,
            p2=p2,
            z_plot=z_plot,
            norm=norm,
            azim=azim,
            x_label=x_label,
            y_label=y_label,
            view_index=view_index,
            output_base=single_output_base,
            dpi=dpi,
        )
        paths.extend(single)

    return paths


def projection_axis_labels(projection: str) -> tuple[str, str]:
    if projection == "pca":
        return "PC1", "PC2"
    if projection == "random_projection":
        return "RP1", "RP2"
    if projection == "tsne":
        return "t-SNE 1", "t-SNE 2"
    if projection == "autoencoder":
        return r"$z_1$", r"$z_2$"
    raise ValueError(f"unknown projection {projection!r}")


def landscape_output_bases(
    *,
    output_dir: Path | None,
    projection: str,
    short_name: str,
    D: int,
) -> tuple[Path, Path, Path]:
    stem = f"{short_name}_{projection}_landscape_d{D}"
    if output_dir is not None:
        base = output_dir / stem
        return base, base, base

    combined = FIGURES_ROOT / FIGURE_OUTPUT_DIRS["combined"] / projection / stem
    single = FIGURES_ROOT / FIGURE_OUTPUT_DIRS["single"] / projection / stem
    heatmap = FIGURES_ROOT / FIGURE_OUTPUT_DIRS["heatmap"] / projection / stem
    return combined, single, heatmap


def build_landscape_grid(
    problem: LSGOProblem,
    samples: np.ndarray,
    args: argparse.Namespace,
    output_base: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    if args.projection == "pca":
        mean, components, _ = fit_pca2(samples)
        lo, hi = pca_grid_bounds(samples, mean, components, args.margin)
        p1_1d = np.linspace(lo[0], hi[0], args.grid)
        p2_1d = np.linspace(lo[1], hi[1], args.grid)
        p1, p2 = np.meshgrid(p1_1d, p2_1d)
        z = evaluate_grid_pca(problem, mean, components, p1, p2)
        return p1, p2, z, "PC1", "PC2"

    if args.projection == "random_projection":
        mean, projection, embedding = fit_random_projection2(samples, seed=args.rp_seed)
        lo, hi = embedding_grid_bounds(embedding, args.margin)
        p1_1d = np.linspace(lo[0], hi[0], args.grid)
        p2_1d = np.linspace(lo[1], hi[1], args.grid)
        p1, p2 = np.meshgrid(p1_1d, p2_1d)
        z = evaluate_grid_random_projection(problem, mean, projection, p1, p2)
        return p1, p2, z, "RP1", "RP2"

    if args.projection == "tsne":
        embedding = fit_tsne2(samples, seed=args.tsne_seed, perplexity=args.tsne_perplexity)
        lo, hi = embedding_grid_bounds(embedding, args.margin)
        p1_1d = np.linspace(lo[0], hi[0], args.grid)
        p2_1d = np.linspace(lo[1], hi[1], args.grid)
        p1, p2 = np.meshgrid(p1_1d, p2_1d)
        z = evaluate_grid_tsne(
            problem,
            samples,
            embedding,
            p1,
            p2,
            k_neighbors=args.tsne_k_neighbors,
        )
        return p1, p2, z, "t-SNE 1", "t-SNE 2"

    if args.projection == "autoencoder":
        print("      training autoencoder ...", flush=True)
        h1 = args.ae_hidden1 if args.ae_hidden1 > 0 else default_ae_hidden_dims(problem.D)[0]
        h2 = args.ae_hidden2 if args.ae_hidden2 > 0 else default_ae_hidden_dims(problem.D)[1]
        ae_config = AE_VARIANT_CONFIG[args.ae_variant]
        if args.ae_variant == AE_LOG_FITNESS_VARIANT:
            autoencoder = fit_autoencoder2_with_log_fitness(
                problem,
                samples,
                seed=args.ae_seed,
                hidden1=h1,
                hidden2=h2,
                epochs=args.ae_epochs,
                batch_size=args.ae_batch_size,
                learning_rate=args.ae_learning_rate,
                device=args.ae_device,
                fitness_lambda=args.ae_fitness_lambda,
            )
        elif args.ae_variant == AE_LOG_FITNESS_MOO_VARIANT:
            autoencoder = fit_autoencoder2_with_log_fitness_moo(
                problem,
                samples,
                seed=args.ae_seed,
                hidden1=h1,
                hidden2=h2,
                epochs=args.ae_epochs,
                batch_size=args.ae_batch_size,
                learning_rate=args.ae_learning_rate,
                device=args.ae_device,
            )
        else:
            autoencoder = fit_autoencoder2(
                samples,
                problem.lb,
                problem.ub,
                seed=args.ae_seed,
                hidden1=h1,
                hidden2=h2,
                epochs=args.ae_epochs,
                batch_size=args.ae_batch_size,
                learning_rate=args.ae_learning_rate,
                device=args.ae_device,
                **ae_config,
            )
        if output_base is not None:
            save_autoencoder_checkpoint(
                autoencoder,
                output_base,
                problem=problem,
                args=args,
                hidden1=h1,
                hidden2=h2,
            )
        embedding = autoencoder.encode(samples)
        lo, hi = embedding_grid_bounds(embedding, args.margin)
        p1_1d = np.linspace(lo[0], hi[0], args.grid)
        p2_1d = np.linspace(lo[1], hi[1], args.grid)
        p1, p2 = np.meshgrid(p1_1d, p2_1d)
        z = evaluate_grid_autoencoder(
            problem,
            autoencoder,
            p1,
            p2,
            decode_batch_size=args.ae_decode_batch_size,
        )
        return p1, p2, z, r"$z_1$", r"$z_2$"

    raise ValueError(f"unknown projection {args.projection!r}")


def prepare_plot_values(
    z: np.ndarray,
    *,
    z_clip_percentile: float | None,
    scale: str,
    log_dynamic_range: float,
) -> tuple[np.ndarray, Normalize | LogNorm, str]:
    z_plot = np.asarray(z, dtype=float).copy()
    if z_clip_percentile is not None:
        cap = float(np.percentile(z_plot, z_clip_percentile))
        z_plot = np.minimum(z_plot, cap)

    finite = z_plot[np.isfinite(z_plot)]
    if finite.size == 0:
        raise ValueError("landscape has no finite values")

    vmin = float(finite.min())
    vmax = float(finite.max())
    if vmax <= vmin:
        vmax = vmin + max(abs(vmin), 1.0) * 1e-9

    use_log = scale == "log"
    if scale == "auto":
        use_log = vmin > 0.0 and (vmax / max(vmin, np.finfo(float).tiny)) >= log_dynamic_range

    if use_log:
        if vmin <= 0.0:
            z_plot = z_plot - vmin + np.finfo(float).tiny
            finite = z_plot[np.isfinite(z_plot)]
            vmin = float(finite.min())
            vmax = float(finite.max())
        z_plot = np.maximum(z_plot, np.finfo(float).tiny)
        if vmax <= vmin:
            vmax = vmin * 1.001
        return z_plot, LogNorm(vmin=vmin, vmax=vmax), "log"

    return z_plot, Normalize(vmin=vmin, vmax=vmax), "linear"


def _linear_contour_levels(z_plot: np.ndarray, n_levels: int = 12) -> np.ndarray:
    finite = z_plot[np.isfinite(z_plot)]
    if finite.size == 0:
        return np.array([], dtype=float)
    vmin = float(finite.min())
    vmax = float(finite.max())
    if vmax <= vmin:
        return np.array([], dtype=float)
    return np.linspace(vmin, vmax, num=n_levels + 2)[1:-1]


def _power_tick_values(vmin: float, vmax: float, *, max_ticks: int = 4) -> np.ndarray:
    if vmin <= 0.0 or vmax <= vmin:
        return np.array([], dtype=float)
    exp_min = max(0, int(np.ceil(np.log10(vmin))))
    exp_max = int(np.floor(np.log10(vmax)))
    if exp_max < exp_min:
        return np.array([], dtype=float)
    if exp_max - exp_min + 1 <= max_ticks:
        exponents = np.arange(exp_min, exp_max + 1)
    else:
        exponents = np.unique(
            np.rint(np.linspace(exp_min, exp_max, max_ticks)).astype(int)
        )
    ticks = 10.0 ** exponents
    return ticks[(ticks >= vmin) & (ticks <= vmax)]


def _configure_log_z_axis(ax, floor: float, vmin: float, vmax: float) -> None:
    ax.set_zlim(floor, vmax)
    ax.set_zscale("log")
    ax.set_zlim(floor, vmax)
    power_ticks = _power_tick_values(vmin, vmax)
    ticks = np.concatenate(([floor], power_ticks))
    ax.zaxis.set_major_locator(FixedLocator(ticks))

    def _format_log_tick(value: float, _pos: int) -> str:
        if np.isclose(value, floor, rtol=1e-6, atol=0.0):
            return "0"
        exponent = int(np.rint(np.log10(value)))
        return rf"$10^{{{exponent}}}$"

    ax.zaxis.set_major_formatter(FuncFormatter(_format_log_tick))
    ax.tick_params(axis="z", labelsize=8, pad=7)


def save_surface(
    *,
    p1: np.ndarray,
    p2: np.ndarray,
    z: np.ndarray,
    x_label: str,
    y_label: str,
    output_base: Path,
    dpi: int,
    z_clip_percentile: float | None,
    scale: str,
    log_dynamic_range: float,
) -> tuple[Path, Path]:
    z_plot, norm, scale_used = prepare_plot_values(
        z,
        z_clip_percentile=z_clip_percentile,
        scale=scale,
        log_dynamic_range=log_dynamic_range,
    )
    finite = z_plot[np.isfinite(z_plot)]
    if finite.size == 0:
        raise ValueError("landscape has no finite values")
    vmin = float(finite.min())
    vmax = float(finite.max())
    z_range = max(vmax - vmin, np.finfo(float).eps)
    if scale_used == "log":
        floor = max(vmin / 1.5, np.finfo(float).tiny)
    else:
        floor = vmin - 0.04 * z_range

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        p1,
        p2,
        z_plot,
        cmap="jet",
        linewidth=0,
        antialiased=True,
        norm=norm,
    )

    levels = _log_contour_levels(z_plot) if scale_used == "log" else _linear_contour_levels(z_plot)
    if levels.size > 0:
        ax.contour(
            p1,
            p2,
            z_plot,
            zdir="z",
            offset=floor,
            levels=levels,
            cmap="jet",
            linewidths=0.7,
            norm=norm,
        )

    ax.set_xlabel(x_label, labelpad=7)
    ax.set_ylabel(y_label, labelpad=7)
    ax.set_zlabel(_Z_LABEL, labelpad=9)
    if scale_used == "log":
        _configure_log_z_axis(ax, floor, vmin, vmax)
    else:
        ax.set_zlim(floor, vmax)
    ax.view_init(elev=25.0, azim=-135.0)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
    return _save_figure(fig, output_base.with_name(f"{output_base.name}_surface"), dpi)


def save_heatmap(
    *,
    p1: np.ndarray,
    p2: np.ndarray,
    z: np.ndarray,
    x_label: str,
    y_label: str,
    output_base: Path,
    dpi: int,
    z_clip_percentile: float | None,
    scale: str,
    log_dynamic_range: float,
) -> tuple[Path, Path]:
    z_plot, norm, scale_used = prepare_plot_values(
        z,
        z_clip_percentile=z_clip_percentile,
        scale=scale,
        log_dynamic_range=log_dynamic_range,
    )
    fig, ax = plt.subplots(figsize=(7.5, 6))
    heatmap = ax.pcolormesh(p1, p2, z_plot, cmap="viridis", norm=norm, shading="auto")
    levels = _log_contour_levels(z_plot) if scale_used == "log" else _linear_contour_levels(z_plot)
    if levels.size > 0:
        contours = ax.contour(
            p1,
            p2,
            z_plot,
            levels=levels,
            colors="white",
            linewidths=0.6,
            alpha=0.85,
        )
        ax.clabel(contours, inline=True, fontsize=7, fmt="%.1e")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    cbar = fig.colorbar(heatmap, ax=ax, pad=0.03)
    cbar.ax.set_title(_Z_LABEL, fontsize=10, pad=6)
    fig.tight_layout()
    return _save_figure(fig, output_base.with_name(f"{output_base.name}_heatmap"), dpi)


def save_data(output_base: Path, p1: np.ndarray, p2: np.ndarray, z: np.ndarray) -> Path:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    path = output_base.with_name(f"{output_base.name}_grid.npz")
    np.savez_compressed(path, p1=p1, p2=p2, fitness=z)
    return path


def output_base_for(func_id: str, args: argparse.Namespace) -> Path:
    if args.projection == "autoencoder":
        projection_label = f"autoencoder_{args.ae_variant}"
    else:
        projection_label = args.projection
    short = func_id.replace("cec2013_lsgo_", "")
    stem = f"{short}_{projection_label}_landscape_d{args.dim}"
    return args.output_dir / projection_label / stem


def plot_one_function(func_id: str, args: argparse.Namespace) -> list[Path]:
    problem = LSGOProblem(
        func_id=func_id,
        D=args.dim,
        seed=args.benchmark_seed,
        group_size=args.group_size,
    )
    rng = np.random.default_rng(args.sample_seed)
    samples = sample_uniform_in_bounds(args.n_samples, problem.lb, problem.ub, rng)
    output_base = output_base_for(func_id, args)
    p1, p2, z, x_label, y_label = build_landscape_grid(problem, samples, args, output_base)
    paths: list[Path] = []
    paths.extend(
        save_surface(
            p1=p1,
            p2=p2,
            z=z,
            x_label=x_label,
            y_label=y_label,
            output_base=output_base,
            dpi=args.dpi,
            z_clip_percentile=args.z_clip_percentile,
            scale=args.scale,
            log_dynamic_range=args.log_dynamic_range,
        )
    )
    paths.extend(
        save_heatmap(
            p1=p1,
            p2=p2,
            z=z,
            x_label=x_label,
            y_label=y_label,
            output_base=output_base,
            dpi=args.dpi,
            z_clip_percentile=args.z_clip_percentile,
            scale=args.scale,
            log_dynamic_range=args.log_dynamic_range,
        )
    )
    paths.append(save_data(output_base, p1, p2, z))
    if args.projection == "autoencoder":
        checkpoint_path = autoencoder_checkpoint_path(output_base)
        if checkpoint_path.exists():
            paths.append(checkpoint_path)
    return paths


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CEC-2013 LSGO projection landscape plots with f(x) as height and color.",
    )
    p.add_argument(
        "--projection",
        type=str,
        choices=PROJECTION_CHOICES,
        default="pca",
        help="2D embedding method.",
    )
    p.add_argument("--dim", type=int, default=1000, help="Problem dimension D.")
    p.add_argument(
        "--benchmark_seed",
        type=int,
        default=0,
        help="Seed for benchmark structural data when D != 1000 or files absent.",
    )
    p.add_argument("--group_size", type=int, default=50, help="Sub-component size.")
    p.add_argument(
        "--functions",
        type=str,
        default="all",
        help="Comma-separated list: all, or f1,f2,..., or full func ids.",
    )
    p.add_argument(
        "--n_samples",
        type=int,
        default=10000,
        help="Random samples in R^D used to fit the embedding.",
    )
    p.add_argument(
        "--sample_seed",
        type=int,
        default=42,
        help="RNG seed for domain samples (independent of benchmark seed).",
    )
    p.add_argument("--grid", type=int, default=50, help="Grid resolution per embedding axis.")
    p.add_argument(
        "--margin",
        type=float,
        default=0.05,
        help="Fractional padding on embedding axis ranges.",
    )
    p.add_argument(
        "--rp_seed",
        type=int,
        default=42,
        help="Random seed for the orthonormal random projection.",
    )
    p.add_argument(
        "--tsne_seed",
        type=int,
        default=42,
        help="Random seed for t-SNE (only used when --projection tsne).",
    )
    p.add_argument(
        "--tsne_perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity; must be < n_samples.",
    )
    p.add_argument(
        "--tsne_k_neighbors",
        type=int,
        default=8,
        help="k-NN neighbors to lift t-SNE grid points back to R^D.",
    )
    p.add_argument(
        "--ae_seed",
        type=int,
        default=42,
        help="Random seed for autoencoder init/training.",
    )
    p.add_argument(
        "--ae_hidden1",
        type=int,
        default=0,
        help="AE encoder hidden size 1 (0 = auto from D).",
    )
    p.add_argument(
        "--ae_hidden2",
        type=int,
        default=0,
        help="AE encoder hidden size 2 (0 = auto from D).",
    )
    p.add_argument(
        "--ae_epochs",
        type=int,
        default=80,
        help="Training epochs for the autoencoder.",
    )
    p.add_argument(
        "--ae_batch_size",
        type=int,
        default=512,
        help="Minibatch size for autoencoder training.",
    )
    p.add_argument(
        "--ae_learning_rate",
        type=float,
        default=1e-3,
        help="Adam learning rate for autoencoder training.",
    )
    p.add_argument(
        "--ae_fitness_lambda",
        type=float,
        default=1.0,
        help="Weight for the standardized log fitness loss in raw_mse_logfit.",
    )
    p.add_argument(
        "--ae_device",
        type=str,
        default="cpu",
        help="Torch device for autoencoder (cpu or cuda).",
    )
    p.add_argument(
        "--ae_decode_batch_size",
        type=int,
        default=512,
        help="Batch size when decoding latent grid points.",
    )
    p.add_argument(
        "--ae_variant",
        choices=AE_VARIANT_CHOICES,
        default="norm_sigmoid_mse",
        help="Autoencoder reconstruction or fitness aware variant.",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root for method subdirectories.",
    )
    p.add_argument("--dpi", type=int, default=220, help="Figure DPI for PNG/PDF.")
    p.add_argument(
        "--save_kind",
        type=str,
        choices=("all", "single"),
        default="all",
        help="Save all view types, or only one single 3D view.",
    )
    p.add_argument(
        "--single_view_index",
        type=int,
        choices=(1, 2, 3),
        default=1,
        help="Single view index to save when --save_kind single.",
    )
    p.add_argument(
        "--z_clip_percentile",
        type=float,
        default=99.0,
        help="Cap displayed z at this percentile (None to disable).",
    )
    p.add_argument(
        "--scale",
        choices=SCALE_CHOICES,
        default="auto",
        help="Display scale for f(x). Auto uses log for positive wide range landscapes.",
    )
    p.add_argument(
        "--log_dynamic_range",
        type=float,
        default=1e3,
        help="Minimum max/min ratio for automatic log scaling.",
    )
    return p


def main() -> None:
    configure_matplotlib()
    args = build_arg_parser().parse_args()
    if args.dim < 1:
        raise ValueError("--dim must be >= 1")
    if args.grid < 5:
        raise ValueError("--grid must be >= 5")
    if args.n_samples < 10:
        raise ValueError("--n_samples must be >= 10")
    if args.tsne_k_neighbors < 1:
        raise ValueError("--tsne_k_neighbors must be >= 1")
    if args.projection == "tsne" and args.tsne_perplexity >= args.n_samples:
        raise ValueError("--tsne_perplexity must be < n_samples")
    if args.ae_epochs < 1:
        raise ValueError("--ae_epochs must be >= 1")
    if args.ae_batch_size < 1:
        raise ValueError("--ae_batch_size must be >= 1")
    if args.ae_decode_batch_size < 1:
        raise ValueError("--ae_decode_batch_size must be >= 1")
    if args.ae_learning_rate <= 0.0:
        raise ValueError("--ae_learning_rate must be > 0")
    if args.ae_fitness_lambda < 0.0:
        raise ValueError("--ae_fitness_lambda must be >= 0")
    if args.log_dynamic_range <= 1.0:
        raise ValueError("--log_dynamic_range must be > 1")

    func_ids = parse_functions(args.functions)
    output_dir = args.output_dir

    z_clip = args.z_clip_percentile
    if z_clip is not None and not (0.0 < z_clip <= 100.0):
        raise ValueError("--z_clip_percentile must be in (0, 100] or disabled")

    print(
        f"CEC-2013 landscapes ({args.projection}): D={args.dim}, "
        f"grid={args.grid}x{args.grid}, n_samples={args.n_samples}"
    )
    print(f"Functions: {', '.join(func_ids)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    saved: list[tuple[str, list[Path]]] = []
    for fid in func_ids:
        print(f"  plotting {fid} ...", flush=True)
        paths = plot_one_function(fid, args)
        saved.append((fid, paths))
        for path in paths:
            print(f"    -> {path.name}")

    print("\nDone. Saved per function as surface, heat map, and grid data.")
    for fid, paths in saved:
        print(f"  {fid}: {len(paths)} files")


if __name__ == "__main__":
    main()
