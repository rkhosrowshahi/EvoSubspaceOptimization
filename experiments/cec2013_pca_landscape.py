r"""Approximate CEC-2013 LSGO landscapes via 2D projections (D=1000 -> 2D).

For each benchmark F1-F15:
  1. Sample points uniformly in the benchmark box [lb, ub]^D.
  2. Reduce to 2D with PCA or t-SNE (--projection).
  3. Build a grid in the 2D embedding and estimate f(x) on the grid.
  4. Plot (dim-1, dim-2, f(x)) as a 3D surface.

Outputs per function: 3 single-view PNG/PDF + 1 combined 3-panel PNG/PDF (8 files).
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
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from problems.lsgo import LSGOProblem
from problems.cec2013lsgo.cec2013lsgo.benchmarks import VALID_FUNC_IDS

# Fixed elevation keeps f(x) as the vertical axis; azimuth rotates around it.
_VIEW_ELEV = 25.0
_VIEW_AZIMS = (45.0, 135.0, 225.0)
_Z_LABEL = r"$f(x)$"
_POSITIVE_Z_FLOOR = 1e-30
PROJECTION_CHOICES = ("pca", "tsne", "autoencoder")
OUTPUT_SUBDIRS = {
    "pca": "cec2013_pca_landscapes",
    "tsne": "cec2013_tsne_landscapes",
    "autoencoder": "cec2013_autoencoder_landscapes",
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

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        span = np.maximum(self.ub - self.lb, 1e-12)
        return (x - self.lb) / span

    def _denormalize(self, x_norm: np.ndarray) -> np.ndarray:
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
) -> TrainedAutoencoder:
    """Train a 2D-latent MLP autoencoder on samples in [lb, ub]^D."""
    torch, nn = _require_torch()

    lb = np.asarray(lb, dtype=np.float32).ravel()
    ub = np.asarray(ub, dtype=np.float32).ravel()
    span = np.maximum(ub - lb, 1e-12)
    x_norm = ((samples - lb) / span).astype(np.float32)
    d = x_norm.shape[1]

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
            self.decoder = nn.Sequential(
                nn.Linear(2, hidden2),
                nn.ReLU(),
                nn.Linear(hidden2, hidden1),
                nn.ReLU(),
                nn.Linear(hidden1, d),
                nn.Sigmoid(),
            )

        def encode(self, x: torch.Tensor) -> torch.Tensor:
            return self.encoder(x)

        def decode(self, z: torch.Tensor) -> torch.Tensor:
            return self.decoder(z)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.decode(self.encode(x))

    torch_device = torch.device(device)
    torch.manual_seed(seed)
    model = MLPAutoencoder().to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    n = x_norm.shape[0]
    batch_size = min(batch_size, n)
    rng = np.random.default_rng(seed)

    def eval_mse() -> float:
        model.eval()
        total = 0.0
        n_batches = 0
        with torch.no_grad():
            for start in range(0, n, batch_size):
                batch = torch.as_tensor(x_norm[start : start + batch_size], device=torch_device)
                recon = model(batch)
                total += float(loss_fn(recon, batch).item())
                n_batches += 1
        return total / max(n_batches, 1)

    print(f"      AE epoch 0/{epochs}, MSE={eval_mse():.6e}", flush=True)

    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(n)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            batch = torch.as_tensor(x_norm[idx], device=torch_device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        if epoch == epochs - 1 or (epoch + 1) % max(epochs // 5, 1) == 0:
            mean_loss = epoch_loss / max(n_batches, 1)
            print(f"      AE epoch {epoch + 1}/{epochs}, MSE={mean_loss:.6e}", flush=True)

    return TrainedAutoencoder(lb=lb, ub=ub, model=model, device=torch_device)


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


def save_all_landscapes(
    *,
    p1: np.ndarray,
    p2: np.ndarray,
    z: np.ndarray,
    x_label: str,
    y_label: str,
    output_base: Path,
    dpi: int,
    z_clip_percentile: float | None,
) -> list[Path]:
    z_plot, norm = prepare_surface(z, z_clip_percentile)
    paths: list[Path] = []

    combined = save_combined_views(
        p1=p1,
        p2=p2,
        z_plot=z_plot,
        norm=norm,
        x_label=x_label,
        y_label=y_label,
        output_base=output_base,
        dpi=dpi,
    )
    paths.extend(combined)

    for view_index, azim in enumerate(_VIEW_AZIMS, start=1):
        single = save_single_view(
            p1=p1,
            p2=p2,
            z_plot=z_plot,
            norm=norm,
            azim=azim,
            x_label=x_label,
            y_label=y_label,
            view_index=view_index,
            output_base=output_base,
            dpi=dpi,
        )
        paths.extend(single)

    return paths


def projection_axis_labels(projection: str) -> tuple[str, str]:
    if projection == "pca":
        return "PC1", "PC2"
    if projection == "tsne":
        return "t-SNE 1", "t-SNE 2"
    if projection == "autoencoder":
        return r"$z_1$", r"$z_2$"
    raise ValueError(f"unknown projection {projection!r}")


def build_landscape_grid(
    problem: LSGOProblem,
    samples: np.ndarray,
    *,
    projection: str,
    grid: int,
    margin: float,
    tsne_seed: int,
    tsne_perplexity: float,
    tsne_k_neighbors: int,
    ae_seed: int,
    ae_hidden1: int,
    ae_hidden2: int,
    ae_epochs: int,
    ae_batch_size: int,
    ae_learning_rate: float,
    ae_device: str,
    ae_decode_batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if projection == "pca":
        mean, components, _ = fit_pca2(samples)
        lo, hi = pca_grid_bounds(samples, mean, components, margin)
        p1_1d = np.linspace(lo[0], hi[0], grid)
        p2_1d = np.linspace(lo[1], hi[1], grid)
        p1, p2 = np.meshgrid(p1_1d, p2_1d)
        z = evaluate_grid_pca(problem, mean, components, p1, p2)
        return p1, p2, z

    if projection == "tsne":
        embedding = fit_tsne2(samples, seed=tsne_seed, perplexity=tsne_perplexity)
        lo, hi = embedding_grid_bounds(embedding, margin)
        p1_1d = np.linspace(lo[0], hi[0], grid)
        p2_1d = np.linspace(lo[1], hi[1], grid)
        p1, p2 = np.meshgrid(p1_1d, p2_1d)
        z = evaluate_grid_tsne(
            problem,
            samples,
            embedding,
            p1,
            p2,
            k_neighbors=tsne_k_neighbors,
        )
        return p1, p2, z

    if projection == "autoencoder":
        print("      training autoencoder ...", flush=True)
        autoencoder = fit_autoencoder2(
            samples,
            problem.lb,
            problem.ub,
            seed=ae_seed,
            hidden1=ae_hidden1,
            hidden2=ae_hidden2,
            epochs=ae_epochs,
            batch_size=ae_batch_size,
            learning_rate=ae_learning_rate,
            device=ae_device,
        )
        embedding = autoencoder.encode(samples)
        lo, hi = embedding_grid_bounds(embedding, margin)
        p1_1d = np.linspace(lo[0], hi[0], grid)
        p2_1d = np.linspace(lo[1], hi[1], grid)
        p1, p2 = np.meshgrid(p1_1d, p2_1d)
        z = evaluate_grid_autoencoder(
            problem,
            autoencoder,
            p1,
            p2,
            decode_batch_size=ae_decode_batch_size,
        )
        return p1, p2, z

    raise ValueError(f"unknown projection {projection!r}")


def plot_one_function(
    func_id: str,
    *,
    D: int,
    seed: int,
    group_size: int,
    n_samples: int,
    grid: int,
    margin: float,
    sample_seed: int,
    projection: str,
    tsne_seed: int,
    tsne_perplexity: float,
    tsne_k_neighbors: int,
    ae_seed: int,
    ae_hidden1: int,
    ae_hidden2: int,
    ae_epochs: int,
    ae_batch_size: int,
    ae_learning_rate: float,
    ae_device: str,
    ae_decode_batch_size: int,
    output_dir: Path,
    dpi: int,
    z_clip_percentile: float | None,
) -> list[Path]:
    problem = LSGOProblem(func_id=func_id, D=D, seed=seed, group_size=group_size)
    rng = np.random.default_rng(sample_seed)

    samples = sample_uniform_in_bounds(n_samples, problem.lb, problem.ub, rng)
    h1 = ae_hidden1 if ae_hidden1 > 0 else default_ae_hidden_dims(D)[0]
    h2 = ae_hidden2 if ae_hidden2 > 0 else default_ae_hidden_dims(D)[1]
    p1, p2, z = build_landscape_grid(
        problem,
        samples,
        projection=projection,
        grid=grid,
        margin=margin,
        tsne_seed=tsne_seed,
        tsne_perplexity=tsne_perplexity,
        tsne_k_neighbors=tsne_k_neighbors,
        ae_seed=ae_seed,
        ae_hidden1=h1,
        ae_hidden2=h2,
        ae_epochs=ae_epochs,
        ae_batch_size=ae_batch_size,
        ae_learning_rate=ae_learning_rate,
        ae_device=ae_device,
        ae_decode_batch_size=ae_decode_batch_size,
    )

    short = func_id.replace("cec2013_lsgo_", "")
    output_base = output_dir / f"{short}_{projection}_landscape_d{D}"
    x_label, y_label = projection_axis_labels(projection)
    return save_all_landscapes(
        p1=p1,
        p2=p2,
        z=z,
        x_label=x_label,
        y_label=y_label,
        output_base=output_base,
        dpi=dpi,
        z_clip_percentile=z_clip_percentile,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CEC-2013 LSGO 2D landscape plots (PCA, t-SNE, or AE + f(x) as height).",
    )
    p.add_argument(
        "--projection",
        type=str,
        choices=PROJECTION_CHOICES,
        default="pca",
        help="2D embedding: pca, tsne (k-NN lift), or autoencoder (MLP latent + decode).",
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
        "--output_dir",
        type=Path,
        default=None,
        help="Directory for PNG/PDF outputs (default depends on --projection).",
    )
    p.add_argument("--dpi", type=int, default=220, help="Figure DPI for PNG/PDF.")
    p.add_argument(
        "--z_clip_percentile",
        type=float,
        default=99.0,
        help="Cap displayed z at this percentile (None to disable).",
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

    func_ids = parse_functions(args.functions)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = REPO_ROOT / "experiments" / "outputs" / OUTPUT_SUBDIRS[args.projection]
    output_dir.mkdir(parents=True, exist_ok=True)

    z_clip = args.z_clip_percentile
    if z_clip is not None and not (0.0 < z_clip <= 100.0):
        raise ValueError("--z_clip_percentile must be in (0, 100] or disabled")

    print(
        f"CEC-2013 landscapes ({args.projection}): D={args.dim}, "
        f"grid={args.grid}x{args.grid}, n_samples={args.n_samples}"
    )
    print(f"Functions: {', '.join(func_ids)}")
    print(f"Output directory: {output_dir}")

    saved: list[tuple[str, list[Path]]] = []
    for fid in func_ids:
        print(f"  plotting {fid} ...", flush=True)
        paths = plot_one_function(
            fid,
            D=args.dim,
            seed=args.benchmark_seed,
            group_size=args.group_size,
            n_samples=args.n_samples,
            grid=args.grid,
            margin=args.margin,
            sample_seed=args.sample_seed,
            projection=args.projection,
            tsne_seed=args.tsne_seed,
            tsne_perplexity=args.tsne_perplexity,
            tsne_k_neighbors=args.tsne_k_neighbors,
            ae_seed=args.ae_seed,
            ae_hidden1=args.ae_hidden1,
            ae_hidden2=args.ae_hidden2,
            ae_epochs=args.ae_epochs,
            ae_batch_size=args.ae_batch_size,
            ae_learning_rate=args.ae_learning_rate,
            ae_device=args.ae_device,
            ae_decode_batch_size=args.ae_decode_batch_size,
            output_dir=output_dir,
            dpi=args.dpi,
            z_clip_percentile=z_clip,
        )
        saved.append((fid, paths))
        for path in paths:
            print(f"    -> {path.name}")

    print("\nDone. Saved per function (combined + 3 views, PNG+PDF):")
    for fid, paths in saved:
        print(f"  {fid}: {len(paths)} files")


if __name__ == "__main__":
    main()
