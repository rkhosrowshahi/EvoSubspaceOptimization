r"""GNBG objective landscapes via 2D projections.

For each GNBG function:
  1. sample full dimensional points uniformly in the GNBG box,
  2. fit PCA, t-SNE, or an autoencoder to a 2D embedding,
  3. evaluate a regular 2D grid by lifting points back to full dimension,
  4. save a 3D surface and a 2D heat map with contours.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib.resources import as_file, files
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.io import loadmat

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from experiments.landscapes.cec2013_projection_landscape import (
    _add_colorbar_right,
    _require_torch,
    configure_matplotlib,
    default_ae_hidden_dims,
    embedding_grid_bounds,
    fit_autoencoder2,
    fit_pca2,
    sample_uniform_in_bounds,
    TrainedAutoencoder,
)

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
VALID_FUNC_IDS = tuple(f"f{i}" for i in range(1, 25))
GNBG_ROOT = files("evo_subspace.problems.gnbg")
FIGURES_ROOT = PROJECT_ROOT / "results" / "figures" / "gnbg"
DEFAULT_OUTPUT_ROOT = FIGURES_ROOT / "gnbg_projection_landscapes"
_Z_LABEL = r"$f(x)$"
_SAVE_PAD_INCHES = 0.18


@dataclass
class GNBGProblem:
    func_id: str
    dimension: int
    comp_num: int
    min_coordinate: float
    max_coordinate: float
    comp_min_pos: np.ndarray
    comp_sigma: np.ndarray
    comp_h: np.ndarray
    mu: np.ndarray
    omega: np.ndarray
    lambda_: np.ndarray
    rotation_matrix: np.ndarray
    optimum_value: float
    optimum_position: np.ndarray

    @property
    def lb(self) -> np.ndarray:
        return np.full(self.dimension, self.min_coordinate, dtype=float)

    @property
    def ub(self) -> np.ndarray:
        return np.full(self.dimension, self.max_coordinate, dtype=float)

    def transform(self, x: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
        y = x.copy()
        pos = x > 0
        y[pos] = np.log(x[pos])
        y[pos] = np.exp(
            y[pos] + alpha[0] * (np.sin(beta[0] * y[pos]) + np.sin(beta[1] * y[pos]))
        )
        neg = x < 0
        y[neg] = np.log(-x[neg])
        y[neg] = -np.exp(
            y[neg] + alpha[1] * (np.sin(beta[2] * y[neg]) + np.sin(beta[3] * y[neg]))
        )
        return y

    def evaluate(self, x: np.ndarray) -> float:
        return float(self.evaluate_batch(np.asarray(x, dtype=float).reshape(1, -1))[0])

    def evaluate_batch(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        values = np.empty(x.shape[0], dtype=float)
        for row_idx, row in enumerate(x):
            x_col = row.reshape(-1, 1)
            component_values = np.empty(self.comp_num, dtype=float)
            for comp_idx in range(self.comp_num):
                if self.rotation_matrix.ndim == 3:
                    rotation = self.rotation_matrix[:, :, comp_idx]
                else:
                    rotation = self.rotation_matrix
                diff = x_col - self.comp_min_pos[comp_idx, :].reshape(-1, 1)
                a = self.transform(
                    (diff.T @ rotation.T).reshape(1, -1),
                    self.mu[comp_idx, :],
                    self.omega[comp_idx, :],
                )
                b = self.transform(
                    (rotation @ diff).reshape(-1, 1),
                    self.mu[comp_idx, :],
                    self.omega[comp_idx, :],
                )
                h = np.diag(self.comp_h[comp_idx, :])
                raw_value = self.comp_sigma[comp_idx] + (a @ h @ b) ** self.lambda_[comp_idx]
                component_values[comp_idx] = float(np.asarray(raw_value).reshape(-1)[0])
            values[row_idx] = float(np.min(component_values))
        return values


def _mat_scalar(field) -> float:
    return float(np.array([item[0] for item in field.flatten()])[0, 0])


def load_gnbg_problem(func_id: str) -> GNBGProblem:
    if func_id not in VALID_FUNC_IDS:
        raise ValueError(f"unknown GNBG function {func_id!r}")
    with as_file(GNBG_ROOT / f"{func_id}.mat") as data_path:
        data = loadmat(data_path)["GNBG"]
    return GNBGProblem(
        func_id=func_id,
        dimension=int(_mat_scalar(data["Dimension"])),
        comp_num=int(_mat_scalar(data["o"])),
        min_coordinate=_mat_scalar(data["MinCoordinate"]),
        max_coordinate=_mat_scalar(data["MaxCoordinate"]),
        comp_min_pos=np.array(data["Component_MinimumPosition"][0, 0], dtype=float),
        comp_sigma=np.array(data["ComponentSigma"][0, 0], dtype=float).reshape(-1),
        comp_h=np.array(data["Component_H"][0, 0], dtype=float),
        mu=np.array(data["Mu"][0, 0], dtype=float),
        omega=np.array(data["Omega"][0, 0], dtype=float),
        lambda_=np.array(data["lambda"][0, 0], dtype=float).reshape(-1),
        rotation_matrix=np.array(data["RotationMatrix"][0, 0], dtype=float),
        optimum_value=_mat_scalar(data["OptimumValue"]),
        optimum_position=np.array(data["OptimumPosition"][0, 0], dtype=float).reshape(-1),
    )


def parse_functions(spec: str) -> list[str]:
    if spec.strip().lower() in ("all", "*"):
        return list(VALID_FUNC_IDS)
    out: list[str] = []
    for part in spec.split(","):
        token = part.strip().lower()
        if not token:
            continue
        if token.startswith("f") and token[1:].isdigit():
            fid = token
        elif token.isdigit():
            fid = f"f{token}"
        else:
            raise ValueError(f"unknown GNBG function token {part!r}")
        if fid not in VALID_FUNC_IDS:
            raise ValueError(f"invalid GNBG function {fid!r}")
        if fid not in out:
            out.append(fid)
    if not out:
        raise ValueError("at least one function is required")
    return out


def evaluate_grid_pca(
    problem: GNBGProblem,
    mean: np.ndarray,
    components: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
) -> np.ndarray:
    flat = np.column_stack([p1.ravel(), p2.ravel()])
    x = mean + flat @ components
    x = np.clip(x, problem.lb, problem.ub)
    return problem.evaluate_batch(x).reshape(p1.shape)


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
    problem: GNBGProblem,
    mean: np.ndarray,
    projection: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
) -> np.ndarray:
    flat = np.column_stack([p1.ravel(), p2.ravel()])
    x = mean + flat @ np.linalg.pinv(projection.T)
    x = np.clip(x, problem.lb, problem.ub)
    return problem.evaluate_batch(x).reshape(p1.shape)


def fit_tsne2(samples: np.ndarray, *, seed: int, perplexity: float, max_iter: int) -> np.ndarray:
    try:
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise ImportError("t-SNE requires scikit-learn") from exc

    if perplexity >= samples.shape[0]:
        raise ValueError("--tsne_perplexity must be < n_samples")
    kwargs = {
        "n_components": 2,
        "perplexity": perplexity,
        "random_state": seed,
        "init": "pca",
        "learning_rate": "auto",
    }
    try:
        tsne = TSNE(max_iter=max_iter, **kwargs)
    except TypeError:
        tsne = TSNE(n_iter=max_iter, **kwargs)
    return tsne.fit_transform(samples)


def evaluate_grid_knn(
    problem: GNBGProblem,
    samples: np.ndarray,
    embedding: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    *,
    k_neighbors: int,
) -> np.ndarray:
    from scipy.spatial import cKDTree

    flat = np.column_stack([p1.ravel(), p2.ravel()])
    tree = cKDTree(embedding)
    k = min(k_neighbors, embedding.shape[0])
    x = np.empty((flat.shape[0], samples.shape[1]), dtype=float)
    for i, point in enumerate(flat):
        dists, idx = tree.query(point, k=k)
        idx = np.atleast_1d(idx)
        dists = np.atleast_1d(dists).astype(float)
        weights = 1.0 / (dists + 1e-12)
        weights /= weights.sum()
        x[i] = weights @ samples[idx]
    x = np.clip(x, problem.lb, problem.ub)
    return problem.evaluate_batch(x).reshape(p1.shape)


def evaluate_grid_autoencoder(
    problem: GNBGProblem,
    autoencoder,
    p1: np.ndarray,
    p2: np.ndarray,
    *,
    decode_batch_size: int,
) -> np.ndarray:
    flat_z = np.column_stack([p1.ravel(), p2.ravel()])
    values = np.empty(flat_z.shape[0], dtype=float)
    for start in range(0, flat_z.shape[0], decode_batch_size):
        batch_z = flat_z[start : start + decode_batch_size]
        x = autoencoder.decode(batch_z)
        x = np.clip(x, problem.lb, problem.ub)
        values[start : start + x.shape[0]] = problem.evaluate_batch(x)
    return values.reshape(p1.shape)


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


def autoencoder_checkpoint_path(output_base: Path) -> Path:
    return output_base.with_name(f"{output_base.name}_autoencoder.pt")


def save_autoencoder_checkpoint(
    autoencoder: TrainedAutoencoder,
    output_base: Path,
    *,
    problem: GNBGProblem,
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
            "dimension": problem.dimension,
            "hidden1": hidden1,
            "hidden2": hidden2,
            "normalize_inputs": bool(ae_config["normalize_inputs"]),
            "decoder_activation": str(ae_config["decoder_activation"]),
            "reconstruction_loss": str(ae_config["reconstruction_loss"]),
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


def _torch_gnbg_transform(x, alpha, beta):
    torch, _ = _require_torch()
    eps = torch.finfo(x.dtype).tiny
    pos_log = torch.log(torch.clamp(x, min=eps))
    neg_log = torch.log(torch.clamp(-x, min=eps))
    pos = torch.exp(pos_log + alpha[0] * (torch.sin(beta[0] * pos_log) + torch.sin(beta[1] * pos_log)))
    neg = -torch.exp(
        neg_log + alpha[1] * (torch.sin(beta[2] * neg_log) + torch.sin(beta[3] * neg_log))
    )
    return torch.where(x > 0, pos, torch.where(x < 0, neg, torch.zeros_like(x)))


def _torch_gnbg_evaluate(problem: GNBGProblem, x):
    torch, _ = _require_torch()
    dtype = x.dtype
    device = x.device
    component_values = []
    comp_min_pos = torch.as_tensor(problem.comp_min_pos, dtype=dtype, device=device)
    comp_sigma = torch.as_tensor(problem.comp_sigma, dtype=dtype, device=device)
    comp_h = torch.as_tensor(problem.comp_h, dtype=dtype, device=device)
    mu = torch.as_tensor(problem.mu, dtype=dtype, device=device)
    omega = torch.as_tensor(problem.omega, dtype=dtype, device=device)
    lambda_ = torch.as_tensor(problem.lambda_, dtype=dtype, device=device)
    rotations = torch.as_tensor(problem.rotation_matrix, dtype=dtype, device=device)

    for comp_idx in range(problem.comp_num):
        if rotations.ndim == 3:
            rotation = rotations[:, :, comp_idx]
        else:
            rotation = rotations
        diff = x - comp_min_pos[comp_idx]
        transformed = _torch_gnbg_transform(
            diff @ rotation.T,
            mu[comp_idx],
            omega[comp_idx],
        )
        quadratic = torch.sum(transformed * comp_h[comp_idx] * transformed, dim=1)
        raw_value = comp_sigma[comp_idx] + torch.pow(torch.clamp(quadratic, min=0.0), lambda_[comp_idx])
        component_values.append(raw_value)
    return torch.min(torch.stack(component_values, dim=1), dim=1).values


def _log_fitness_stats(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    shift = 0.0
    min_value = float(np.min(values))
    if min_value <= 0.0:
        shift = min_value - 1.0
    log_values = np.log(np.maximum(values - shift, np.finfo(float).tiny))
    return shift, float(np.mean(log_values)), float(np.std(log_values) + 1e-12)


def fit_autoencoder2_with_log_fitness(
    problem: GNBGProblem,
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
    """Train a raw MSE AE with an added standardized log fitness preservation term."""
    torch, nn = _require_torch()

    lb = np.asarray(problem.lb, dtype=np.float32).ravel()
    ub = np.asarray(problem.ub, dtype=np.float32).ravel()
    x_train = np.asarray(samples, dtype=np.float32)
    d = x_train.shape[1]
    target_fitness = problem.evaluate_batch(samples)
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

    def transformed_fitness_loss(recon, target):
        recon_clipped = torch.clamp(
            recon,
            min=torch.as_tensor(lb, dtype=recon.dtype, device=recon.device),
            max=torch.as_tensor(ub, dtype=recon.dtype, device=recon.device),
        )
        recon_fitness = _torch_gnbg_evaluate(problem, recon_clipped)
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
    problem: GNBGProblem,
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
            "raw_mse_logfit_moo requires TorchJD. Install with: "
            'pip install "torchjd[quadprog_projector]"'
        ) from exc

    lb = np.asarray(problem.lb, dtype=np.float32).ravel()
    ub = np.asarray(problem.ub, dtype=np.float32).ravel()
    x_train = np.asarray(samples, dtype=np.float32)
    d = x_train.shape[1]
    target_fitness = problem.evaluate_batch(samples)
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

    def transformed_fitness_loss(recon, target):
        recon_clipped = torch.clamp(
            recon,
            min=torch.as_tensor(lb, dtype=recon.dtype, device=recon.device),
            max=torch.as_tensor(ub, dtype=recon.dtype, device=recon.device),
        )
        recon_fitness = _torch_gnbg_evaluate(problem, recon_clipped)
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


def build_landscape_grid(
    problem: GNBGProblem,
    samples: np.ndarray,
    args: argparse.Namespace,
    output_base: Path | None = None,
):
    if args.projection == "pca":
        mean, components, _ = fit_pca2(samples)
        embedding = (samples - mean) @ components.T
        lo, hi = embedding_grid_bounds(embedding, args.margin)
        p1, p2 = np.meshgrid(
            np.linspace(lo[0], hi[0], args.grid),
            np.linspace(lo[1], hi[1], args.grid),
        )
        z = evaluate_grid_pca(problem, mean, components, p1, p2)
        return p1, p2, z, "PC1", "PC2"

    if args.projection == "random_projection":
        mean, projection, embedding = fit_random_projection2(samples, seed=args.rp_seed)
        lo, hi = embedding_grid_bounds(embedding, args.margin)
        p1, p2 = np.meshgrid(
            np.linspace(lo[0], hi[0], args.grid),
            np.linspace(lo[1], hi[1], args.grid),
        )
        z = evaluate_grid_random_projection(problem, mean, projection, p1, p2)
        return p1, p2, z, "RP1", "RP2"

    if args.projection == "tsne":
        embedding = fit_tsne2(
            samples,
            seed=args.tsne_seed,
            perplexity=args.tsne_perplexity,
            max_iter=args.tsne_max_iter,
        )
        lo, hi = embedding_grid_bounds(embedding, args.margin)
        p1, p2 = np.meshgrid(
            np.linspace(lo[0], hi[0], args.grid),
            np.linspace(lo[1], hi[1], args.grid),
        )
        z = evaluate_grid_knn(
            problem,
            samples,
            embedding,
            p1,
            p2,
            k_neighbors=args.tsne_k_neighbors,
        )
        return p1, p2, z, "t-SNE 1", "t-SNE 2"

    if args.projection == "autoencoder":
        h1 = args.ae_hidden1 if args.ae_hidden1 > 0 else default_ae_hidden_dims(problem.dimension)[0]
        h2 = args.ae_hidden2 if args.ae_hidden2 > 0 else default_ae_hidden_dims(problem.dimension)[1]
        cache_key = (
            problem.dimension,
            tuple(np.asarray(problem.lb, dtype=float)),
            tuple(np.asarray(problem.ub, dtype=float)),
            args.n_samples,
            args.sample_seed,
            args.ae_seed,
            h1,
            h2,
            args.ae_epochs,
            args.ae_batch_size,
            args.ae_learning_rate,
            args.ae_device,
            args.ae_variant,
            args.ae_fitness_lambda if args.ae_variant == AE_LOG_FITNESS_VARIANT else None,
            problem.func_id if args.ae_variant in AE_FITNESS_AWARE_VARIANTS else None,
            args.margin,
            args.grid,
        )
        ae_cache = getattr(args, "_ae_cache", {})
        if cache_key not in ae_cache:
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
            embedding = autoencoder.encode(samples)
            lo, hi = embedding_grid_bounds(embedding, args.margin)
            p1, p2 = np.meshgrid(
                np.linspace(lo[0], hi[0], args.grid),
                np.linspace(lo[1], hi[1], args.grid),
            )
            ae_cache[cache_key] = (autoencoder, p1, p2)
            setattr(args, "_ae_cache", ae_cache)
        else:
            autoencoder, p1, p2 = ae_cache[cache_key]
        if output_base is not None:
            save_autoencoder_checkpoint(
                autoencoder,
                output_base,
                problem=problem,
                args=args,
                hidden1=h1,
                hidden2=h2,
            )
        z = evaluate_grid_autoencoder(
            problem,
            autoencoder,
            p1,
            p2,
            decode_batch_size=args.ae_decode_batch_size,
        )
        return p1, p2, z, r"$z_1$", r"$z_2$"

    raise ValueError(f"unknown projection {args.projection!r}")


def _log_contour_levels(z_plot: np.ndarray, n_levels: int = 12) -> np.ndarray:
    finite = z_plot[np.isfinite(z_plot)]
    if finite.size == 0:
        return np.array([], dtype=float)
    vmin = float(finite.min())
    vmax = float(finite.max())
    if vmax <= vmin:
        return np.array([], dtype=float)
    return np.geomspace(vmin, vmax, num=n_levels + 2)[1:-1]


def _linear_contour_levels(z_plot: np.ndarray, n_levels: int = 12) -> np.ndarray:
    finite = z_plot[np.isfinite(z_plot)]
    if finite.size == 0:
        return np.array([], dtype=float)
    vmin = float(finite.min())
    vmax = float(finite.max())
    if vmax <= vmin:
        return np.array([], dtype=float)
    return np.linspace(vmin, vmax, num=n_levels + 2)[1:-1]


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


def _save_figure(fig: plt.Figure, output_base: Path, dpi: int) -> tuple[Path, Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_base.with_suffix(".pdf")
    png_path = output_base.with_suffix(".png")
    fig.savefig(pdf_path, dpi=dpi, pad_inches=_SAVE_PAD_INCHES)
    fig.savefig(png_path, dpi=dpi, pad_inches=_SAVE_PAD_INCHES)
    plt.close(fig)
    return pdf_path, png_path


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
    floor = 0.0 if vmin >= 0.0 else vmin - 0.04 * z_range

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
    stem = f"{func_id}_{projection_label}_landscape_d{args.dim}"
    return args.output_dir / projection_label / stem


def plot_one_function(func_id: str, args: argparse.Namespace) -> list[Path]:
    problem = load_gnbg_problem(func_id)
    if args.dim is not None and args.dim != problem.dimension:
        raise ValueError(f"{func_id} has dimension {problem.dimension}, not {args.dim}")
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
    parser = argparse.ArgumentParser(
        description="GNBG projected objective landscapes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--projection", choices=PROJECTION_CHOICES, default="pca")
    parser.add_argument("--functions", type=str, default="all")
    parser.add_argument("--dim", type=int, default=30)
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--grid", type=int, default=250)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--rp_seed", type=int, default=42)
    parser.add_argument("--tsne_seed", type=int, default=42)
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--tsne_max_iter", type=int, default=10000)
    parser.add_argument("--tsne_k_neighbors", type=int, default=8)
    parser.add_argument("--ae_seed", type=int, default=42)
    parser.add_argument("--ae_hidden1", type=int, default=0)
    parser.add_argument("--ae_hidden2", type=int, default=0)
    parser.add_argument("--ae_epochs", type=int, default=80)
    parser.add_argument("--ae_batch_size", type=int, default=512)
    parser.add_argument("--ae_learning_rate", type=float, default=1e-3)
    parser.add_argument(
        "--ae_fitness_lambda",
        type=float,
        default=1.0,
        help="Weight for the standardized log fitness loss in the raw_mse_logfit AE variant.",
    )
    parser.add_argument("--ae_device", type=str, default="cpu")
    parser.add_argument("--ae_decode_batch_size", type=int, default=512)
    parser.add_argument(
        "--ae_variant",
        choices=AE_VARIANT_CHOICES,
        default="norm_sigmoid_mse",
        help="Autoencoder reconstruction variant.",
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--z_clip_percentile", type=float, default=99.0)
    parser.add_argument(
        "--scale",
        choices=SCALE_CHOICES,
        default="auto",
        help="f(x) display scale. Auto uses log only for positive wide-range landscapes.",
    )
    parser.add_argument(
        "--log_dynamic_range",
        type=float,
        default=1e3,
        help="Minimum max/min ratio for auto log scaling.",
    )
    return parser


def main() -> None:
    configure_matplotlib()
    args = build_arg_parser().parse_args()
    if args.n_samples < 10:
        raise ValueError("--n_samples must be >= 10")
    if args.grid < 5:
        raise ValueError("--grid must be >= 5")
    if args.tsne_perplexity >= args.n_samples:
        raise ValueError("--tsne_perplexity must be < n_samples")
    if args.tsne_k_neighbors < 1:
        raise ValueError("--tsne_k_neighbors must be >= 1")
    if args.tsne_max_iter < 250:
        raise ValueError("--tsne_max_iter must be >= 250")
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
    if args.z_clip_percentile is not None and not (0.0 < args.z_clip_percentile <= 100.0):
        raise ValueError("--z_clip_percentile must be in (0, 100] or disabled")
    if args.log_dynamic_range <= 1.0:
        raise ValueError("--log_dynamic_range must be > 1")

    func_ids = parse_functions(args.functions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"GNBG landscapes ({args.projection}): functions={len(func_ids)}, "
        f"samples={args.n_samples}, grid={args.grid}x{args.grid}"
    )
    print(f"Output root: {args.output_dir}")

    saved: list[tuple[str, list[Path]]] = []
    for fid in func_ids:
        print(f"  plotting {fid} ...", flush=True)
        paths = plot_one_function(fid, args)
        saved.append((fid, paths))
        for path in paths:
            print(f"    -> {path.relative_to(args.output_dir)}")

    print("\nDone.")
    for fid, paths in saved:
        print(f"  {fid}: {len(paths)} files")


if __name__ == "__main__":
    main()
