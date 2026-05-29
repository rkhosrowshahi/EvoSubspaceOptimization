r"""Coverage experiment: map low-dimensional uniforms into full 3D space.

Compares subspace expands from shared 2D draws:
1) RandomProjection.expand(z)
2) LoRA latent (tied u,v duplication) then expand — Monte Carlo stats use this 2D to 3D path
3) RandomBlocking.expand(z)

Figures: random projection / blocking shown in 3D with default and right-side cameras.
The LoRA column shows two adjacent 2D scatter plots for packed latent coefficients:
vec(A) and vec(B) (each lives in R^{M*r} here M*r=2, full latent dimension 4).
"""

from __future__ import annotations

from collections.abc import Callable
import argparse
from dataclasses import dataclass, field
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np

# Allow running as: python experiments/reverse_map_2d_to_3d_coverage.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from subspace.lora import LoRA
from subspace.random_blocking import RandomBlocking
from subspace.random_projection import RandomProjection


@dataclass
class StreamingStats3D:
    """Online stats for 3D points."""

    count: int = 0
    mean: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    m2: np.ndarray = field(default_factory=lambda: np.zeros((3, 3), dtype=np.float64))
    min_vals: np.ndarray = field(default_factory=lambda: np.full(3, np.inf, dtype=np.float64))
    max_vals: np.ndarray = field(default_factory=lambda: np.full(3, -np.inf, dtype=np.float64))

    def update(self, x: np.ndarray) -> None:
        if x.size == 0:
            return
        self.min_vals = np.minimum(self.min_vals, x.min(axis=0))
        self.max_vals = np.maximum(self.max_vals, x.max(axis=0))

        for row in x:
            self.count += 1
            delta = row - self.mean
            self.mean = self.mean + delta / self.count
            delta2 = row - self.mean
            self.m2 += np.outer(delta, delta2)

    def covariance(self) -> np.ndarray:
        if self.count < 2:
            return np.zeros((3, 3), dtype=np.float64)
        return self.m2 / (self.count - 1)


def iter_uniform_2d(
    n_points: int,
    batch_size: int,
    seed: int,
    low: float,
    high: float,
):
    rng = np.random.default_rng(seed)
    remaining = n_points
    while remaining > 0:
        n = min(batch_size, remaining)
        yield rng.uniform(low, high, size=(n, 2))
        remaining -= n


def linear_rank_from_cov(cov: np.ndarray, rel_tol: float = 1e-10) -> tuple[np.ndarray, int]:
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)[::-1]
    threshold = max(eigvals[0] * rel_tol, 1e-16) if eigvals.size else 0.0
    rank = int(np.sum(eigvals > threshold))
    return eigvals, rank


def voxel_occupancy(
    mapper,
    n_points: int,
    batch_size: int,
    seed: int,
    low: float,
    high: float,
    bins: int,
    min_vals: np.ndarray,
    max_vals: np.ndarray,
) -> tuple[int, int]:
    eps = 1e-15
    span = np.maximum(max_vals - min_vals, eps)
    occupied = np.zeros((bins, bins, bins), dtype=bool)

    for z in iter_uniform_2d(n_points, batch_size, seed, low, high):
        x = mapper(z)
        t = (x - min_vals) / span
        t = np.clip(t, 0.0, 1.0 - np.finfo(np.float64).eps)
        idx = np.floor(t * bins).astype(np.int64)
        occupied[idx[:, 0], idx[:, 1], idx[:, 2]] = True

    occupied_count = int(np.count_nonzero(occupied))
    total_count = int(occupied.size)
    return occupied_count, total_count


def build_random_projection_mapper(seed: int, device: str):
    rp = RandomProjection(
        D=3,
        d=2,
        subspace_assignment="absolute",
        seed=seed,
        device=device,
    )

    def mapper(z2: np.ndarray) -> np.ndarray:
        return rp.expand(z2)

    return mapper


def build_lora_tied_mapper(seed: int, device: str | None) -> tuple[Callable[[np.ndarray], np.ndarray], LoRA]:
    lora = LoRA(
        D=3,
        d=1,  # rank r=1 -> internal search_dim = 2 * M * r = 4 for D=3
        subspace_assignment="absolute",
        seed=seed,
        device=device,
    )

    if lora.search_dim != 4:
        raise RuntimeError(f"Unexpected LoRA search_dim={lora.search_dim}, expected 4 for D=3,r=1.")

    def mapper(z2: np.ndarray) -> np.ndarray:
        # Tie LoRA parameters to keep a 2D input:
        # z_lora = [u, v, u, v], interpreted as A=[u,v]^T and B=[u,v].
        # This follows LoRA's A@B structure while constraining to 2D latent input.
        u = z2[:, 0]
        v = z2[:, 1]
        z_lora = np.column_stack((u, v, u, v))
        return lora.expand(z_lora)

    return mapper, lora


def build_random_blocking_mapper(seed: int):
    rb = RandomBlocking(
        D=3,
        d=2,
        subspace_assignment="absolute",
        seed=seed,
    )

    def mapper(z2: np.ndarray) -> np.ndarray:
        return rb.expand(z2)

    return mapper


def analyze_method(
    name: str,
    mapper,
    n_points: int,
    batch_size: int,
    seed: int,
    low: float,
    high: float,
    bins: int,
) -> dict[str, object]:
    stats = StreamingStats3D()
    for z in iter_uniform_2d(n_points, batch_size, seed, low, high):
        x = mapper(z)
        stats.update(x)

    cov = stats.covariance()
    eigvals, linear_rank = linear_rank_from_cov(cov)
    occupied, total = voxel_occupancy(
        mapper=mapper,
        n_points=n_points,
        batch_size=batch_size,
        seed=seed,
        low=low,
        high=high,
        bins=bins,
        min_vals=stats.min_vals,
        max_vals=stats.max_vals,
    )

    return {
        "name": name,
        "count": stats.count,
        "min_vals": stats.min_vals,
        "max_vals": stats.max_vals,
        "eigvals": eigvals,
        "linear_rank": linear_rank,
        "occupied": occupied,
        "total_voxels": total,
        "occupancy_ratio": occupied / total if total > 0 else 0.0,
    }


def print_report(result: dict[str, object]) -> None:
    name = result["name"]
    min_vals = result["min_vals"]
    max_vals = result["max_vals"]
    eigvals = result["eigvals"]
    linear_rank = result["linear_rank"]
    occupied = result["occupied"]
    total_voxels = result["total_voxels"]
    occupancy_ratio = result["occupancy_ratio"]

    print(f"\n=== {name} ===")
    print(f"samples: {result['count']}")
    print(f"Bounding box min: {np.array2string(min_vals, precision=6)}")
    print(f"Bounding box max: {np.array2string(max_vals, precision=6)}")
    print(f"Covariance eigenvalues: {np.array2string(eigvals, precision=6)}")
    print(f"estimated linear rank: {linear_rank}")
    print(f"occupied voxels: {occupied}/{total_voxels} ({100.0 * occupancy_ratio:.3f}%)")

    if linear_rank < 3:
        print("verdict: points are confined to a lower-dimensional (<=2D) region in R^3.")
    elif occupancy_ratio < 0.2:
        print("verdict: points occupy only a small part of the 3D volume (nonlinear subregion).")
    else:
        print("verdict: points show broad 3D volume coverage.")


def _format_subplot_caption(result: dict[str, object]) -> str:
    eig = result["eigvals"]
    rank = result["linear_rank"]
    occ = result["occupied"]
    total = result["total_voxels"]
    occ_ratio = result["occupancy_ratio"]
    return (
        f"Covariance eigenvalues={np.array2string(eig, precision=2)}\n"
        f"Rank={rank}, Voxel occupancy={occ}/{total} ({100.0 * occ_ratio:.2f}%)"
    )


def _draw_lora_ab_latent_planes(
    fig,
    gs_cell,
    z4_latent: np.ndarray,
    *,
    split: int,
    M: int,
    rank: int,
    title_suffix: str,
    lora_result: dict[str, object],
    show_caption: bool,
) -> None:
    """Scatter packed latent blocks z = [vec(A); vec(B)] (length 4 here) into two ℝ² planes.

    Unpack matches subspace LoRA.expand: vec(A) is the first split = M·r coefficients.
    """
    A_packed = z4_latent[:, :split]
    B_packed = z4_latent[:, split:]
    gs_inner = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_cell, wspace=0.38)

    ax_a = fig.add_subplot(gs_inner[0, 0])
    ax_a.scatter(A_packed[:, 0], A_packed[:, 1], s=1, alpha=0.28)
    ax_a.set_aspect("equal", adjustable="box")
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.set_xlabel(r"$[\mathrm{vec}(A)]_1$")
    ax_a.set_ylabel(r"$[\mathrm{vec}(A)]_2$")
    prefix = title_suffix + " " if title_suffix.strip() else ""
    ax_a.set_title(
        rf"{prefix}LoRA: $A\in\mathbb{{R}}^{{{M}\times{rank}}}$, "
        rf"packed as $\mathrm{{vec}}(A)\in\mathbb{{R}}^{{{M * rank}}}$"
    )

    ax_b = fig.add_subplot(gs_inner[0, 1])
    ax_b.scatter(B_packed[:, 0], B_packed[:, 1], s=1, alpha=0.28)
    ax_b.set_aspect("equal", adjustable="box")
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.set_xlabel(r"$[\mathrm{vec}(B)]_1$")
    ax_b.set_ylabel(r"$[\mathrm{vec}(B)]_2$")
    ax_b.set_title(
        rf"{prefix}LoRA: $B\in\mathbb{{R}}^{{{rank}\times{M}}}$, "
        rf"packed as $\mathrm{{vec}}(B)\in\mathbb{{R}}^{{{M * rank}}}$"
    )

    if show_caption:
        pos_a = ax_a.get_position()
        pos_b = ax_b.get_position()
        xc_fig = (pos_a.x0 + pos_b.x1) / 2
        yt_fig = min(pos_a.y0, pos_b.y0) - 0.02
        fig.text(
            xc_fig,
            yt_fig,
            _format_subplot_caption(lora_result),
            ha="center",
            va="top",
            fontsize=10,
            transform=fig.transFigure,
        )


def _draw_3d_scatter_panel(
    ax: plt.Axes,
    xyz: np.ndarray,
    title: str,
    result: dict[str, object],
    *,
    elev: float | None,
    azim: float | None,
    show_caption: bool,
) -> None:
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=1, alpha=0.2)
    ax.set_title(title)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_zlabel(r"$x_3$")
    if elev is not None and azim is not None:
        ax.view_init(elev=elev, azim=azim)
    if show_caption:
        ax.text2D(
            0.5,
            -0.10,
            _format_subplot_caption(result),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=12,
        )


def save_joint_figure(
    z2_plot: np.ndarray,
    x_rp_plot: np.ndarray,
    x_rb_plot: np.ndarray,
    z4_lora_plot: np.ndarray,
    lora: LoRA,
    rp_result: dict[str, object],
    rb_result: dict[str, object],
    lora_result: dict[str, object],
    output_path: Path,
) -> None:
    base = Path(output_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = base.with_suffix(".pdf")
    png_path = base.with_suffix(".png")
    fig = plt.figure(figsize=(22, 16))
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1.0, 1.0])

    ax2d = fig.add_subplot(gs[0, 0])
    ax2d.scatter(z2_plot[:, 0], z2_plot[:, 1], s=1, alpha=0.25)
    ax2d.set_title(r"Input uniform samples in $\mathbb{R}^2$")
    ax2d.set_xlabel(r"$z_1$")
    ax2d.set_ylabel(r"$z_2$")
    ax2d.set_aspect("equal", adjustable="box")
    ax2d.spines["top"].set_visible(False)
    ax2d.spines["right"].set_visible(False)

    ax_rp = fig.add_subplot(gs[0, 1], projection="3d")
    _draw_3d_scatter_panel(
        ax_rp,
        x_rp_plot,
        r"Random Projection: $\mathbb{R}^2 \to \mathbb{R}^3$",
        rp_result,
        elev=None,
        azim=None,
        show_caption=True,
    )

    ax_rb = fig.add_subplot(gs[0, 2], projection="3d")
    _draw_3d_scatter_panel(
        ax_rb,
        x_rb_plot,
        r"Random Blocking: $\mathbb{R}^2 \to \mathbb{R}^3$",
        rb_result,
        elev=None,
        azim=None,
        show_caption=True,
    )

    _draw_lora_ab_latent_planes(
        fig,
        gs[0, 3],
        z4_lora_plot,
        split=lora._split,
        M=lora.M,
        rank=lora.r,
        title_suffix="",
        lora_result=lora_result,
        show_caption=True,
    )

    # Right-side camera on RP/RB 3D panels (lower row). LoRA: same latent A/B planes repeated.
    right_elev, right_azim = 12.0, 90.0
    ax_key = fig.add_subplot(gs[1, 0])
    ax_key.axis("off")
    ax_key.text(
        0.5,
        0.58,
        "Lower grid",
        ha="center",
        va="center",
        fontsize=13,
        transform=ax_key.transAxes,
    )
    ax_key.text(
        0.5,
        0.38,
        f"cols 2-3: RP & RB,\nright-side camera\n(elev={right_elev:.0f}\u00b0, azim={right_azim:.0f}\u00b0)",
        ha="center",
        va="center",
        fontsize=10,
        transform=ax_key.transAxes,
        color="0.35",
    )
    ax_key.text(
        0.5,
        0.06,
        "col 4: LoRA $\\mathbb{R}^4$ latent\n(two $\\mathbb{R}^2$ coefficient planes)",
        ha="center",
        va="center",
        fontsize=9,
        transform=ax_key.transAxes,
        color="0.35",
    )

    ax_rp_r = fig.add_subplot(gs[1, 1], projection="3d")
    _draw_3d_scatter_panel(
        ax_rp_r,
        x_rp_plot,
        r"Random Projection (right view): $\mathbb{R}^2 \to \mathbb{R}^3$",
        rp_result,
        elev=right_elev,
        azim=right_azim,
        show_caption=False,
    )

    ax_rb_r = fig.add_subplot(gs[1, 2], projection="3d")
    _draw_3d_scatter_panel(
        ax_rb_r,
        x_rb_plot,
        r"Random Blocking (right view): $\mathbb{R}^2 \to \mathbb{R}^3$",
        rb_result,
        elev=right_elev,
        azim=right_azim,
        show_caption=False,
    )

    _draw_lora_ab_latent_planes(
        fig,
        gs[1, 3],
        z4_lora_plot,
        split=lora._split,
        M=lora.M,
        rank=lora.r,
        title_suffix=r"(same $\mathbb{R}^4$ draws) ",
        lora_result=lora_result,
        show_caption=False,
    )

    fig.subplots_adjust(left=0.05, right=0.97, bottom=0.07, top=0.91, hspace=0.32, wspace=0.25)

    fig.savefig(pdf_path, dpi=220)
    fig.savefig(png_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2D -> 3D reverse-map coverage experiment")
    parser.add_argument("--n_points", type=int, default=5_000_000, help="Number of 2D uniform samples.")
    parser.add_argument("--batch_size", type=int, default=250_000, help="Batch size for mapping.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for 2D samples and projection.")
    parser.add_argument("--low", type=float, default=-1.0, help="Lower bound of uniform sampling range.")
    parser.add_argument("--high", type=float, default=1.0, help="Upper bound of uniform sampling range.")
    parser.add_argument("--bins", type=int, default=32, help="Bins per axis for 3D voxel occupancy.")
    parser.add_argument(
        "--plot_samples",
        type=int,
        default=50_000,
        help="Number of points to render in the joint 2D/3D figure.",
    )
    parser.add_argument(
        "--plot_path",
        type=str,
        default="results/figures/reverse_map_2d_to_3d",
        help="Output figure path for the combined 2D+3D visualization.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for RandomProjection/LoRA (e.g. 'cpu', 'cuda:0', or 'none' for LoRA NumPy path).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_points <= 0:
        raise ValueError("--n_points must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.bins < 2:
        raise ValueError("--bins must be >= 2")
    if args.plot_samples <= 0:
        raise ValueError("--plot_samples must be positive")
    if not args.low < args.high:
        raise ValueError("Require low < high")

    device = args.device.strip()
    if not device:
        raise ValueError("--device cannot be empty")
    lora_device: str | None = None if device.lower() == "none" else device

    print("Running 2D -> 3D coverage experiment")
    print(
        f"N={args.n_points}, batch_size={args.batch_size}, "
        f"uniform_range=[{args.low}, {args.high}], bins={args.bins}, seed={args.seed}, "
        f"device={device}"
    )

    rp_mapper = build_random_projection_mapper(seed=args.seed, device=device)
    rb_mapper = build_random_blocking_mapper(seed=args.seed)
    lora_mapper, lora = build_lora_tied_mapper(seed=args.seed, device=lora_device)

    rp_result = analyze_method(
        name="RandomProjection (d=2 -> D=3)",
        mapper=rp_mapper,
        n_points=args.n_points,
        batch_size=args.batch_size,
        seed=args.seed,
        low=args.low,
        high=args.high,
        bins=args.bins,
    )
    print_report(rp_result)

    rb_result = analyze_method(
        name="RandomBlocking (d=2 -> D=3)",
        mapper=rb_mapper,
        n_points=args.n_points,
        batch_size=args.batch_size,
        seed=args.seed,
        low=args.low,
        high=args.high,
        bins=args.bins,
    )
    print_report(rb_result)

    lora_result = analyze_method(
        name="LoRA-style tied map (2D latent, D=3)",
        mapper=lora_mapper,
        n_points=args.n_points,
        batch_size=args.batch_size,
        seed=args.seed,
        low=args.low,
        high=args.high,
        bins=args.bins,
    )
    print_report(lora_result)

    rng_plot = np.random.default_rng(args.seed + 1)
    z2_plot = rng_plot.uniform(args.low, args.high, size=(args.plot_samples, 2))
    z4_lora_plot = rng_plot.uniform(args.low, args.high, size=(args.plot_samples, lora.search_dim))
    x_rp_plot = rp_mapper(z2_plot)
    x_rb_plot = rb_mapper(z2_plot)
    plot_path = REPO_ROOT / args.plot_path
    save_joint_figure(
        z2_plot=z2_plot,
        x_rp_plot=x_rp_plot,
        x_rb_plot=x_rb_plot,
        z4_lora_plot=z4_lora_plot,
        lora=lora,
        rp_result=rp_result,
        rb_result=rb_result,
        lora_result=lora_result,
        output_path=plot_path,
    )
    print(f"saved figures: {plot_path.with_suffix('.pdf')} and {plot_path.with_suffix('.png')}")

    print("\nDone.")


if __name__ == "__main__":
    main()
