"""Optional PyTorch helpers for GPU-accelerated subspace maps."""

from __future__ import annotations

import torch

def parse_torch_device(device: str | None):
    """Return ``torch.device`` if ``device`` is set, else ``None`` (NumPy path)."""
    if device is None:
        return None
    return torch.device(device)
