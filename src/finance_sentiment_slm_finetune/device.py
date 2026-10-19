"""Torch device selection."""

from __future__ import annotations

import torch


def resolve_device() -> torch.device:
    """Picks CUDA, then MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    print("No GPU found, falling back to CPU (much slower).")
    return torch.device("cpu")
