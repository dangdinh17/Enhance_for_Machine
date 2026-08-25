"""Small, reusable helpers for reproducible PyTorch training."""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any

import torch


class AverageMeter:
    """Track a sample-weighted running average."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Clear all accumulated values."""
        self.total = 0.0
        self.count = 0

    @property
    def average(self) -> float:
        """Return the current average, or zero before the first update."""
        return self.total / self.count if self.count else 0.0

    def update(self, value: float, sample_count: int = 1) -> None:
        """Add a scalar measured over ``sample_count`` samples."""
        if sample_count < 1:
            raise ValueError("sample_count must be positive")
        self.total += float(value) * sample_count
        self.count += sample_count


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python and PyTorch random number generators."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve an explicit device or select CUDA when it is available."""
    resolved = torch.device(
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return resolved


def save_checkpoint(state: dict[str, Any], path: str | Path) -> None:
    """Atomically save a checkpoint, avoiding partially written files."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(file_descriptor)
    try:
        torch.save(state, temporary_name)
        os.replace(temporary_name, destination)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
