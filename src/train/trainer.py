"""Reusable training and validation loop for MFQEv2."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from src.utils import AverageMeter, resolve_device, save_checkpoint


@dataclass(frozen=True)
class EpochMetrics:
    """Aggregated metrics from one data-loader pass."""

    loss: float
    samples: int


class MFQEv2Trainer:
    """Train an MFQEv2-style model from triplets of neighboring frames.

    Each batch must map the keys ``previous``, ``current``, ``following`` and
    ``target`` to tensors. The loss receives ``(enhanced, target)`` and may be
    :class:`src.loss.FeatureLoss` or any compatible PyTorch loss.
    """

    REQUIRED_KEYS = ("previous", "current", "following", "target")

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str | torch.device | None = None,
        use_amp: bool = True,
        gradient_clip_norm: float | None = None,
    ) -> None:
        if gradient_clip_norm is not None and gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")
        self.device = resolve_device(device)
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.gradient_clip_norm = gradient_clip_norm
        self.amp_enabled = use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.amp_enabled)

    def train_epoch(self, batches: Iterable[Mapping[str, Tensor]]) -> EpochMetrics:
        """Run one optimization epoch."""
        self.model.train()
        return self._run_epoch(batches, training=True)

    @torch.no_grad()
    def validate_epoch(self, batches: Iterable[Mapping[str, Tensor]]) -> EpochMetrics:
        """Evaluate the model without updating parameters."""
        self.model.eval()
        return self._run_epoch(batches, training=False)

    def save(
        self,
        path: str | Path,
        epoch: int,
        metrics: EpochMetrics | None = None,
    ) -> None:
        """Save model, optimizer, scaler, epoch and optional metrics."""
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        state: dict[str, object] = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
        }
        if metrics is not None:
            state["metrics"] = {
                "loss": metrics.loss,
                "samples": metrics.samples,
            }
        save_checkpoint(state, path)

    def _run_epoch(
        self,
        batches: Iterable[Mapping[str, Tensor]],
        training: bool,
    ) -> EpochMetrics:
        loss_meter = AverageMeter()
        for batch in batches:
            tensors = self._prepare_batch(batch)
            batch_size = tensors["target"].shape[0]
            if training:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(training):
                with torch.amp.autocast(self.device.type, enabled=self.amp_enabled):
                    enhanced = self.model(
                        tensors["previous"],
                        tensors["current"],
                        tensors["following"],
                    )
                    loss = self.criterion(enhanced, tensors["target"])
                if loss.ndim != 0:
                    raise ValueError("criterion must return a scalar loss")

            if training:
                self.scaler.scale(loss).backward()
                if self.gradient_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip_norm
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()

            loss_meter.update(loss.detach().item(), batch_size)

        if loss_meter.count == 0:
            raise ValueError("The data loader produced no batches")
        return EpochMetrics(loss=loss_meter.average, samples=loss_meter.count)

    def _prepare_batch(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        missing = set(self.REQUIRED_KEYS) - set(batch)
        if missing:
            raise KeyError(f"Batch is missing required keys: {sorted(missing)}")
        tensors: dict[str, Tensor] = {}
        for name in self.REQUIRED_KEYS:
            value = batch[name]
            if not isinstance(value, Tensor):
                raise TypeError(f"Batch value '{name}' must be a torch.Tensor")
            tensors[name] = value.to(self.device, non_blocking=True)
        return tensors
