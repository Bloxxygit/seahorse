"""Reusable PyTorch training primitives for future Kaggle GPU runs.

This module deliberately does not load data, start training, or write
checkpoints. Callers supply already-batched token tensors when a dataset is
available on Kaggle.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
import json
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from config import TrainingConfig


Batch = Mapping[str, Tensor]


@dataclass(frozen=True)
class TrainMetrics:
    """Metrics emitted by one micro-batch without retaining computation graphs."""

    loss: float
    optimizer_step: bool
    global_step: int


@dataclass(frozen=True)
class ValidationMetrics:
    loss: float
    batches: int
    global_step: int


def build_adamw(model: nn.Module, config: TrainingConfig) -> torch.optim.AdamW:
    """Create AdamW groups, excluding biases and normalization scales from decay."""
    decay, no_decay = [], []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
    )


class Trainer:
    """Train a causal language model from ``input_ids``/``labels`` batches.

    Batches must be mappings with integer tensors shaped ``[batch, sequence]``:
    ``input_ids`` are model inputs and ``labels`` are next-token targets. Use
    ``-100`` in labels for positions excluded from cross-entropy loss.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.config = config or TrainingConfig()
        if self.config.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be at least one.")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.optimizer = build_adamw(self.model, self.config)
        self._use_amp = self.config.mixed_precision and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self._use_amp)
        self.global_step = 0
        self._pending_micro_batches = 0
        self.optimizer.zero_grad(set_to_none=True)

    def _move_batch(self, batch: Batch) -> tuple[Tensor, Tensor]:
        try:
            input_ids, labels = batch["input_ids"], batch["labels"]
        except KeyError as error:
            raise KeyError("Each batch must provide 'input_ids' and 'labels'.") from error
        if input_ids.shape != labels.shape:
            raise ValueError("input_ids and labels must have the same shape.")
        if input_ids.ndim != 2:
            raise ValueError("input_ids and labels must have shape [batch, sequence].")
        return input_ids.to(self.device, dtype=torch.long), labels.to(self.device, dtype=torch.long)

    def _optimizer_step(self) -> None:
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step += 1

    def _loss_for_batch(self, batch: Batch) -> Tensor:
        input_ids, labels = self._move_batch(batch)
        logits = self.model(input_ids)
        return F.cross_entropy(
            logits.flatten(end_dim=-2),
            labels.flatten(),
            ignore_index=self.config.ignore_index,
        )

    def train_batch(self, batch: Batch) -> TrainMetrics:
        """Process one micro-batch and step when accumulation is complete."""
        self.model.train()
        input_ids, labels = self._move_batch(batch)
        autocast = torch.autocast(device_type="cuda", dtype=torch.float16) if self._use_amp else nullcontext()
        with autocast:
            logits = self.model(input_ids)
            loss = F.cross_entropy(
                logits.flatten(end_dim=-2),
                labels.flatten(),
                ignore_index=self.config.ignore_index,
            )
            scaled_loss = loss / self.config.gradient_accumulation_steps
        self.scaler.scale(scaled_loss).backward()

        self._pending_micro_batches += 1
        optimizer_step = self._pending_micro_batches == self.config.gradient_accumulation_steps
        if optimizer_step:
            self._optimizer_step()
            self._pending_micro_batches = 0
        return TrainMetrics(loss=loss.detach().item(), optimizer_step=optimizer_step, global_step=self.global_step)

    @torch.no_grad()
    def validate(self, batches: Iterable[Batch]) -> ValidationMetrics:
        """Evaluate mean next-token loss without changing optimizer state."""
        was_training = self.model.training
        self.model.eval()
        losses: list[float] = []
        for batch in batches:
            losses.append(float(self._loss_for_batch(batch).item()))
        if was_training:
            self.model.train()
        if not losses:
            raise ValueError("Validation requires at least one batch.")
        return ValidationMetrics(sum(losses) / len(losses), len(losses), self.global_step)

    def save_checkpoint(self, path: str | Path, *, data_state: Mapping[str, object] | None = None) -> None:
        """Save all state needed to resume a deterministic training run."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "global_step": self.global_step,
            "pending_micro_batches": self._pending_micro_batches,
            "training_config": self.config.__dict__,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "data_state": dict(data_state or {}),
        }
        torch.save(payload, destination)

    def load_checkpoint(self, path: str | Path) -> dict[str, object]:
        """Restore trainer state and return the checkpoint's data-resume state."""
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if payload.get("format_version") != 1:
            raise ValueError("Unsupported trainer checkpoint format.")
        if payload.get("training_config") != self.config.__dict__:
            raise ValueError("Checkpoint training configuration does not match the trainer.")
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scaler.load_state_dict(payload["scaler"])
        self.global_step = int(payload["global_step"])
        self._pending_micro_batches = int(payload["pending_micro_batches"])
        torch.set_rng_state(payload["rng_state"])
        if torch.cuda.is_available() and payload.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
        return dict(payload.get("data_state", {}))

    def finish_accumulation(self) -> bool:
        """Apply a final update if an epoch ends mid-accumulation."""
        if self._pending_micro_batches == 0:
            return False
        self._optimizer_step()
        self._pending_micro_batches = 0
        return True

    def fit(
        self,
        batches: Iterable[Batch],
        max_optimizer_steps: int | None = None,
        *,
        validation_batches: Iterable[Batch] | None = None,
        checkpoint_dir: str | Path | None = None,
        checkpoint_interval: int | None = None,
        validation_interval: int | None = None,
        log_path: str | Path | None = None,
    ) -> list[TrainMetrics]:
        """Run the configured epochs when explicitly invoked by a Kaggle job."""
        if max_optimizer_steps is not None and max_optimizer_steps < 1:
            raise ValueError("max_optimizer_steps must be positive when supplied.")
        if checkpoint_interval is not None and checkpoint_interval < 1:
            raise ValueError("checkpoint_interval must be positive when supplied.")
        if validation_interval is not None and validation_interval < 1:
            raise ValueError("validation_interval must be positive when supplied.")
        if (checkpoint_interval or validation_interval) and validation_batches is None and validation_interval:
            raise ValueError("validation_batches are required when validation is enabled.")
        if log_path is not None:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        history: list[TrainMetrics] = []
        for _ in range(self.config.epochs):
            for batch in batches:
                metrics = self.train_batch(batch)
                history.append(metrics)
                if log_path is not None:
                    with Path(log_path).open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({"type": "train", **metrics.__dict__}) + "\n")
                if validation_interval and metrics.optimizer_step and self.global_step % validation_interval == 0:
                    validation = self.validate(validation_batches)  # type: ignore[arg-type]
                    if log_path is not None:
                        with Path(log_path).open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps({"type": "validation", **validation.__dict__}) + "\n")
                if checkpoint_interval and metrics.optimizer_step and self.global_step % checkpoint_interval == 0:
                    self.save_checkpoint(Path(checkpoint_dir or "checkpoints") / f"step-{self.global_step}.pt")
                if max_optimizer_steps is not None and self.global_step >= max_optimizer_steps:
                    return history
            if self.finish_accumulation() and history:
                history[-1] = TrainMetrics(history[-1].loss, True, self.global_step)
        return history


class TrainingLoop:
    """Deprecated no-op compatibility shim for the original scaffold entrypoint."""

    def run(self, model: object, examples: object) -> dict[str, int]:
        del model
        try:
            count = len(examples)  # type: ignore[arg-type]
        except TypeError:
            count = 0
        return {"examples_seen": count, "steps_completed": 0}
