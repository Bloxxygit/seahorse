"""Reusable PyTorch training primitives for future Kaggle GPU runs.

This module deliberately does not load data, start training, or write
checkpoints. Callers supply already-batched token tensors when a dataset is
available on Kaggle.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass

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

    def finish_accumulation(self) -> bool:
        """Apply a final update if an epoch ends mid-accumulation."""
        if self._pending_micro_batches == 0:
            return False
        self._optimizer_step()
        self._pending_micro_batches = 0
        return True

    def fit(self, batches: Iterable[Batch], max_optimizer_steps: int | None = None) -> list[TrainMetrics]:
        """Run the configured epochs when explicitly invoked by a Kaggle job."""
        if max_optimizer_steps is not None and max_optimizer_steps < 1:
            raise ValueError("max_optimizer_steps must be positive when supplied.")
        history: list[TrainMetrics] = []
        for _ in range(self.config.epochs):
            for batch in batches:
                metrics = self.train_batch(batch)
                history.append(metrics)
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
