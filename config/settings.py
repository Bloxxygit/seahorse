"""The single source of truth for model and training defaults."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocabulary_size: int = 256
    hidden_size: int = 128
    context_length: int = 128


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 8
    learning_rate: float = 3e-4
    epochs: int = 1


SETTINGS = {"model": ModelConfig(), "training": TrainingConfig()}
