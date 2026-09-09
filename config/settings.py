"""The single source of truth for model and training defaults."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocabulary_size: int = 32_000
    hidden_size: int = 512
    num_layers: int = 8
    num_attention_heads: int = 8
    mlp_hidden_size: int = 2_048
    context_length: int = 512
    dropout: float = 0.0


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 8
    learning_rate: float = 3e-4
    epochs: int = 1
    weight_decay: float = 0.1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    mixed_precision: bool = True
    ignore_index: int = -100


SETTINGS = {"model": ModelConfig(), "training": TrainingConfig()}
