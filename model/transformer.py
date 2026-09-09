"""SeaGlass's small causal decoder-only Transformer implemented in PyTorch."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from config import ModelConfig


class RMSNorm(nn.Module):
    """Root-mean-square normalization with a learned per-channel scale."""

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = x.float().pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(rms + self.eps).to(dtype=x.dtype) * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary positional embeddings for a fixed maximum context length."""

    def __init__(self, head_dim: int, max_context_length: int, base: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE requires an even attention head dimension.")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_context_length, dtype=torch.float32)
        angles = torch.outer(positions, inv_freq)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """Apply RoPE to ``[batch, heads, sequence, head_dim]`` queries or keys."""
        sequence_length = x.size(-2)
        cos = self.cos[:sequence_length].to(dtype=x.dtype, device=x.device)[None, None, :, :]
        sin = self.sin[:sequence_length].to(dtype=x.dtype, device=x.device)[None, None, :, :]
        even, odd = x[..., ::2], x[..., 1::2]
        rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
        return rotated.flatten(start_dim=-2)


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a lower-triangular causal mask."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads.")
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.rope = RotaryEmbedding(self.head_dim, config.context_length)
        self.register_buffer(
            "causal_mask",
            torch.ones(config.context_length, config.context_length, dtype=torch.bool).tril(),
            persistent=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        batch_size, sequence_length, hidden_size = x.shape
        qkv = self.qkv(x).view(batch_size, sequence_length, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = self.rope(query.transpose(1, 2))
        key = self.rope(key.transpose(1, 2))
        value = value.transpose(1, 2)

        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = self.causal_mask[:sequence_length, :sequence_length]
        scores = scores.masked_fill(~mask, float("-inf"))
        attention = F.softmax(scores, dim=-1, dtype=torch.float32).to(dtype=query.dtype)
        attention = self.dropout(attention)
        output = attention @ value
        output = output.transpose(1, 2).contiguous().view(batch_size, sequence_length, hidden_size)
        return self.out_proj(output)


class FeedForward(nn.Module):
    """The Transformer block's GELU MLP."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.mlp_hidden_size)
        self.fc2 = nn.Linear(config.mlp_hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Pre-RMSNorm Transformer block with attention and MLP residual paths."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config.hidden_size)
        self.mlp = FeedForward(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.attention_norm(x))
        return x + self.mlp(self.mlp_norm(x))


class SeaGlassTransformer(nn.Module):
    """An 8-layer causal language model that returns next-token logits.

    The input and output token embedding weights are tied, as specified in
    ``MODEL_SPEC.md``. Supply integer token IDs with shape ``[batch, sequence]``.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.token_embedding = nn.Embedding(self.config.vocabulary_size, self.config.hidden_size)
        self.blocks = nn.ModuleList(TransformerBlock(self.config) for _ in range(self.config.num_layers))
        self.final_norm = RMSNorm(self.config.hidden_size)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocabulary_size, bias=False)
        self.apply(self._initialize_weights)
        self.lm_head.weight = self.token_embedding.weight
        self._scale_residual_projections()

    def _initialize_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _scale_residual_projections(self) -> None:
        scale = 0.02 / math.sqrt(2 * self.config.num_layers)
        for name, parameter in self.named_parameters():
            if name.endswith("attention.out_proj.weight") or name.endswith("mlp.fc2.weight"):
                nn.init.normal_(parameter, mean=0.0, std=scale)

    @property
    def parameter_count(self) -> int:
        """Count unique trainable parameters (the tied output is counted once)."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(self, token_ids: Tensor) -> Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence].")
        _, sequence_length = token_ids.shape
        if sequence_length == 0:
            raise ValueError("token_ids must contain at least one token.")
        if sequence_length > self.config.context_length:
            raise ValueError(f"Sequence length {sequence_length} exceeds context length {self.config.context_length}.")
        x = self.token_embedding(token_ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x))
