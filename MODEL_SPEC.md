# SeaGlass AI: Initial Language-Model Specification

## Goal

The first SeaGlass model should be a small, decoder-only language model that
can be trained repeatedly on free Kaggle GPU sessions. It is an experimental
baseline, not a pretrained-model replacement: this repository will not
download weights or data, and this document does not introduce any ML
dependencies.

## Proposed architecture

Use a causal, decoder-only Transformer (GPT-style). Given preceding tokens,
it predicts the next token. The model consists of a token embedding table,
eight identical pre-normalized Transformer blocks, and a tied output
projection.

| Setting | Initial value | Rationale |
| --- | ---: | --- |
| Architecture | Decoder-only causal Transformer | The simplest well-understood architecture for next-token language modelling and generation. |
| Parameter target | about 42M | Large enough to expose real training behaviour while remaining inexpensive to iterate on. |
| Transformer layers | 8 | Provides meaningful depth without making experiment cycles impractical. |
| Hidden size (`d_model`) | 512 | A compact, GPU-friendly width with many future scaling multiples. |
| Attention heads | 8 | 64 dimensions per head, a conventional efficient head size. |
| Attention type | Multi-head causal self-attention | Each token can attend only to earlier tokens. |
| MLP width | 2,048 | Four times the hidden size, the standard dense Transformer expansion. |
| Context length | 512 tokens | Useful for initial language experiments while keeping attention memory low. |
| Vocabulary size | 32,000 tokens | A reasonable language-model vocabulary that avoids excessive embedding cost. |
| Position encoding | RoPE (rotary positional embeddings) | Adds no learned position-table parameters and extrapolates more naturally than learned absolute embeddings. |
| Normalization | Pre-RMSNorm | Stable for training and slightly simpler than LayerNorm. Apply before attention and MLP, plus a final RMSNorm. |
| Activation | GELU | A dependable baseline with broadly available implementations. |
| Output/input weight tying | Yes | Reuse the token embedding matrix as the output projection; reduces parameters and commonly improves small-model efficiency. |

### Tokenizer

Train a byte-level BPE tokenizer from the eventual training corpus, with a
32,000-token vocabulary and explicit special tokens such as `<bos>`, `<eos>`,
`<pad>`, and `<unk>`. Byte-level coverage means arbitrary Unicode text is
representable without an unknown-character failure. The tokenizer is trained
only when data and an appropriate tokenizer dependency are intentionally added;
no tokenizer or pretrained artifact is to be downloaded for this phase.

### Initialization

Initialize linear and embedding weights from a zero-mean normal distribution
with standard deviation `0.02`; initialize biases to zero. Scale residual
projection weights (the attention output and MLP output projections) by
`1 / sqrt(2 * n_layers)` to keep the residual stream well behaved at depth.
RMSNorm scale parameters begin at one.

## Parameter estimate

With tied input/output embeddings, the approximate count is:

| Component | Approximate parameters |
| --- | ---: |
| Token embeddings (`32,000 × 512`) | 16.38M |
| Eight attention blocks | 8.40M |
| Eight MLPs (`512 → 2,048 → 512`) | 16.79M |
| Norms and biases | 0.02M |
| **Total** | **about 41.6M** |

The tied output head adds no separate `32,000 × 512` matrix. The exact total
will vary slightly with implementation details such as linear-layer biases.

## Expected resource use

The inference/checkpoint footprint is small:

| Artifact | Expected size |
| --- | ---: |
| Model weights in FP32 | about 166 MB (roughly 160 MiB) |
| Model weights in FP16/BF16 | about 83 MB (roughly 80 MiB) |
| Checkpoint with AdamW states, FP32-equivalent | roughly 0.5–0.8 GB, before metadata and dataloader state |

For training on a 16 GB free Kaggle GPU (for example, a T4 or P100), expect
roughly **4–8 GB VRAM** for this model with mixed precision, AdamW, gradient
checkpointing disabled, and a conservative per-device micro-batch of 8–16 at
512 tokens. The exact peak depends substantially on framework kernels,
precision, sequence packing, and activation handling. Start with micro-batch
8, use gradient accumulation to reach the desired global batch size, and
measure peak allocation before increasing it. Gradient checkpointing provides
headroom if a larger context or batch is needed.

## Why this fits Kaggle

The 42M-parameter target is compact enough for 16 GB GPUs without exotic
memory techniques, so data, learning rate, loss curves, and tokenizer choices
can be iterated quickly. A 512-token context prevents quadratic attention cost
from dominating early experiments. Mixed-precision training and gradient
accumulation are standard rather than mandatory for the initial configuration.
The checkpoint is also small enough to save and version comfortably within a
notebook workflow.

## Scaling path

Keep the same tokenizer format, causal objective, RoPE, RMSNorm, and module
interfaces as the model grows. The first scaling step can increase depth to
12 layers and width to 768 (12 heads), then increase context to 1,024 tokens.
This preserves the training pipeline while increasing capacity. Larger runs
can later adopt grouped-query attention, FlashAttention-compatible kernels,
sequence packing, activation checkpointing, and multi-GPU training—but none
is required for the initial implementation.

## Non-goals for this phase

- No neural-network implementation.
- No dependency installation.
- No model, tokenizer, or dataset download.
- No training run.
