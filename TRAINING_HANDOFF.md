# SeaGlass AI — Training Handoff

Copy this document into the destination project, notebook, or planning tool. It
describes the SeaGlass AI baseline and its training constraints.

## Objective

Build and train SeaGlass AI, an experimental, decoder-only language model that
is practical to iterate on with a free Kaggle GPU. This is a from-scratch
baseline, not a replacement for a pretrained model. Keep the project
self-contained: do not download pretrained weights, tokenizers, or data
implicitly.

The model should be useful as a broad, conversational builder assistant with
particular strength in Roblox/Luau, real-time systems, game-engine mechanics,
debugging, networking, physics, and performance reasoning.

## Model requirements

Implement a causal GPT-style Transformer with these initial settings:

| Item | Required value |
| --- | --- |
| Parameter target | About 41.6M (roughly 42M) |
| Layers | 8 pre-normalized Transformer blocks |
| Hidden size | 512 |
| Attention heads | 8 (64 dimensions per head) |
| MLP width | 2,048 |
| Context length | 512 tokens |
| Vocabulary | 32,000 tokens |
| Positional encoding | RoPE |
| Normalization | Pre-RMSNorm, plus final RMSNorm |
| Activation | GELU |
| Weight tying | Tie input embedding and output projection |
| Objective | Next-token cross-entropy loss |

Initialize linear and embedding weights with a zero-mean normal distribution
of standard deviation `0.02`; initialize biases to zero. Scale attention-output
and MLP-output residual projections by `1 / sqrt(2 * n_layers)`. Initialize
RMSNorm scales to one.

## Tokenizer requirements

Use a locally trained byte-level BPE tokenizer with a target vocabulary of
32,000. It must encode arbitrary UTF-8 text and reserve these IDs:

| Token | ID | Purpose |
| --- | ---: | --- |
| `<pad>` | 256 | Batch padding |
| `<bos>` | 257 | Beginning of sequence |
| `<eos>` | 258 | End of sequence |
| `<unk>` | 259 | Compatibility reserve |

IDs `0` through `255` represent raw UTF-8 bytes. Train the tokenizer only on
approved local corpus files, then freeze it before model training.

```bash
python3 -m tokenizer.train \
  --input /path/to/corpus.txt /path/to/more-text/ \
  --output artifacts/seaglass-bpe-32000.json \
  --vocab-size 32000
```

## Training-data policy

Use only attributable, rights-cleared, high-signal sources. Keep a provenance
record containing source, license, collection date, quality score, and split.
Never include private conversations, personal data, credentials, or material
with unclear training permission.

Target corpus mixture:

- **20% Roblox and Luau technical content:** API and engine behavior, client/
  server boundaries, replication, remote validation, networking, physics,
  animation, rendering, terrain, pathfinding, profiling, memory, streaming,
  performance, debugging, tests, and robust code design.
- **80% broad capability content:** licensed general knowledge, reasoning,
  mathematics, science, stories, dialogue, programming, systems, graphics,
  physics, machine learning, and game-engine design.

Reject or heavily deprioritize clickbait, listicles, game popularity chatter,
SEO spam, scraped or duplicate pages, repetitive update news, low-quality
tutorials, and insecure cargo-cult remote-handling examples.

Before training: normalize text; remove boilerplate; deduplicate exact and
near duplicates; score depth, clarity, correctness, license status, and domain
fit; then split by source/document to prevent leakage. Maintain held-out
evaluation sets for Luau/API reasoning, client/server safety, replication,
physics, optimization, and technical-explanation quality.

## Dataset and batch contract

Stream local JSONL or blank-line-delimited plain-text documents rather than
loading the full corpus into RAM. A JSONL record must include `text` and may
include `id`, `split`, and metadata. Filter empty, malformed, duplicate, and
low-quality records. Use deterministic train/validation splits and bounded-
buffer deterministic train shuffling.

Wrap each document with `<bos>` and `<eos>`, then pack causal sequences of 512
tokens. Yield batches with integer tensors named `input_ids` and `labels`, each
shaped `[batch, sequence]`.

## Kaggle baseline run

Use AdamW parameter grouping, CUDA mixed precision when available, gradient
accumulation, gradient clipping, periodic validation, and checkpointing. Start
on a 16 GB Kaggle GPU with micro-batch size 8 and measure actual memory before
increasing it. A 42M-parameter model should generally require roughly 4–8 GB
VRAM under this baseline, but verify on the chosen hardware.

```bash
python3 kaggle_train.py \
  --corpus /kaggle/input/approved-corpus/documents.jsonl \
  --tokenizer /kaggle/input/seaglass/seaglass-bpe-32000.json \
  --output /kaggle/working/seaglass \
  --batch-size 8 \
  --gradient-accumulation-steps 1 \
  --max-steps 48828
```

The corpus and tokenizer paths must be supplied explicitly. Write checkpoints
and `metrics.jsonl` under the output directory. Do not silently fetch data or
models.

## Persona (apply separately from the knowledge corpus)

SeaGlass should sound like a collaborative builder and player: curious,
technically enthusiastic, direct, conversational, calm, clever, and
occasionally playful. Lead with the useful answer, then provide the reasoning
needed to act. Use plain language, concrete examples, and small testable next
steps. State uncertainty instead of inventing facts, APIs, or results.

In Roblox discussion, prioritize architecture, debugging, performance,
networking, and engine mechanics over popularity commentary. Humor is optional
and brief; never let it obscure safety, privacy, or technical accuracy.

Do not imitate a specific person, reproduce private details or conversations,
claim actions that did not occur, or mix persona examples into technical
training data. Version and evaluate the persona layer independently from the
knowledge corpus.

## Acceptance checks

```bash
python3 main.py
python3 -m unittest discover -s tests -v
```

The sanity check and tests must pass before a Kaggle run. The repository should
contain code and documentation only; do not commit corpora, checkpoints,
outputs, virtual environments, or Python cache files.
