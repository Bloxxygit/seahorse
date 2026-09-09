# SeaGlass AI

SeaGlass AI is a compact language-model project. It contains a local tokenizer,
a PyTorch Transformer, and reusable training utilities, but intentionally has
no downloaded models or training data.

## Quick start

From this directory, run:

```bash
python3 main.py
```

The command constructs the placeholder pipeline and prints a successful sanity
check. It does not train or download anything.

## Layout

- `config/` — one central settings module for model and training defaults.
- `model/` — the causal decoder-only PyTorch Transformer and parameter report.
- `tokenizer/` — tokenizer interface, byte-level BPE implementation, and local training CLI.
- `dataset/` — dataset record type and loader interface.
- `training/` — AdamW, mixed precision, gradient accumulation, and clipping utilities.
- `inference/` — inference interface and a placeholder response path.
- `evaluation/` — evaluator interface and an empty-result implementation.
- `main.py` — dependency-free end-to-end sanity check.

## Training pipeline

`training.Trainer` accepts a future iterable of batches containing
`input_ids` and `labels`, both integer tensors shaped `[batch, sequence]`.
It supplies AdamW parameter grouping, CUDA mixed precision, gradient
accumulation, gradient clipping, and cross-entropy next-token loss. It does
not load a dataset or save checkpoints itself.

The full 41.6M-parameter model is intended for Kaggle GPU execution. Do not
invoke `Trainer.fit()` for full runs on a local Chromebook. A future Kaggle
entrypoint can construct `SeaGlassTransformer`, prepare an iterable of batches,
and explicitly call `Trainer.fit()` there.

## Dataset pipeline

The dataset package streams local JSONL or blank-line-delimited plain-text
documents; it never loads an entire corpus into RAM. JSONL records require a
`text` field and may include `id`, `split`, and arbitrary metadata. Future
corpus providers—including separately curated Roblox technical and general
knowledge sources—only need to yield `SourceDocument` objects, so the training
loop remains unchanged.

The pipeline filters empty, duplicate, malformed, and basic low-quality
documents; accepts custom filtering hooks; assigns deterministic train and
validation splits; performs bounded-buffer deterministic train shuffling; and
adds `<bos>`/`<eos>` around every document before packing causal 512-token
sequences. It yields `input_ids`/next-token `labels` ready for `Trainer`.

Example Kaggle notebook wiring (after providing approved local Kaggle input
files and a trained SeaGlass tokenizer):

```python
from torch.utils.data import DataLoader
from dataset import DatasetConfig, JsonlDocumentSource, PackedSequenceDataset, build_sequence_pipeline
from tokenizer import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer.load("/kaggle/input/seaglass/tokenizer.json")
source = JsonlDocumentSource("/kaggle/input/approved-corpus/documents.jsonl")
config = DatasetConfig(sequence_length=512, seed=17)

def train_sequences():
    return build_sequence_pipeline(source, tokenizer, split="train", config=config)

loader = DataLoader(PackedSequenceDataset(train_sequences), batch_size=8)
# Pass `loader` to Trainer.fit(...) only in the Kaggle training job.
```

## Tokenizer

SeaGlass includes a dependency-free byte-level BPE tokenizer. It learns merge
rules only from text files you supply; it never downloads a pretrained
tokenizer or model. The target vocabulary is 32,000 tokens. IDs `0`--`255`
represent raw UTF-8 bytes, while these special tokens are reserved:

| Token | ID | Purpose |
| --- | ---: | --- |
| `<pad>` | 256 | Padding for batches of unequal length. |
| `<bos>` | 257 | Beginning of sequence. |
| `<eos>` | 258 | End of sequence. |
| `<unk>` | 259 | Reserved compatibility token; byte encoding itself does not require unknown tokens. |

Train from individual local text files or directories (directories are searched
recursively for `.txt` files):

```bash
python3 -m tokenizer.train \
  --input /path/to/corpus.txt /path/to/more-text/ \
  --output artifacts/seaglass-bpe-32000.json \
  --vocab-size 32000
```

Use a saved tokenizer from Python:

```python
from tokenizer import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer.load("artifacts/seaglass-bpe-32000.json")
token_ids = tokenizer.encode("<bos>Hello, SeaGlass!<eos>")
assert tokenizer.decode(token_ids) == "<bos>Hello, SeaGlass!<eos>"
```

The JSON tokenizer file normally needs well under 1 MB at 32,000 tokens;
roughly 0.3--1 MB is a sensible planning estimate, depending on formatting and
the number of merges actually supported by the corpus. Run its tests with:

```bash
python3 -m unittest discover -s tests -v
```
