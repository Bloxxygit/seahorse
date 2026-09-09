# SeaGlass AI

SeaGlass AI is a minimal, dependency-free scaffold for experimenting with a
language-model-style system. It intentionally contains no downloaded models,
training data, or machine-learning framework yet.

## Quick start

From this directory, run:

```bash
python3 main.py
```

The command constructs the placeholder pipeline and prints a successful sanity
check. It does not train or download anything.

## Layout

- `config/` — one central settings module for model and training defaults.
- `model/` — the model interface and a deliberately unimplemented baseline.
- `tokenizer/` — tokenizer interface plus a small character-level placeholder.
- `dataset/` — dataset record type and loader interface.
- `training/` — training-loop interface; it validates inputs but never trains.
- `inference/` — inference interface and a placeholder response path.
- `evaluation/` — evaluator interface and an empty-result implementation.
- `main.py` — dependency-free end-to-end sanity check.

When an ML framework is selected later, replace the placeholder implementations
while preserving these small interfaces.
