"""Command-line training for a local SeaGlass byte-level BPE tokenizer."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from .bpe import ByteLevelBPETokenizer


def _text_files(inputs: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for raw_path in inputs:
        path = Path(raw_path)
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(candidate for candidate in path.rglob("*.txt") if candidate.is_file()))
        else:
            raise FileNotFoundError(f"Input path does not exist: {path}")
    if not files:
        raise ValueError("No text files found. Supply files or directories containing .txt files.")
    return files


def _read_texts(paths: Iterable[Path]) -> Iterable[str]:
    for path in paths:
        yield path.read_text(encoding="utf-8", errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a SeaGlass byte-level BPE tokenizer from local text files.")
    parser.add_argument("--input", nargs="+", required=True, help="Text files or directories of .txt files.")
    parser.add_argument("--output", required=True, help="Destination tokenizer JSON path.")
    parser.add_argument("--vocab-size", type=int, default=32_000, help="Target vocabulary size (default: 32000).")
    args = parser.parse_args()

    files = _text_files(args.input)
    tokenizer = ByteLevelBPETokenizer.train(_read_texts(files), args.vocab_size)
    tokenizer.save(args.output)
    print(f"Saved tokenizer with {tokenizer.vocab_size} tokens to {args.output} (requested {args.vocab_size}).")


if __name__ == "__main__":
    main()
