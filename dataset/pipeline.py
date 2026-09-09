"""Streaming split, sampling, token packing, and PyTorch dataset adapters."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
import hashlib
import random

import torch
from torch.utils.data import IterableDataset

from tokenizer import ByteLevelBPETokenizer

from .documents import DocumentProvider, SourceDocument
from .filters import DocumentHook, FilterConfig, filter_documents


@dataclass(frozen=True)
class DatasetConfig:
    sequence_length: int = 512
    validation_fraction: float = 0.02
    seed: int = 17
    shuffle_buffer_size: int = 10_000
    sample_fraction: float = 1.0
    include_final_partial_sequence: bool = False
    filter_config: FilterConfig = field(default_factory=FilterConfig)


@dataclass(frozen=True)
class PackedSequence:
    """A fixed-length causal-LM pair suitable for ``training.Trainer``."""

    input_ids: tuple[int, ...]
    labels: tuple[int, ...]


def _stable_fraction(value: str, seed: int) -> float:
    digest = hashlib.blake2b(f"{seed}:{value}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def assign_split(document: SourceDocument, validation_fraction: float, seed: int) -> str:
    """Respect an explicit split or deterministically assign train/validation."""
    if document.split in {"train", "validation"}:
        return document.split
    return "validation" if _stable_fraction(document.document_id, seed) < validation_fraction else "train"


def select_split(documents: Iterable[SourceDocument], split: str, config: DatasetConfig) -> Iterator[SourceDocument]:
    if split not in {"train", "validation"}:
        raise ValueError("split must be 'train' or 'validation'.")
    if not 0.0 <= config.validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1).")
    for document in documents:
        if assign_split(document, config.validation_fraction, config.seed) == split:
            yield document


def deterministic_sample(documents: Iterable[SourceDocument], fraction: float, seed: int) -> Iterator[SourceDocument]:
    """Keep a stable hash-selected sample without materializing the corpus."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("sample_fraction must be in (0, 1].")
    for document in documents:
        if _stable_fraction(document.document_id, seed) < fraction:
            yield document


def buffered_shuffle(documents: Iterable[SourceDocument], buffer_size: int, seed: int) -> Iterator[SourceDocument]:
    """Deterministically shuffle with bounded memory, suitable for streams."""
    if buffer_size < 1:
        raise ValueError("shuffle_buffer_size must be positive.")
    rng = random.Random(seed)
    buffer: list[SourceDocument] = []
    for document in documents:
        if len(buffer) < buffer_size:
            buffer.append(document)
            continue
        index = rng.randrange(buffer_size)
        yield buffer[index]
        buffer[index] = document
    rng.shuffle(buffer)
    yield from buffer


def pack_documents(
    documents: Iterable[SourceDocument],
    tokenizer: ByteLevelBPETokenizer,
    config: DatasetConfig,
) -> Iterator[PackedSequence]:
    """Add BOS/EOS to every document and pack a continuous causal token stream."""
    if config.sequence_length < 1:
        raise ValueError("sequence_length must be positive.")
    special = tokenizer.special_tokens
    try:
        bos, eos, pad = special["<bos>"], special["<eos>"], special["<pad>"]
    except KeyError as error:
        raise ValueError("Tokenizer must define <bos>, <eos>, and <pad> special tokens.") from error

    tokens: list[int] = []
    needed = config.sequence_length + 1
    for document in documents:
        tokens.extend((bos, *tokenizer.encode(document.text), eos))
        while len(tokens) >= needed:
            yield PackedSequence(tuple(tokens[: config.sequence_length]), tuple(tokens[1:needed]))
            del tokens[: config.sequence_length]

    if config.include_final_partial_sequence and len(tokens) >= 2:
        input_ids = tokens[:-1]
        labels = tokens[1:]
        padding = config.sequence_length - len(input_ids)
        yield PackedSequence(
            tuple(input_ids + [pad] * padding),
            tuple(labels + [-100] * padding),
        )


def build_sequence_pipeline(
    provider: DocumentProvider,
    tokenizer: ByteLevelBPETokenizer,
    split: str,
    config: DatasetConfig | None = None,
    hooks: tuple[DocumentHook, ...] = (),
) -> Iterator[PackedSequence]:
    """Build a lazy provider-to-token-sequence pipeline for one split."""
    config = config or DatasetConfig()
    documents: Iterable[SourceDocument] = filter_documents(provider, config.filter_config, hooks)
    documents = select_split(documents, split, config)
    documents = deterministic_sample(documents, config.sample_fraction, config.seed)
    if split == "train":
        documents = buffered_shuffle(documents, config.shuffle_buffer_size, config.seed)
    return pack_documents(documents, tokenizer, config)


class PackedSequenceDataset(IterableDataset[dict[str, torch.Tensor]]):
    """PyTorch/Kaggle adapter; factory keeps streams re-iterable per worker."""

    def __init__(self, sequence_factory: Callable[[], Iterable[PackedSequence]]) -> None:
        super().__init__()
        self.sequence_factory = sequence_factory

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        for sequence in self.sequence_factory():
            yield {
                "input_ids": torch.tensor(sequence.input_ids, dtype=torch.long),
                "labels": torch.tensor(sequence.labels, dtype=torch.long),
            }
