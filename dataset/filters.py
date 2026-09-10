"""Composable streaming quality filters for source documents."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
import hashlib

from .documents import SourceDocument


DocumentHook = Callable[[SourceDocument], bool]


@dataclass(frozen=True)
class FilterConfig:
    min_characters: int = 1
    max_repeated_character_fraction: float = 0.90
    deduplicate: bool = True


@dataclass
class FilterStats:
    input_documents: int = 0
    rejected_quality: int = 0
    rejected_hooks: int = 0
    rejected_duplicates: int = 0
    accepted_documents: int = 0


def basic_quality_filter(document: SourceDocument, config: FilterConfig) -> bool:
    """Reject empty/short text and near-single-character low-quality content."""
    stripped = document.text.strip()
    if len(stripped) < config.min_characters:
        return False
    most_common_count = max((stripped.count(character) for character in set(stripped)), default=0)
    return most_common_count / len(stripped) <= config.max_repeated_character_fraction


def filter_documents(
    documents: Iterable[SourceDocument],
    config: FilterConfig | None = None,
    hooks: Sequence[DocumentHook] = (),
    stats: FilterStats | None = None,
) -> Iterator[SourceDocument]:
    """Yield quality-approved documents one at a time.

    Custom hooks can enforce source licenses, Roblox technical-topic relevance,
    or any future provider-specific policy. Exact deduplication uses a local
    hash set; for huge corpora callers can disable it and substitute an
    external/scalable deduplication hook.
    """
    config = config or FilterConfig()
    seen: set[bytes] = set()
    for document in documents:
        if stats is not None:
            stats.input_documents += 1
        if not basic_quality_filter(document, config):
            if stats is not None:
                stats.rejected_quality += 1
            continue
        if not all(hook(document) for hook in hooks):
            if stats is not None:
                stats.rejected_hooks += 1
            continue
        fingerprint = hashlib.blake2b(document.text.strip().encode("utf-8"), digest_size=16).digest()
        if config.deduplicate and fingerprint in seen:
            if stats is not None:
                stats.rejected_duplicates += 1
            continue
        if config.deduplicate:
            seen.add(fingerprint)
        if stats is not None:
            stats.accepted_documents += 1
        yield document
