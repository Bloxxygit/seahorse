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
        if not basic_quality_filter(document, config) or not all(hook(document) for hook in hooks):
            continue
        fingerprint = hashlib.blake2b(document.text.strip().encode("utf-8"), digest_size=16).digest()
        if config.deduplicate and fingerprint in seen:
            continue
        if config.deduplicate:
            seen.add(fingerprint)
        yield document
