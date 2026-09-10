"""Fail-closed corpus provenance and deterministic manifest generation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from tokenizer import ByteLevelBPETokenizer

from .documents import SourceDocument
from .filters import DocumentHook, FilterConfig, FilterStats, filter_documents


ALLOWED_LICENSES = frozenset({"public-domain", "CC0-1.0", "CC-BY-4.0", "MIT", "BSD", "Apache-2.0"})


@dataclass(frozen=True)
class ManifestConfig:
    source_id: str
    domain: str
    license: str
    rights_evidence: str
    collection_date: str
    split_seed: int = 17


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_corpus_manifest(
    documents: Iterable[SourceDocument],
    output_jsonl: str | Path,
    output_manifest: str | Path,
    config: ManifestConfig,
    *,
    tokenizer: ByteLevelBPETokenizer | None = None,
    filter_config: FilterConfig | None = None,
    hooks: tuple[DocumentHook, ...] = (),
) -> dict[str, object]:
    """Write accepted documents and an auditable count/checksum manifest.

    Rights metadata is mandatory and licenses outside the allowlist fail closed.
    The source iterator should contain only one registered source at a time.
    """
    if config.license not in ALLOWED_LICENSES:
        raise ValueError(f"License is not allowlisted: {config.license}")
    if not config.rights_evidence.strip():
        raise ValueError("rights_evidence is required")
    destination = Path(output_jsonl)
    manifest_path = Path(output_manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stats = FilterStats()
    characters = tokens = 0
    with destination.open("w", encoding="utf-8") as handle:
        for document in filter_documents(documents, filter_config, hooks, stats):
            record = {
                "id": document.document_id,
                "text": document.text,
                "split": document.split,
                "metadata": dict(document.metadata),
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            characters += len(document.text)
            if tokenizer is not None:
                tokens += len(tokenizer.encode(document.text))
    # The filter API is streaming and intentionally does not materialize rejects;
    # report rejected counts supplied by the caller's source manifest when known.
    manifest: dict[str, object] = {
        "format_version": 1,
        "source": asdict(config),
        "input_documents": stats.input_documents,
        "accepted_documents": stats.accepted_documents,
        "rejected_documents": stats.input_documents - stats.accepted_documents,
        "rejections": {
            "quality": stats.rejected_quality,
            "hooks": stats.rejected_hooks,
            "duplicates": stats.rejected_duplicates,
        },
        "characters": characters,
        "tokens": tokens,
        "split_seed": config.split_seed,
        "output_sha256": _sha256(destination),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
