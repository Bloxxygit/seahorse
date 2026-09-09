"""Streaming document providers and a provider-neutral document contract."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class SourceDocument:
    """One attributable source document before filtering or tokenization."""

    document_id: str
    text: str
    split: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


class DocumentProvider(Protocol):
    """Interface future Roblox and general-corpus providers must implement."""

    def __iter__(self) -> Iterator[SourceDocument]: ...


class JsonlDocumentSource:
    """Stream JSONL records containing a required string ``text`` field.

    Optional ``id`` and ``split`` fields are preserved. Malformed records and
    records without string text are skipped by default rather than crashing a
    long Kaggle preprocessing job.
    """

    def __init__(self, path: str | Path, split: str | None = None) -> None:
        self.path = Path(path)
        self.split = split

    def __iter__(self) -> Iterator[SourceDocument]:
        with self.path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                    continue
                document_id = str(record.get("id", f"{self.path.name}:{line_number}"))
                record_split = record.get("split", self.split)
                split = str(record_split) if record_split in {"train", "validation"} else self.split
                metadata = {key: value for key, value in record.items() if key not in {"id", "text", "split"}}
                yield SourceDocument(document_id, record["text"], split, metadata)


class PlainTextDocumentSource:
    """Stream blank-line-delimited documents from a UTF-8 plain-text file."""

    def __init__(self, path: str | Path, split: str | None = None) -> None:
        self.path = Path(path)
        self.split = split

    def __iter__(self) -> Iterator[SourceDocument]:
        document_lines: list[str] = []
        document_number = 0
        with self.path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.strip():
                    document_lines.append(line)
                    continue
                if document_lines:
                    document_number += 1
                    text = "".join(document_lines).strip()
                    yield SourceDocument(f"{self.path.name}:{document_number}", text, self.split)
                    document_lines.clear()
        if document_lines:
            document_number += 1
            yield SourceDocument(f"{self.path.name}:{document_number}", "".join(document_lines).strip(), self.split)
