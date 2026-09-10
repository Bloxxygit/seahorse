"""Streaming document-to-token-sequence dataset pipeline."""

from .loader import Example, DatasetLoader
from .documents import DocumentProvider, JsonlDocumentSource, PlainTextDocumentSource, SourceDocument
from .filters import DocumentHook, FilterConfig, FilterStats, filter_documents
from .pipeline import DatasetConfig, PackedSequence, PackedSequenceDataset, build_sequence_pipeline
from .manifest import ALLOWED_LICENSES, ManifestConfig, write_corpus_manifest

__all__ = [
    "DatasetConfig",
    "DatasetLoader",
    "DocumentHook",
    "DocumentProvider",
    "Example",
    "FilterConfig",
    "FilterStats",
    "JsonlDocumentSource",
    "PackedSequence",
    "PackedSequenceDataset",
    "PlainTextDocumentSource",
    "SourceDocument",
    "build_sequence_pipeline",
    "filter_documents",
    "ALLOWED_LICENSES",
    "ManifestConfig",
    "write_corpus_manifest",
]
