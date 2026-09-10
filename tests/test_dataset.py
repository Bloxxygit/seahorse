"""Tiny synthetic tests for the streaming SeaGlass dataset pipeline."""

import json
import tempfile
import unittest
from pathlib import Path

from dataset import (
    DatasetConfig,
    FilterConfig,
    JsonlDocumentSource,
    PlainTextDocumentSource,
    SourceDocument,
    filter_documents,
    ManifestConfig,
    write_corpus_manifest,
)
from dataset.pipeline import pack_documents, select_split
from tokenizer import ByteLevelBPETokenizer


class CountingProvider:
    def __init__(self, documents: list[SourceDocument]) -> None:
        self.documents = documents
        self.emitted = 0

    def __iter__(self):
        for document in self.documents:
            self.emitted += 1
            yield document


class DatasetPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = ByteLevelBPETokenizer()

    def test_packing_preserves_bos_eos_and_cross_document_boundary(self) -> None:
        documents = [SourceDocument("one", "A"), SourceDocument("two", "B")]
        sequence = next(pack_documents(documents, self.tokenizer, DatasetConfig(sequence_length=4)))
        bos = self.tokenizer.special_tokens["<bos>"]
        eos = self.tokenizer.special_tokens["<eos>"]
        self.assertEqual(sequence.input_ids, (bos, ord("A"), eos, bos))
        self.assertEqual(sequence.labels, (ord("A"), eos, bos, ord("B")))

    def test_filtering_removes_empty_duplicate_and_low_quality_documents(self) -> None:
        documents = [
            SourceDocument("empty", "   "),
            SourceDocument("good", "Useful technical explanation."),
            SourceDocument("duplicate", "Useful technical explanation."),
            SourceDocument("noise", "zzzzzzzzzz"),
        ]
        accepted = list(filter_documents(documents, FilterConfig(min_characters=3)))
        self.assertEqual([document.document_id for document in accepted], ["good"])

    def test_jsonl_and_plain_text_sources_keep_document_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jsonl_path = root / "tiny.jsonl"
            jsonl_path.write_text(
                json.dumps({"id": "json-1", "text": "first", "split": "validation"}) + "\nnot json\n",
                encoding="utf-8",
            )
            text_path = root / "tiny.txt"
            text_path.write_text("one line\nsecond line\n\nthird document\n", encoding="utf-8")
            json_documents = list(JsonlDocumentSource(jsonl_path))
            text_documents = list(PlainTextDocumentSource(text_path, split="train"))
        self.assertEqual([(document.document_id, document.split) for document in json_documents], [("json-1", "validation")])
        self.assertEqual([document.text for document in text_documents], ["one line\nsecond line", "third document"])

    def test_split_assignment_is_deterministic_and_disjoint(self) -> None:
        documents = [SourceDocument(f"doc-{index}", "content") for index in range(20)]
        config = DatasetConfig(validation_fraction=0.35, seed=9)
        train = [document.document_id for document in select_split(documents, "train", config)]
        validation = [document.document_id for document in select_split(documents, "validation", config)]
        self.assertEqual(train, [document.document_id for document in select_split(documents, "train", config)])
        self.assertFalse(set(train) & set(validation))
        self.assertEqual(set(train) | set(validation), {document.document_id for document in documents})

    def test_packing_consumes_documents_lazily(self) -> None:
        provider = CountingProvider([SourceDocument("one", "A"), SourceDocument("two", "B"), SourceDocument("three", "C")])
        iterator = pack_documents(provider, self.tokenizer, DatasetConfig(sequence_length=4))
        next(iterator)
        self.assertEqual(provider.emitted, 2)

    def test_manifest_records_rights_and_filter_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = write_corpus_manifest(
                [
                    SourceDocument("good", "Useful technical explanation."),
                    SourceDocument("duplicate", "Useful technical explanation."),
                    SourceDocument("empty", " "),
                ],
                root / "documents.jsonl",
                root / "manifest.json",
                ManifestConfig("source-1", "general", "CC0-1.0", "https://example.test/license", "2026-09-09"),
            )
        self.assertEqual(manifest["accepted_documents"], 1)
        self.assertEqual(manifest["rejections"]["duplicates"], 1)
        self.assertEqual(manifest["rejections"]["quality"], 1)


if __name__ == "__main__":
    unittest.main()
