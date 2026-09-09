"""Tests for the dependency-free byte-level BPE tokenizer."""

import tempfile
import unittest
from pathlib import Path

from tokenizer import ByteLevelBPETokenizer, DEFAULT_SPECIAL_TOKENS


class ByteLevelBPETokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = ByteLevelBPETokenizer.train(
            ["SeaGlass AI learns from local text.\n", "Repeated words words words."],
            target_vocab_size=280,
        )

    def assert_round_trip(self, text: str) -> None:
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), text)

    def test_ascii_round_trip(self) -> None:
        self.assert_round_trip("SeaGlass AI")

    def test_unicode_round_trip(self) -> None:
        self.assert_round_trip("café — 海 glass 🐚")

    def test_whitespace_round_trip(self) -> None:
        self.assert_round_trip("  tabs\tand\nnewlines  ")

    def test_empty_string(self) -> None:
        self.assertEqual(self.tokenizer.encode(""), [])
        self.assertEqual(self.tokenizer.decode([]), "")

    def test_special_tokens_are_stable(self) -> None:
        text = "<bos>Hello<eos>"
        self.assertEqual(self.tokenizer.encode(text)[0], DEFAULT_SPECIAL_TOKENS["<bos>"])
        self.assert_round_trip(text)

    def test_save_and_load_preserve_encoding(self) -> None:
        text = "Saved tokenizer: café <eos>"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seaglass_tokenizer.json"
            self.tokenizer.save(path)
            restored = ByteLevelBPETokenizer.load(path)
        self.assertEqual(restored.encode(text), self.tokenizer.encode(text))
        self.assertEqual(restored.decode(restored.encode(text)), text)

    def test_unknown_token_id_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.tokenizer.decode([99_999])


if __name__ == "__main__":
    unittest.main()
