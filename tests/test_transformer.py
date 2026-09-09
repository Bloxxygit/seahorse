"""Unit tests for the SeaGlass causal decoder-only Transformer."""

import unittest

import torch

from config import ModelConfig
from model import SeaGlassTransformer


class SeaGlassTransformerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(7)
        cls.config = ModelConfig()
        cls.model = SeaGlassTransformer(cls.config).eval()

    def test_forward_returns_next_token_logits(self) -> None:
        token_ids = torch.randint(0, self.config.vocabulary_size, (2, 11))
        logits = self.model(token_ids)
        self.assertEqual(tuple(logits.shape), (2, 11, self.config.vocabulary_size))

    def test_causal_attention_cannot_see_future_tokens(self) -> None:
        prefix = torch.tensor([[11, 12, 13, 14]])
        changed_future = torch.tensor([[11, 12, 13, 29_999]])
        with torch.no_grad():
            prefix_logits = self.model(prefix)
            changed_logits = self.model(changed_future)
        torch.testing.assert_close(prefix_logits[:, :-1], changed_logits[:, :-1], rtol=0, atol=1e-6)

    def test_tied_input_and_output_embeddings(self) -> None:
        self.assertEqual(self.model.token_embedding.weight.data_ptr(), self.model.lm_head.weight.data_ptr())

    def test_parameter_count_matches_specification(self) -> None:
        self.assertEqual(self.model.parameter_count, 41_595_392)
        self.assertLess(abs(self.model.parameter_count - 41_600_000), 100_000)

    def test_context_limit_is_enforced(self) -> None:
        token_ids = torch.zeros((1, self.config.context_length + 1), dtype=torch.long)
        with self.assertRaises(ValueError):
            self.model(token_ids)


if __name__ == "__main__":
    unittest.main()
