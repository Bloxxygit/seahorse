"""Lightweight unit tests for the training pipeline using a tiny model only."""

import unittest

import torch

from config import ModelConfig, TrainingConfig
from model import SeaGlassTransformer
from training import Trainer, build_adamw


class TrainerTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(3)
        model_config = ModelConfig(
            vocabulary_size=64,
            hidden_size=16,
            num_layers=1,
            num_attention_heads=4,
            mlp_hidden_size=32,
            context_length=8,
        )
        training_config = TrainingConfig(
            learning_rate=1e-3,
            mixed_precision=False,
            gradient_accumulation_steps=1,
        )
        self.model = SeaGlassTransformer(model_config)
        self.trainer = Trainer(self.model, training_config, device="cpu")
        self.batch = {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "labels": torch.tensor([[2, 3, 4]], dtype=torch.long),
        }

    def test_train_batch_performs_one_optimizer_step(self) -> None:
        before = self.model.token_embedding.weight.detach().clone()
        metrics = self.trainer.train_batch(self.batch)
        self.assertGreater(metrics.loss, 0.0)
        self.assertTrue(metrics.optimizer_step)
        self.assertEqual(metrics.global_step, 1)
        self.assertFalse(torch.equal(before, self.model.token_embedding.weight.detach()))

    def test_invalid_batch_is_rejected(self) -> None:
        with self.assertRaises(KeyError):
            self.trainer.train_batch({"input_ids": self.batch["input_ids"]})
        with self.assertRaises(ValueError):
            self.trainer.train_batch({"input_ids": self.batch["input_ids"], "labels": torch.ones((1, 2), dtype=torch.long)})

    def test_adamw_excludes_normalization_scales_from_weight_decay(self) -> None:
        optimizer = build_adamw(self.model, self.trainer.config)
        self.assertEqual([group["weight_decay"] for group in optimizer.param_groups], [0.1, 0.0])

    def test_partial_accumulation_resets_after_final_step(self) -> None:
        config = TrainingConfig(mixed_precision=False, gradient_accumulation_steps=2)
        trainer = Trainer(self.model, config, device="cpu")
        self.assertFalse(trainer.train_batch(self.batch).optimizer_step)
        self.assertTrue(trainer.finish_accumulation())
        self.assertEqual(trainer.global_step, 1)
        self.assertFalse(trainer.train_batch(self.batch).optimizer_step)


if __name__ == "__main__":
    unittest.main()
