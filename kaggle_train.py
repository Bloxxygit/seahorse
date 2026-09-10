"""Kaggle entrypoint for a rights-cleared SeaGlass pretraining run.

The notebook wrapper can call this module directly. All inputs are explicit so
the job never downloads a corpus or silently changes the tokenizer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import ModelConfig, TrainingConfig
from dataset import DatasetConfig, JsonlDocumentSource, PackedSequenceDataset, build_sequence_pipeline
from model import SeaGlassTransformer
from tokenizer import ByteLevelBPETokenizer
from training import Trainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SeaGlass training on an approved Kaggle input dataset.")
    parser.add_argument("--corpus", required=True, help="Approved JSONL corpus path.")
    parser.add_argument("--tokenizer", required=True, help="Frozen SeaGlass tokenizer JSON path.")
    parser.add_argument("--output", default="/kaggle/working/seaglass", help="Checkpoint and log directory.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=48_828, help="Approximate 200M-token budget at batch 8 and length 512.")
    parser.add_argument("--checkpoint-interval", type=int, default=250)
    parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    torch.manual_seed(args.seed)
    tokenizer = ByteLevelBPETokenizer.load(args.tokenizer)
    model_config = ModelConfig()
    if tokenizer.target_vocab_size != model_config.vocabulary_size:
        raise ValueError("Tokenizer target vocabulary does not match ModelConfig.vocabulary_size")

    dataset_config = DatasetConfig(sequence_length=model_config.context_length, seed=args.seed)
    source = JsonlDocumentSource(args.corpus)

    def make_loader(split: str) -> DataLoader:
        def sequences():
            return build_sequence_pipeline(source, tokenizer, split=split, config=dataset_config)

        dataset = PackedSequenceDataset(sequences)
        return DataLoader(dataset, batch_size=args.batch_size, num_workers=0, pin_memory=torch.cuda.is_available())

    train_loader = make_loader("train")
    validation_loader = make_loader("validation")
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=1,
        mixed_precision=torch.cuda.is_available(),
    )
    output = Path(args.output)
    trainer = Trainer(SeaGlassTransformer(model_config), training_config)
    trainer.fit(
        train_loader,
        max_optimizer_steps=args.max_steps,
        validation_batches=validation_loader,
        checkpoint_dir=output / "checkpoints",
        checkpoint_interval=args.checkpoint_interval,
        validation_interval=args.validation_interval,
        log_path=output / "metrics.jsonl",
    )


if __name__ == "__main__":
    main()

