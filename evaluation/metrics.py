"""Evaluation placeholder; metrics arrive with a real model and dataset."""

from collections.abc import Sequence

from dataset import Example
from model import LanguageModel


class Evaluator:
    def evaluate(self, model: LanguageModel, examples: Sequence[Example]) -> dict[str, float]:
        del model, examples
        return {"placeholder_score": 0.0}
