"""Training-loop placeholder; no optimization is performed."""

from collections.abc import Sequence

from dataset import Example
from model import LanguageModel


class TrainingLoop:
    def run(self, model: LanguageModel, examples: Sequence[Example]) -> dict[str, int]:
        """Validate pipeline wiring and report zero training steps."""
        del model
        return {"examples_seen": len(examples), "steps_completed": 0}
