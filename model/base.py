"""Framework-agnostic model contracts."""

from abc import ABC, abstractmethod
from collections.abc import Sequence


class LanguageModel(ABC):
    """Minimal contract to implement once a model backend is chosen."""

    @abstractmethod
    def predict_next(self, token_ids: Sequence[int]) -> int:
        """Return the next token ID for a sequence."""


class PlaceholderModel(LanguageModel):
    """Non-ML model used solely to prove that components connect."""

    def __init__(self, vocabulary_size: int) -> None:
        self.vocabulary_size = vocabulary_size

    def predict_next(self, token_ids: Sequence[int]) -> int:
        return token_ids[-1] if token_ids else 0
