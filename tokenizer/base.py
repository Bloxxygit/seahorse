"""Small tokenizer interface with a dependency-free placeholder."""

from abc import ABC, abstractmethod
from collections.abc import Sequence


class Tokenizer(ABC):
    @abstractmethod
    def encode(self, text: str) -> list[int]: ...

    @abstractmethod
    def decode(self, token_ids: Sequence[int]) -> str: ...


class CharacterTokenizer(Tokenizer):
    """UTF-8-byte fallback retained for the initial application sanity check."""

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_ids: Sequence[int]) -> str:
        return bytes(token_ids).decode("utf-8", errors="replace")
