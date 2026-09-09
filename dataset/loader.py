"""Dataset contracts; no files are read by this starter implementation."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Example:
    text: str


class DatasetLoader(Protocol):
    def load(self) -> list[Example]:
        """Load examples from a future data source."""
