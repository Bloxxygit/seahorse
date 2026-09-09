"""A small, dependency-free byte-level BPE tokenizer.

The serialized format contains only special-token metadata and merge pairs, so
tokenizers are reproducible and do not depend on a third-party runtime.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
import json
from pathlib import Path
import re

from .base import Tokenizer


DEFAULT_SPECIAL_TOKENS = {
    "<pad>": 256,
    "<bos>": 257,
    "<eos>": 258,
    "<unk>": 259,
}
BASE_VOCAB_SIZE = 256
FORMAT_VERSION = 1


def _pretokenize(text: str) -> list[tuple[int, ...]]:
    """Split into whitespace/non-whitespace chunks, then represent UTF-8 bytes."""
    return [tuple(chunk.encode("utf-8")) for chunk in re.findall(r"\S+|\s+", text)]


def _replace_pair(tokens: tuple[int, ...], pair: tuple[int, int], replacement: int) -> tuple[int, ...]:
    result: list[int] = []
    index = 0
    while index < len(tokens):
        if index + 1 < len(tokens) and (tokens[index], tokens[index + 1]) == pair:
            result.append(replacement)
            index += 2
        else:
            result.append(tokens[index])
            index += 1
    return tuple(result)


class ByteLevelBPETokenizer(Tokenizer):
    """UTF-8 byte BPE with explicitly reserved special-token IDs.

    IDs 0--255 always represent individual bytes. IDs 256--259 are reserved
    for ``<pad>``, ``<bos>``, ``<eos>``, and ``<unk>``. Learned merge-token IDs
    begin at 260 and follow the order in ``merges``.
    """

    def __init__(
        self,
        merges: Sequence[tuple[int, int]] = (),
        special_tokens: dict[str, int] | None = None,
        target_vocab_size: int = 32_000,
    ) -> None:
        self.special_tokens = dict(special_tokens or DEFAULT_SPECIAL_TOKENS)
        self._validate_special_tokens()
        self.merges = [tuple(pair) for pair in merges]
        self.target_vocab_size = target_vocab_size
        self._first_merge_id = max(self.special_tokens.values()) + 1
        self._merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        if len(self._merge_ranks) != len(self.merges):
            raise ValueError("Merge pairs must be unique.")

        self._token_bytes: dict[int, bytes] = {token_id: bytes([token_id]) for token_id in range(BASE_VOCAB_SIZE)}
        for offset, (left, right) in enumerate(self.merges):
            token_id = self._first_merge_id + offset
            if left not in self._token_bytes or right not in self._token_bytes:
                raise ValueError("A merge may refer only to an earlier byte or merge token.")
            self._token_bytes[token_id] = self._token_bytes[left] + self._token_bytes[right]

        escaped = "|".join(re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True))
        self._special_pattern = re.compile(f"({escaped})") if escaped else None
        self._special_by_id = {token_id: token for token, token_id in self.special_tokens.items()}

    def _validate_special_tokens(self) -> None:
        ids = list(self.special_tokens.values())
        if not self.special_tokens or len(ids) != len(set(ids)):
            raise ValueError("Special tokens must be non-empty and have unique IDs.")
        if min(ids) < BASE_VOCAB_SIZE:
            raise ValueError("Special-token IDs must not overlap byte IDs 0--255.")

    @property
    def vocab_size(self) -> int:
        """Actual number of usable token IDs, including reserved special tokens."""
        return self._first_merge_id + len(self.merges)

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text:
            return []

        segments = self._special_pattern.split(text) if self._special_pattern else [text]
        encoded: list[int] = []
        for segment in segments:
            if not segment:
                continue
            special_id = self.special_tokens.get(segment)
            if special_id is not None:
                encoded.append(special_id)
                continue
            for byte_tokens in _pretokenize(segment):
                encoded.extend(self._apply_merges(byte_tokens))
        return encoded

    def _apply_merges(self, tokens: tuple[int, ...]) -> tuple[int, ...]:
        tokens = tuple(tokens)
        while len(tokens) > 1:
            ranked_pairs = [
                (self._merge_ranks[pair], pair)
                for pair in zip(tokens, tokens[1:])
                if pair in self._merge_ranks
            ]
            if not ranked_pairs:
                break
            _, pair = min(ranked_pairs)
            replacement = self._first_merge_id + self._merge_ranks[pair]
            tokens = _replace_pair(tokens, pair, replacement)
        return tokens

    def decode(self, token_ids: Sequence[int]) -> str:
        output: list[str] = []
        byte_buffer = bytearray()
        for token_id in token_ids:
            if token_id in self._special_by_id:
                if byte_buffer:
                    output.append(bytes(byte_buffer).decode("utf-8", errors="replace"))
                    byte_buffer.clear()
                output.append(self._special_by_id[token_id])
            elif token_id in self._token_bytes:
                byte_buffer.extend(self._token_bytes[token_id])
            else:
                raise ValueError(f"Unknown token ID: {token_id}")
        if byte_buffer:
            output.append(bytes(byte_buffer).decode("utf-8", errors="replace"))
        return "".join(output)

    def save(self, path: str | Path) -> None:
        """Save the learned merge list and metadata as portable JSON."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": FORMAT_VERSION,
            "type": "byte_level_bpe",
            "target_vocab_size": self.target_vocab_size,
            "special_tokens": self.special_tokens,
            "merges": [list(pair) for pair in self.merges],
        }
        destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ByteLevelBPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format_version") != FORMAT_VERSION or payload.get("type") != "byte_level_bpe":
            raise ValueError("Unsupported tokenizer file format.")
        merges = [tuple(pair) for pair in payload["merges"]]
        special_tokens = {token: int(token_id) for token, token_id in payload["special_tokens"].items()}
        return cls(merges, special_tokens, int(payload["target_vocab_size"]))

    @classmethod
    def train(cls, texts: Iterable[str], target_vocab_size: int = 32_000) -> "ByteLevelBPETokenizer":
        """Learn BPE merges from text supplied by the caller.

        Training is intentionally simple and deterministic. It aggregates equal
        pre-tokenized chunks before each merge, which keeps small and medium
        local-corpus experiments practical without external dependencies.
        """
        minimum_vocab = max(DEFAULT_SPECIAL_TOKENS.values()) + 1
        if target_vocab_size < minimum_vocab:
            raise ValueError(f"target_vocab_size must be at least {minimum_vocab}")

        chunks: Counter[tuple[int, ...]] = Counter()
        for text in texts:
            if not isinstance(text, str):
                raise TypeError("All training texts must be strings")
            chunks.update(_pretokenize(text))

        merges: list[tuple[int, int]] = []
        next_token_id = minimum_vocab
        while next_token_id < target_vocab_size:
            pair_counts: Counter[tuple[int, int]] = Counter()
            for chunk, frequency in chunks.items():
                pair_counts.update({pair: frequency for pair in zip(chunk, chunk[1:])})
            if not pair_counts:
                break

            pair = min(pair_counts, key=lambda candidate: (-pair_counts[candidate], candidate))
            merges.append(pair)
            updated_chunks: Counter[tuple[int, ...]] = Counter()
            for chunk, frequency in chunks.items():
                updated_chunks[_replace_pair(chunk, pair, next_token_id)] += frequency
            chunks = updated_chunks
            next_token_id += 1

        return cls(merges=merges, target_vocab_size=target_vocab_size)
