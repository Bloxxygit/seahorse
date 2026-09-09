"""Simple inference adapter."""

from model import LanguageModel
from tokenizer import Tokenizer


class InferenceService:
    def __init__(self, model: LanguageModel, tokenizer: Tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def generate_one_token(self, prompt: str) -> str:
        token_ids = self.tokenizer.encode(prompt)
        next_id = self.model.predict_next(token_ids)
        return self.tokenizer.decode([next_id])
