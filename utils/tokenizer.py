from __future__ import annotations

from typing import Any, Protocol


class Tokenizer(Protocol):
    vocab_size: int

    def encode(self, s: str) -> list[int]: ...
    def decode(self, tokens: list[int]) -> str: ...
    def train_from_text(self, text: str, **kwargs: Any) -> None: ...


class CharTokenizer:
    def __init__(self) -> None:
        self.stoi: dict[str, int] = {}
        self.itos: dict[int, str] = {}
        self.vocab_size: int = 0

    def train_from_text(self, text: str, **kwargs: Any) -> None:
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, tokens: list[int]) -> str:
        return ''.join(self.itos[i] for i in tokens)


class BPETokenizer:
    def __init__(self) -> None:
        from tokenizers import Tokenizer, models, pre_tokenizers
        self.tokenizer = Tokenizer(models.BPE())
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
        self.vocab_size: int = 0

    def train_from_text(self, text: str, vocab_size: int = 1000, **kwargs: Any) -> None:
        from tokenizers import trainers
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
        )
        self.tokenizer.train_from_iterator([text], trainer=trainer)
        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, s: str) -> list[int]:
        return self.tokenizer.encode(s).ids

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)


def get_tokenizer(tokenizer_type: str) -> Tokenizer:
    if tokenizer_type == "char":
        return CharTokenizer()
    elif tokenizer_type == "bpe":
        return BPETokenizer()
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
