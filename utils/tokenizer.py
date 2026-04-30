from __future__ import annotations

import json
from typing import Any, Protocol


class Tokenizer(Protocol):
    vocab_size: int

    def encode(self, s: str) -> list[int]: ...
    def decode(self, tokens: list[int]) -> str: ...
    def train_from_text(self, text: str, **kwargs: Any) -> None: ...
    def save(self, path: str) -> None: ...


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

    def save(self, path: str) -> None:
        payload = {"type": "char", "vocab": self.stoi}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)


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

    def save(self, path: str) -> None:
        payload = {
            "type": "bpe",
            "vocab_size": self.vocab_size,
            "tokenizer": self.tokenizer.to_str(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)


def get_tokenizer(tokenizer_type: str) -> Tokenizer:
    if tokenizer_type == "char":
        return CharTokenizer()
    elif tokenizer_type == "bpe":
        return BPETokenizer()
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")


def load_tokenizer_from_file(path: str) -> Tokenizer:
    """Load a tokenizer that was previously saved with ``tokenizer.save()``."""
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    tok_type = payload["type"]
    if tok_type == "char":
        char_tok = CharTokenizer()
        char_tok.stoi = {k: int(v) for k, v in payload["vocab"].items()}
        char_tok.itos = {int(v): k for k, v in payload["vocab"].items()}
        char_tok.vocab_size = len(char_tok.stoi)
        return char_tok
    if tok_type == "bpe":
        from tokenizers import Tokenizer as HFTokenizer
        bpe_tok = BPETokenizer()
        bpe_tok.tokenizer = HFTokenizer.from_str(payload["tokenizer"])
        bpe_tok.vocab_size = payload["vocab_size"]
        return bpe_tok
    raise ValueError(f"Unknown tokenizer type in file: {tok_type!r}")
