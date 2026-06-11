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
            special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]"]
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


class T5SentencePieceTokenizer:
    """T5's pre-trained SentencePiece tokenizer (vocab_size=32128).

    Token IDs from this tokenizer correctly index into the T5 embedding matrix,
    making it the only valid choice for ELF which uses frozen T5 embeddings.
    No training is needed — the vocabulary is fixed by the T5 model.
    """

    def __init__(self, model_name: str = "t5-base") -> None:
        self.model_name = model_name
        self._tok = self._load(model_name)

    @staticmethod
    def _load(model_name: str):
        # Try transformers T5TokenizerFast (tokenizer-only import, no torch models)
        try:
            from transformers import T5TokenizerFast
            return T5TokenizerFast.from_pretrained(model_name)
        except Exception:
            pass
        # Fallback: raw sentencepiece (avoids transformers entirely)
        try:
            from sentencepiece import SentencePieceProcessor
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo_id=model_name, filename="spiece.model")
            sp = SentencePieceProcessor()
            sp.Load(path)
            return sp
        except Exception as exc:
            raise RuntimeError(
                f"Cannot load T5 tokenizer for '{model_name}'. "
                "pip install transformers sentencepiece"
            ) from exc

    def train_from_text(self, text: str, **kwargs: Any) -> None:
        pass  # pre-trained — no training needed

    def encode(self, text: str) -> list[int]:
        if hasattr(self._tok, "encode"):
            # T5TokenizerFast: returns token IDs, no special tokens
            return self._tok.encode(text, add_special_tokens=False)
        # SentencePieceProcessor fallback
        return self._tok.EncodeAsIds(text)

    def decode(self, tokens) -> str:
        if not isinstance(tokens, list):
            tokens = list(tokens)
        if hasattr(self._tok, "decode"):
            return self._tok.decode(tokens, skip_special_tokens=True)
        return self._tok.Decode(tokens)

    @property
    def vocab_size(self) -> int:
        v = self._tok.vocab_size
        return v() if callable(v) else int(v)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type": "t5", "model_name": self.model_name}, f)

    @classmethod
    def load(cls, path: str) -> "T5SentencePieceTokenizer":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(model_name=data.get("model_name", "t5-base"))


def get_tokenizer(tokenizer_type: str, **kwargs) -> Tokenizer:
    if tokenizer_type == "char":
        return CharTokenizer()
    elif tokenizer_type == "bpe":
        return BPETokenizer()
    elif tokenizer_type == "t5":
        return T5SentencePieceTokenizer(kwargs.get("model_name", "t5-base"))
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type!r}")


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
    if tok_type == "t5":
        return T5SentencePieceTokenizer.load(path)
    raise ValueError(f"Unknown tokenizer type in file: {tok_type!r}")
