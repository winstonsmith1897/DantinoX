import json
import os
from typing import List, Union
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

class BaseTokenizer:
    def encode(self, text: str) -> List[int]:
        raise NotImplementedError
    
    def decode(self, ids: List[int]) -> str:
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

class CharTokenizer(BaseTokenizer):
    def __init__(self, text: str = None, vocab_path: str = None):
        if vocab_path and os.path.exists(vocab_path):
            with open(vocab_path, 'r') as f:
                self.chars = json.load(f)
        elif text:
            self.chars = sorted(list(set(text)))
        else:
            raise ValueError("Give a text or a path to retrieve the vocabulary.")
        
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        
    def encode(self, s: str) -> List[int]:
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, l: List[int]) -> str:
        return ''.join([self.itos[i] for i in l])

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.chars, f)

class BPETokenizer(BaseTokenizer):
    def __init__(self, vocab_path: str = None):
        if vocab_path and os.path.exists(vocab_path):
            self.tokenizer = HFTokenizer.from_file(vocab_path)
        else:
            self.tokenizer = HFTokenizer(BPE(unk_token="[UNK]"))
            self.tokenizer.pre_tokenizer = Whitespace()

    def train(self, files: List[str], vocab_size: int = 5000):
        trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], 
                             vocab_size=vocab_size)
        self.tokenizer.train(files, trainer)

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def save(self, path: str):
        self.tokenizer.save(path)

def get_tokenizer(config_type: str, **kwargs) -> BaseTokenizer:
    if config_type == "char":
        return CharTokenizer(**kwargs)
    elif config_type == "bpe":
        return BPETokenizer(**kwargs)
    else:
        raise ValueError(f"This tokenizer {config_type} is not supported.")