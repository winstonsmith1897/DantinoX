import jax.numpy as jnp

class CharTokenizer:
    def __init__(self):
        self.stoi = {}
        self.itos = {}
        self.vocab_size = 0

    def train_from_text(self, text):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])

class BPETokenizer:
    def __init__(self):
        from tokenizers import Tokenizer, models, pre_tokenizers
        self.tokenizer = Tokenizer(models.BPE())
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
        self.vocab_size = 0

    def train_from_text(self, text, vocab_size=1000):
        from tokenizers import trainers
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size, 
            special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
        )
        self.tokenizer.train_from_iterator([text], trainer=trainer)
        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, s):
        return self.tokenizer.encode(s).ids

    def decode(self, l):
        return self.tokenizer.decode(l)

def get_tokenizer(tokenizer_type):
    if tokenizer_type == "char":
        return CharTokenizer()
    elif tokenizer_type == "bpe":
        return BPETokenizer()
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")