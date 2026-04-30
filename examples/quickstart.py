"""
DantinoX quickstart — train a small character-level model, then generate text.

Run from the repository root:

    python examples/quickstart.py

Requirements: a text corpus at data/corpus.txt (or edit DATA_PATH below).
The script trains for 200 steps (~30 seconds on CPU with this tiny config)
then prints a short generated continuation.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import Config
from dantinox.generator import Generator
from dantinox.trainer import Trainer

DATA_PATH = "data/corpus.txt"

# A tiny model that fits comfortably on CPU for demonstration purposes.
config = Config(
    dim=128,
    n_heads=4,
    head_size=32,         # dim == n_heads * head_size: 4 * 32 = 128
    num_blocks=2,
    vocab_size=256,       # updated automatically by the tokenizer
    max_context=128,
    kv_heads=2,
    tokenizer_type="char",
    epochs=4,             # increase for real training
    batch_size=8,
    grad_accum=2,
    lr=3e-4,
    warmup_steps=50,
    grad_clip=1.0,        # gradient clipping — strongly recommended
    patience=0,           # set > 0 to enable early stopping
    eval_iters=5,
    gradient_checkpointing=False,
    dropout_rate=0.0,
)

print("Training…")
trainer = Trainer(config)
run_dir = trainer.fit(DATA_PATH)
print(f"\nCheckpoint saved to: {run_dir}")

print("\nGenerating…")
gen = Generator(run_dir, seed=42)
prompt = "Nel mezzo del cammin "
output = gen.generate(prompt, max_new_tokens=200, temperature=0.8, top_k=40)
print(f"\nPrompt: {prompt!r}")
print("-" * 60)
print(output)
