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

# ── GQA baseline — trains in ~5 min on T4 ────────────────────────────────────
config_gqa = Config(
    # ── Architecture ─────────────────────────────────────────────────────────
    dim=256,             # model width (embedding + hidden size)
    n_heads=8,           # number of query heads
    head_size=32,        # per-head dimension  (dim == n_heads * head_size)
    num_blocks=6,        # transformer depth
    max_context=256,     # maximum sequence length
    kv_heads=2,          # GQA: 4 query heads share each KV head (set == n_heads for MHA)
    weight_tying=True,   # tie embedding and output projection weights

    # ── MLP ──────────────────────────────────────────────────────────────────
    use_swiglu=True,     # SwiGLU activation (better than GELU for language)
    expansion=4,         # MLP hidden = dim * expansion
    activation="gelu",   # used only when use_swiglu=False: "gelu" | "relu" | "silu"

    # ── Normalisation ─────────────────────────────────────────────────────────
    norm_type="rmsnorm", # "rmsnorm" (faster) | "layernorm"

    # ── Positional encoding ───────────────────────────────────────────────────
    use_rotary_pos=True, # RoPE (recommended)
    rope_scale_factor=1.0, # >1 compresses RoPE frequencies for longer contexts (NTK-aware)
    trainable_pos=False, # learnable absolute bias (combines with RoPE when True)
    absolute_pos=False,  # simple absolute positional embeddings (legacy)

    # ── Attention ────────────────────────────────────────────────────────────
    use_flash_attention=True,   # memory-efficient attention kernel
    sliding_window=False,       # restrict attention to a local window (see config_swa)

    # ── Regularisation ───────────────────────────────────────────────────────
    dropout_rate=0.1,
    gradient_checkpointing=True, # recompute activations on backward (saves VRAM)

    # ── Dataset — streamed from HuggingFace ──────────────────────────────────
    dataset_source="huggingface",  # "huggingface" | "local"
    dataset_name="Daniele/dante-corpus",
    dataset_config="",             # HF subset, e.g. "en" for allenai/c4
    dataset_text_field="text",
    dataset_split="train",
    streaming=True,                # stream from Hub without downloading

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tokenizer_type="char",   # "char" (fast) | "bpe" (better compression)
    # tokenizer_path=None,   # path to a pre-trained BPE tokenizer JSON

    # ── Optimiser ────────────────────────────────────────────────────────────
    optimizer="adamw",   # "adamw" | "adam" | "lion" | "adafactor"
    lr=3e-4,
    lr_schedule="wsd",   # "wsd" (warmup→stable→cosine) | "cosine" | "linear" | "constant"
    warmup_steps=100,
    grad_clip=1.0,
    use_bf16=True,       # bfloat16 mixed precision (faster + less VRAM on Ampere/T4)

    # ── Training loop ─────────────────────────────────────────────────────────
    epochs=1,
    batch_size=64,
    grad_accum=4,        # effective batch = batch_size * grad_accum = 256
    patience=10,         # early stopping: stop if val loss doesn't improve for N epochs
    seed=42,

    # ── Evaluation ───────────────────────────────────────────────────────────
    eval_iters=20,       # batches averaged for each val-loss estimate

    # ── Multi-GPU (optional) ─────────────────────────────────────────────────
    n_devices=0,         # 0 = use all available GPUs; set e.g. 2 to limit
)

print("GQA config:")
print(config_gqa)

config = config_gqa
print("Training…")

trainer = Trainer(config)
run_dir = trainer.fit()
print(f"\nCheckpoint saved to: {run_dir}")

print("\nGenerating…")
gen = Generator(run_dir, seed=42)
prompt = "Nel mezzo del cammin "
output = gen.generate(prompt, max_new_tokens=200, temperature=0.8, top_k=40)

print(f"\nPrompt: {prompt!r}")
print("-" * 60)
print(output)
