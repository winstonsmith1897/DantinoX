---
title: Embedder Training
---

# Embedder Training

This guide covers all three ways to train a DantinoX model as a sentence encoder for RAG.

| Mode | When to use | Entry point |
|------|-------------|-------------|
| **Unsupervised (SimCSE)** | No labelled data, any flat text corpus | `dx.fit("embedder", ...)` / `dx.train(paradigm, ...)` |
| **Supervised pairs** | (anchor, positive) pairs available | `dx.EmbedderTrainer.fit_pairs(data)` |
| **Fine-tuning** | Pretrained AR / Discrete model → encoder | `dx.EmbedderTrainer.fit_pairs(data, model=dx.load(...))` |

---

## 1. Unsupervised — SimCSE

`EmbedderParadigm` plugs directly into the stock `Trainer`.  
Each `[B, T]` token window is encoded twice with **different dropout masks** — the two views act as anchor / positive in the InfoNCE loss. No labelled pairs required.

### One-liner

```python
import dantinox as dx

run_dir = dx.fit(
    "embedder",                       # paradigm string
    "data/corpus.txt",                # flat text file or HuggingFace dataset name
    # ── model ──────────────────────────────────────────────
    dim=256, n_heads=4, head_size=64, num_blocks=6,
    vocab_size=32_000,
    causal=False,   # bidirectional — strongly recommended
    dropout=0.1,    # REQUIRED: SimCSE needs dropout > 0
    # ── embedder-specific ──────────────────────────────────
    pooling="mean", temperature=0.05,
    # ── training ───────────────────────────────────────────
    lr=3e-4, epochs=10, batch_size=64,
)
```

### With a HuggingFace dataset (no local file needed)

```python
import dantinox as dx

cfg = dx.ModelConfig(
    dim=256, n_heads=4, head_size=64, num_blocks=6,
    vocab_size=32_000, causal=False, dropout=0.1,
)
paradigm = dx.EmbedderParadigm(cfg, pooling="mean", temperature=0.05)

train_cfg = dx.TrainingConfig(
    dataset_source="huggingface",
    dataset_name="wikitext",
    dataset_config="wikitext-103-raw-v1",
    dataset_text_field="text",
    tokenizer_type="bpe",
    lr=3e-4, epochs=10, batch_size=64,
    warmup_steps=500,
)

run_dir = dx.train(paradigm, training_config=train_cfg)
```

### Full control via Trainer

```python
trainer = dx.Trainer(paradigm, train_cfg)
run_dir = trainer.fit("data/corpus.txt")
```

:::{admonition} Batch size and in-batch negatives
:class: tip

Larger batch sizes mean more in-batch negatives and a stronger InfoNCE signal. On a single 40 GB GPU, `batch_size=256` is usually achievable with `dim≤512`.
:::

---

## 2. Supervised Pairs

Use `EmbedderTrainer` when you have labelled (anchor, positive) pairs. The trainer is **single-device** — it reads CUDA_VISIBLE_DEVICES automatically from the environment.

### Data formats

=== "List of tuples"
    ```python
    pairs = [
        ("How does JAX work?", "JAX is a NumPy-compatible library with JIT on GPU/TPU."),
        ("What is diffusion?", "Diffusion iteratively denoises a noisy input."),
        # ...
    ]
    ```

=== "JSONL file"
    ```json
    {"anchor": "How does JAX work?", "positive": "JAX is a NumPy-compatible library..."}
    {"anchor": "What is diffusion?", "positive": "Diffusion iteratively denoises..."}
    ```

=== "TSV file"
    ```
    How does JAX work?\tJAX is a NumPy-compatible library with JIT on GPU/TPU.
    What is diffusion?\tDiffusion iteratively denoises a noisy input.
    ```

### Training

```python
import dantinox as dx
from dantinox.utils.tokenizer import BPETokenizer

tok = BPETokenizer()
tok.train_from_text(open("data/corpus.txt").read(), vocab_size=32_000)

cfg = dx.ModelConfig(
    dim=256, n_heads=4, head_size=64, num_blocks=6,
    vocab_size=tok.vocab_size, causal=False, dropout=0.1, max_context=256,
)
paradigm = dx.EmbedderParadigm(cfg, pooling="mean", temperature=0.05)

trainer = dx.EmbedderTrainer(
    paradigm, tok,
    dx.TrainingConfig(lr=2e-4, epochs=20, batch_size=64),
)

run_dir = trainer.fit_pairs("data/pairs.jsonl")  # or list of tuples
```

### Checkpoint layout

`fit_pairs()` writes a run directory compatible with `Embedder.from_run()`:

```
runs/embedder_supervised/
  checkpoint_best.msgpack     # best epoch by training loss
  checkpoint_latest.msgpack   # most recent epoch
  tokenizer.json              # saved tokenizer for Embedder.from_run()
  config.yaml                 # ModelConfig
```

---

## 3. Fine-Tuning a Pretrained Model

Any DantinoX AR or Discrete checkpoint can be converted into an embedder by wrapping it in `EmbedderParadigm` and running a few supervised epochs. The pretrained weights give the model a strong language prior, so a small number of pairs is sufficient.

```python
import dantinox as dx
from dantinox.utils.tokenizer import load_tokenizer_from_file

pretrained_run = "runs/my_discrete_run"

cfg   = dx.ModelConfig.from_yaml(f"{pretrained_run}/config.yaml")
model = dx.load(pretrained_run)   # loads weights from checkpoint_best.msgpack
tok   = load_tokenizer_from_file(f"{pretrained_run}/tokenizer.json")

# Wrap in EmbedderParadigm — same architecture, new contrastive loss
paradigm = dx.EmbedderParadigm(cfg, pooling="mean", temperature=0.05)

trainer = dx.EmbedderTrainer(
    paradigm, tok,
    dx.TrainingConfig(
        lr=5e-5,    # lower LR for fine-tuning
        epochs=5,
        batch_size=32,
    ),
)

# Pass model= to inject pretrained weights instead of starting from scratch
run_dir = trainer.fit_pairs(pairs, model=model, run_dir="runs/embedder_finetuned")
```

:::{admonition} How many pairs do you need?
:class: tip

- Fine-tuning from a pretrained checkpoint: **1 000 – 10 000** pairs typically suffice.
- Training from scratch (supervised only): aim for **50 000+** pairs.
- Unsupervised SimCSE on a large corpus is a good substitute for labelled pairs when they are unavailable.
:::

---

## Hyperparameter Guide

### Model architecture

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `causal` | `False` | Bidirectional attention gives better embeddings |
| `dropout` | `0.1` | Required for SimCSE; keep at 0.1 even for supervised |
| `dim` | 256–768 | Larger = better quality, higher inference cost |
| `num_blocks` | 4–12 | Deeper = richer representations |
| `max_context` | 128–512 | Must cover your longest document |

### InfoNCE / contrastive

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `temperature` | `0.05–0.10` | Lower = sharper, stronger gradient |
| `batch_size` | 64–256 | More in-batch negatives → better loss |
| `pooling` | `"mean"` | Best for bidirectional models |

### Optimisation

| Parameter | Unsupervised | Supervised | Fine-tuning |
|-----------|-------------|------------|-------------|
| `lr` | `3e-4` | `2e-4` | `5e-5` |
| `epochs` | 5–20 | 10–30 | 3–10 |
| `warmup_steps` | 200–500 | 100–200 | 50–100 |
| `optimizer` | `"adamw"` | `"adamw"` | `"adamw"` |

---

## Multi-GPU Note

`EmbedderTrainer` is **single-device** by design — it runs on whichever GPU is selected by `CUDA_VISIBLE_DEVICES`. For multi-GPU training, use `EmbedderParadigm` with the stock `Trainer`, which supports full data-parallel sharding:

```python
# Multi-GPU: use stock Trainer with EmbedderParadigm
trainer = dx.Trainer(paradigm, dx.TrainingConfig(lr=3e-4, epochs=10, n_devices=4))
run_dir = trainer.fit("data/corpus.txt")
```

See [Multi-GPU Training](multi-gpu.md) for details.

---

## See Also

- [Retriever Paradigm](../paradigms/retriever.md) — architecture and InfoNCE math
- [RAG Tutorial](../tutorials/retriever-rag.md) — FAISS, LangChain, ChromaDB
- [API — EmbedderParadigm](../api/paradigms.md#sentence-embedder)
- [API — EmbedderTrainer](../api/training.md#embeddertrainer)
