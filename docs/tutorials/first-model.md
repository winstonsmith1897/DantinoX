---
title: Training Your First Model
---

# Training Your First Model

This tutorial trains a small Grouped-Query Attention (GQA) Transformer on a local text file, evaluates it, and generates text. It takes roughly **5–10 minutes** on a single GPU (e.g. T4 or RTX 3090).

---

## 1. Installation

```bash
git clone https://github.com/winstonsmith1897/DantinoX.git
cd DantinoX

conda create -n dantinox python=3.12 -y
conda activate dantinox

pip install -U "jax[cuda12]"
pip install -e ".[all]"
```

Verify that JAX sees your GPU:

```python
import jax
print(jax.devices())   # should list at least one CudaDevice
```

---

## 2. Prepare a Text Corpus

DantinoX can load text from a local file or directly from HuggingFace Hub. For this tutorial we use a local file.

```bash
# Use any plain-text corpus. Here we grab the complete works of Shakespeare (~5 MB).
wget -O data/corpus.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

Alternatively, point DantinoX at a HuggingFace dataset by setting `dataset_source = "huggingface"` in the config (see step 3).

---

## 3. Define a Config

Every aspect of the model, training, and inference is expressed in a single `Config` dataclass. Create a file `configs/tutorial.yaml`:

```yaml
# configs/tutorial.yaml

model:
  dim: 256              # embedding + hidden dimension
  n_heads: 8            # query heads
  head_size: 32         # per-head dimension (dim = n_heads × head_size)
  num_blocks: 6         # transformer depth
  max_context: 256      # maximum sequence length
  kv_heads: 2           # GQA: 4 query heads per KV head
  norm_type: rmsnorm    # faster than LayerNorm; used in LLaMA, Mistral
  use_swiglu: true
  weight_tying: true

attention:
  use_rotary_pos: true
  use_flash_attention: true   # fused SDPA kernel (JAX ≥ 0.4.25)

training:
  optimizer: adamw
  lr: 3e-4
  lr_schedule: wsd      # warmup → stable → cosine decay
  warmup_steps: 100
  grad_clip: 1.0
  use_bf16: true
  batch_size: 64
  grad_accum: 4         # effective batch = 64 × 4 = 256
  epochs: 1
  patience: 5           # stop early if val loss doesn't improve

tokenizer:
  tokenizer_type: char  # character-level; no pre-trained vocabulary needed

data:
  dataset_source: local
```

!!! note "dim = n_heads × head_size"
    DantinoX enforces `dim == n_heads × head_size`. For the config above: 256 = 8 × 32. The library raises a `ValueError` at startup if this constraint is violated.

---

## 4. Train

### Using the CLI

```bash
dantinox train \
  --config configs/tutorial.yaml \
  --data_path data/corpus.txt
```

### Using the Python API

```python
from core.config import Config
from dantinox.trainer import Trainer

config = Config.from_yaml("configs/tutorial.yaml")
run_dir = Trainer(config).fit("data/corpus.txt")
print(f"Checkpoint saved to: {run_dir}")
```

Training logs `train_loss` and `val_loss` to the console and writes them to `{run_dir}/training_log.csv`. A model summary is saved to `{run_dir}/model_summary.json`.

!!! tip "W&B integration"
    Pass `wandb_project="my-project"` to `Trainer.fit()` to log all metrics to Weights & Biases automatically. No other changes are required.

---

## 5. Understanding the Run Directory

After training, `run_dir` contains:

```text
runs/run_20260101_120000/
├── config.yaml          # exact config used (reproducibility)
├── weights.msgpack      # model weights (Flax serialization)
├── tokenizer.json       # fitted tokenizer (char-level vocabulary)
├── training_log.csv     # per-epoch train/val loss
└── model_summary.json   # parameter count, FLOPs estimate
```

The checkpoint is self-contained: `config.yaml` and `weights.msgpack` together fully specify the model.

---

## 6. Generate Text

### Single prompt

```python
from dantinox.generator import Generator

gen = Generator(run_dir)   # or pass the path as a string
text = gen.generate(
    "To be, or not to be,",
    max_new_tokens=200,
    temperature=0.8,
    top_k=40,
)
print(text)
```

### Batched generation

```python
prompts = [
    "To be, or not to be,",
    "All the world's a stage,",
    "What a piece of work is man,",
]
results = gen.generate_batch(prompts, max_new_tokens=100, temperature=0.8)
for prompt, result in zip(prompts, results):
    print(f"[{prompt[:20]}...]\n{result}\n")
```

### Streaming

```python
for token in gen.stream("To be, or not to be,", max_new_tokens=200):
    print(token, end="", flush=True)
print()
```

---

## 7. Switching to MHA or MLA

The attention family is controlled by two config fields:

| Attention | Config |
| :--- | :--- |
| MHA | `attention_type: mha` (or `kv_heads == n_heads`) |
| GQA | `attention_type: gqa` (or `kv_heads < n_heads`) |
| MLA | `attention_type: mla` |

To switch the tutorial model to MLA:

```yaml
# Add to configs/tutorial.yaml under a new `mla:` section
mla:
  mla: true
  down_dim_q: 128
  down_dim_kv: 64
  rope_dim: 32
```

MLA caches only the compressed latent `c_KV` per token (64 scalars vs. 256 for GQA in this config), reducing KV-cache memory by ~4×.

---

## 8. Next Steps

| Goal | Tutorial |
| :--- | :--- |
| Adapt the model to a new domain | [LoRA Fine-Tuning](lora-fine-tuning.md) |
| Train a non-autoregressive model | [Masked Diffusion LM](diffusion-lm.md) |
| Publish the checkpoint | [Pushing to HuggingFace Hub](hub.md) |
| Scale to multiple GPUs | [Multi-GPU Training](../training/multi-gpu.md) |
| Understand the attention math | [Architecture](../architecture.md) |
