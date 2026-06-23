---
title: Autoregressive Training
---

# Autoregressive Training

Set `model_type: "autoregressive"` (the default) to train a causal language model.

---

## Loss Function

Standard next-token cross-entropy with teacher-forcing:

$$
\mathcal{L}_{\text{AR}} = -\frac{1}{T} \sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})
$$

For MoE models, the load-balancing auxiliary loss is added:

$$
\mathcal{L} = \mathcal{L}_{\text{AR}} + \alpha_{\text{bal}} \cdot \mathcal{L}_{\text{bal}}
$$

---

## Quick Start

```bash
dantinox train \
  --config configs/default_config.yaml \
  --use_bf16 true \
  --n_devices 2 \
  --dataset_source huggingface \
  --dataset_name wikitext \
  --dataset_config wikitext-103-raw-v1
```

---

## Key Config Fields

```yaml
model:
  model_type: "autoregressive"
  dim: 256
  n_heads: 8
  head_size: 32
  num_blocks: 12
  max_context: 512
  weight_tying: true       # tie embedding ↔ LM head weights
  use_swiglu: true         # SwiGLU FFN (better than GELU)
  norm_type: "layernorm"   # or "rmsnorm"
  dropout_rate: 0.15
```

---

## Gradient Clipping

```yaml
training:
  grad_clip: 1.0   # default — recommended for all runs
```

---

## LR Finder

Before a long run, find the optimal learning rate:

```bash
dantinox find-lr \
  --config configs/default_config.yaml \
  --min_lr 1e-6 --max_lr 1e-2 \
  --num_steps 100 --plot
```

Pick the LR just **before** the loss minimum on the output chart.

---

## LR Schedules

| Value | Behaviour |
|---|---|
| `"cosine"` | Smooth cosine decay from peak to `lr × 0.01` (default) |
| `"linear"` | Linear ramp down |
| `"constant"` | Flat after warmup |
| `"wsd"` | Warmup → stable (40 %) → cosine decay |

---

## LoRA Fine-Tuning

Fine-tune a pre-trained AR checkpoint by training only low-rank adapter weights:

```bash
dantinox train \
  --config runs/ar_mha_256d_12b_Dense/config.yaml \
  --use_lora true \
  --lora_rank 8 \
  --lora_alpha 16.0 \
  --lora_targets attention
```

Only ~0.1–0.5 % of parameters are trained.
See the [LoRA Fine-Tuning tutorial](../tutorials/lora-fine-tuning.md) and the [Architecture reference](../architecture.md#lora-fine-tuning) for full details.

---

## Checkpoint Loading

```python
from dantinox.core.model import Transformer

model = Transformer.from_pretrained("runs/ar_mha_256d_12b_Dense")
# or load from HuggingFace Hub:
model = Transformer.from_pretrained("my-org/dantinox-model")
```

---

## Resume Training

```bash
dantinox train \
  --config configs/default_config.yaml \
  --run_dir runs/ar_mha_256d_12b_Dense \
  --resume
```

Restores weights and step counter; optimizer moments restart.
