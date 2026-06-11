---
title: LoRA Fine-Tuning
---

# LoRA Fine-Tuning

This tutorial fine-tunes a pretrained DantinoX checkpoint on a new text corpus using Low-Rank Adaptation (LoRA). LoRA freezes the base model and trains only a small set of rank-decomposed adapter weights — approximately **0.2 % of total parameters** — making it practical on a single GPU.

**Prerequisites:** a trained checkpoint from the [Training Your First Model](first-model.md) tutorial, or any DantinoX run directory.

---

## Background

LoRA (Hu et al., 2022) inserts a trainable low-rank delta alongside each frozen linear projection:

$$W_{\text{eff}} = W_{\text{base}} + \frac{\alpha}{r} \cdot AB$$

where $A \in \mathbb{R}^{d \times r}$ is initialised with scaled Gaussian noise and $B \in \mathbb{R}^{r \times k}$ is zero-initialised. At the start of fine-tuning the adapter contributes nothing ($B = 0$), so training starts from the exact pretrained behaviour.

DantinoX implements freezing at the **type level** via a custom `LoRAParam(nnx.Variable)` subclass. Base weights are registered as `nnx.Param`; adapters as `LoRAParam`. The optimizer is constructed with `wrt=LoRAParam`, so only adapter weights receive gradient updates — no manual masking or `stop_gradient` required.

---

## 1. Prepare the Fine-Tuning Dataset

```bash
# Example: fine-tune on Italian poetry (vs. the English Shakespeare used in pre-training)
wget -O data/finetune.txt https://raw.githubusercontent.com/BrunoSilvestrini/Datasets/main/divina_commedia.txt
```

Any plain-text file works. The tokenizer fitted during pre-training is reused automatically — the vocabulary does not change.

---

## 2. Create a Fine-Tuning Config

Load the pretrained config and enable LoRA:

```python
# finetune.py
from dantinox.core.config import Config
from dantinox.trainer import Trainer

# Load the exact config the pretrained model was trained with
run_dir = "runs/run_20260101_120000"   # your pretrained checkpoint
config  = Config.from_yaml(f"{run_dir}/config.yaml")

# ── LoRA settings ─────────────────────────────────────────────────────────────
config.use_lora     = True
config.lora_rank    = 8       # adapter rank r — larger = more capacity, more params
config.lora_alpha   = 16.0    # scaling constant α (effective scale = α / r = 2.0)
config.lora_dropout = 0.05    # dropout on the LoRA delta path (0 = disabled)
config.lora_targets = "attention"   # which layers to adapt: "attention" | "mlp" | "all"

# ── Fine-tuning hyperparameters ────────────────────────────────────────────────
config.lr           = 1e-4    # lower LR than pre-training
config.lr_schedule  = "cosine"
config.warmup_steps = 50
config.epochs       = 3
config.patience     = 3

ft_run_dir = Trainer(config).fit(
    data_path="data/finetune.txt",
    run_dir=f"{run_dir}/lora_ft",   # save inside the base checkpoint directory
)
print(f"Fine-tuned checkpoint: {ft_run_dir}")
```

Or equivalently via the CLI:

```bash
dantinox train \
  --config runs/run_20260101_120000/config.yaml \
  --data_path data/finetune.txt \
  --use_lora true \
  --lora_rank 8 \
  --lora_alpha 16.0 \
  --lora_targets attention \
  --lr 1e-4 \
  --epochs 3
```

---

## 3. Understanding `lora_targets`

| `lora_targets` | Adapted layers | Parameters (rank=8, dim=256) |
| :--- | :--- | :--- |
| `"attention"` | `qkv`, `o_proj` in every Attention block | ~0.2 % |
| `"mlp"` | `up_proj`, `down_proj` in every MLP block | ~0.4 % |
| `"all"` | All of the above | ~0.6 % |

For most domain-adaptation tasks, `"attention"` is sufficient and cheapest. Use `"all"` if you need to also adapt the feed-forward knowledge.

---

## 4. Monitor Training

The fine-tuning run logs to the same `training_log.csv` format. You can compare loss curves between pre-training and fine-tuning:

```python
import pandas as pd
import matplotlib.pyplot as plt

base = pd.read_csv("runs/run_20260101_120000/training_log.csv")
ft   = pd.read_csv("runs/run_20260101_120000/lora_ft/training_log.csv")

fig, ax = plt.subplots()
ax.plot(base["val_loss"], label="Pre-training val loss")
ax.plot(ft["val_loss"],   label="Fine-tuning val loss")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend()
plt.savefig("lora_comparison.pdf", bbox_inches="tight")
```

---

## 5. Generate with the Fine-Tuned Model

```python
from dantinox.generator import Generator

gen = Generator("runs/run_20260101_120000/lora_ft")
print(gen.generate("Nel mezzo del cammin di nostra vita", max_new_tokens=200))
```

The `Generator` loads the LoRA adapters automatically. The base weights remain unchanged in `weights.msgpack`; the adapters are stored as a separate key in the same file.

---

## 6. Merging LoRA Weights for Deployment

Once fine-tuning is complete you can merge the adapters into the base weights to eliminate the LoRA overhead at inference time:

```python
from dantinox.core.lora import LoRALinear
from dantinox.core.model import Transformer
from dantinox.core.config import Config
import flax.serialization

config = Config.from_yaml("runs/run_20260101_120000/lora_ft/config.yaml")
model  = Transformer.from_pretrained("runs/run_20260101_120000/lora_ft")

# Merge every LoRALinear layer in-place
for path, module in model.iter_modules():
    if isinstance(module, LoRALinear):
        merged = module.merge_weights()   # W_base + (α/r) · A·B
        module.kernel.value = merged
        module.lora_A = None
        module.lora_B = None

# Save merged weights to a new directory
import os
out_dir = "runs/run_20260101_120000/lora_merged"
os.makedirs(out_dir, exist_ok=True)
config.use_lora = False
config.to_yaml(f"{out_dir}/config.yaml")
flax.serialization.to_bytes(model)   # or use dantinox push directly
```

!!! note
    Merged checkpoints are identical in size to the original base model and have zero inference overhead. Use them when you want to deploy without any LoRA machinery.

---

## 7. Choosing the Rank

| Rank | Parameters | Use case |
| :--- | :--- | :--- |
| 4 | ~0.1 % | Style transfer, minimal domain shift |
| 8 | ~0.2 % | Standard domain adaptation (recommended starting point) |
| 16 | ~0.4 % | Significant distribution shift |
| 32 | ~0.8 % | Near full fine-tuning quality; diminishing returns |

A higher rank increases adapter expressiveness but also the risk of overfitting on small datasets. Start with `rank=8` and adjust based on validation loss.

---

## Next Steps

| Goal | Reference |
| :--- | :--- |
| Understand the LoRA math and type system | [Architecture — LoRA Fine-Tuning](../architecture.md#lora-fine-tuning) |
| Push the fine-tuned model to HuggingFace Hub | [Pushing to HuggingFace Hub](hub.md) |
| Fine-tune a diffusion model | [Masked Diffusion LM](diffusion-lm.md) |
