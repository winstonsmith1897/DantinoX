# Optimizers & Schedules

DantinoX ships five optimizers and four LR schedules, all configurable from `TrainingConfig`. The optimizer is built by `build_optimizer(model, config, total_steps)` which returns an `nnx.Optimizer` ready for use in the training loop.

---

## Optimizers

| `config.optimizer` | Algorithm | Notes |
| :--- | :--- | :--- |
| `"adamw"` | AdamW | Default. `weight_decay=0.1`. Good all-around baseline. |
| `"adafactor"` | Adafactor | Memory-efficient: no second-moment state. Use for large models on tight VRAM. |
| `"lion"` | Lion | Sign-gradient update. Typically needs 3–10× lower LR than AdamW. |
| `"adam"` | Adam | No weight decay. Useful for fine-tuning when decay hurts. |
| `"muon"` | Muon | Newton-Schulz orthogonalization for 2D weights. Strong empirical results on LM pre-training. |

### Muon notes

Muon (`optax.contrib.muon`) applies Newton-Schulz iterations to orthogonalize 2D parameter updates before applying them. This is equivalent to computing the unitary factor of the polar decomposition and has been shown to improve training stability and convergence speed on language model pre-training.

```python
# Muon skips the outer grad-clip chain — it handles normalization internally.
# All other optimizers:  optax.chain(clip_by_global_norm, base_optimizer)
# Muon:                  optax.contrib.muon(lr_schedule)  (no chain)
```

---

## LR schedules

### Cosine (`"cosine"`)

Linear warmup → cosine decay to 1 % of peak.

```
lr
▲
│    ████
│  ██    ██
│ █        ██
│█           ████████
└──────────────────────► step
   warmup   decay
```

### Linear (`"linear"`)

Linear warmup → linear decay to 1 % of peak.

### Constant (`"constant"`)

Linear warmup → flat plateau at peak LR.

### WSD — Warm-Stable-Decay (`"wsd"`)

Three phases: linear warmup → 40 % stable plateau → cosine decay.

```
lr
▲
│    ████████████████
│  ██               ████
│ █                     ████████
│█
└──────────────────────────────► step
   warmup    stable      decay
```

!!! tip "WSD for long runs"
    WSD tends to outperform cosine on long pre-training runs (>100 K steps) because the stable phase allows the optimizer to settle before the final decay.

---

## Usage

```python
from core.config import TrainingConfig
from dantinox.training.optimizer import build_optimizer, build_schedule

cfg = TrainingConfig(
    lr=3e-4,
    optimizer="muon",
    lr_schedule="wsd",
    grad_clip=1.0,
    epochs=10,
)
optimizer = build_optimizer(model, cfg, total_steps=50_000)
```

### LoRA-aware optimization

When the model contains `LoRAParam` variables, `build_optimizer` automatically applies `optax.multi_transform` so that only adapter parameters receive gradient updates. Base `nnx.Param` weights are frozen at the type level — no manual masking required.

```python
# Verified by build_optimizer internals:
# if _model_has_lora(model):
#     tx = optax.multi_transform(
#         {"lora": base_optimizer, "frozen": optax.set_to_zero()},
#         label_fn,
#     )
```

---

## CLI

```bash
dantinox train --config configs/default_config.yaml \
               --optimizer muon \
               --lr_schedule wsd \
               --lr 1e-4
```

Any `TrainingConfig` field can be overridden via CLI flags. Config YAML values are the base; CLI flags override on top.
