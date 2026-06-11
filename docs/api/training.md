# `dantinox.training`

The training module is paradigm-agnostic. It calls `paradigm.loss_fn` at every step and owns checkpointing, logging, and multi-device replication.

## Overview

```
Trainer.fit(data_path)
    ↓
  loads data  →  builds paradigm  →  JIT-compiles step
    ↓
  for each epoch:
    train step  →  paradigm.loss_fn  →  optimizer update  →  checkpoint
    eval step   →  val_loss  →  early stopping check
```

The `Trainer` interacts with the paradigm through a single interface:

```python
loss, metrics = paradigm.loss_fn(model, batch)
```

Everything else (data loading, checkpointing, LR scheduling, gradient accumulation, multi-device replication) is handled by `Trainer` independently of the paradigm.

---

## Quick start

```python
from dantinox.trainer import Trainer
from core.config import Config

cfg     = Config.from_yaml("configs/default_config.yaml")
trainer = Trainer(cfg)
run_dir = trainer.fit("data/corpus.txt")
print("Checkpoint saved to:", run_dir)
```

With W&B logging:

```python
run_dir = trainer.fit(
    "data/corpus.txt",
    wandb_project="DantinoX",
)
```

---

## Trainer

::: dantinox.training.trainer.Trainer
    options:
      show_source: true
      members:
        - __init__
        - fit

---

## Optimizer factory

::: dantinox.training.optimizer.build_optimizer
    options:
      show_source: true
      heading_level: 3

::: dantinox.training.optimizer.build_schedule
    options:
      show_source: true
      heading_level: 3

---

## Supported optimizers

| Name | `config.optimizer` | Notes |
| :--- | :--- | :--- |
| AdamW | `"adamw"` | Default. Weight decay 0.1. |
| Adafactor | `"adafactor"` | Memory-efficient; no `lr²` state. |
| Lion | `"lion"` | Sign-gradient update; often needs lower LR. |
| Adam | `"adam"` | No weight decay. |
| Muon | `"muon"` | Newton-Schulz orthogonalization for 2D weights. |

## Supported schedules

| Name | `config.lr_schedule` | Shape |
| :--- | :--- | :--- |
| Cosine | `"cosine"` | Warmup → cosine decay to 1 % of peak |
| Linear | `"linear"` | Warmup → linear decay to 1 % of peak |
| Constant | `"constant"` | Warmup → flat plateau |
| WSD | `"wsd"` | Warmup → 40 % stable → cosine decay |

!!! note "Muon and grad clipping"
    Muon handles gradient normalization internally. When `optimizer="muon"`, the outer `optax.clip_by_global_norm` chain is skipped. All other optimizers apply `clip_by_global_norm(config.grad_clip)` before the update.

!!! note "LoRA masking"
    When the model contains `LoRAParam` variables, `build_optimizer` automatically applies an `optax.multi_transform` mask that zeroes gradients for base `nnx.Param` weights. No manual filtering required.

---

## Checkpointing

The trainer saves two files per run:

| File | Contents |
|---|---|
| `best_model_weights.msgpack` | Weights at the epoch with lowest val loss |
| `training_log.csv` | Step-by-step loss, val_loss, lr, elapsed |
| `config.yaml` | Exact config used (reproducible) |
| `model_summary.json` | Parameter count, architecture summary |

Checkpoints use `msgpack` serialisation compatible with Flax NNX state dicts. Load with:

```python
import msgpack
from flax import nnx
from flax.serialization import _msgpack_ext_unpack

with open("runs/my_run/best_model_weights.msgpack", "rb") as f:
    state = msgpack.unpackb(f.read(), ext_hook=_msgpack_ext_unpack, strict_map_key=False)
nnx.update(model, state)
```

---

## Multi-device training

`Trainer` uses JAX SPMD data parallelism via `jax.device_put_replicated`:

```python
cfg = Config.from_yaml("configs/large.yaml")
cfg.n_devices  = 4    # use 4 GPUs
cfg.batch_size = 32   # per-device micro-batch
cfg.grad_accum = 8    # effective batch = 32 × 8 × 4 = 1024
cfg.use_bf16   = True

trainer = Trainer(cfg)
trainer.fit("wiki.txt")
```

Set `n_devices=0` to use all available GPUs automatically.

---

## LR range test

```python
suggested_lr, lr_hist, loss_hist = trainer.find_lr(
    "data/corpus.txt",
    min_lr=1e-7,
    max_lr=1.0,
    num_steps=100,
)
print(f"Suggested LR: {suggested_lr:.2e}")
```

The finder does an exponential sweep from `min_lr` to `max_lr` over `num_steps` steps, records the smoothed loss, and suggests the LR at the point of steepest descent.

---

## See also

- [Configuration Reference](../configuration.md) — all `Config` fields
- [CLI Reference](../cli.md) — `dantinox train` and `dantinox find-lr`
- [Cookbook](../cookbook.md) — end-to-end training recipes
- [Paradigms API](paradigms.md) — `loss_fn` contract
