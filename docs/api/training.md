# `dantinox.training`

The training module is paradigm-agnostic. It calls `paradigm.loss_fn` at every step and owns checkpointing, logging, and multi-device replication.

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
