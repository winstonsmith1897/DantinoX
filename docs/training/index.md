---
title: Training
---

# Training

DantinoX supports two training paradigms from a single `Trainer` class.
The paradigm is selected via `model_type` in the config; everything else
— optimiser, schedule, checkpointing, multi-GPU — is shared.

---

## Quick Start

=== "Autoregressive"

    ```bash
    dantinox train \
      --config configs/default_config.yaml \
      --use_bf16 true --n_devices 2
    ```

=== "Diffusion"

    ```bash
    dantinox train \
      --config configs/diffusion_base.yaml \
      --use_bf16 true --n_devices 2
    ```

=== "Python API"

    ```python
    from dantinox import Trainer
    from core.config import Config

    config  = Config.from_yaml("configs/diffusion_base.yaml")
    run_dir = Trainer(config).fit()   # dataset from config
    ```

---

## Pages in this Section

<div class="grid cards" markdown>

-   :material-arrow-right-circle: **[Autoregressive Training](autoregressive.md)**

    Cross-entropy loss, KV-cache, LR finder, LoRA fine-tuning.

-   :material-wave: **[Diffusion Training](diffusion.md)**

    ELBO loss, noise schedules, AdaLayerNorm, time embeddings.

-   :material-tune: **[Hyperparameter Sweeps](sweeps.md)**

    Bayesian W&B sweeps, attention-type comparison sweep.

-   :material-gpu: **[Multi-GPU](multi-gpu.md)**

    SPMD data parallelism with JAX mesh sharding.

</div>

---

## Common Configuration

These options apply to **both** AR and Diffusion training:

| Field | Default | Description |
|---|---|---|
| `epochs` | `3` | Training epochs |
| `batch_size` | `64` | Global batch size |
| `grad_accum` | `4` | Gradient accumulation steps |
| `lr` | `0.001` | Peak learning rate |
| `lr_schedule` | `"cosine"` | `cosine` · `linear` · `constant` · `wsd` |
| `warmup_steps` | `420` | Linear warmup steps |
| `optimizer` | `"adamw"` | `adamw` · `adafactor` · `lion` |
| `grad_clip` | `1.0` | Gradient norm clip (0 = disabled) |
| `use_bf16` | `true` | bfloat16 mixed precision |
| `n_devices` | `2` | GPUs for data-parallel training |
| `patience` | `0` | Early stopping patience (0 = off) |
| `seed` | `42` | PRNG seed |

---

## Run Directory

Every training run saves its artifacts to an isolated directory:

```
runs/<run_name>/
├── config.yaml                 ← full config snapshot
├── tokenizer.json              ← vocabulary (not needed for inference reload)
├── model_weights.msgpack       ← latest checkpoint
├── best_model_weights.msgpack  ← best val-loss checkpoint
├── training_cursor.json        ← resume pointer
├── model_summary.json          ← parameter count and memory estimates
└── training_log.csv            ← step, train_loss, val_loss, ms/step
```

---

## Dataset Pre-Tokenisation Cache

The first training run for a given dataset downloads the corpus from
HuggingFace and tokenises it, then saves:

```
data/<dataset>_<config>_<tokenizer>.npy    ← token ID array (int32)
data/<dataset>_<config>_<tokenizer>.json   ← shared tokenizer
```

All subsequent runs load from these files directly — no re-download,
no re-tokenisation.  This reduces per-run startup from ~60s to ~2s.
