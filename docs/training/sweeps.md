---
title: Hyperparameter Sweeps
---

# Hyperparameter Sweeps

DantinoX integrates with **Weights & Biases** Bayesian sweeps.
The `dantinox sweep` subcommand launches a W&B agent that samples
hyperparameters according to a YAML specification.

---

## Quick Start

```bash
dantinox sweep \
  --sweep_config configs/sweep.yaml \
  --wandb_project DantinoX \
  --count 50
```

---

## Sweep Configs

### Attention-type comparison sweep

`configs/attention_sweep.yaml` runs all three attention types
(MHA · GQA · MLA) with random hyperparameter combinations:

```yaml
program: train_sweep_attention_comparison.py
method: random
metric:
  name: val_loss
  goal: minimize

parameters:
  attention_type:
    values: ["standard_mha", "standard_gqa", "mla"]
  dim:
    values: [256, 512]
  num_blocks:
    values: [8, 12, 16]
  lr:
    distribution: log_uniform_values
    min: 0.0001
    max: 0.0015
  optimizer:
    values: ["adamw", "lion"]
  use_moe:
    values: [true, false]
```

### Full ablation sweep

`configs/sweep.yaml` covers all major hyperparameters:

```yaml
method: bayes
parameters:
  lr:
    distribution: log_uniform_values
    min: 0.0001
    max: 0.005
  batch_size:
    values: [16, 32, 64]
  optimizer:
    values: ["adamw", "adafactor", "lion"]
  dropout_rate:
    values: [0.0, 0.1, 0.15]
  use_moe:
    values: [true, false]
  use_swiglu:
    values: [true, false]
  norm_type:
    values: ["layernorm", "rmsnorm"]
  lr_schedule:
    values: ["cosine", "wsd"]
```

---

## EMNLP Training Suite

For the systematic 180-run comparison (84 AR + 96 Diffusion), use the
pre-built shell scripts that sweep across attention type, model size,
FFN variant, and 14 ablation axes:

```bash
# Dry run — see all commands without executing
bash scripts/train_ar_suite.sh --dry-run
bash scripts/train_diffusion_suite.sh --dry-run

# Full run (2 GPUs, WikiText-103)
bash scripts/train_ar_suite.sh          # 84 runs
bash scripts/train_diffusion_suite.sh   # 96 runs
```

### Filter by axis

```bash
PART=A   bash scripts/train_ar_suite.sh   # size × attention matrix only
PART=B   bash scripts/train_ar_suite.sh   # ablations only
ATTN=mla bash scripts/train_ar_suite.sh   # MLA only
DIM=256  bash scripts/train_ar_suite.sh   # 256-dim only
```

### Ablation axes (Part B)

| Suffix | What changes vs baseline (256d 12b Dense) |
|---|---|
| `RMSNorm` | `norm_type: rmsnorm` |
| `Drop0` / `Drop20` | `dropout_rate: 0.0` / `0.20` |
| `GELU` | `use_swiglu: false` |
| `SlidingWin64` | `sliding_window: true, context_window: 64` |
| `NoSink` | `no_sink: true` |
| `SchedWSD` | `lr_schedule: wsd` |
| `SchedLinear` / `SchedSqrt` | diffusion noise schedule |
| `T500` | `diffusion_steps: 500` |
| `BS128` | `batch_size: 128` |
| `Ctx256` / `Ctx1024` | `max_context: 256` / `1024` |
| `MoE8exp` | `use_moe: true, n_experts: 8, top_k_mlp: 2` |
