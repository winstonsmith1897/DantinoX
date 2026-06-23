---
title: Training
---

# Training Guide

DantinoX uses a single `Trainer` class for all three paradigms (AR, Diffusion, ELF). This page explains every training option, what it does, and when to change it.

---

## Quick start

=== "CLI"

    ```bash
    # Autoregressive
    dantinox train \
        --config configs/default_config.yaml \
        --data_path data/wiki.txt

    # Diffusion
    dantinox train \
        --config configs/diffusion_base.yaml \
        --data_path data/wiki.txt \
        --use_bf16 true \
        --n_devices 2

    # Override any field inline
    dantinox train \
        --config configs/default_config.yaml \
        --data_path data/wiki.txt \
        --lr 1e-4 \
        --batch_size 128 \
        --optimizer muon
    ```

=== "Python"

    ```python
    from dantinox import Trainer
    from dantinox.core.config import Config

    config  = Config.from_yaml("configs/default_config.yaml")
    run_dir = Trainer(config).fit("data/wiki.txt")
    # run_dir is e.g. "runs/run_20260611_142301"
    ```

=== "One-liner"

    ```python
    import dantinox as dx

    run_dir = dx.fit(
        "ar",                    # paradigm
        "data/wiki.txt",         # data
        dim=512, n_heads=8, head_size=64,
        num_blocks=12, vocab_size=32_000,
        lr=3e-4, epochs=5,
    )
    ```

---

## Training configuration fields

Every field below can be set in a YAML config or overridden on the CLI with `--field value`.

### Learning rate

| Field | Default | Description |
|:------|:-------:|:------------|
| `lr` | `0.001` | **Peak learning rate.** The highest LR reached after warmup. Typical range: `1e-5` (fine-tuning) to `3e-3` (training from scratch). See [find-lr](#find-lr) to auto-detect. |
| `lr_schedule` | `"cosine"` | Schedule applied after warmup. See [Schedules](#schedules). |
| `warmup_steps` | `420` | Linear warmup from LR=0 to `lr` over this many steps. A common rule is 1–5% of total training steps. |

### Batch size and gradient accumulation

| Field | Default | Description |
|:------|:-------:|:------------|
| `batch_size` | `64` | Total number of sequences per optimiser update. This is the **effective** batch size: `batch_size = micro_batch_size × grad_accum`. |
| `grad_accum` | `4` | Number of gradient accumulation micro-steps. Allows simulating a large batch when GPU memory is limited. |

!!! note "Effective batch size"
    **Effective batch size** = `batch_size × n_devices`.

    `batch_size` is the total size summed across all accumulation steps. The micro-batch fed to each step is `batch_size / grad_accum`.

    Example: `batch_size=256, grad_accum=4, n_devices=2` → micro-batch per step = 256/4 = 64, total effective batch = 256 × 2 = 512.

### Optimisers

Set via `optimizer: "..."`:

=== "AdamW (default)"

    ```yaml
    optimizer: "adamw"
    ```

    Adam with decoupled weight decay. The standard choice for most transformer training. Reliable across architectures and scales. Uses per-parameter adaptive learning rates and momentum.

    **When to use:** default for everything. Start here.

=== "Lion"

    ```yaml
    optimizer: "lion"
    ```

    Sign-based gradient update. Uses less memory than AdamW (stores only momentum, not variance) and often trains faster. From the paper *"Symbolic Discovery of Optimization Algorithms"* (Chen et al., 2023).

    **When to use:** when memory is tight, or when you want to experiment with faster convergence. May need a lower LR than AdamW (divide your AdamW LR by ~3–10).

=== "Muon"

    ```yaml
    optimizer: "muon"
    ```

    Momentum Orthogonalized by Newton-Schulz. Applies Newton-Schulz orthogonalization to 2D weight gradients, which makes the update geometry-aware. Falls back to Adam for biases and norms. From optax >= 0.2.6.

    **When to use:** state-of-the-art for transformer pre-training. Often achieves lower loss than AdamW at equivalent compute. Recommended for research runs.

=== "Adafactor"

    ```yaml
    optimizer: "adafactor"
    ```

    Memory-efficient approximation of Adam. Factorises the second-moment matrix instead of storing it per-parameter, saving O(√(m×n)) vs O(m×n) memory per matrix.

    **When to use:** when training very large models (billions of parameters) where AdamW's optimizer state would not fit in memory.

=== "Adam"

    ```yaml
    optimizer: "adam"
    ```

    Classic Adam without weight decay. Avoid for training from scratch (weight decay is important). Useful for final fine-tuning steps.

### Schedules

Set via `lr_schedule: "..."`. All schedules include a linear warmup from 0 to peak LR:

=== "cosine (default)"

    ```yaml
    lr_schedule: "cosine"
    ```

    After warmup, the LR follows a half-cosine decay from `lr` to `lr × 0.01`.

    ```
    LR
    ▲
    │      ╭───╮
    │    ╭─╯   ╰─╮
    │  ╭─╯       ╰───────────
    └──────────────────────── step
      warmup  decay
    ```

    Best default. Smooth decay avoids abrupt LR changes near the end of training.

=== "linear"

    ```yaml
    lr_schedule: "linear"
    ```

    After warmup, the LR decreases linearly from `lr` to `lr × 0.01`. Simpler than cosine.

=== "constant"

    ```yaml
    lr_schedule: "constant"
    ```

    After warmup, the LR stays at `lr` until training ends. Useful for fine-tuning on small datasets or as a baseline.

=== "wsd (warmup-stable-decay)"

    ```yaml
    lr_schedule: "wsd"
    ```

    Three phases: (1) linear warmup → (2) stable plateau at peak LR for 40% of budget → (3) cosine decay to `lr × 0.01`. Shown to work well for very long training runs (Hu et al., 2024).

    ```
    LR
    ▲
    │      ╭───────────╮
    │    ╭─╯           ╰─╮
    │  ╭─╯               ╰────
    └──────────────────────── step
      warmup  stable   decay
    ```

### Gradient clipping

| Field | Default | Description |
|:------|:-------:|:------------|
| `grad_clip` | `1.0` | Maximum global gradient norm. Gradients are rescaled if their global L2 norm exceeds this value. Set to `0` to disable. |

Gradient clipping prevents "exploding gradients" — rare but catastrophic events where a single bad batch causes the gradient to be huge, blowing up the weights. A value of 1.0 is safe for most cases.

### Precision

| Field | Default | Description |
|:------|:-------:|:------------|
| `use_bf16` | `true` | If true, model parameters are cast to `bfloat16` before training. Halves memory usage with minimal quality loss. Requires a GPU with BF16 support (Ampere A100/H100 or newer). |

BFloat16 (bfloat16) has the same exponent range as float32 but fewer mantissa bits. Unlike float16, it rarely causes numerical overflow/underflow in transformer training.

### Multi-GPU

| Field | Default | Description |
|:------|:-------:|:------------|
| `n_devices` | `0` | Number of GPUs to use. `0` = auto-detect and use all available. `1` = single GPU. Must divide `batch_size`. |

DantinoX uses **JAX SPMD data parallelism**: the model is replicated on all devices, the batch is sharded (split) across devices, and gradients are reduced (averaged) automatically.

### Regularisation

| Field | Default | Description |
|:------|:-------:|:------------|
| `dropout_rate` | `0.0` | Dropout applied inside attention and FFN. `0.0` = disabled (standard for LLMs trained on large corpora). Use `0.1`–`0.3` only for small models or fine-tuning. |
| `patience` | `0` | Early stopping: stop training if validation loss has not improved for this many evaluation intervals (each 500 steps). `0` = disabled. |

### Dataset

| Field | Default | Description |
|:------|:-------:|:------------|
| `dataset_source` | `"local"` | `"local"` = read from `data_path` file; `"huggingface"` = load from HuggingFace datasets. |
| `dataset_name` | `""` | HF dataset identifier (e.g. `"wikitext"`) or local file path fallback. |
| `dataset_config` | `""` | HF dataset config (e.g. `"wikitext-103-raw-v1"`). |
| `dataset_split` | `"train"` | HF dataset split to use. |
| `dataset_text_field` | `"text"` | Column name containing the text in a HF dataset. |
| `max_train_tokens` | `10_000_000` | Cap the number of tokens used for training. Useful to run a fixed compute budget regardless of corpus size. Set to `0` to use the full corpus. |
| `tokenizer_type` | `"char"` | `"char"` = character-level tokenizer (simple, built-in); `"bpe"` = Byte-Pair Encoding (requires training); `"t5"` = T5 pre-trained SentencePiece tokenizer. |

### Checkpointing

| Field | Default | Description |
|:------|:-------:|:------------|
| `checkpoint_every` | `2000` | Save a resumable checkpoint every N steps (in addition to the best-val checkpoint). |
| `eval_iters` | `10` | Number of batches averaged for each validation loss estimate. |
| `seed` | `42` | Random seed for data sampling and model initialisation. Set for reproducibility. |

---

## The training loop — what happens at every step

1. **Sample a micro-batch** of `batch_size / grad_accum` sequences from the training corpus
2. **Forward pass**: feed the batch through the model, compute the loss
3. **Backward pass**: compute gradients with `nnx.value_and_grad`
4. **Accumulate**: add scaled gradients (`/ grad_accum`) to the accumulator
5. **Repeat** steps 1–4 `grad_accum` times
6. **Clip gradients** (if `grad_clip > 0`)
7. **Update parameters** with the optimiser
8. Every 500 steps: **evaluate** on a validation subset, log to CSV, save best checkpoint

---

## Run directory structure

Every training run saves its artifacts to an isolated directory:

```
runs/
└── run_20260611_142301/
    ├── config.yaml                ← complete config snapshot (use to reproduce!)
    ├── tokenizer.json             ← tokenizer vocabulary
    ├── best_model_weights.msgpack ← checkpoint with lowest validation loss
    ├── model_weights.msgpack      ← latest resume checkpoint (deleted on completion)
    ├── training_cursor.json       ← resume pointer (step number, deleted on completion)
    ├── model_summary.json         ← parameter count and memory breakdown
    └── training_log.csv           ← step, train_loss, val_loss, ms/step
```

The `model_weights.msgpack` and `training_cursor.json` files exist only during training. When training completes normally, the cursor is deleted. This lets you distinguish a completed run from an interrupted one.

### `model_summary.json` — memory breakdown

```json
{
    "total_params_M": 48.23,
    "dtype": "bfloat16",
    "weights_mem_MB": 96.47,
    "optimizer_mem_MB": 96.47,
    "est_activations_MB": 3276.8
}
```

| Field | Meaning |
|:------|:--------|
| `total_params_M` | Total trainable parameters in millions |
| `dtype` | Floating-point type of stored weights |
| `weights_mem_MB` | Approximate VRAM for model weights |
| `optimizer_mem_MB` | VRAM for optimiser state (AdamW stores m+v → 2× weights) |
| `est_activations_MB` | Estimated VRAM for activations during the backward pass |

---

## Dataset tokenisation cache

The first training run for a given dataset tokenises the corpus and saves:

```
data/<dataset>_<config>_<tokenizer_type>.npy    ← token ID array (int32)
data/<dataset>_<config>_<tokenizer_type>.json   ← shared tokenizer vocab
```

All subsequent runs with the same `(dataset_name, dataset_config, tokenizer_type)` load directly from these files — no re-download, no re-tokenisation. This reduces per-run startup from ~60s to ~2s.

---

## Find-LR — auto-detect the optimal learning rate {#find-lr}

Before committing to a full training run, use the LR range test to find a good starting LR:

```bash
dantinox find-lr \
    --config configs/default_config.yaml \
    --data_path data/wiki.txt \
    --min_lr 1e-7 \
    --max_lr 1.0 \
    --num_steps 100 \
    --plot                # save lr_finder.png
```

**How it works (Smith 2015):**

1. Start with LR = `min_lr`
2. Train for one step, measure loss
3. Multiply LR by a constant factor (exponential sweep)
4. Repeat for `num_steps` steps
5. Plot smoothed loss vs LR
6. Find the LR at the steepest downward slope — this is the suggested LR
7. If the loss diverges (> 4× best loss), stop early

**Rule of thumb:** use a LR that is 3–10× smaller than the LR where loss starts exploding.

---

## Resuming training

If a run is interrupted, resume it from the last saved checkpoint:

```bash
dantinox train \
    --config configs/default_config.yaml \
    --data_path data/wiki.txt \
    --run_dir runs/run_20260611_142301 \
    --resume
```

The trainer will:
1. Read `training_cursor.json` to find the last completed step
2. Load `model_weights.msgpack` into the model
3. Continue training from `start_step + 1`

!!! warning "Optimizer state is not restored"
    When resuming, the optimiser state (momentum, variance) is re-initialised from zero. This causes a brief spike in loss for the first few hundred steps until the optimiser re-warms.

---

## W&B logging

Pass `--wandb_project MyProject` to log to Weights & Biases:

```bash
dantinox train \
    --config configs/default_config.yaml \
    --data_path data/wiki.txt \
    --wandb_project DantinoX_experiments
```

Logged metrics: `train_loss`, `val_loss`, `step`.

---

## Pages in this section

| Page | What it covers |
|:-----|:--------------|
| [Autoregressive Training](autoregressive.md) | AR-specific details: causal mask, teacher-forcing, cross-entropy loss |
| [Diffusion Training](diffusion.md) | Masked diffusion: ELBO loss, noise schedules, continuous t, 1/t weighting |
| [Optimisers & Schedules](optimizers.md) | Deep dive into AdamW, Lion, Muon, Adafactor, WSD |
| [Hyperparameter Sweeps](sweeps.md) | Bayesian W&B sweeps, sweep YAML format |
| [Multi-GPU](multi-gpu.md) | SPMD data parallelism, mesh sharding, batch divisibility |
| [Full Training Suite](emnlp-suite.md) | Reproducing the benchmark results |
