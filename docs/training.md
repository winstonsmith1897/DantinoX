# Training & Sweeps

## Quick Start

```bash
# Minimal — auto-generates a timestamped run directory
dantinox train --config configs/default_config.yaml --data_path data/corpus.txt

# With bfloat16 and gradient clipping (recommended for GPU training)
dantinox train \
  --config configs/default_config.yaml \
  --data_path data/corpus.txt \
  --use_bf16 True --grad_clip 1.0

# Resume an interrupted run from the last saved checkpoint
dantinox train \
  --config configs/default_config.yaml \
  --data_path data/corpus.txt \
  --run_dir runs/run_20260101_120000 --resume
```

Or via the Python API:

```python
from dantinox import Config, Trainer

config = Config.from_yaml("configs/default_config.yaml")
run_dir = Trainer(config).fit("data/corpus.txt")
print(f"Saved to: {run_dir}")
```

---

## Run Directory

Every training run writes its artifacts to an isolated directory (`runs/run_YYYYMMDD_HHMMSS/` by default, or whatever you pass via `--run_dir`):

| File | Contents |
| :--- | :--- |
| `config.yaml` | Full config snapshot |
| `tokenizer.json` | Serialised vocabulary (no corpus needed at inference) |
| `model_weights.msgpack` | Latest checkpoint (updated every 50 steps) |
| `best_model_weights.msgpack` | Best checkpoint by validation loss |
| `training_cursor.json` | Last saved step (used by `--resume`) |
| `model_summary.json` | Parameter count and memory estimates |
| `training_log.csv` | Per-eval step, train/val loss, bal loss, ms/step |

---

## Gradient Clipping

Enabled by default (`grad_clip = 1.0`). The optimizer is wrapped with `optax.clip_by_global_norm` before the weight update, preventing gradient explosions with large models or high learning rates:

```yaml
# configs/default_config.yaml
grad_clip: 1.0   # set to 0 to disable
```

```python
config = Config(grad_clip=1.0)   # default
config = Config(grad_clip=0.0)   # disabled
```

!!! tip
    For very small models (dim < 128) or low LRs, clipping is rarely needed. For anything larger, keep it at `1.0`.

---

## bfloat16 / Mixed Precision

Set `use_bf16: true` to halve GPU memory use with negligible loss quality impact. All learnable parameters — and their optimizer moments — are cast to `bfloat16` immediately after model construction:

```yaml
use_bf16: true
```

```python
config = Config(use_bf16=True)
run_dir = Trainer(config).fit("data/corpus.txt")
```

The `model_summary.json` will report `"dtype": "bfloat16"` and halved `weights_mem_MB` and `optimizer_mem_MB` estimates.

!!! note "Checkpoints are dtype-preserving"
    Weights are saved in whatever dtype the model is running in. A `bfloat16` checkpoint loads as `bfloat16` in `Generator` automatically — no extra flags needed.

---

## Early Stopping

Set `patience > 0` to stop training automatically when the validation loss has not improved for `patience` consecutive evaluation intervals (every 50 steps):

```yaml
patience: 5   # stop after 5 evals with no improvement (0 = disabled)
```

```python
config = Config(patience=5)
```

The best-ever checkpoint is always written to `best_model_weights.msgpack`, independently of whether early stopping fires.

---

## Checkpoint Resumption

If a run is interrupted, resume from the last saved step with `--resume`:

```bash
dantinox train \
  --config configs/default_config.yaml \
  --data_path data/corpus.txt \
  --run_dir runs/run_20260101_120000 \
  --resume
```

Python API:

```python
run_dir = Trainer(config).fit(
    "data/corpus.txt",
    run_dir="runs/run_20260101_120000",
    resume=True,
)
```

!!! warning "Optimizer state"
    The model weights and step cursor are restored exactly. Optimizer moments (Adam's first and second moments) are **not** preserved — they restart from zero. The learning rate schedule resumes from the saved step, so the LR is correct; only the warm-up of the moments is lost (typically negligible after a few steps).

---

## Learning Rate Finder

Before committing to a long run, use the LR range test (Smith 2015) to identify a good peak learning rate. It sweeps from `min_lr` to `max_lr` over `num_steps` steps and reports the point of steepest loss descent:

```bash
dantinox find-lr \
  --config configs/default_config.yaml \
  --data_path data/corpus.txt \
  --min_lr 1e-6 --max_lr 1e-2 \
  --num_steps 100 --plot
```

```python
from dantinox import Trainer, Config

config = Config.from_yaml("configs/default_config.yaml")
suggested_lr, lr_history, loss_history = Trainer(config).find_lr(
    "data/corpus.txt",
    min_lr=1e-6,
    max_lr=1e-2,
    num_steps=100,
)
print(f"Suggested LR: {suggested_lr:.2e}")
```

The `--plot` flag saves `lr_finder.png` with the smoothed loss curve and a vertical marker at the suggested LR.

!!! tip "How to read the chart"
    Pick the LR just **before** the loss bottoms out — not the minimum itself. The minimum is typically already past the regime where training is stable.

---

## Monitored Metrics

Every 50 steps, training evaluates `eval_iters` random batches on both splits and logs:

| Metric | Description |
| :--- | :--- |
| `train_loss` | Cross-entropy on a random training batch |
| `val_loss` | Cross-entropy on held-out validation data |
| `train_bal` / `val_bal` | MoE routing balance loss (zero for non-MoE models) |
| `ms_per_step` | Milliseconds per training step (wall-clock) |

---

## Hyperparameter Sweeps (W&B)

DantinoX integrates with **Weights & Biases** Bayesian sweeps via the `dantinox sweep` subcommand.

```bash
dantinox sweep \
  --sweep_config configs/sweep.yaml \
  --data_path data/corpus.txt \
  --wandb_project DantinoX \
  --count 50
```

### Example sweep config

```yaml
# configs/sweep.yaml
method: bayes
metric:
  name: val_loss
  goal: minimize
parameters:
  lr:
    distribution: log_uniform_values
    min: 0.0001
    max: 0.005
  batch_size:
    values: [16, 32, 64]
  grad_accum:
    values: [2, 4, 8]
  warmup_steps:
    values: [50, 100, 200]
  dim:
    values: [256, 512]
  num_blocks:
    values: [4, 8, 12]
  optimizer:
    values: ["adamw", "adafactor", "lion"]
  tokenizer_type:
    values: ["char", "bpe"]
  dropout_rate:
    values: [0.0, 0.1, 0.15]
  use_moe:
    values: [true, false]
```

!!! warning "GQA shape consistency"
    Ensure `dim == n_heads × head_size` holds for every trial. Pin `n_heads` and `head_size` in the sweep config and let `dim` derive from them, or add a validation step in your sweep agent.

---

## Training MLA Models

MLA introduces a training/inference split controlled by the `inference` flag:

| Flag | Training | Inference |
| :--- | :--- | :--- |
| `mla` | `true` | `true` |
| `inference` | **`false`** | **`true`** |

Train with `inference: false`. At generation time, `Generator` automatically sets `inference = True` when it detects `mla = True` in the saved config — no manual intervention needed.

!!! note
    The `inference` flag only affects the computation graph, not the saved weights. Switching modes does not require re-saving the checkpoint.

---

## Training Loop: Deep Dive

### Functional State Management

Flax NNX models are stateful Python objects. `jax.jit` requires pure functions, so the training step splits the model into a static graph definition and a dynamic state pytree, operates on the pytree inside JIT, then merges back:

```python
graphdef, state = nnx.split((model, optimizer, metrics))
_, _, new_state = train_step(graphdef, state, x, y)
nnx.update((model, optimizer, metrics), new_state)
```

### Gradient Accumulation

Gradient accumulation is implemented as a manual micro-batch loop inside `@jax.jit`. This keeps the compiled graph fixed while effectively multiplying the batch size by `grad_accum`:

```python
for i in range(config.grad_accum):
    (loss, bal), grads = grad_fn(model, xs[i], ys[i])
    acc = tree_map(lambda a, g: a + g / config.grad_accum, acc, grads)
```

The accumulated gradient `acc` is applied in a single optimizer update step.
