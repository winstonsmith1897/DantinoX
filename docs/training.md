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

## LR Schedules

Set `lr_schedule` in the config to choose how the learning rate decays after the warmup phase:

| Schedule | Config value | Behaviour |
| :--- | :--- | :--- |
| **Cosine** (default) | `"cosine"` | Smooth cosine decay from peak to `lr × 0.01` |
| **Linear** | `"linear"` | Linear ramp down to `lr × 0.01` |
| **Constant** | `"constant"` | Flat at peak LR after warmup |
| **WSD** | `"wsd"` | Warmup → stable (40 %) → cosine decay |

```yaml
# configs/default_config.yaml
lr_schedule: "cosine"    # default
warmup_steps: 420        # linear warmup before the schedule kicks in
```

```python
config = Config(lr_schedule="wsd", warmup_steps=500)
run_dir = Trainer(config).fit("data/corpus.txt")
```

!!! tip "Which schedule to pick"
    **Cosine** is the safe default for most runs. **WSD** (Warmup-Stable-Decay) is a good choice for longer runs where you want a sustained high-LR phase before decay. **Constant** is useful when you want full manual control over the LR after warmup.

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

## Loading a Trained Model

Use `Transformer.from_pretrained` to load a trained checkpoint in one line — no need to reconstruct the config or tokenizer manually:

```python
from core import Transformer

# Loads config.yaml + best_model_weights.msgpack from the run directory
model = Transformer.from_pretrained("runs/run_20260101_120000")

# Or load the latest checkpoint instead of the best one
model = Transformer.from_pretrained("runs/run_20260101_120000", best=False)
```

`from_pretrained` automatically:

1. Reads `config.yaml` from the run directory.
2. Constructs the `Transformer` with those settings.
3. Deserialises `best_model_weights.msgpack` (or `model_weights.msgpack` when `best=False`).

!!! note "For text generation use `Generator`"
    `Transformer.from_pretrained` gives you the raw model for custom inference loops, fine-tuning, or probing. For simple text generation, `Generator(run_dir)` is easier — it handles tokenisation and decoding automatically.

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

---

## LoRA Fine-Tuning

LoRA (Low-Rank Adaptation) lets you fine-tune a pre-trained model on new data by training only a small fraction of the parameters — the base weights stay completely frozen.

### How it works

For each adapted linear layer, the effective weight is:

\[
W_{\text{eff}} = W_{\text{base}} + \frac{\alpha}{r} \cdot A B
\]

where \(W_{\text{base}} \in \mathbb{R}^{d \times k}\) is frozen, \(A \in \mathbb{R}^{d \times r}\) and \(B \in \mathbb{R}^{r \times k}\) are trainable (rank \(r \ll \min(d,k)\)), and \(\alpha\) is a scaling constant. \(B\) is initialised to zero so the adapter contributes nothing at the start of fine-tuning.

### Configuration

| Field | Default | Description |
|---|---|---|
| `use_lora` | `False` | Enable LoRA adapters |
| `lora_rank` | `8` | Adapter rank \(r\) |
| `lora_alpha` | `16.0` | Scaling constant \(\alpha\) (effective scale = \(\alpha/r\)) |
| `lora_dropout` | `0.0` | Dropout on the LoRA path |
| `lora_targets` | `"attention"` | Which layers to adapt: `"attention"`, `"mlp"`, or `"all"` |

### Usage

```python
from dantinox import Config, Trainer
from core import Transformer

# 1. Load a pre-trained model
model = Transformer.from_pretrained("runs/run_20260101_120000")

# 2. Create a fine-tuning config — only LoRA fields differ
ft_config = Config.from_yaml("runs/run_20260101_120000/config.yaml")
ft_config.use_lora     = True
ft_config.lora_rank    = 8
ft_config.lora_alpha   = 16.0
ft_config.lora_targets = "attention"   # only Q/K/V/O projections

# 3. Fine-tune — only LoRAParam variables are trained
run_dir = Trainer(ft_config).fit("data/finetune_corpus.txt")
```

```bash
# CLI equivalent
dantinox train \
  --config runs/run_20260101_120000/config.yaml \
  --data_path data/finetune_corpus.txt \
  --use_lora True --lora_rank 8 --lora_targets attention
```

### Trainable parameter count

With `lora_rank=8` and `lora_targets="attention"`, only ~0.1–0.5 % of parameters are trained — making fine-tuning on a single GPU practical even for large models.

### Merge and export

After fine-tuning, merge the LoRA delta back into the base weight for deployment (no runtime overhead):

```python
from core.lora import LoRALinear

for module in model.modules():
    if isinstance(module, LoRALinear):
        merged_kernel = module.merge_weights()   # W_base + (α/r) * A @ B
```

---

## Multi-GPU Data-Parallel Training

DantinoX supports data-parallel training across multiple GPUs using JAX's SPMD sharding. The implementation uses `jax.sharding.Mesh` — no `pmap`, no manual AllReduce. XLA handles gradient synchronisation automatically.

### How it works

| What | How |
|---|---|
| Model weights | Replicated on every device (`NamedSharding(mesh, P())`) |
| Input batch | Sharded along axis 0 (`NamedSharding(mesh, P("data"))`) |
| Gradients | AllReduced automatically by XLA |

Each device computes its share of the forward+backward pass; XLA fuses the AllReduce into the compiled program.

### Configuration

| Field | Default | Description |
|---|---|---|
| `n_devices` | `0` | Number of GPUs to use. `0` = all available, `1` = single-device |

**Constraint:** `batch_size` must be divisible by `n_devices`.

### Usage

```python
config = Config(
    dim=512, n_heads=16, head_size=32, num_blocks=8,
    batch_size=256,   # split evenly across 4 GPUs → 64 per device
    n_devices=4,
)
run_dir = Trainer(config).fit("data/corpus.txt")
```

```bash
dantinox train \
  --config configs/default_config.yaml \
  --data_path data/corpus.txt \
  --n_devices 4 --batch_size 256
```

### Sharding utilities (low-level API)

```python
from core.sharding import make_mesh, replicate, shard_batch, num_devices

mesh = make_mesh(n_devices=4)
print(f"Training on {num_devices(mesh)} GPUs")

# Replicate any pytree across all devices
model_state_replicated = replicate(model_state, mesh)

# Shard a batch along axis 0
x_sharded = shard_batch(x, mesh)   # x.shape = (batch_size, seq_len)
```
