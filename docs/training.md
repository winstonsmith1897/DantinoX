
# Training & Sweeps

## Training Pipeline

The training loop uses Flax NNX functional state management. The core update step fuses the forward pass, loss computation, and gradient updates into a single XLA kernel via `@jax.jit`.

!!! tip "Why `@jax.jit` over `@nnx.jit`?"
    Manually splitting the model (`nnx.split`) and passing only the `state` pytree into a `@jax.jit`-decorated function reduces Python dispatch overhead. For models up to ~100M parameters this yields measurably faster step times than the higher-level `@nnx.jit` wrapper. See the [Flax performance guide](https://flax.readthedocs.io/en/stable/guides/performance.html) for details.

### Execution

```bash
# Run using the default configuration file
python train.py --config configs/default_config.yaml

# Dynamically override parameters via CLI
python train.py --batch_size 64 --lr 5e-4 --use_moe True
```

---

## Monitoring & Logging

Every execution generates an isolated artifact directory (`runs/run_YYYYMMDD_HHMMSS/`) containing the state of the experiment: `config.yaml`, `model_summary.json`, `training_log.csv`, and the serialized `model_weights.msgpack`.

**Live Console Output:**

```text
Step   50/4200 | Train: 4.1204 (Bal: 0.0452) | Val: 4.1560 (Bal: 0.0461) | VRAM: 3.42GB
Step  100/4200 | Train: 3.8901 (Bal: 0.0421) | Val: 3.9102 (Bal: 0.0415) | VRAM: 3.42GB
```

**Tracked Metrics:**

| Metric | Description |
| :--- | :--- |
| **Train / Val Loss** | Cross-Entropy for autoregressive next-token prediction |
| **Balancing Loss** | Auxiliary penalty for MoE expert routing |
| **VRAM GB** | Peak device memory footprint |
| **ms_per_step** | XLA kernel execution speed and throughput |


## Hyperparameter Tuning (W&B Sweeps)

DantinoX natively supports automated hyperparameter search using **Weights & Biases (W&B)**. The search relies on a Bayesian optimization strategy designed to minimize the validation loss (`val_loss`) by efficiently exploring architectural and training configurations.

To launch a sweep, use the provided configuration.

### Sweep Configuration (`sweep.yaml`)

```yaml
program: train_sweep.py
method: bayes
metric:
  name: val_loss
  goal: minimize
parameters:
  epochs:
    values: [12, 16, 20, 24]
  optimizer:
    values: ["adamw", "adafactor", "lion"]
  tokenizer_type:
    values: ["char", "bpe"]
  max_context:
    values: [256, 512]
  weight_tying:
    values: [true, false]
  dropout_rate:
    values: [0.0, 0.1, 0.15]
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
  kv_heads:
    values: [2, 4]
  use_moe:
    values: [true, false]
  n_experts:
    values: [4]
  top_k_mlp:
    values: [1, 2]
  expansion:
    values: [2, 4]
  alpha_balance:
    distribution: uniform
    min: 0.01
    max: 0.15
  sliding_window:
    values: [true, false]
  context_window:
    values: [32, 64, 128]
  no_sink:
    values: [true, false]
  pos_encoding:
    values: ["rotary", "absolute"]
command:
  - ${env}
  - python
  - ${program}
  - ${args}
```

### Execution

Initialize the sweep and start the agent:

```bash
wandb sweep sweep.yaml
wandb agent <USERNAME/PROJECT/SWEEP_ID>
```

!!! warning "GQA Shape Consistency"
    To prevent tensor shape mismatches and XLA compilation crashes during the automated search, `n_heads` and `head_size` are **dynamically derived** from `dim` and `kv_heads` inside `train_sweep.py`. This guarantees that `dim == n_heads × head_size` holds for every Bayesian trial.

---

## Training MLA Models

MLA introduces two additional config flags that interact with the training/inference split:

| Flag | Training | Inference |
| :--- | :--- | :--- |
| `mla` | `true` | `true` |
| `inference` | **`false`** | **`true`** |

During training (`inference: false`) the model materialises full $K$ and $V$ tensors from the latent compression to compute gradients normally. Weight absorption is not differentiable in the same sense and is irrelevant to parameter updates.

At generation time, reload the checkpoint with `inference: true` to activate weight absorption without changing the saved weights:

```python
config = Config.from_yaml("runs/<run>/config.yaml")
config.inference = True          # override for decode — weights are unchanged
model = Transformer(config, rngs=nnx.Rngs(0))
# ... load weights ...
```

!!! note
    The `inference` flag is **not** saved into the checkpoint weights — it only affects the forward-pass computation graph. You can safely switch between training and inference modes without re-saving the model.

---

## Deep Dive: JAX/Flax NNX Training Loop

Training in JAX requires bridging the gap between stateful model architectures and pure, functional transformations like `jax.grad` and `jax.jit`. DantinoX implements a highly optimized update step, explicitly managing the functional state to maximize XLA compilation efficiency.

### 1. Functional State Management

Flax NNX models are stateful Python objects, but `jax.grad` requires pure functions. The solution is to split the model into a static graph definition and a dynamic state pytree, pass only the state into the JIT-compiled function, and merge them back inside `loss_fn`:

```python
graphdef, state = nnx.split(model)   # once, outside the training loop
```

### 2. The Core Update Step

The entire forward pass, loss computation, and backpropagation are fused into a single XLA kernel:

```python
@jax.jit
def train_step(graphdef, state, opt_state, batch):
    def loss_fn(current_state):
        model = nnx.merge(graphdef, current_state)

        logits, _, balancing_loss = model(
            batch['input_ids'], use_cache=False,
            kv_caches=None, cache_index=None, deterministic=False
        )

        # Autoregressive next-token prediction
        ce_loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits[:, :-1, :], labels=batch['labels'][:, 1:]
        ).mean()

        _, new_state = nnx.split(model)
        return ce_loss + balancing_loss, (new_state, balancing_loss, ce_loss)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (new_state, bal_loss, ce_loss)), grads = grad_fn(state)

    updates, new_opt_state = optimizer.update(grads, opt_state, new_state)
    new_state = optax.apply_updates(new_state, updates)
    return new_state, new_opt_state, loss, bal_loss, ce_loss
```

The model is merged and re-split *inside* `loss_fn` so that `jax.value_and_grad` can trace gradient flow through the full state pytree. `has_aux=True` lets the function return the updated state (needed to propagate Dropout PRNG keys) alongside the scalar loss.

### 3. Gradient Accumulation

Python-level accumulation loops break XLA's static graph requirements. DantinoX delegates accumulation entirely to Optax via `MultiSteps`, keeping the compiled graph fixed:

```python
optimizer = optax.MultiSteps(
    optax.adamw(learning_rate=config.lr),
    every_k_schedule=config.grad_accum    # weight update every N micro-steps
)
```

`optax.apply_updates` tracks micro-step state internally and emits a weight update exactly every `grad_accum` calls, with no VRAM spike from intermediate gradient buffers.