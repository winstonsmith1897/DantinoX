

## Training Pipeline

The training loop leverages Flax NNX functional state management. The core update step uses `@jax.jit`  to fuse the forward pass, loss computation, and optimizer updates into a single, highly optimized **XLA kernel**. There is also the "not splitted version", which uses `@nnx.jit`. However, as per flax documentation (https://flax.readthedocs.io/en/stable/guides/performance.html), the one with `@jax.jit` is faster for smaller model/batch size.

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

> ⚠️ **Technical Note on Grouped Query Attention (GQA):** > To prevent tensor shape mismatches and XLA compilation crashes during the automated search, the total number of query heads (`n_heads`) and head dimensions (`head_size`) are **dynamically calculated** inside `train_sweep.py` based on the selected `dim` and `kv_heads`. This ensures that the attention projections remain mathematically consistent across all Bayesian trials.

## Empirical Results & Ablation Studies

Through extensive hyperparameter sweeps (ID: `cacbxc69`) logged via **Weights & Biases**, we conducted a comprehensive ablation study. By isolating individual architectural choices, we evaluated their direct impact on convergence stability and memory efficiency.

### Architectural Impact on Convergence

| ⚙️ Core Optimization & Routing | 🧠 Attention Mechanisms |
| :---: | :---: |
| ![Optimizer Convergence](assets/loss_by_optimizer.png){ width="100%" } | ![MoE Impact](assets/loss_by_moe.png){ width="100%" } |
| **Convergence by Optimizer:** Isolating the impact of the optimization algorithm across identical architectures. | **Sparse MoE vs Dense:** Evaluating the convergence speed when routing parameters through Top-K experts. |
| ![Sliding Window](assets/loss_by_sliding_window.png){ width="100%" } | ![Attention Sink](assets/loss_by_no_sink.png){ width="100%" } |
| **Sliding Window:** Impact of restricting the attention receptive field on the learning trajectory. | **Attention Sink Gating:** Training stability achieved by applying a sigmoid gate (`no_sink`) to attention outputs. |

---

### Memory & Parameter Efficiency

| 🔗 Parameter Sharing | 💾 Memory Footprint |
| :---: | :---: |
| ![Weight Tying](assets/loss_by_weight_tying.png){ width="100%" } | ![VRAM Footprint](assets/vram_comparison.png){ width="100%" } |
| **Weight Tying:** Convergence behavior when tying the embedding matrix to the output language modeling head. | **Peak VRAM (Dense vs Sparse MoE):** Scaling capacity via MoE while maintaining a constrained VRAM footprint. |

> *Charts generated automatically from W&B Sweep telemetry using the internal plotting scripts.*

---

## 🔬 Deep Dive: JAX/Flax NNX Training Loop

Training in JAX requires bridging the gap between stateful model architectures and pure, functional transformations like `jax.grad` and `jax.jit`. DantinoX implements a highly optimized update step, explicitly managing the functional state to maximize XLA compilation efficiency.

### 1. Functional State Management (Graph vs State)
Flax NNX allows models to be written like standard Python objects, but JAX transformations strictly require pure functions. While DantinoX supports `@nnx.jit` for simplicity, the core training loop uses an explicit `@jax.jit` implementation.

```python
# Extracting the static graph and the dynamic state/weights
graphdef, state = nnx.split(model)
```
**Why do this?** As per the official Flax performance guides, manually splitting the model and passing only the `state` into a `@jax.jit` compiled function significantly reduces Python overhead. For smaller models or smaller batch sizes, this explicit functional split yields noticeably faster step times than the higher-level `@nnx.jit` wrapper.

### 2. The Core Update Step (`@jax.jit`)
The entire forward pass, loss computation, and backpropagation are fused into a single XLA kernel.

```python
@jax.jit
def train_step(graphdef, state, opt_state, batch):
    def loss_fn(current_state):
        # 1. Reconstruct the stateful model inside the pure function
        model = nnx.merge(graphdef, current_state)
        
        # 2. Forward pass (returns logits, kv_cache, and MoE balancing loss)
        logits, _, balancing_loss = model(batch['input_ids'], 
                                          use_cache=False, 
                                          kv_caches=None, 
                                          cache_index=None, 
                                          deterministic=False)
        
        # 3. Autoregressive Cross-Entropy Loss
        # Shifting logits and labels for next-token prediction
        shift_logits = logits[:, :-1, :]
        shift_labels = batch['labels'][:, 1:]
        
        ce_loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=shift_logits, labels=shift_labels
        ).mean()
        
        # 4. Total Loss Calculation
        total_loss = ce_loss + balancing_loss
        
        # 5. Extract the updated state (e.g., updated PRNGs for dropout)
        _, new_state = nnx.split(model)
        
        return total_loss, (new_state, balancing_loss, ce_loss)

    # Compute gradients and extract auxiliary outputs
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (new_state, bal_loss, ce_loss)), grads = grad_fn(state)
    
    # Apply gradients via Optax
    updates, new_opt_state = optimizer.update(grads, opt_state, new_state)
    new_state = optax.apply_updates(new_state, updates)
    
    return new_state, new_opt_state, loss, bal_loss, ce_loss
```

**Key Mechanics:**
* **`loss_fn` purity:** The model is merged and re-split entirely *inside* the loss function. This ensures `jax.value_and_grad` can trace the exact flow of gradients through the state without side-effects.
* **`has_aux=True`:** Allows the loss function to return the updated model state (crucial for updating Dropout PRNG keys) and individual loss metrics alongside the total scalar loss.

### 3. Gradient Accumulation in XLA
For training on standard hardware, matching a large target global batch size requires gradient accumulation. In PyTorch, this is usually a manual loop. In JAX, writing custom accumulation loops breaks the static execution graph.

DantinoX handles this elegantly by pushing the accumulation logic directly into the Optax optimizer definition:

```python
# Handled cleanly via Optax wrapper
optimizer = optax.MultiSteps(
    optax.adamw(learning_rate=config.lr), 
    every_k_schedule=config.grad_accum
)
```
This guarantees that `optax.apply_updates` tracks the micro-steps internally and only alters the weights once every `grad_accum` steps, keeping the XLA graph perfectly static and avoiding unnecessary VRAM spikes.