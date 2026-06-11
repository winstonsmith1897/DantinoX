---
title: Diffusion Training
---

# Diffusion Model Training

Set `model_type: "diffusion"` to train a masked discrete diffusion model.

---

## Loss Function

Masked cross-entropy ELBO, evaluated only at `[MASK]` positions:

$$
\mathcal{L}_{\text{ELBO}} = -\frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \log p_\theta(x_0^{(i)} \mid x_t, t)
$$

At each training step:

1. Sample a random timestep $t \sim \text{Uniform}(1, T)$ per sample.
2. Corrupt the input $x_0 \to x_t$ using the noise schedule.
3. Feed $(x_t, t)$ to the bidirectional `DiffusionTransformer`.
4. Compute masked CE on the predicted $p_\theta(x_0 \mid x_t, t)$.

---

## Quick Start

```bash
dantinox train \
  --config configs/diffusion_base.yaml \
  --use_bf16 true \
  --n_devices 2
```

---

## Config Reference

```yaml
model:
  model_type: "diffusion"
  dim: 256
  n_heads: 8
  head_size: 32
  num_blocks: 12
  max_context: 512

diffusion:
  diffusion_steps: 1000       # total forward-process steps T
  noise_schedule: "cosine"    # "cosine" | "linear" | "sqrt"
  mask_token_id: 0            # vocabulary ID of [MASK]
  num_sampling_steps: 50      # fast reverse-diffusion steps at inference
  time_emb_dim: 256           # TimeEmbedding MLP output dimension

training:
  lr: 0.001
  batch_size: 64
  grad_accum: 4
  epochs: 3
  optimizer: "adamw"
  n_devices: 2
  use_bf16: true
```

---

## Noise Schedule Choice

The schedule affects how quickly tokens are masked during the forward process.

```python
from dantinox.core.diffusion import make_noise_schedule
from dantinox.core.config import Config

config   = Config(diffusion_steps=1000, noise_schedule="cosine")
schedule = make_noise_schedule(config)   # NoiseSchedule(alpha_bar=[T+1])
```

| Schedule | Training stability | Inference quality | Notes |
|---|---|---|---|
| **cosine** | ✓✓ | ✓✓ | Default — slow masking near boundaries |
| linear | ✓ | ✓ | Simple; over-masks at large $t$ |
| sqrt | ✓ | ✓ | Intermediate; decelerating mask rate |

---

## Time Embedding

Each transformer block receives a time-step conditioning vector computed by:

```
t (integer) → sinusoidal(t, model_dim) → Linear → SiLU → Linear → c
```

where `c ∈ ℝ^{time_emb_dim}` is used by `AdaLayerNorm` to modulate scale and shift.

Larger `time_emb_dim` improves the model's ability to distinguish fine-grained
timestep differences; 256 is a good default for models up to ~50M parameters.

---

## Training Loop Internals

The diffusion `train_step` (simplified):

```python
# Sample random timestep per sample
t   = jax.random.randint(rng, (B,), 1, config.diffusion_steps + 1)

# Corrupt: replace tokens with [MASK] at rate 1 - alpha_bar[t]
x_t = corrupt(x0, t, rng, schedule, config.mask_token_id)

# Forward pass (bidirectional, with AdaLayerNorm conditioning on t)
out  = model(x_t, t, deterministic=False)

# ELBO loss — only at masked positions
loss = masked_cross_entropy(out.logits, x0, x_t, config.mask_token_id,
                            out.aux_loss, model.alpha_balance)
```

---

## Monitoring Training

The same `training_log.csv` is written as for AR:

| Column | Description |
|---|---|
| `train_loss` | ELBO at randomly sampled $t$ on training data |
| `val_loss` | ELBO on held-out validation data |
| `train_bal` | MoE balance loss (0 for dense models) |
| `ms_per_step` | Wall-clock time per step |

A decreasing `val_loss` means the model is learning to predict masked tokens
more accurately — equivalent to decreasing perplexity.

!!! note "Comparing AR and Diffusion val_loss"
    AR val_loss and Diffusion val_loss are not directly comparable because they
    measure different objectives (next-token CE vs masked CE at random $t$).
    Use bits-per-byte (bpb) from `benchmarks/perplexity_eval.py` for fair
    cross-paradigm quality comparison.

---

## Checkpoint Loading

```python
from dantinox.core.model import DiffusionTransformer

model = DiffusionTransformer.from_pretrained("runs/diff_mha_256d_12b_Dense")
```

After loading, run `fast_dllm_generate` for inference
(see [Diffusion Inference](../inference/diffusion.md)).
