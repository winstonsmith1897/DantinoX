---
title: Diffusion Training
---

# Diffusion Model Training

Set `model_type: "diffusion"` to train a masked discrete diffusion model.

---

## Loss Function

$(1/t)$-weighted masked cross-entropy ELBO, evaluated only at `[MASK]` positions
(the model is conditioned on $x_t$ only — **not** on $t$ itself):

$$
\mathcal{L}_{\text{ELBO}} = \frac{1}{t}\cdot\frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} -\log p_\theta(x_0^{(i)} \mid x_t)
$$

At each training step:

1. Sample a continuous noise level $t \sim \text{Uniform}(0, 1)$ per sample.
2. Corrupt the input $x_0 \to x_t$ using the noise schedule (mask each token with probability $p_{\text{mask}}(t)$).
3. Feed $x_t$ **alone** — not $t$ — to the bidirectional `DiffusionTransformer`.
4. Compute the $(1/t)$-weighted masked CE on the predicted $p_\theta(x_0 \mid x_t)$.

!!! warning "The model does not see t"
    DantinoX's discrete diffusion has **no time conditioning at all** — no
    `AdaLayerNorm`, no time-embedding MLP. The model learns to denoise purely
    from the pattern of `[MASK]` tokens in the input, which implicitly
    encodes the noise level. `t` is only used to weight the loss and to
    determine the masking probability during corruption — it is never passed
    into the model's forward pass. See
    [Discrete Diffusion](../paradigms/diffusion.md#the-denoising-model) for
    details.

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
  mask_token_id: 4            # vocabulary ID of [MASK]
  num_sampling_steps: 50      # fast reverse-diffusion steps at inference

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

## No Time Conditioning

Unlike many diffusion models (and unlike DantinoX's own continuous
flow-matching paradigm), discrete diffusion here has **no time-conditioning
pathway at all**: no sinusoidal time embedding, no `AdaLayerNorm`, no
control tokens. `time_emb_dim` is a real config field, but it belongs to the
continuous flow-matching paradigm (see
[Continuous Flow-Matching Training](continuous.md#config-reference)) — it has
no effect here even if set on a shared `Config` object. The model
distinguishes noise levels purely from *how many* tokens are masked in the
input, which is itself a function of `t` through the noise schedule.

---

## Training Loop Internals

The diffusion `train_step` (simplified):

```python
# Sample a continuous per-sample noise level t in [t_min, 1]
t = jax.random.uniform(rng_t, (B,), minval=t_min, maxval=1.0)

# Corrupt: mask each token independently with probability p_mask(t)
x_t = corrupt(x0, t, rng_c, config.noise_schedule, config.mask_token_id)

# Forward pass — bidirectional, no time conditioning of any kind
out = model(x_t, deterministic=False)

# (1/t)-weighted ELBO — cross-entropy only at masked positions
loss = masked_cross_entropy(out.logits, x0, x_t, config.mask_token_id,
                            t_float=t, aux_loss=out.aux_loss,
                            alpha_balance=model.alpha_balance)
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
