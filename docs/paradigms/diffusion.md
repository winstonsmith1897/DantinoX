---
title: Masked Diffusion
---

# Masked Discrete Diffusion

DantinoX implements **Masked Language-Model Diffusion** (MDLM, Austin et al. 2021)
for discrete sequences.  Unlike continuous diffusion models, the forward process
replaces tokens with a special `[MASK]` token; the reverse process predicts the
original token at each masked position.

---

## Forward Process

Each token $x_0^{(i)}$ is independently masked at timestep $t$:

$$
q(x_t^{(i)} \mid x_0^{(i)}) =
\begin{cases}
x_0^{(i)} & \text{with probability } \bar\alpha_t \\
\texttt{[MASK]} & \text{with probability } 1 - \bar\alpha_t
\end{cases}
$$

$\bar\alpha_t$ is the **noise schedule** — a monotonically decreasing function
from $\bar\alpha_0 = 1$ (clean) to $\bar\alpha_T \approx 0$ (fully masked).

### Noise Schedules

Three schedules are available via `noise_schedule`:

=== "cosine (default)"

    $$\bar\alpha_t = \cos^2\!\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right) \bigg/ \bar\alpha_0$$

    Slow masking near $t=0$ and $t=T$; best empirical quality.

=== "linear"

    $$\bar\alpha_t = 1 - \frac{t}{T}$$

    Simple baseline; tends to mask too aggressively at large $t$.

=== "sqrt"

    $$\bar\alpha_t = 1 - \sqrt{\frac{t}{T} + \epsilon}$$

    Intermediate; masking decelerates over time.

---

## Reverse Process (Denoising Model)

A **bidirectional** `DiffusionTransformer` is trained to predict the clean sequence
$x_0$ from the noisy sequence $(x_t, t)$:

$$
p_\theta(x_0 \mid x_t, t)
$$

Key differences from the AR transformer:

| | AR | Diffusion |
|---|---|---|
| Attention mask | Causal | None (full) |
| Time conditioning | — | AdaLayerNorm (DiT-style) |
| Block type | `ARBlock` | `DiffusionBlock` |
| Cache | Static KV-cache | DualCache |

### AdaLayerNorm

Each transformer block modulates its normalisation with a learned time-step embedding:

$$
\text{AdaLN}(\mathbf{x}, \mathbf{c}) = (1 + \gamma(\mathbf{c})) \cdot \text{LN}(\mathbf{x}) + \beta(\mathbf{c})
$$

where $\mathbf{c}$ is the output of the `TimeEmbedding` MLP:

$$
\mathbf{c} = \text{MLP}\!\left(\text{sinusoidal}(t)\right), \quad \mathbf{c} \in \mathbb{R}^{d_{\text{emb}}}
$$

---

## Training Loss (ELBO)

The loss is the masked cross-entropy evaluated only at masked positions:

$$
\mathcal{L}_{\text{ELBO}} = -\frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \log p_\theta(x_0^{(i)} \mid x_t, t)
$$

where $\mathcal{M} = \{i : x_t^{(i)} = \texttt{[MASK]}\}$.

```yaml
model:
  model_type: "diffusion"
  dim: 256
  n_heads: 8
  head_size: 32
  num_blocks: 12
  max_context: 512

diffusion:
  diffusion_steps: 1000
  noise_schedule: "cosine"
  mask_token_id: 0
  num_sampling_steps: 50
  time_emb_dim: 256
```

---

## Reverse Sampling (Inference)

### Simple MDLM Sampler

`diffusion_generate` runs `num_sampling_steps` denoising steps over the full sequence:

```python
from core.generation import diffusion_generate
from core.diffusion import make_noise_schedule

schedule = make_noise_schedule(config)
tokens   = diffusion_generate(
    model,
    prefix,                       # [B, T_prefix]
    gen_len=128,
    schedule=schedule,
    mask_token_id=0,
    num_sampling_steps=50,
    temperature=1.0,
)
```

At each step $t$:

1. Predict $p_\theta(x_0 \mid x_t, t)$.
2. Sample $\hat x_0$ from the predicted distribution.
3. Re-corrupt $\hat x_0$ to $x_{t-1}$ using the schedule.

### Fast-dLLM Sampler (recommended)

For large sequences, use the block-wise DualCache sampler
(see [Fast-dLLM](fast-dllm.md)):

```python
from core.generation import fast_dllm_generate

tokens = fast_dllm_generate(
    model, prefix, gen_len=256,
    schedule=schedule,
    mask_token_id=0,
    block_size=32,
    steps_per_block=20,
    confidence_threshold=0.9,
)
```

---

## Configuration Reference

| Field | Default | Description |
|---|---|---|
| `diffusion_steps` | `1000` | Forward process timesteps $T$ |
| `noise_schedule` | `"cosine"` | `"cosine"` · `"linear"` · `"sqrt"` |
| `mask_token_id` | `0` | Vocabulary ID of `[MASK]` |
| `num_sampling_steps` | `50` | Fast reverse-diffusion steps |
| `time_emb_dim` | `256` | Output dimension of TimeEmbedding MLP |
