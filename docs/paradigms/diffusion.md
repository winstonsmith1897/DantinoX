---
title: Masked Diffusion (LLaDA)
---

# Masked Discrete Diffusion (LLaDA)

DantinoX implements **LLaDA** (Large Language Diffusion with mAsking, arXiv:2502.09992): a masked discrete diffusion model for language generation. Unlike autoregressive models that generate one token at a time, LLaDA generates **all tokens in parallel**, iteratively revealing them from a fully masked starting point.

---

## The core idea

Instead of generating token by token from left to right, the model starts with a completely masked sequence and iteratively *unmasks* positions until the entire sequence is revealed. This has two key consequences:

1. **Bidirectional context**: unlike AR, the model sees the whole (partially masked) sequence when predicting a token. This allows it to condition a token on what comes *after* it — which AR cannot do.

2. **Parallel generation**: all unmasked tokens are predicted simultaneously per step, so generation time scales with the number of *steps*, not the number of *tokens*.

---

## Forward process (adding noise)

The forward process is simple: at noise level $t \in [0, 1]$, each token is independently replaced by `[MASK]` with probability $t$, or kept as the original with probability $1 - t$:

$$
q(x_t^{(i)} \mid x_0^{(i)}) =
\begin{cases}
x_0^{(i)} & \text{with probability } 1 - t \\
\texttt{[MASK]} & \text{with probability } t
\end{cases}
$$

At $t = 0$: the sequence is completely clean (no masks).  
At $t = 1$: the sequence is completely masked.

This is much simpler than continuous diffusion (no score function, no SDE/ODE). The "noise level" $t$ directly equals the expected fraction of masked tokens.

!!! note "Continuous t in DantinoX"
    DantinoX uses **continuous** $t \in [0, 1]$, not discrete timesteps. During training, a different $t$ is sampled per sequence in the batch, allowing the model to learn denoising at all noise levels simultaneously.

### Noise schedules

The `noise_schedule` setting controls whether $t$ is sampled uniformly or with a skewed distribution. In DantinoX, the schedule affects the `corrupt` function which transforms `t` into a masking probability.

| `noise_schedule` | Masking probability from $t$ | Notes |
|:----------------|:----------------------------:|:------|
| `"linear"` | $p_{\text{mask}} = t$ | Uniform masking, simplest baseline |
| `"cosine"` | $p_{\text{mask}} = 1 - \cos^2(\pi t / 2)$ | Slow near $t=0$, faster near $t=1$; better quality |
| `"sqrt"` | $p_{\text{mask}} = 1 - \sqrt{1 - t}$ | Intermediate behaviour |

---

## The denoising model

A **bidirectional** `DiffusionTransformer` is trained to predict the original clean token $x_0^{(i)}$ at every masked position, given the partially masked sequence $x_t$:

$$
p_\theta(x_0^{(i)} \mid x_t), \quad \forall i : x_t^{(i)} = \texttt{[MASK]}
$$

**Key architectural difference from AR:** there is **no causal mask**. Every position attends to every other position (full bidirectional attention). The model sees the entire context — both what is already unmasked and where the masks are.

**No time conditioning.** Unlike many diffusion models, the DantinoX LLaDA implementation does **not** pass $t$ to the model as a conditioning input (no AdaLayerNorm, no time embedding MLP). The model learns to denoise purely from the pattern of masks in the input — which implicitly encodes the noise level. This simplification follows the LLaDA paper and works well in practice.

---

## Training loss — the 1/t ELBO weighting

The loss is masked cross-entropy, but with a critical weighting factor of $1/t$:

$$
\mathcal{L} = \frac{1}{t} \cdot \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} -\log p_\theta(x_0^{(i)} \mid x_t)
$$

where $\mathcal{M} = \{i : x_t^{(i)} = \texttt{[MASK]}\}$ is the set of masked positions.

**Why $1/t$?** This comes from the ELBO (Evidence Lower BOund) derivation in the LLaDA paper (Eq. 3). At small $t$ (few masks), only a few tokens are masked, but each one carries a lot of information — the $1/t$ factor amplifies these rare but informative training signals so that all noise levels contribute equally in expectation.

### t_min — preventing gradient instability

At very small $t$ (e.g. $t = 1/512 = 0.002$ for a sequence of 512 tokens), the $1/t$ factor becomes ~500, causing enormous gradient variance. DantinoX clamps the minimum $t$ to:

$$
t_{\text{min}} = \max\!\left(\frac{1}{L}, 0.05\right)
$$

where $L$ is `max_context`. For typical context lengths (≥128), this resolves to `t_min = 0.05`, meaning at least ~26 tokens are masked per sequence. This keeps gradients stable while still covering the full denoising range.

### Per-sequence noise levels

Each sequence in the micro-batch gets its own independently sampled $t$:

```python
t_batch = jax.random.uniform(key, (batch_size,), minval=t_min, maxval=1.0)
```

This means each step optimises the ELBO at multiple noise levels simultaneously, reducing gradient variance compared to using a single $t$ for the entire batch.

---

## Training configuration

```yaml title="configs/diffusion_base.yaml"
# Architecture
model_type: "diffusion"
dim: 512
n_heads: 8
head_size: 64
num_blocks: 12
max_context: 512
causal: false              # ← MUST be false: bidirectional attention

# Tokenizer
tokenizer_type: "bpe"
mask_token_id: 4           # vocabulary ID reserved for [MASK]
vocab_size: 32000

# Diffusion settings
noise_schedule: "cosine"   # "cosine" | "linear" | "sqrt"

# Training
lr: 3e-4
batch_size: 64
grad_accum: 4
epochs: 20                 # diffusion typically needs more epochs than AR
optimizer: "adamw"
lr_schedule: "cosine"
warmup_steps: 400
use_bf16: true
```

!!! warning "`causal: false` is mandatory"
    The `DiffusionTransformer` must use bidirectional (non-causal) attention.
    Setting `causal: true` will apply a lower-triangular mask, preventing the model
    from seeing future unmasked tokens — this destroys the denoising signal.

!!! warning "`mask_token_id` must be reserved"
    `mask_token_id` must be an ID that does not appear as a regular token in your vocabulary.
    If using BPE, add it as a special token. For character tokenizers, use an index ≥ vocab_size - 1.

---

## Inference — reverse process

Generation starts from a fully masked sequence and runs $N$ denoising steps, each step revealing some tokens.

### Simple sampler (`diffusion_generate`)

```python
from dantinox.core.generation import diffusion_generate
from dantinox.core.diffusion import make_noise_schedule

schedule = make_noise_schedule(cfg)

tokens = diffusion_generate(
    model,
    prefix=prefix_ids,         # [B, T_prefix] — already-known prefix (can be empty)
    gen_len=128,               # number of tokens to generate
    schedule=schedule,
    mask_token_id=cfg.mask_token_id,
    num_sampling_steps=50,     # number of denoising steps
    temperature=1.0,
)
```

**What happens at each step $s$ (from $t=1$ down to $t=0$):**

1. The model predicts $p_\theta(x_0 \mid x_t)$ for every masked position
2. A sample $\hat{x}_0$ is drawn from the predicted distribution
3. $\hat{x}_0$ is re-masked at the noise level of the next step $t_{s-1}$ — tokens that have high confidence are kept unmasked, lower-confidence ones are re-masked
4. Repeat until all positions are unmasked

### Fast-dLLM sampler (`fast_dllm_generate`) — recommended

For long sequences, the block-wise sampler gives 1.4–2.1× speedup:

```python
from dantinox.core.generation import fast_dllm_generate

tokens = fast_dllm_generate(
    model,
    prefix=prefix_ids,
    gen_len=256,
    schedule=schedule,
    mask_token_id=cfg.mask_token_id,
    block_size=32,             # process 32 tokens per block
    use_dual_cache=True,       # cache KV from already-unmasked blocks
    confidence_threshold=0.9,  # commit a token once confidence ≥ 90%
    steps_per_block=20,        # denoising steps applied within each block
)
```

**How DualCache works:**

The key insight is that once a block of tokens is committed (unmasked with high confidence), their KV representations do not change. The dual cache:

- **Cache A**: KV from already-committed (left) blocks — never recomputed
- **Cache B**: KV from the current block being decoded — recomputed each step

This avoids recomputing attention over the entire sequence for every denoising step, achieving speedups proportional to the number of committed blocks.

---

## CLI generation

```bash
# Standard diffusion generation
dantinox generate \
    --run_dir runs/diffusion_512d \
    --prompt "In the beginning" \
    --n_steps 50 \
    --max_new_tokens 128

# Fast-dLLM with DualCache
dantinox generate \
    --run_dir runs/diffusion_512d \
    --prompt "In the beginning" \
    --n_steps 50 \
    --block_size 32 \
    --use_dual_cache \
    --confidence_threshold 0.9 \
    --max_new_tokens 256
```

---

## Comparison: LLaDA vs standard AR

| Property | Autoregressive | LLaDA (Masked Diffusion) |
|:----------|:-------------:|:------------------------:|
| Attention | Causal (left only) | Full bidirectional |
| Generation order | Left to right, one token | All positions, parallel |
| Time conditioning | None | None (in DantinoX) |
| Inference cost | O(T) steps of O(T) each | N steps, each O(T) in parallel |
| Can revise tokens | No | Yes (re-masking) |
| Quality | Good baseline | Better on long-range coherence |
| Speed (wall clock) | Fast with KV-cache | Slower, improves with Fast-dLLM |

---

## Configuration reference

| Field | Type | Default | Description |
|:------|:----:|:-------:|:------------|
| `model_type` | `str` | `"autoregressive"` | Must be `"diffusion"` |
| `causal` | `bool` | `true` | Must be `false` for diffusion |
| `noise_schedule` | `str` | `"cosine"` | `"cosine"` \| `"linear"` \| `"sqrt"` |
| `mask_token_id` | `int` | `0` | Vocabulary ID of the `[MASK]` token |
| `num_sampling_steps` | `int` | `50` | Reverse-diffusion steps at inference |
| `diffusion_steps` | `int` | `1000` | Forward-process steps (for schedule) |

---

## See also

- [Fast-dLLM DualCache](fast-dllm.md) — detailed explanation of the block-wise caching speedup
- [Confidence-Aware Decoding](confidence.md) — how confidence thresholds work
- [AR vs Diffusion Comparison](comparison.md) — throughput and quality benchmarks
- [Training Guide](../training/index.md) — optimisers, schedules, multi-GPU
