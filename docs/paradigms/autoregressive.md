---
title: Autoregressive Paradigm
---

# Autoregressive (AR) Generation

The autoregressive paradigm is the classical approach to language modelling: the model generates one token at a time, reading all previously generated tokens to predict the next one. It is the simplest paradigm to train and often the fastest to run at inference, thanks to the KV-cache.

---

## How it works — the core idea

Given an input sequence of tokens $x_1, x_2, \ldots, x_t$, the model predicts the probability distribution over the next token $x_{t+1}$:

$$
p_\theta(x_{t+1} \mid x_1, x_2, \ldots, x_t)
$$

To generate a full sequence of length $N$, this prediction is applied repeatedly:

1. Feed the prompt tokens to the model → get distribution over $x_{t+1}$
2. Sample (or take the argmax) of $x_{t+1}$
3. Append $x_{t+1}$ to the context
4. Feed the new context → get distribution over $x_{t+2}$
5. Repeat until `max_new_tokens` tokens are generated or an EOS token appears

This is called **autoregressive decoding** because each output token is conditioned on all previously generated tokens.

---

## Architecture

```
Input tokens  x₁  x₂  …  xₜ
       │
       ▼
  Token Embedding  [vocab_size → dim]
       │  (weight-tied to the LM head projection)
       ▼
  + Positional Encoding  (RoPE / Sinusoidal / Learned)
       │
       ▼
  Dropout  (only during training; disabled at inference)
       │
 ┌─────────────────────────────────────────┐
 │  Transformer Block  ×  num_blocks       │
 │                                         │
 │  ┌─ Pre-Norm (LayerNorm or RMSNorm) ────┤
 │  │                                      │
 │  ├─ Attention (MHA / GQA / MLA)        │
 │  │   └─ Causal mask: position i can    │
 │  │      only attend to positions ≤ i   │
 │  │                                      │
 │  ├─ Residual connection                 │
 │  │                                      │
 │  ├─ Pre-Norm (LayerNorm or RMSNorm) ────┤
 │  │                                      │
 │  └─ Feed-Forward (Dense MLP or MoE)    │
 └─────────────────────────────────────────┘
       │
       ▼
  Final LayerNorm
       │
       ▼
  LM Head  [dim → vocab_size]
       │  (shares weights with Token Embedding if weight_tying=true)
       ▼
  Logits  [batch_size, seq_len, vocab_size]
```

The causal mask is what makes the model "autoregressive": token at position $i$ can only see tokens at positions $0, 1, \ldots, i$. This is implemented as a lower-triangular boolean mask applied to the attention weights before softmax.

---

## Training objective

Training uses **teacher-forcing**: the entire ground-truth sequence is fed as input, and the model predicts every next token simultaneously. This avoids the slow left-to-right generation at training time.

**Loss function:**

$$
\mathcal{L}_{AR} = -\frac{1}{T} \sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})
$$

This is the average cross-entropy loss across all positions in the sequence. The causal mask ensures the model never "cheats" by looking at future tokens during training.

**What "teacher-forcing" means:** instead of using the model's own predictions as the next input (which would accumulate errors), we always feed the ground-truth token as input. This makes training much faster and more stable, though it creates a slight train/inference discrepancy (the model never sees its own errors during training).

### Training configuration

```yaml title="configs/default_config.yaml"
# Architecture
model_type: "autoregressive"
dim: 512
n_heads: 8
head_size: 64           # dim = n_heads × head_size  ← strict constraint
num_blocks: 12
max_context: 1024       # maximum sequence length
vocab_size: 32000

# Attention variant
attention_type: "mha"   # "mha" | "gqa" | "mla"
kv_heads: 2             # used only for GQA (n_heads > kv_heads)

# FFN
use_swiglu: true        # SwiGLU activation (recommended)
use_moe: false          # sparse Mixture of Experts (off by default)

# Normalisation
norm_type: "rmsnorm"    # "rmsnorm" | "layernorm"

# Positional encoding
use_rotary_pos: true    # RoPE (recommended)

# Regularisation
dropout_rate: 0.0       # 0 = disabled (common for LLMs)
weight_tying: true      # tie embedding ↔ LM head (saves vocab_size × dim params)

# Training
lr: 3e-4
batch_size: 64
grad_accum: 4
epochs: 5
optimizer: "adamw"
lr_schedule: "cosine"
warmup_steps: 400
grad_clip: 1.0
use_bf16: true
```

---

## KV-Cache — how it works and why it matters

Without a KV-cache, generating token $t+1$ would require recomputing the attention keys and values for tokens $1, \ldots, t$ from scratch. This is O(T²) work per token. With the KV-cache, those keys and values are computed once and reused.

### Prefill phase (the prompt)

When the prompt is fed to the model, all positions are computed in a single forward pass. The keys and values for every prompt token are stored in the cache:

```
K-cache[layer, :, 0:T_prompt, :] = K_prompt    # shape: [B, H_kv, T_prompt, d_h]
V-cache[layer, :, 0:T_prompt, :] = V_prompt
cache_index = T_prompt                           # write pointer
```

### Decode phase (one token at a time)

For each new token, only a single forward pass of the query is needed. The new key and value are written at `cache_index` and all previous entries are read:

```python
# Inside the attention layer, decode step:
k_cache = jax.lax.dynamic_update_slice(k_cache, k_new, (0, 0, 0, cache_index, 0))
v_cache = jax.lax.dynamic_update_slice(v_cache, v_new, (0, 0, 0, cache_index, 0))
# Attention over all past + current tokens
attn = query @ k_cache[:, :, :cache_index+1, :]  # O(T) not O(T²)
```

This brings inference from O(T²) per token down to O(T) per token.

### Cache memory usage

The cache size depends on the attention variant:

$$
\text{KV-MB} = L \times 2 \times S \times H_{kv} \times d_h \times \text{bytes per value}
$$

where $L$ = num_blocks, $S$ = sequence length, $H_{kv}$ = number of KV heads.

| Variant | $H_{kv}$ | Cache at 1024 tok, 12 blocks, fp32 |
|:--------|:--------:|:----------------------------------:|
| MHA (H=8) | 8 | 768 KB |
| GQA (H_kv=2) | 2 | 192 KB |
| MLA | $d_c^{KV}/d_h$ ≈ 1 | ~48 KB |

---

## Generation — Python API

### `generate` function

```python
from dantinox.core.generation import generate
from dantinox.core.model import Transformer
from dantinox.core.config import Config
from flax import nnx

cfg   = Config.from_yaml("runs/ar_512d/config.yaml")
model = Transformer(cfg, rngs=nnx.Rngs(0))
# ... load weights ...

tokens = generate(
    model,
    prompt_ids,            # int32 array of shape [B, T_prompt]
    max_generations=200,   # how many new tokens to generate
    use_cache=True,        # use KV-cache (strongly recommended for speed)
    top_p=0.9,             # nucleus sampling: keep tokens summing to 90% prob
    temperature=1.0,       # 1.0 = unchanged, <1.0 = sharper, >1.0 = flatter
    top_k=None,            # if set, keep only top-k tokens before sampling
    seed=42,               # random seed for reproducibility
)
# tokens: int32 array of shape [B, T_prompt + max_generations]
```

**Parameters explained:**

| Parameter | Effect |
|:----------|:-------|
| `top_p` (nucleus) | Sort tokens by probability, keep the smallest set whose cumulative probability ≥ p, sample from it. `top_p=1.0` = no filtering. |
| `temperature` | Divide logits by temperature before softmax. Low value (e.g. 0.3) makes the model very focused. High value (e.g. 1.5) makes it more creative/random. |
| `top_k` | Keep only the k most probable tokens. Often combined with temperature. |
| Greedy | Set `top_p=None, top_k=1, temperature=1.0` to always pick the most probable token. |

### `Generator` class (high-level)

```python
from dantinox.generator import Generator

gen = Generator("runs/ar_512d_12b")  # loads config + weights automatically

# Standard generation
text = gen.generate(
    "In the beginning",
    max_new_tokens=200,
    top_p=0.9,
    temperature=0.8,
)
print(text)

# Streaming (print tokens as they are produced)
for chunk in gen.stream("Chapter 1:", top_p=0.95):
    print(chunk, end="", flush=True)
```

The `Generator` class handles:

1. Loading `config.yaml` from the run directory
2. Instantiating the correct model class based on `model_type`
3. Loading `best_model_weights.msgpack` into the model
4. Running the tokenizer for encoding prompts and decoding outputs

---

## Generation — CLI

```bash
# Basic generation
dantinox generate \
    --run_dir runs/ar_512d_12b \
    --prompt "In the beginning" \
    --max_new_tokens 200

# Streaming with nucleus sampling
dantinox generate \
    --run_dir runs/ar_512d_12b \
    --prompt "Once upon a time" \
    --top_p 0.9 \
    --temperature 0.8 \
    --stream

# Greedy decoding (deterministic)
dantinox generate \
    --run_dir runs/ar_512d_12b \
    --prompt "The capital of France is" \
    --greedy

# Disable KV-cache (for debugging; much slower)
dantinox generate \
    --run_dir runs/ar_512d_12b \
    --prompt "Hello" \
    --no_cache
```

---

## Sampling strategies explained

### Greedy decoding
Always picks $\arg\max_v p_\theta(v \mid x_{<t})$. Fast and deterministic but tends to produce repetitive, boring output.

### Temperature scaling
Divides logits by $T$ before softmax:
$$p'_v = \frac{\exp(\ell_v / T)}{\sum_u \exp(\ell_u / T)}$$
- $T < 1$: sharpens the distribution (model is more confident, less creative)
- $T = 1$: standard softmax
- $T > 1$: flattens the distribution (more random, more diverse)

### Nucleus (top-p) sampling
1. Sort tokens by probability (descending)
2. Keep the smallest prefix of tokens whose cumulative probability ≥ p
3. Renormalise and sample
This avoids selecting very unlikely tokens while keeping diversity. Recommended value: `top_p=0.9` or `top_p=0.95`.

### Top-k sampling
Keep only the k highest-probability tokens, renormalise, sample. `top_k=50` is a common default.

---

## Multi-GPU data-parallel training

```yaml title="configs/multi_gpu.yaml"
n_devices: 4
use_bf16: true
batch_size: 256    # total batch split across 4 GPUs → 64 per device
grad_accum: 1
```

!!! warning "Batch size divisibility"
    `batch_size` must be divisible by `n_devices`.
    With 4 GPUs and `batch_size=256`, each device gets 64 samples.

DantinoX uses JAX's **SPMD data parallelism**: the model is replicated on every device, and each device processes a different shard of the batch. Gradients are averaged across devices automatically via `jax.lax.pmean`.

---

## Weight tying

When `weight_tying: true` (default), the token embedding matrix and the LM head projection share the same weights. This saves `vocab_size × dim` parameters — for a 32K vocabulary and dim=512, that's 16M fewer parameters. It also tends to improve perplexity because the model learns to encode and decode tokens in the same space.

---

## See also

- [KV-Cache Mechanics](../inference/kv-cache.md) — memory formulas, batch size calculator
- [Training Guide](../training/index.md) — optimisers, schedules, gradient accumulation
- [Configuration Reference](../configuration.md) — all fields for `ModelConfig`
- [Benchmarks](../benchmarks.md) — throughput numbers for MHA vs GQA vs MLA
