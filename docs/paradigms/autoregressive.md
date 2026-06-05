---
title: Autoregressive Paradigm
---

# Autoregressive Generation

The `Transformer` class implements a standard causal (decoder-only) language model.
Given a prefix $x_{1:t}$, it predicts the next token $x_{t+1}$ by attending only
to previous positions via a causal mask.

---

## Architecture

```
Input tokens  x₁ … xₜ
       ↓ Embedding (weight-tied to LM head)
       ↓ + Positional encoding (RoPE / absolute / trainable)
       ↓ Dropout
 ┌─────────────────────────┐
 │  ARBlock × L            │
 │  ├─ LayerNorm / RMSNorm │
 │  ├─ MHA / GQA / MLA     │  ← causal mask
 │  ├─ Residual             │
 │  ├─ LayerNorm / RMSNorm │
 │  └─ MLP / MoE            │
 └─────────────────────────┘
       ↓ Final LayerNorm
       ↓ LM head  →  logits [B, T, V]
```

The model is configured by setting `model_type: "autoregressive"` (default).

---

## KV-Cache

During autoregressive decoding the model keeps a static pre-allocated
KV-cache of shape `[B, H_kv, max_context, head_size]` per layer.

- **Prefill** (`use_cache=False`): full forward pass on the prompt, fills the cache.
- **Decode** (`use_cache=True`): each new token attends to the growing cache using `cache_index` as the write pointer.

```python
from core.generation import generate

tokens = generate(
    model,
    prompt_ids,           # [B, T_prompt]
    max_generations=200,
    use_cache=True,       # enable KV-cache
    top_p=0.9,
    temperature=1.0,
)
```

### Cache memory

$$
\text{KV-MB} = L \times 2 \times S \times H_{kv} \times d_h \times \text{bytes/value}
$$

| Attention | $H_{kv}$ | Cache @ 512 tok, 12L, fp32 |
|---|---|---|
| MHA | $H$ | 384 KB |
| GQA (×4) | $H/4$ | 96 KB |
| MLA | $d_c^{KV} + d_r$ | ~23 KB |

---

## Training

```yaml
model:
  model_type: "autoregressive"
  dim: 256
  n_heads: 8
  head_size: 32
  num_blocks: 12
  max_context: 512
  weight_tying: true

training:
  lr: 0.0012
  batch_size: 64
  grad_accum: 4
  epochs: 3
  optimizer: "lion"
```

Training objective: cross-entropy on the next token at every position
(teacher-forcing, causal mask enforced by the attention layer).

$$
\mathcal{L}_{AR} = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})
$$

---

## Streaming

The CLI `generate` subcommand supports token-by-token streaming:

```bash
dantinox generate \
  --run_dir runs/ar_mha_256d_12b_Dense \
  --prompt "Nel mezzo del cammin " \
  --max_new_tokens 200 \
  --stream
```

Internally this calls `Generator.stream()`, which yields decoded text chunks
as each token is produced.

---

## Multi-GPU

Set `n_devices: 2` (or higher) for data-parallel training:

```yaml
training:
  n_devices: 2
  use_bf16: true
```

JAX's SPMD mesh replicates model parameters across devices and shards each batch.
Gradient reduction is handled automatically via `jax.lax.pmean`.
