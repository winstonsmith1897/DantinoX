---
title: Fast-dLLM DualCache
---

# Fast-dLLM Block-wise DualCache

DantinoX integrates the **DualCache** inference optimisation from
*Fast-dLLM: Training-free Acceleration of Diffusion LLM Inference via Causal KV Cache*
(Wu et al., arXiv:2505.22618).

DualCache reduces the per-step cost of diffusion generation from
$O(T_{\text{total}}^2)$ to roughly $O(B \cdot T_{\text{total}})$ per inner step,
where $B$ is the block size and $T_{\text{total}}$ is the total sequence length.

---

## Motivation

A naïve diffusion sampler runs the full-sequence model at every denoising step:

```
for t in T → 0:
    logits = model([prompt | x_t])   # O((T_prefix + T_gen)²) attention
    x_t    = unmask(logits, x_t)
```

With $T_{\text{gen}} = 256$ and $T_{\text{prefix}} = 64$, every step processes
320 tokens through bidirectional attention — expensive even with KV-cache.

---

## Block-wise Generation

Fast-dLLM divides the generated region into $K$ non-overlapping blocks of size $B$:

```
[  prompt  |  block 0  |  block 1  |  …  |  block K-1  ]
```

For each block $k$, an **inner loop** of `steps_per_block` denoising steps
operates only on that block's tokens.  The model attends to:

1. **Prefix KV** — cached from the static prompt (computed once, never recomputed).
2. **Fresh block KV** — recomputed each inner step from the current block tokens.
3. **Suffix KV** — cached from the remaining all-`[MASK]` blocks after block $k$.

```
Inner step on block k:
  context = [prefix_KV | fresh_block_KV | suffix_KV]
  logits  = model_on_block_k(x[s:e], context)
  x[s:e]  = confidence_unmask(logits, x[s:e])
```

The suffix KV barely changes within a single block's inner loop
(cosine similarity > 0.99 between adjacent steps in Fast-dLLM §3.2),
so it is safely reused and refreshed only at each block boundary.

---

## DualCache Data Structure

```python
class DualCache(NamedTuple):
    prefix_kvs: tuple   # per-layer (k, v) for the prompt
    suffix_kvs: tuple   # per-layer (k, v) for remaining MASK blocks
```

| Field | Shape | Refresh |
|---|---|---|
| `prefix_kvs[l]` | `[B, H_kv, 1, T_prefix, d_h]` | Once, before all blocks |
| `suffix_kvs[l]` | `[B, H_kv, 1, T_suffix, d_h]` | Once per block boundary |

For MLA the KV tensors use the compressed latent dimension $d_c^{KV}$;
`suffix_kvs` is set to `None` when using prefix-only caching.

---

## Python API

```python
from dantinox.core.generation import fast_dllm_generate
from dantinox.core.diffusion import make_noise_schedule

schedule = make_noise_schedule(config)

tokens = fast_dllm_generate(
    model,
    prefix,                    # [B, T_prefix]  — empty OK
    gen_len   = 256,
    schedule  = schedule,
    mask_token_id = 0,

    # Block-wise parameters
    block_size        = 32,    # tokens per block  (default: 32)
    steps_per_block   = 20,    # denoising steps per block

    # Confidence-aware unmasking
    decoding_strategy = "threshold",   # "threshold" | "factor"
    confidence_threshold = 0.9,
    factor               = 1.5,

    # Cache mode
    use_dual_cache    = True,  # False → prefix-only (slower)
    refresh_interval  = None,  # None = refresh at block boundary only
    seed              = 42,
)
# tokens: [B, gen_len]
```

---

## Effect of Block Size

Larger blocks amortise the cache-refresh cost but introduce more approximation:

| Block size $B$ | Inner steps saved | Approximation error | Net speedup |
|---|---|---|---|
| 4 | low | very low | ~1.1× |
| 16 | medium | low | ~1.5× |
| **32** (default) | high | low | **~1.8×** |
| 64 | high | medium | ~1.6× |
| 128 | max | high | ~1.3× |

DualCache delivers 1.4–2.1× speedup over prefix-only caching across model sizes
(see [Experiments & Results](../paper.md)).

---

## Suffix Cache Refresh

By default, the suffix KV is refreshed once per block boundary.
Pass `refresh_interval=r` to refresh every $r$ inner steps for higher accuracy:

```python
tokens = fast_dllm_generate(
    ...,
    refresh_interval=5,   # refresh suffix KV every 5 inner steps
)
```

Lower `refresh_interval` → less approximation error, higher wall-clock time.

---

## Building the Dual Cache Manually

For fine-grained control, use the `DiffusionTransformer` methods directly:

```python
# Build or refresh the dual cache for block k
dual_cache = model.compute_block_dual_cache(
    x_full,      # [B, T_total] — full sequence including MASK blocks
    t,           # [B] — current timestep
    block_start, # absolute token index of block start
    block_end,   # absolute token index of block end
)

# Inner loop: run only on the current block
logits = model.decode_block(
    x_block,      # [B, block_size]
    t,
    dual_cache,
    block_start,  # for correct RoPE offset
)
```

---

## Speedup Summary

Measured on a 256-dim, 12-layer model, `gen_len=256`, `block_size=32`:

| Method | tok/s (BS=1) | tok/s (BS=8) |
|---|---|---|
| Naïve (no cache) | baseline | baseline |
| PrefixCache only | ~1.4× | ~1.4× |
| **DualCache** (default) | **~2.1×** | **~1.9×** |
