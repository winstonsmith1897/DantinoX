---
title: KV-Cache
---

# KV-Cache

## AR Static KV-Cache

The autoregressive KV-cache is **statically pre-allocated** at model construction.
Shape: `[B, H_kv, max_context, head_size]` per layer.

```python
# Prefill: full forward pass, fills the cache
logits, kv_cache, _ = model(
    prompt_ids,
    use_cache=True, kv_caches=init_cache, cache_index=0
)

# Decode: single new token uses the cache
logits, kv_cache, _ = model(
    next_token,
    use_cache=True, kv_caches=kv_cache, cache_index=T_prompt
)
```

Static allocation means **no dynamic shapes** and **no XLA recompilation** between steps.

---

## Memory Footprint

$$
\text{KV-MB} = L \times 2 \times S \times H_{kv} \times d_h \times \text{bytes}
$$

| Attention | $H_{kv}$ | fp32 @ 512 tok, 12L, $d_h=32$ |
|---|---|---|
| MHA ($H=8$) | 8 | 384 KB |
| GQA ($H_{kv}=H/4=2$) | 2 | 96 KB |
| MLA ($d_c^{KV}=96, d_r=16$) | — | ~27 KB |

MLA reduces the cache by storing a compressed latent vector instead of
full K/V tensors.  The decompressed K/V are computed on-the-fly during attention.

---

## Diffusion DualCache

For diffusion, see the [DualCache page](../paradigms/fast-dllm.md).
The suffix KV is cached across inner denoising steps within a block,
adding ~20–40% peak memory vs the static AR cache.

---

## Cache Memory vs Throughput

At large batch sizes, the cache footprint determines how many sequences
fit in VRAM simultaneously:

| Attention | Max BS @ 40 GB VRAM, 512 tok |
|---|---|
| MHA | ~100 |
| GQA (×4) | ~400 |
| MLA | ~1500 |

This is why MLA achieves higher throughput at large batch sizes despite
having similar per-step latency to GQA — more sequences fit in cache.
