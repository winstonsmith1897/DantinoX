---
title: KV-Cache
---

# KV-Cache Mechanics

DantinoX uses three distinct caching strategies depending on the model type and generation paradigm.

---

## AR Static KV-Cache

The autoregressive KV-cache is **statically pre-allocated** at model construction:

```
shape: [B, H_kv, max_context, head_size]  per layer
```

Static allocation means **no dynamic shapes** and **no XLA recompilation** between decode steps.

```python
# Prefill — full forward pass fills the cache
logits, kv_cache, _ = model(
    prompt_ids,
    use_cache=True, kv_caches=init_cache, cache_index=0
)

# Decode — each step processes a single new token
logits, kv_cache, _ = model(
    next_token,
    use_cache=True, kv_caches=kv_cache, cache_index=T_prompt
)
```

---

## Memory Formula

$$
\text{KV-MB} = L \times 2 \times S \times H_{kv} \times d_h \times \text{bytes\_per\_element}
$$

where:

- $L$ = number of layers
- $S$ = sequence length (tokens)
- $H_{kv}$ = number of KV heads
- $d_h$ = head dimension
- `bytes_per_element` = 2 for bfloat16, 4 for float32

### Reference table (12 layers, head_size=32, bfloat16)

| Attention | $H_{kv}$ | KV-MB @ 512 tok | KV-MB @ 1024 tok | KV-MB @ 4096 tok |
|---|---|---|---|---|
| MHA ($H=8$) | 8 | 0.375 MB | 0.750 MB | 3.0 MB |
| GQA ($H_{kv}=H/4=2$) | 2 | 0.094 MB | 0.188 MB | 0.75 MB |
| GQA ($H_{kv}=H/8=1$) | 1 | 0.047 MB | 0.094 MB | 0.375 MB |
| MLA ($d_c^{KV}=96$) | — | ~0.027 MB | ~0.054 MB | ~0.216 MB |

MLA stores a compressed latent vector per token instead of full K/V tensors, then decompresses on the fly during attention.

### Batch size calculator

For a single A100 40 GB, estimate the maximum batch size:

$$
\text{Max BS} \approx \frac{\text{VRAM}_\text{available} - \text{Weights MB}}{\text{KV-MB per sequence}}
$$

| Attention | Weights (256d 12b) | KV/seq @ 512 tok | Max BS @ 40 GB |
|---|---|---|---|
| MHA | ~32 MB | 0.375 MB | ~100 |
| GQA (×4) | ~30 MB | 0.094 MB | ~400 |
| MLA | ~32 MB | ~0.027 MB | ~1500 |

!!! note
    These are rough estimates. Actual VRAM usage depends on activations, intermediate buffers, and JAX XLA padding. Use `jax.devices()[0].memory_stats()` for real measurements.

---

## MHA vs GQA vs MLA Comparison

| | MHA | GQA | MLA |
|---|---|---|---|
| KV heads | $H$ | $H / r$ | compressed latent |
| KV cache size | $L \cdot 2 \cdot S \cdot H \cdot d_h$ | $L \cdot 2 \cdot S \cdot (H/r) \cdot d_h$ | $L \cdot S \cdot d_c^{KV}$ |
| Decode throughput | baseline | ~1.0× (similar) | ~0.8× (absorb overhead) |
| Prefill speed | baseline | ~same | ~10–30% slower |
| Cache at 512 tok, 12L, bf16 | 0.375 MB | 0.094 MB (r=4) | ~0.027 MB |
| Max batch @ 40 GB | ~100 | ~400 | ~1500 |
| Config flag | `attention="mha"` | `attention="gqa", kv_heads=2` | `attention="mla"` |

MLA's slower per-step latency is offset by fitting more sequences in VRAM simultaneously, making it the highest-throughput option at large batch sizes.

---

## Diffusion DualCache

For diffusion, the cache consists of two parts:

```python
class DualCache(NamedTuple):
    prefix_kvs: tuple   # per-layer (k, v) for the prompt — computed once
    suffix_kvs: tuple   # per-layer (k, v) for remaining MASK tokens — refreshed per block
```

The suffix KV adds overhead proportional to the number of remaining MASK blocks. Averaged over a full generation this adds approximately **20–40%** to the peak cache size vs. the static AR cache.

See [Fast-dLLM DualCache](../paradigms/fast-dllm.md) for the full architecture description.

---

## Cache Memory vs Throughput

At large batch sizes, the cache footprint determines how many sequences fit in VRAM simultaneously.

Measured throughput on A100 40 GB, 256d 12-layer model, bfloat16, `seq_len=512`:

| Attention | Decode tok/s (BS=1) | Decode tok/s (BS=8) | Max BS |
|---|---|---|---|
| MHA | 89 | ~540 | ~100 |
| GQA (×4) | 90 | ~600 | ~400 |
| MLA | 70 | ~650 | ~1500 |

GQA and MHA have similar single-sequence throughput; MLA pays a per-step overhead (~20%) but excels when many sequences are batched together because more of them fit in cache.

---

## Disabling the Cache

Disable the KV cache for debugging or for short sequences where re-allocation overhead is negligible:

```python
# Python
tokens = generate(model, prompt_ids, use_cache=False)

# CLI
dantinox generate --run_dir runs/my_run --no_cache
```

---

## Cache Warmup

JAX JIT-compiles the decode step on first call. The second call (and all subsequent) use the compiled kernel. DantinoX's `Generator` class does one warmup forward automatically:

```python
gen = Generator("runs/my_run")
# First call triggers JIT compilation (slow)
_ = gen.generate("warmup", max_new_tokens=1)
# Second call uses cached compilation
text = gen.generate("Real prompt", max_new_tokens=200)
```

The XLA compilation cache (stored in `~/.cache/jax_xla/dantinox`) persists across processes, so the overhead is only paid once per unique model architecture.
