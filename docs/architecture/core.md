# Core Layers

This page is a deep-dive into the `core/` neural-network primitives. The existing [Architecture](../architecture.md) page covers the system-level design; this page focuses on implementation details, math, and configuration knobs for each layer.

---

## Attention: MHA, GQA, MLA

All three attention variants share the same causal mask and KV-cache infrastructure. The variant is selected by `config.mla` and `config.kv_heads`.

### Selection guide

| Mode | `config.mla` | `config.kv_heads` | KV cache / token / layer |
| :--- | :---: | :---: | :--- |
| Multi-Head (MHA) | `false` | = `n_heads` | `2 · H · d_h` values |
| Grouped-Query (GQA) | `false` | < `n_heads` | `2 · H_kv · d_h` values |
| Multi-Head Latent (MLA) | `true` | any | `d_c^{KV} + d_r` values |

### Flash Attention

Set `use_flash_attention: true` to use JAX's fused kernel (`jax.nn.dot_product_attention`, JAX ≥ 0.4.25). Activates only when `mla=false`, `use_cache=false`, and `sliding_window=false`.

!!! tip
    Enable for medium–large models training with long sequences. Avoids materialising the O(T²) attention matrix.

### Multi-Head Latent Attention (MLA) math

Latent compression:

$$\mathbf{c}_{KV} = \mathrm{Norm}(W_{DKV}\,\mathbf{x}), \quad d_c^{KV} \ll d$$

Only `c_KV` is cached — `d_c^{KV}` scalars per token instead of `2·H_kv·d_h`.

Weight absorption (inference, `config.inference=true`):

$$A_{QK} = W_{UQ}^\top W_{UK}, \quad A_{VO} = W_{UV} W_O$$

Full K/V tensors are never materialised at decode time.

!!! warning "Training vs inference"
    `inference: false` during training, `inference: true` for generation. The weights are identical; only the computation graph differs.

---

## Feed-Forward Network

| Variant | Config | Formula |
| :--- | :--- | :--- |
| Dense GELU | `use_moe=false`, `use_swiglu=false` | `W₂·GELU(W₁x)` |
| Dense SwiGLU | `use_moe=false`, `use_swiglu=true` | `W₂·(W₁x ⊙ σ(W_gate·x))` |
| Sparse MoE | `use_moe=true` | Top-K routing + load-balancing loss |

MoE load-balancing: $\mathcal{L}_\mathrm{bal} = \alpha \cdot N \sum_i f_i P_i$

---

## Normalisation

| `norm_type` | Formula | Notes |
| :--- | :--- | :--- |
| `"layernorm"` | $(x - \mu) / \sigma \cdot \gamma + \beta$ | Default |
| `"rmsnorm"` | $x / \mathrm{RMS}(x) \cdot \gamma$ | Faster, used in LLaMA/Mistral |

---

## Positional Encoding

| Mode | Config key | Notes |
| :--- | :--- | :--- |
| RoPE (default) | `use_rotary_pos: true` | NTK scaling via `rope_scale_factor` |
| Sinusoidal | `absolute_pos: true` | Fixed, no parameters |
| Learned | `trainable_pos: true` | Standard learned embeddings |

NTK-aware scaling ($\lambda$ = `rope_scale_factor`):

$$\theta_i = \frac{1}{(10000 \cdot \lambda)^{2i/d}}$$

---

## LoRA

`LoRALinear` is a drop-in for `nnx.Linear`. Base weights are stored as `nnx.Param` and frozen; adapter matrices use `LoRAParam` — a distinct variable subclass.

```python
optimizer = nnx.Optimizer(model, tx, wrt=LoRAParam)  # only LoRAParam updated
```

$W_\mathrm{eff} = W_\mathrm{base} + \frac{\alpha}{r} \cdot AB$

Merge for zero-overhead inference: `layer.merge_weights()`.

---

## Static KV-Cache (XLA-compatible)

Fixed-size buffer pre-allocated at prefill; surgical writes via `jax.lax.dynamic_update_slice` — no recompilation on every token.

```python
# Decode: insert at cache_index
k_cache = jax.lax.dynamic_update_slice(k_cache, k_new, (0, 0, 0, cache_index, 0))
```

---

## Gradient checkpointing

`nnx.remat` wraps each block when `gradient_checkpointing=true`. Activations are discarded on forward and recomputed on backward — trades FLOPS for VRAM.

Automatically disabled during inference (`use_cache=true`).
