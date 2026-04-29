# Core Architecture

DantinoX implements a decoder-only Transformer with a modular design: every major component — attention type, feed-forward network, positional encoding — is toggled via configuration without changing any source code.

---

## Attention Mechanisms

Three attention families are supported, all sharing the same causal mask and KV-cache infrastructure. The choice is driven by the `mla` flag in the configuration.

### Comparison

| | **MHA** | **GQA** | **MLA** |
| :--- | :--- | :--- | :--- |
| Config | `mla: false`, `kv_heads = n_heads` | `mla: false`, `kv_heads < n_heads` | `mla: true` |
| KV cache per token per layer | $2 \cdot H_{\text{kv}} \cdot d_h$ values | $2 \cdot H_{\text{kv}} \cdot d_h$ values | $d_c^{KV} + d_r$ values |
| KV cache at 512 tok, 12 layers¹ | **384 KB** | **96 KB** | **~23 KB** |
| Decoupled RoPE | ✗ | ✗ | ✓ |
| Weight absorption at decode | ✗ | ✗ | ✓ |
| Extra parameters vs GQA | — | — | +4 projections +2 norms |

> ¹ With $H=16$, $H_{\text{kv}}=4$, $d_h=32$, $d_c^{KV}=64$, $d_r=32$ — all in fp32.

---

### Multi-Head Attention (MHA) & Grouped-Query Attention (GQA)

Standard MHA projects the input $\mathbf{x}$ to queries, keys, and values via a fused `qkv` projection.
GQA is MHA with $H_{\text{kv}} < H$: KV heads are repeated to match query heads during the dot-product, reducing cache by a factor of $H / H_{\text{kv}}$.

```python
# core/attention.py — fused QKV projection (MHA / GQA)
self.qkv = nnx.Linear(
    dim,
    dim + 2 * kv_heads * head_size,   # Q + K + V in a single matmul
    use_bias=False, rngs=rngs
)
```

Set `kv_heads = n_heads` for MHA or `kv_heads < n_heads` for GQA.

---

### Multi-Head Latent Attention (MLA)

MLA (introduced in DeepSeek-V2) replaces the standard KV projection with a low-rank bottleneck. Instead of caching $K$ and $V$ tensors directly, only a small latent vector $\mathbf{c}_{KV}$ is stored per token. Full keys and values are reconstructed on-the-fly during training, or bypassed entirely at decode time via weight absorption.

#### Latent Compression

$$
\mathbf{c}_Q = \text{Norm}(W_{DQ}\,\mathbf{x}), \quad \mathbf{c}_{KV} = \text{Norm}(W_{DKV}\,\mathbf{x})
$$

$$
\mathbf{q} = W_{UQ}\,\mathbf{c}_Q, \quad
\mathbf{k} = W_{UK}\,\mathbf{c}_{KV}, \quad
\mathbf{v} = W_{UV}\,\mathbf{c}_{KV}
$$

where $W_{DQ} \in \mathbb{R}^{d \times d_c^Q}$, $W_{DKV} \in \mathbb{R}^{d \times d_c^{KV}}$, and the up-projections restore the full multi-head dimensionality. **Only $\mathbf{c}_{KV}$ is cached** — a vector of $d_c^{KV}$ scalars instead of $2 \cdot H_{\text{kv}} \cdot d_h$.

#### Decoupled RoPE

Rotary embeddings cannot be applied inside the latent space because the compressed representation must remain position-independent for the cache to be reusable. MLA adds parallel lightweight projections that carry positional information separately:

$$
\mathbf{q}^r = \text{RoPE}(W_{Q}^r\,\mathbf{x}), \quad \mathbf{k}^r = \text{RoPE}(W_{K}^r\,\mathbf{x})
$$

```python
# core/attention.py — decoupled RoPE projections (MLA)
self.q_pe = nnx.Linear(dim, rope_dim, rngs=rngs)   # W_Q^r
self.k_pe = nnx.Linear(dim, rope_dim, rngs=rngs)   # W_K^r
```

The final attention score combines content and position channels:

$$
s = \frac{\mathbf{q} \cdot \mathbf{k}^\top + \mathbf{q}^r \cdot (\mathbf{k}^r)^\top}{\sqrt{d_h + d_r}}
$$

#### Weight Absorption (Inference Path)

During decode (`inference=True`), up-projecting the cached $\mathbf{c}_{KV}$ back to full multi-head $K$ and $V$ would be wasteful. Instead, the associativity of matrix multiplication allows pre-fusing the projections into absorbed weight matrices that operate directly on the latent cache:

$$
A_{QK} = W_{UQ}^\top W_{UK} \in \mathbb{R}^{d_c^Q \times d_c^{KV}}
\quad\Rightarrow\quad
\mathbf{q} \cdot \mathbf{k}^\top = \mathbf{c}_Q \cdot A_{QK} \cdot \mathbf{c}_{KV}^\top
$$

$$
A_{VO} = W_{UV} W_O \in \mathbb{R}^{d_c^{KV} \times d}
\quad\Rightarrow\quad
\text{out} = \sum_s \alpha_s\,\mathbf{c}_{KV}^{(s)} \cdot A_{VO}
$$

The full multi-head $K$, $V$ tensors are never materialised. Only the compressed latent cache ($d_c^{KV}$ scalars per token) is read from HBM:

```python
# core/attention.py — weight absorption at decode
q_proj    = self.up_q.kernel.reshape(down_dim_q, kv_heads, n_heads // kv_heads, head_size)
k_proj    = self.up_k.kernel.reshape(down_dim_kv, kv_heads, head_size)
attn_proj = jnp.einsum('qngh, knh -> ngqk', q_proj, k_proj)   # pre-fuse W_UQ · W_UK
attn_proj = jnp.einsum('btq,  ngqk -> btngk', q, attn_proj)   # project latent Q
attn      = jnp.einsum('btngk, bsk -> bngts', attn_proj, k)   # attend on latent K cache

W_v  = self.up_v.kernel.reshape(down_dim_kv, kv_heads, head_size)
W_o  = self.o_proj.kernel.reshape(kv_heads, n_heads // kv_heads, head_size, dim)
W_vo = jnp.einsum('dnh, nghc -> dngc', W_v, W_o)              # pre-fuse W_UV · W_O
out  = jnp.einsum('bngtd, dngc -> btc', L, W_vo)              # project from latent V cache
```

!!! warning "Training vs. Inference"
    Set `inference: false` during training — weight absorption is decode-only. After training, reload the checkpoint with `inference: true` to activate the optimised decode path. The saved weights are identical; only the forward-pass computation graph changes.

---

## Feed-Forward Network

The FFN is selected per `use_moe`:

=== "Dense MLP"

    A standard two-layer feed-forward block with optional SwiGLU gating
    (`use_swiglu: true` replaces GELU with a gated linear unit for better gradient flow):

    ```
    hidden = activation(W₁ x) ⊙ (W_gate x)   # SwiGLU
    out    = W₂ hidden
    ```

=== "Sparse MoE"

    A top-K router selects `top_k_mlp` out of `n_experts` expert MLPs per token.
    An auxiliary load-balancing loss prevents expert collapse:

    $$\mathcal{L}_{\text{bal}} = \alpha \cdot N \sum_{i=1}^{N} f_i \cdot P_i$$

    where $f_i$ is the fraction of tokens routed to expert $i$ and $P_i$ is the mean router probability for expert $i$.

---

## Positional Encoding

| Mode | Config | Notes |
| :--- | :--- | :--- |
| **Rotary (RoPE)** | `use_rotary_pos: true` | Default. Decoupled variant used with MLA. |
| **Absolute sinusoidal** | `absolute_pos: true` | Fixed frequencies, no learned parameters. |
| **Learned** | `trainable_pos: true` | Standard learned position embeddings. |

RoPE frequencies are pre-computed at init and cached as a static array. At each forward pass, `jax.lax.dynamic_slice_in_dim` extracts the relevant sub-sequence without triggering recompilation:

```python
# core/attention.py — dynamic RoPE slice (XLA-safe)
angle = jax.lax.dynamic_slice_in_dim(self.angle, start_index=cache_index, slice_size=T, axis=3)
```

---

## Configuration Reference

All parameters live in a single `Config` dataclass and are loaded from YAML. CLI overrides are merged at runtime.

??? note "Full annotated YAML"

    ```yaml
    model:
      dim: 512                      # Hidden dimension d; must equal n_heads × head_size
      n_heads: 16                   # Number of query heads H
      kv_heads: 4                   # KV heads H_kv (H_kv = H → MHA; H_kv < H → GQA)
      head_size: 32                 # Head dimension d_h; dim = n_heads × head_size
      num_blocks: 12                # Number of Transformer layers L
      max_context: 512              # Maximum sequence length for KV cache allocation
      weight_tying: true            # Tie lm_head weights to token embedding matrix
      activation: gelu              # FFN activation: "gelu" or "swiglu"
      use_swiglu: true              # Use SwiGLU gating in the FFN
      gradient_checkpointing: true  # Recompute activations on backward (nnx.remat)
      dropout_rate: 0.15            # Dropout probability (attention, residual, embedding)

    mla:
      mla: false                    # Enable Multi-Head Latent Attention
      inference: false              # Activate weight absorption — set true for generation only
      down_dim_q: 256               # Query latent dimension d_c^Q
      down_dim_kv: 64               # KV latent dimension d_c^KV (= cache size per token)
      rope_dim: 32                  # Decoupled RoPE dimension d_r (≤ head_size)

    moe:
      use_moe: false                # Replace Dense FFN with Sparse MoE
      n_experts: 4                  # Total number of expert MLPs N
      top_k_mlp: 2                  # Experts activated per token K
      expansion: 4                  # FFN expansion factor inside each expert
      alpha_balance: 0.1            # Load-balancing loss weight α

    attention:
      use_rotary_pos: true          # Enable Rotary Positional Embeddings
      trainable_pos: false          # Learned absolute positional embeddings
      absolute_pos: false           # Fixed sinusoidal embeddings
      sliding_window: false         # Restrict attention to a local causal window
      context_window: 64            # Window size (tokens) when sliding_window: true
      no_sink: true                 # Sigmoid gate on attention output (prevents attention sink)

    tokenizer:
      tokenizer_type: "char"        # "char" for character-level, "bpe" for Byte-Pair Encoding
      vocab_size: 2000              # Maximum vocabulary size
      tokenizer_path: "configs/vocab.json"

    data:
      dataset_source: "huggingface" # "huggingface" or "local"
      dataset_name: "Daniele/dante-corpus"
      streaming: true               # Stream from HuggingFace to avoid local RAM pressure

    training:
      lr: 0.0015                    # Peak learning rate (cosine decay)
      batch_size: 64                # Per-device batch size
      grad_accum: 4                 # Gradient accumulation steps
      seed: 42
      optimizer: "adamw"            # "adamw", "adafactor", or "lion"
      epochs: 100
      warmup_steps: 0               # Linear LR warmup steps

    generation:
      use_cache: true               # Static KV cache for autoregressive decode
      greedy: false                 # Greedy decoding (overrides sampling)
      temperature: 1.3
      top_p: null                   # Nucleus sampling threshold (null = disabled)
      top_k: null                   # Top-K sampling (null = disabled)
      max_generations: 150
      seed: 42

    logging:
      eval_iters: 20
      log_file: "training_log.csv"
      summary_file: "model_summary.json"
    ```

---

## Implementation Details

### Static KV-Cache (XLA-Compatible)

JAX's XLA compiler requires all array shapes to be known at trace time. Dynamic concatenation forces recompilation on every new token, which is unacceptable for autoregressive decode. DantinoX pre-allocates a fixed-size cache buffer at prefill and uses `jax.lax.dynamic_update_slice` for O(1) positional writes:

=== "MHA / GQA"

    ```python
    # Prefill: allocate zeros and fill the prompt slice
    k_cache = jnp.zeros((B, kv_heads, 1, max_context, head_size), dtype=k.dtype)
    k_cache = k_cache.at[:, :, :, :T, :].set(k)

    # Decode: surgical insert at cache_index — no recompilation
    k_cache = jax.lax.dynamic_update_slice(k_cache, k, (0, 0, 0, cache_index, 0))
    ```

=== "MLA"

    The cache stores the compressed latent $\mathbf{c}_{KV}$ and the decoupled RoPE keys — not the full $K$/$V$ tensors.

    ```python
    # Prefill
    c_cache     = jnp.zeros((B, max_context, down_dim_kv), dtype=c_kv.dtype)
    c_cache     = c_cache.at[:, :T, :].set(c_kv)
    k_rope_cache = jnp.zeros((B, 1, 1, max_context, rope_dim), dtype=c_kv.dtype)

    # Decode
    c_cache = jax.lax.dynamic_update_slice(c_cache, c_kv, (0, cache_index, 0))
    ```

    Cache footprint per token per layer: `(down_dim_kv + rope_dim) × 4 bytes` vs `2 × kv_heads × head_size × 4 bytes` for GQA.

### Sliding Window Attention & Attention Gating

**Sliding window** restricts each token to attend only to the previous `context_window` tokens, preventing quadratic memory growth during long-context generation:

```python
table  = jnp.arange(max_context)[:, None] - jnp.arange(max_context)[None, :]
mask   = (table <= context_window) & (table >= 0)
window = jnp.where(mask, 0.0, -1e9)
# Applied via dynamic_slice_in_dim at each forward pass
```

**Attention gating** (`no_sink`) multiplies the attention output by a sigmoid-projected gate computed from the input residual. This prevents the degenerate "attention sink" pattern — where initial tokens accumulate disproportionate attention mass — that degrades generation quality at long sequences:

```python
if self.no_sink:
    y = y * jax.nn.sigmoid(self.W(x))
```

### Sparse Mixture of Experts

```python
# core/attention.py — MoE routing
probs          = jax.nn.softmax(self.router(x))
values, idx    = jax.lax.top_k(probs, self.top_k_mlp)
values         = values / jnp.sum(values, axis=-1, keepdims=True)   # renormalise

# Load-balancing loss (auxiliary, added to cross-entropy)
f   = jnp.mean(jnp.sum(jax.nn.one_hot(idx, n_experts), axis=2), axis=(0, 1))
P   = jnp.mean(probs.reshape(B * T, n_experts), axis=0)
moe_loss = jnp.sum(f * P) * n_experts * alpha_balance
```

Expert outputs are accumulated in a static zero buffer to keep array shapes fixed for XLA:

```python
y = jnp.zeros_like(x)
for i in range(n_experts):
    w   = jnp.sum(jnp.where(idx == i, values, 0), axis=-1, keepdims=True)
    out, _ = self.experts[i](x, deterministic=deterministic)
    y   = y + w * out
```

### Gradient Checkpointing

`nnx.remat` discards intermediate activations during the forward pass and recomputes them on demand during backpropagation. This trades compute for memory, enabling larger batch sizes or deeper models on a fixed VRAM budget. Checkpointing is automatically disabled in inference mode (where `jax.grad` is never called):

```python
if self.gradient_checkpointing and not use_cache:
    block_fn = nnx.remat(lambda m, h, c: m(h, use_cache=False, kv_cache=c, ...))
else:
    block_fn = lambda m, h, c: m(h, use_cache=use_cache, kv_cache=c, ...)
```
