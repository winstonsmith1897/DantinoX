# Core Architecture

Every major component — attention type, normalisation, feed-forward network, positional encoding — is selected by a single field in `Config`. No subclassing, no source edits.

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

#### Flash Attention (opt-in)

Set `use_flash_attention: true` to use JAX's fused scaled-dot-product kernel (`jax.nn.dot_product_attention`, JAX ≥ 0.4.25) during training. This is off by default so existing configs require no changes.

The Flash Attention path activates when all of the following hold:
- `use_flash_attention: true`
- `mla: false` (MHA/GQA only)
- `use_cache: false` (training pass — cache path uses the manual kernel)
- `sliding_window: false`

```python
# core/attention.py — Flash Attention fast path
if self.use_flash_attention and not self.mla and not use_cache and not self.sliding_window:
    q_fa = q.reshape(B, T, self.n_heads,  self.head_size)   # [B, T, H, D]
    k_fa = k.reshape(B, T, self.kv_heads, self.head_size)
    v_fa = v.reshape(B, T, self.kv_heads, self.head_size)
    if self.use_rotary:
        q_fa, k_fa = self._apply_rope_thd(q_fa, 0), self._apply_rope_thd(k_fa, 0)
    # GQA: broadcast K/V to full head count for JAX < 0.4.31 compat
    if self.kv_heads < self.n_heads:
        g    = self.n_heads // self.kv_heads
        k_fa = jnp.repeat(k_fa, g, axis=2)
        v_fa = jnp.repeat(v_fa, g, axis=2)
    y = jax.nn.dot_product_attention(q_fa, k_fa, v_fa, is_causal=True)
```

!!! tip "When to enable"
    Enable Flash Attention for medium-to-large models training with long sequences on GPU. The fused kernel avoids materialising the full $[B, H, T, T]$ attention matrix, reducing memory from $O(T^2)$ to $O(T)$.

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

## Normalisation

The normalisation applied before attention and the feed-forward block is controlled by `norm_type`:

| `norm_type` | Formula | Notes |
| :--- | :--- | :--- |
| `layernorm` (default) | $\frac{x - \mu}{\sigma} \cdot \gamma + \beta$ | Standard LayerNorm — mean-centred, learned bias |
| `rmsnorm` | $\frac{x}{\text{RMS}(x)} \cdot \gamma$ | Faster — no mean subtraction, no bias; used in LLaMA, Mistral, Gemma |

```python
# core/block.py — RMSNorm
class RMSNorm(nnx.Module):
    def __call__(self, x):
        rms = jnp.sqrt(jnp.mean(x * x, axis=-1, keepdims=True) + 1e-6)
        return (x / rms) * self.scale[...]
```

Both `Block.ln1`, `Block.ln2` (pre-attention and pre-FFN), and `Transformer.ln_f` (final output norm) respect `norm_type` via a `_build_norm` factory. Switching is a one-line config change with no code edits.

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

#### NTK-Aware RoPE Scaling

Setting `rope_scale_factor > 1` compresses the RoPE base frequency, allowing the model to generalise to contexts longer than `max_context` without fine-tuning (Neural Tangent Kernel-aware interpolation):

$$\theta_i = \frac{1}{(10000 \cdot \lambda)^{2i/C}}$$

where $\lambda$ = `rope_scale_factor`. A value of 2 approximately doubles the effective context window.

```python
# core/attention.py — NTK-aware frequency compression
base     = 10000.0 * self._rope_scale   # compressed if rope_scale_factor > 1
inv_freq = 1.0 / (base ** (jnp.arange(0, C, 2) / C))
```

!!! note
    `rope_scale_factor = 1.0` (default) gives the standard RoPE behaviour. The angle table is computed once at init, so inference speed is unaffected.

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

    normalization:
      norm_type: "layernorm"        # Normalisation type: "layernorm" or "rmsnorm"

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
      use_flash_attention: false    # Fused scaled-dot-product (jax.nn.dot_product_attention, JAX ≥ 0.4.25)
      rope_scale_factor: 1.0        # NTK-aware RoPE scaling: >1 compresses base frequency for long-context extrapolation

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
      batch_size: 64                # Total batch size (must be divisible by n_devices)
      grad_accum: 4                 # Gradient accumulation steps
      seed: 42
      optimizer: "adamw"            # "adamw", "adafactor", or "lion"
      epochs: 100
      warmup_steps: 0               # Linear LR warmup steps
      lr_schedule: "cosine"         # LR schedule after warmup: "cosine" | "linear" | "constant" | "wsd"
      grad_clip: 1.0                # Gradient clipping max norm (0 = disabled)
      patience: 0                   # Early stopping patience (0 = disabled)
      use_bf16: false               # Cast parameters to bfloat16

    lora:
      use_lora: false               # Enable LoRA adapters (freezes base nnx.Param weights)
      lora_rank: 8                  # Adapter rank r (smaller = fewer params)
      lora_alpha: 16.0              # Scaling constant α (effective scale = α / r)
      lora_dropout: 0.0             # Dropout on the LoRA δ path
      lora_targets: "attention"     # Which layers to adapt: "attention" | "mlp" | "all"

    multi_gpu:
      n_devices: 0                  # GPUs to use: 0 = all available, 1 = single-device

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

## Typed Model Outputs

`Transformer.__call__` returns a `ModelOutput` NamedTuple instead of a plain tuple. This is fully backward-compatible — existing code that unpacks the tuple continues to work unchanged:

```python
from core import ModelOutput

# Named access (preferred)
out = model(x, use_cache=False, kv_caches=None, cache_index=0)
loss = cross_entropy(out.logits, targets) + config.alpha_balance * out.aux_loss

# Positional unpacking (backward-compatible)
logits, kv_caches, aux_loss = model(x, ...)
```

`ModelOutput` is a native JAX pytree (NamedTuples are handled by `jax.tree_util` without registration), so it passes through `jax.jit`, `jax.grad`, and `nnx.value_and_grad` transparently.

| Field | Type | Description |
| :--- | :--- | :--- |
| `logits` | `jnp.ndarray [B, T, V]` | Token logits |
| `kv_caches` | `tuple` | Per-layer KV/latent caches |
| `aux_loss` | `float` | MoE load-balancing loss (0.0 for dense models) |

---

## LoRA Fine-Tuning

LoRA (Hu et al. 2022) inserts a trainable low-rank delta alongside each frozen linear projection. The effective weight is:

\[
W_{\text{eff}} = \underbrace{W_{\text{base}}}_{\text{frozen}} + \underbrace{\frac{\alpha}{r} \cdot A B}_{\text{trainable}}
\]

where \(A \in \mathbb{R}^{d \times r}\) is initialised with scaled Gaussian noise and \(B \in \mathbb{R}^{r \times k}\) is zero-initialised, so the adapter contributes nothing at the start of fine-tuning.

### Type-Level Freezing

DantinoX uses a custom `LoRAParam(nnx.Variable)` subclass — distinct from `nnx.Param` — to freeze base weights **at the type level**, not by masking or filtering:

```python
optimizer = nnx.Optimizer(model, tx, wrt=LoRAParam)          # only LoRAParam updated
grad_fn   = nnx.value_and_grad(loss, argnums=DiffState(0, LoRAParam))  # only LoRA grads
```

No `stop_gradient`, no manual filtering — the type system enforces the freeze.

### LoRALinear

`LoRALinear` is a drop-in replacement for `nnx.Linear`:

```python
from core.lora import LoRALinear, LoRAParam

layer = LoRALinear(in_features=512, out_features=512, rank=8, alpha=16.0, rngs=rngs)

# Forward: W_base(x) + (alpha/r) * dropout(x @ A) @ B
y = layer(x)

# Merge delta into base weight for deployment (zero inference overhead)
merged_kernel = layer.merge_weights()   # shape (in, out)
```

### Targets

| `lora_targets` | Adapted layers |
|---|---|
| `"attention"` | `qkv`, `o_proj` in every `Attention` block |
| `"mlp"` | `up_proj`, `down_proj` in every `MLP` block |
| `"all"` | All of the above |

### Trainable parameter count

With `lora_rank=8`, a 512-dim model adapting only attention projections trains ≈ 0.2 % of total parameters — practical fine-tuning on a single GPU.

---

## Multi-GPU Data-Parallel Sharding

DantinoX uses JAX's SPMD sharding (`jax.sharding.Mesh`) for data-parallel training. There is no `pmap`, no manual `jax.lax.pmean` — XLA infers and fuses the AllReduce automatically.

### Sharding strategy

| What | Sharding | Why |
|---|---|---|
| Model weights | `NamedSharding(mesh, P())` — replicated | Every device needs the full model for the forward pass |
| Input batch | `NamedSharding(mesh, P("data"))` — split on axis 0 | Each device processes a different slice |
| Gradients | Automatically AllReduced by XLA | `@jax.jit` compiles a single SPMD program |

```
Device 0 │ batch slice 0 → forward → ∂L/∂W ──┐
Device 1 │ batch slice 1 → forward → ∂L/∂W ──┤ AllReduce → W_new (replicated)
Device 2 │ batch slice 2 → forward → ∂L/∂W ──┤
Device 3 │ batch slice 3 → forward → ∂L/∂W ──┘
```

### Usage

Set `n_devices` in config — everything else is automatic:

```python
config = Config(
    dim=512, n_heads=16, head_size=32, num_blocks=8,
    batch_size=256,   # total; split to 64 per GPU across 4 devices
    n_devices=4,
)
Trainer(config).fit("data/corpus.txt")
```

**Constraint:** `batch_size % n_devices == 0`. Checked at startup.

### Low-level API

```python
from core.sharding import make_mesh, replicate, shard_batch, num_devices

mesh = make_mesh(n_devices=4)                # jax.sharding.Mesh over 4 GPUs
state_replicated = replicate(model_state, mesh)
x_sharded        = shard_batch(x, mesh)     # x.shape = (batch, seq_len)
print(num_devices(mesh))                    # 4
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
