# Core Layers — Deep Reference

This page is the complete implementation reference for the `core/` neural-network
primitives. Every layer, every formula, every configuration knob is explained
here. For the system-level design see the [Architecture overview](../architecture.md).

---

## Attention: MHA, GQA, MLA

All three attention variants live in `core/attention.py` and share the same
abstract base class `BaseAttention`. The variant is selected by
`config.attention_type` (`"mha"`, `"gqa"`, `"mla"`, or `"auto"`).

### Selection guide

| Mode | `config.attention_type` | `config.kv_heads` | KV-cache memory per token per layer |
|:---|:---:|:---:|:---|
| Multi-Head (MHA) | `"mha"` | `= n_heads` | `2 × H × d_h × bpp` bytes |
| Grouped-Query (GQA) | `"gqa"` | `< n_heads` | `2 × H_kv × d_h × bpp` bytes |
| Multi-Head Latent (MLA) | `"mla"` | any | `d_c^{KV} × bpp` bytes |

The `"auto"` mode derives the attention type from the legacy `config.mla` flag
and the relative values of `n_heads` and `kv_heads`.

---

### Multi-Head Attention (MHA)

MHA is the original attention mechanism (Vaswani et al. 2017). Every attention
head has its own set of key and value projections. The full forward pass:

$$\mathbf{Q} = \mathbf{x} W_Q, \quad \mathbf{K} = \mathbf{x} W_K, \quad \mathbf{V} = \mathbf{x} W_V$$

$$\text{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_h}}\right) \mathbf{V}$$

where $d_h$ = `head_size` is the dimension per head. The scaling by
$1/\sqrt{d_h}$ prevents the dot products from growing so large that the
softmax saturates into regions with vanishing gradients.

In DantinoX, all three projections are fused into a single `qkv` linear layer
that maps from `dim` to `dim + 2 × kv_heads × head_size`:

```python
qkv_out = self.dim + 2 * self.kv_heads * self.head_size
self.qkv = nnx.Linear(self.dim, qkv_out, use_bias=False, rngs=rngs)
```

The output is split with `jax.lax.split` at inference time.

---

### Grouped-Query Attention (GQA)

GQA (Ainslie et al. 2023) reduces the number of key-value projections relative
to query projections. Instead of `n_heads` independent KV heads, there are
`kv_heads < n_heads` heads. Each KV head is shared by a group of
`n_heads // kv_heads` query heads.

**Why it saves KV-cache memory.** In autoregressive generation, the K and V
tensors for all previous tokens must be kept in memory. With MHA this grows as
`2 × n_heads × d_h` values per token per layer. With GQA it is
`2 × kv_heads × d_h`. For a model with `n_heads=16`, `kv_heads=4`, the KV
cache is 4× smaller, which directly translates to longer generation contexts
or smaller VRAM requirements.

The implementation uses `MHAAttention` and `GQAAttention` classes which are
both subclasses of `_StandardAttention` and share the same forward pass code —
the only difference is the value of `kv_heads`.

During Flash Attention, GQA is handled by expanding the KV heads to match
the query count before calling the fused kernel:

```python
if self.kv_heads < self.n_heads:
    g    = self.n_heads // self.kv_heads
    k_fa = jnp.repeat(k_fa, g, axis=2)
    v_fa = jnp.repeat(v_fa, g, axis=2)
```

In the general (non-Flash) path the grouped layout is preserved in memory and
the `q @ k.T` matmul is broadcast automatically.

---

### Multi-Head Latent Attention (MLA)

MLA (DeepSeek-V2) compresses keys and values to a low-dimensional latent space
before caching. This makes the KV cache dramatically smaller.

#### Latent compression

$$\mathbf{c}_{KV} = \mathrm{Norm}(W_{DKV}\,\mathbf{x})$$

where $W_{DKV}$ maps from `dim` to `down_dim_kv` (`config.down_dim_kv`,
default 256). The compressed latent $\mathbf{c}_{KV}$ is `down_dim_kv`
scalars per token — typically 3 to 20 times smaller than the full KV tensors.

Only $\mathbf{c}_{KV}$ is stored in the KV cache, not the full K and V
matrices. Full K and V are re-expanded from $\mathbf{c}_{KV}$ on demand
during the forward pass:

```python
self.down_kv = nnx.Linear(config.dim, config.down_dim_kv, rngs=rngs)
self.up_k    = nnx.Linear(config.down_dim_kv, head_size * kv_heads, rngs=rngs)
self.up_v    = nnx.Linear(config.down_dim_kv, head_size * kv_heads, rngs=rngs)
```

Queries undergo the same treatment with `down_dim_q` (`config.down_dim_q`):

$$\mathbf{c}_{Q} = \mathrm{Norm}(W_{DQ}\,\mathbf{x}), \quad \mathbf{q} = W_{UQ}\,\mathbf{c}_{Q}$$

A separate small RoPE component (`rope_dim`, default 32) is appended to Q and K
to preserve relative positional information that would otherwise be lost in the
latent projection.

#### Weight absorption at inference

When `config.inference=True`, MLA avoids materialising the full K and V tensors
entirely, instead absorbing the up-projection weights into the attention score
computation. The K up-projection is folded into the Q-K dot product:

$$A_{QK} = W_{UQ}^\top W_{UK}$$

and the V up-projection into the context-output product:

$$W_{VO} = W_{UV} W_O$$

Both absorbed products are computed once from the weight tensors and applied
via `jnp.einsum`. The cache stores only the low-dimensional latent
$\mathbf{c}_{KV}$, never the full K or V:

```python
# Absorbed path (inference):
q_proj    = self.up_q.kernel.reshape(down_dim_q, kv_heads, g, head_size)
k_proj    = self.up_k.kernel.reshape(down_dim_kv, kv_heads, head_size)
attn_proj = jnp.einsum("qngh, knh -> ngqk", q_proj, k_proj)
attn_proj = jnp.einsum("btq, ngqk -> btngk", q, attn_proj)
attn      = jnp.einsum("btngk, bsk -> bngts", attn_proj, k)  # k is c_kv cache
```

!!! warning "Training vs inference mode"
    Set `config.inference=False` during training (the default). Set
    `config.inference=True` for generation. The weights are identical; only
    the computation graph changes. Forgetting to switch modes gives correct
    results but suboptimal performance.

---

### Flash Attention

Flash Attention (Dao et al. 2022) is a hardware-aware algorithm that computes
the exact same attention output as the standard formulation but avoids
materialising the full $T \times T$ attention matrix.

**Tiling algorithm.** The $Q$, $K$, and $V$ matrices are divided into tiles
that fit in SRAM (the on-chip memory of the GPU SM). The softmax is computed
incrementally across tiles using a running maximum trick. At no point is the
full $O(T^2)$ matrix written to HBM. Memory complexity drops from $O(T^2)$ to
$O(T)$.

In DantinoX, Flash Attention is activated when all of the following hold:

1. `config.use_flash_attention=True`
2. `use_cache=False` (training, not AR generation)
3. `sliding_window=False`
4. `is_causal=True`
5. `prefix_kv=None` (no diffusion dual-cache)

Under these conditions `fit` calls the JAX fused kernel directly:

```python
y = jax.nn.dot_product_attention(q_fa, k_fa, v_fa, is_causal=True)
```

`jax.nn.dot_product_attention` dispatches to the `cudnn` or `xla` Flash
Attention backend depending on the hardware and JAX version (≥ 0.4.25).

!!! tip "When to enable Flash Attention"
    Enable for medium-to-large models training with long sequences
    (`max_context >= 512`). The memory saving grows quadratically with
    sequence length, so for `max_context=2048` Flash Attention uses ~16×
    less activation memory for the attention computation than the standard path.

---

### Static KV-Cache (XLA-compatible)

The KV-cache enables efficient autoregressive generation by reusing the K and V
tensors computed for all previous tokens.

**Pre-allocated buffers.** The cache is a pair of zero-filled arrays of shape
`[B, kv_heads, 1, max_context, head_size]` allocated the first time
`use_cache=True` is passed:

```python
if kv_cache[0] is None:
    kc = jnp.zeros((B, self.kv_heads, 1, self.max_context, self.head_size), dtype=k.dtype)
    vc = jnp.zeros_like(kc)
    kc = kc.at[:, :, :, :T, :].set(k)   # prefill
    vc = vc.at[:, :, :, :T, :].set(v)
```

**`dynamic_update_slice` for decode.** At each decode step (one new token,
`T=1`), the new K and V vectors are inserted at position `cache_index` using
`jax.lax.dynamic_update_slice`:

```python
kc = jax.lax.dynamic_update_slice(kc, k, (0, 0, 0, cache_index, 0))
vc = jax.lax.dynamic_update_slice(vc, v, (0, 0, 0, cache_index, 0))
```

`dynamic_update_slice` is a primitive that XLA understands natively. It writes
a small tensor into a position of a larger tensor that is determined at runtime
(`cache_index` is a dynamic value).

**Why "static" shapes?** XLA requires every tensor's shape to be known at
compile time. If the cache array grew dynamically (like a Python list), XLA
would need to recompile for every new length. Using a fixed-size pre-allocated
buffer means the shape is always `[B, kv_heads, 1, max_context, head_size]`,
so compilation happens exactly once. The `cache_index` counter advances at
runtime without triggering recompilation.

**MLA cache.** For MLA the cache stores the compressed latent
$\mathbf{c}_{KV}$ of shape `[B, max_context, down_dim_kv]` plus a small RoPE
component `[B, 1, 1, max_context, rope_dim]`. This is the storage saving:
instead of `2 × kv_heads × head_size` values per token, MLA stores
`down_dim_kv + rope_dim` values.

---

## Feed-Forward Network (FFN)

The FFN in `core/mlp.py` sits after the attention sub-layer inside each
`Block`. It is a position-wise two-layer network applied identically to every
token vector.

### GELU

GELU (Gaussian Error Linear Unit) is a smooth approximation of ReLU:

$$\text{GELU}(x) = x \cdot \Phi(x) \approx 0.5 x \left(1 + \tanh\!\left[\sqrt{2/\pi}(x + 0.044715 x^3)\right]\right)$$

where $\Phi$ is the standard normal CDF. Compared to ReLU, GELU is differentiable
at zero and has non-zero gradient for slightly negative inputs, which improves
gradient flow.

In DantinoX the standard GELU is used via `jax.nn.gelu` when
`config.use_swiglu=False` and `config.activation="gelu"`.

### SwiGLU

SwiGLU (Shazeer 2020) is the default FFN activation in DantinoX
(`config.use_swiglu=True`). It is a gated linear unit variant used in
LLaMA, PaLM, and most modern large language models.

The formula is:

$$\text{FFN}(x) = W_2 \cdot \big(W_1 x \odot \sigma(W_{\text{gate}} x)\big)$$

where $\sigma$ is the SiLU (Sigmoid Linear Unit, also called Swish):
$\sigma(x) = x \cdot \text{sigmoid}(x)$.

In practice, $W_1$ and $W_{\text{gate}}$ are implemented as a single fused
linear layer of output size `2 × intermediate_dim`, then split:

```python
class Swiglu(nnx.Module):
    def __call__(self, x):
        gate, data = jnp.split(x, 2, axis=-1)
        return jax.nn.silu(gate) * data
```

The `up_proj` maps from `dim` to `2 × intermediate_dim` (when `use_swiglu=True`)
so the split produces two halves of size `intermediate_dim`:

```python
up_proj_dim = intermediate_dim * 2 if config.use_swiglu else intermediate_dim
self.up_proj = nnx.Linear(config.dim, up_proj_dim, rngs=rngs)
self.down_proj = nnx.Linear(intermediate_dim, config.dim, rngs=rngs)
```

**Why SwiGLU improves training.** The gating mechanism allows the network to
selectively suppress individual feature channels, giving it more expressive
power than a plain ReLU for the same parameter count. Empirically, SwiGLU
consistently achieves lower perplexity than GELU at the same FLOP budget.

---

### Sparse Mixture-of-Experts (MoE)

MoE (`core/moe.py`) replaces the dense FFN with `n_experts` independent MLP
modules and a learned router that selects the top-K experts for each token.

```python
class MoE(nnx.Module):
    def __init__(self, config, rngs):
        self.n_experts = config.n_experts          # (1)!
        self.experts   = nnx.List([MLP(config, rngs) for _ in range(n_experts)])
        self.router    = nnx.Linear(config.dim, n_experts, use_bias=False, rngs=rngs)
        self.top_k_mlp = config.top_k_mlp         # (2)!
```

1. Total number of experts (default `4`).
2. Number of experts activated per token (default `2`).

**Router.** For each token vector $\mathbf{x} \in \mathbb{R}^d$, the router
computes a softmax distribution over experts:

$$\mathbf{p} = \text{softmax}(W_r \mathbf{x}) \in \mathbb{R}^N$$

The top-K indices and their probabilities are selected, then re-normalised
so the weights sum to 1:

```python
probs = jax.nn.softmax(self.router(x))
values, indices = jax.lax.top_k(probs, self.top_k_mlp)
values = values / jnp.sum(values, axis=-1, keepdims=True)
```

The output is a weighted sum of the K selected expert outputs:

$$\mathbf{y} = \sum_{i \in \text{top-K}} p_i \cdot \text{Expert}_i(\mathbf{x})$$

**Load-balancing loss.** Without regularisation, the router collapses to
always routing every token to the same expert. The auxiliary loss prevents this:

$$\mathcal{L}_{\text{bal}} = \alpha \cdot N \sum_{i=1}^{N} f_i \cdot P_i$$

where $N$ is the number of experts, $f_i$ is the fraction of tokens routed to
expert $i$, $P_i$ is the mean router probability for expert $i$, and $\alpha$ =
`config.alpha_balance` (default `0.1`) is a coefficient that controls the
strength of the regularisation.

In DantinoX:

```python
expert_mean_prob = jnp.mean(probs.reshape(B*T, N), axis=0)  # P_i
freq = jnp.mean(jnp.sum(one_hot(indices, N), axis=2), axis=(0, 1))  # f_i
moe_loss = jnp.sum(freq * expert_mean_prob) * N              # * N = α already included
```

The loss is added to the main loss with `model.alpha_balance` as the coefficient
inside `train_step`.

**Why K=2 is standard.** Using K=1 gives no gradient through the discrete
top-1 selection (only the selected expert receives a gradient). K=2 provides
a soft gradient through both selected experts and empirically achieves better
load balance. K > 2 reduces the sparsity benefit; K=2 is the Mixtral and
Switch-Transformer default.

---

## Normalisation

Normalisation layers appear twice per transformer block: before attention
(`norm1`) and before the FFN (`norm2`). DantinoX calls this configuration
"pre-norm", which is more stable than the original "post-norm" Transformer.

### LayerNorm

Standard Layer Normalisation (Ba et al. 2016). Normalises over the feature
dimension $d$ of each token independently:

$$\hat{\mathbf{x}} = \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y = \gamma \odot \hat{\mathbf{x}} + \beta$$

where $\mu$ and $\sigma^2$ are the mean and variance computed over the $d$
feature channels of each token, $\gamma \in \mathbb{R}^d$ is a learned gain,
and $\beta \in \mathbb{R}^d$ is a learned bias.

Selected with `config.norm_type="layernorm"`. Implemented by `nnx.LayerNorm`.

### RMSNorm

Root Mean Square Layer Normalisation (Zhang & Sennrich 2019). Drops the mean
subtraction, computing only the RMS:

$$\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}$$

$$y = \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})} \odot \gamma$$

The gain $\gamma \in \mathbb{R}^d$ is learnable; there is no bias term.

Selected with `config.norm_type="rmsnorm"` (the default for modern models).
RMSNorm is used in LLaMA, Mistral, Llama 2/3, and most current open-source LLMs.

**Why RMSNorm is faster.** Computing mean and variance requires two passes over
the feature vector. RMSNorm only requires one pass. In practice RMSNorm
achieves similar training behaviour to LayerNorm at ~10-15 % lower wall-clock
time per step.

Implementation in `core/block.py`:

```python
class RMSNorm(nnx.Module):
    def __init__(self, dim, *, eps=1e-6, rngs):
        self.scale = nnx.Param(jnp.ones(dim))
        self.eps   = eps

    def __call__(self, x):
        rms = jnp.sqrt(jnp.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return (x / rms) * self.scale[...]
```

---

## Positional Encoding

Positional encoding is selected via `config.use_rotary_pos`,
`config.trainable_pos`, and `config.absolute_pos` (or the cleaner
`ModelConfig.pos_encoding` field with values `"rotary"`, `"absolute"`,
`"learned"`, `"none"`).

### RoPE (Rotary Positional Embedding) — default

RoPE (Su et al. 2022) applies a rotation in 2-D subspaces of the query and key
vectors. For each pair of dimensions $(2i, 2i+1)$ and position $p$:

$$\begin{pmatrix} q_{2i}' \\ q_{2i+1}' \end{pmatrix} = \begin{pmatrix} \cos(p\,\theta_i) & -\sin(p\,\theta_i) \\ \sin(p\,\theta_i) & \cos(p\,\theta_i) \end{pmatrix} \begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}$$

where $\theta_i = \frac{1}{10000^{2i/d_h}}$ is a frequency.

The dot product $q' \cdot k'^{(m)}$ for tokens at positions $p$ and $m$
depends only on their *relative* distance $p - m$, not their absolute
positions. This is the key property: RoPE encodes relative positions
implicitly through the rotation, without needing separate relative position
embeddings.

**NTK-aware scaling.** For sequences longer than `max_context`, the base
frequency can be scaled:

$$\theta_i = \frac{1}{(10000 \times \lambda)^{2i/d_h}}$$

where $\lambda$ = `config.rope_scale_factor` (default `1.0`). Increasing
$\lambda$ reduces the rotation speed of each frequency, effectively stretching
the positional encoding to longer contexts. This is the NTK-aware interpolation
method (bloc97 2023).

DantinoX precomputes a frequency table at construction time:

```python
def _compute_angle(self, T, C):
    P        = jnp.arange(T, dtype=jnp.float32)
    base     = 10_000.0 * self._rope_scale
    inv_freq = 1.0 / (base ** (jnp.arange(0, C, 2, dtype=jnp.float32) / C))
    degree   = jnp.einsum("i,j->ij", P, inv_freq)
    return degree[None, None, None, :, :]  # [1, 1, 1, T, C//2]
```

The rotation is applied in a numerically stable way:

```python
out[..., 0::2] = x[..., 0::2] * cos_a - x[..., 1::2] * sin_a
out[..., 1::2] = x[..., 0::2] * sin_a + x[..., 1::2] * cos_a
```

Two variants exist for different tensor layouts:
- `_apply_rope_grouped`: for `[B, H, G, T, D]` tensors (general attention path).
- `_apply_rope_thd`: for `[B, T, H, D]` tensors (Flash Attention path).

### Sinusoidal (absolute, fixed)

Classic absolute positional encoding from the original Transformer. No
learnable parameters:

$$\text{pe}(p, 2i) = \sin\!\left(\frac{p}{10000^{2i/d}}\right), \quad \text{pe}(p, 2i+1) = \cos\!\left(\frac{p}{10000^{2i/d}}\right)$$

Added to the token embedding table at the input of the transformer. Selected
with `config.absolute_pos=True` (or `pos_encoding="absolute"`).

Because it has no learned parameters, sinusoidal encoding can generalise to
lengths beyond `max_context` at inference (with some degradation). However,
it has largely been superseded by RoPE for language modelling.

### Learned positional embeddings

An `nnx.Embed` table of shape `[max_context, dim]` is added to the input
embeddings. Selected with `config.trainable_pos=True` (or
`pos_encoding="learned"`). Simple and effective for tasks where the sequence
length is always bounded by `max_context`.

---

## LoRA — Low-Rank Adaptation

LoRA (Hu et al. 2022) enables parameter-efficient fine-tuning by injecting a
low-rank update into existing linear layers, leaving the base weights frozen.

### `LoRALinear` — how it works

`LoRALinear` in `core/lora.py` is a drop-in replacement for `nnx.Linear`. It
wraps a frozen base linear layer and adds two small adapter matrices A and B:

```python
class LoRALinear(nnx.Module):
    def __init__(self, in_features, out_features, *, rank=8, alpha=16.0, ...):
        self.base  = nnx.Linear(in_features, out_features, ...)  # base weight = nnx.Param
        self.scale = alpha / rank

        k_a, k_b = jax.random.split(rngs.params())
        self.lora_A = LoRAParam(jax.random.normal(k_a, (in_features, rank)) / sqrt(in_features))
        self.lora_B = LoRAParam(jnp.zeros((rank, out_features)))  # zero init → delta=0 at t=0

    def __call__(self, x):
        out   = self.base(x)          # frozen forward
        delta = x @ self.lora_A[...]  # low-rank projection
        return out + (delta @ self.lora_B[...]) * self.scale
```

**Effective weight.** The computation is equivalent to:

$$W_{\text{eff}} = W_{\text{base}} + \frac{\alpha}{r} \cdot AB$$

where $W_{\text{base}} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$
is the frozen pre-trained weight, $A \in \mathbb{R}^{d_{\text{in}} \times r}$
and $B \in \mathbb{R}^{r \times d_{\text{out}}}$ are the adapter matrices, and
$\alpha/r$ is a scalar that rescales the adapter contribution. $r$ is the rank.

**Why rank-$r$.** The product $AB$ is a rank-$r$ matrix, so it can capture
at most $r$ linearly independent directions in the weight space. Because
$r \ll \min(d_{\text{in}}, d_{\text{out}})$, the adapter has far fewer
parameters than the base weight: $r \times (d_{\text{in}} + d_{\text{out}})$
vs $d_{\text{in}} \times d_{\text{out}}$. For a typical linear layer
(512 → 512, r=8): 8 × 1024 = 8192 vs 512 × 512 = 262144 — a 32× reduction.

**Initialisation.** $B$ is initialised to zero so that at the start of training
$AB = 0$ and `LoRALinear` is identical to the frozen base layer. $A$ is
initialised with small random values scaled by $1/\sqrt{d_{\text{in}}}$.

### `LoRAParam` — the type-system freezing mechanism

```python
class LoRAParam(nnx.Variable):
    """Trainable LoRA variable — distinct type so base nnx.Param weights stay frozen."""
    pass
```

`LoRAParam` is a subclass of `nnx.Variable`. When the optimizer is constructed
with `wrt=LoRAParam`, NNX's state-extraction machinery (`nnx.state(model, LoRAParam)`)
collects *only* variables of this type. The base weights are `nnx.Param` and are
never included in the optimizer's tracked state, so they receive no gradient
updates.

This is the entire freezing mechanism — no `stop_gradient`, no masking, no
manual filtering. The type hierarchy enforces it.

### `merge_weights()` — fold adapters for inference

```python
def merge_weights(self) -> jnp.ndarray:
    return self.base.kernel[...] + self.scale * (self.lora_A[...] @ self.lora_B[...])
```

`merge_weights()` returns the fused weight $W_{\text{base}} + \frac{\alpha}{r} AB$
as a single array. After merging, the model can be loaded into a standard
`nnx.Linear` with no runtime overhead from the adapter computation.

!!! tip "Merging for deployment"
    After fine-tuning, call `merge_weights()` on each `LoRALinear`, replace it
    with a standard `nnx.Linear` initialised from the merged kernel, and save.
    The deployed model runs exactly as fast as a fully fine-tuned model.

### LoRA targets

`config.lora_targets` controls which layers receive `LoRALinear` adapters:

| `config.lora_targets` | Layers adapted |
|---|---|
| `"attention"` (default) | `qkv` and `o_proj` in every attention block |
| `"mlp"` | `up_proj` and `down_proj` in every FFN block |
| `"all"` | Both attention and FFN layers |

---

## `Block` — the unified transformer block

`core/block.py` defines a single `Block` class that handles both AR and
diffusion models:

```python
class Block(nnx.Module):
    def __init__(self, config, rngs):
        self.attention = build_attention(config, rngs)  # MHA | GQA | MLA
        self.norm1     = _build_norm(config, config.dim, rngs)
        self.norm2     = _build_norm(config, config.dim, rngs)
        self.causal    = config.causal
        self.ffn       = MoE(config, rngs) if config.use_moe else MLP(config, rngs)
```

The forward pass is a standard pre-norm residual:

```
x → norm1 → attention → residual add → norm2 → FFN → residual add → x_out
```

```python
def __call__(self, x, *, cache=None, cache_index=0, ...):
    x_norm = self.norm1(x)
    x_attn, new_cache = self.attention(x_norm, use_cache=(cache is not None), ...)
    x = x + x_attn

    ff, aux = self.ffn(self.norm2(x), deterministic=deterministic)
    x_out   = x + ff
    return x_out, new_cache, aux
```

The `causal` flag (from `config.causal`, which is `True` when
`model_type == "autoregressive"`) is passed to the attention layer as
`is_causal`. When `False`, attention is bidirectional — every position can
attend to every other position. This is the correct mode for masked diffusion
models (where the model must "fill in" masked tokens by attending to surrounding
context in both directions).

---

## Gradient Checkpointing

Gradient checkpointing (also called activation rematerialisation) trades
compute for memory. It is controlled by `config.gradient_checkpointing`.

### How `nnx.remat` works

`nnx.remat` wraps an NNX module with JAX's `jax.checkpoint` primitive. During
the forward pass, intermediate activations inside the wrapped module are
*discarded* — they are not held in memory for the backward pass.

```python
# In Transformer.__call__ (pseudocode):
for block in self.blocks:
    if self.gradient_checkpointing:
        block_fn = nnx.remat(block)
    else:
        block_fn = block
    x, cache, aux = block_fn(x, ...)
```

**Forward pass.** The block computes its output and returns it, but XLA is told
not to hold on to the intermediate values (attention weights, intermediate FFN
activations, etc.) in device memory.

**Backward pass.** When the backward pass needs the activations for gradient
computation, it *re-runs the forward pass* of the block from the block's input
(which is retained). This recomputation adds approximately one extra forward
pass worth of FLOP per block.

### Memory vs compute trade-off

| Metric | Without checkpointing | With checkpointing |
|---|---|---|
| Activation memory | $O(\text{num\_blocks})$ | $O(\sqrt{\text{num\_blocks}})$ (selective) or $O(1)$ (full) |
| Extra compute | 0 % | ~33 % |
| Typical VRAM saving | — | 50–80 % |

The factor of $O(\sqrt{L})$ applies to the optimal selective checkpointing
strategy (checkpoint every $\sqrt{L}$-th block). DantinoX uses full per-block
remat (`nnx.remat` on every block), which achieves the best memory saving at
the cost of recomputing every block's internals.

**When is checkpointing automatically disabled?** When `use_cache=True`
(AR generation), the KV cache stores the K and V activations for all previous
tokens. Using remat during generation would force recomputation of the entire
prefix at every decode step, which is far worse than just keeping the
activations. The `Block` is always called without `nnx.remat` during generation.

!!! tip "Practical recommendation"
    Enable `gradient_checkpointing=True` whenever `batch_size × grad_accum`
    is large or `max_context > 512`. For short sequences and small batches
    it may not be necessary and the 33 % extra compute has a measurable impact
    on step throughput.

---

## `build_attention` factory

```python
def build_attention(config, rngs) -> BaseAttention:
    t = getattr(config, "attention_type", "auto")
    if t == "mla":  return MLAAttention(config, rngs)
    if t == "gqa":  return GQAAttention(config, rngs)
    if t == "mha":  return MHAAttention(config, rngs)
    # "auto" fallback:
    if getattr(config, "mla", False):          return MLAAttention(config, rngs)
    if (config.kv_heads or n_heads) < n_heads: return GQAAttention(config, rngs)
    return MHAAttention(config, rngs)
```

The factory handles both the new explicit `attention_type` string and the legacy
`mla` / `kv_heads` flags so that existing YAML configs continue to work.

---

## No-sink gating (attention sink suppression)

When `config.no_sink=True`, a learned gating signal is applied to the attention
output before the residual connection:

```python
if self.no_sink:
    self.W = nnx.Linear(self.dim, self.dim, rngs=rngs)

def _apply_gate(self, y, x):
    return y * jax.nn.sigmoid(self.W(x)) if self.no_sink else y
```

This is inspired by the "attention sink" phenomenon (Xiao et al. 2023), where
certain tokens attract disproportionately large attention weights and act as
"sink" tokens. The gating mechanism allows the model to suppress the output
from such positions when the original input `x` does not warrant a strong
response.

---

## Sliding window attention

When `config.sliding_window=True`, each token attends only to a local window
of `config.context_window` preceding tokens:

```python
table = jnp.arange(T)[:, None] - jnp.arange(T)[None, :]
mask  = (table <= config.context_window) & (table >= 0)
self.window = jnp.where(mask, 0.0, -1e9)
```

The window bias is added to the attention logits, masking out all positions
outside the window with $-10^9$. This reduces the effective receptive field
but allows the model to handle sequences much longer than `max_context` at
a fixed compute cost.

---

## See also

- [Training API Reference](../api/training.md) — `Trainer.fit`, optimizers, schedules
- [Architecture Overview](../architecture.md) — system-level design
- [Configuration Reference](../configuration.md) — all `Config` and `ModelConfig` fields
- [Generation](generation.md) — autoregressive sampling, diffusion sampling
