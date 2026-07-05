# Core Layers — Deep Reference

This page is the complete implementation reference for the `core/` neural-network
primitives. Every layer, every formula, every configuration knob is explained
here. For the system-level design see the [Architecture overview](../architecture.md).

---

## Attention: MHA, GQA, MLA

All three attention variants live in `core/attention.py` and share the same
abstract base class `BaseAttention`. The variant is selected by
`config.attention_type` (`"mha"`, `"gqa"`, `"mla"`, or `"auto"`). The module provides shared infrastructure for RoPE positional encoding, causal/sliding-window masking, no-sink gating, dropout, and native LoRA integration.

### Selection guide

| Mode | `config.attention_type` | `config.kv_heads` | KV-cache memory per token per layer |
|:---|:---:|:---:|:---|
| Multi-Head (MHA) | `"mha"` | `= n_heads` | `2 × H × d_h × bpp` bytes |
| Grouped-Query (GQA) | `"gqa"` | `< n_heads` | `2 × H_kv × d_h × bpp` bytes |
| Multi-Head Latent (MLA) | `"mla"` | any | `d_c^{KV} × bpp` bytes |

The `"auto"` mode derives the attention type from the legacy `config.mla` flag
and the relative values of `n_heads` and `kv_heads`.

---

### Multi-Head (MHA) and Grouped-Query Attention (GQA)

MHA is the original attention mechanism (Vaswani et al. 2017) where every attention head has its own set of key and value projections. GQA (Ainslie et al. 2023) reduces the number of key-value projections relative to query projections, sharing each KV head across a group of `n_heads // kv_heads` query heads to drastically save KV-cache memory during autoregressive generation.

In DantinoX, both MHA and GQA share the exact same forward pass via the `_StandardAttention` class. The distinction is purely structural based on the `kv_heads` parameter. 

The projections are fused into a single `qkv` linear layer, which natively supports **Low-Rank Adaptation (LoRA)** if `config.use_lora` targets the attention blocks:

```python
qkv_out = self.dim + 2 * self.kv_heads * self.head_size
self.qkv = (
    LoRALinear(self.dim, qkv_out, use_bias=False, **_lk)
    if _use_lora else nnx.Linear(self.dim, qkv_out, use_bias=False, rngs=rngs)
)
```

During Flash Attention, GQA is handled by repeating the KV heads to match the query count before calling the fused kernel. In the general (non-Flash) path, the grouped layout `[B, kv_heads, G, T, head_size]` is preserved, and the matmul is broadcast automatically.

---

### Differential Attention

If `config.differential=True`, `_StandardAttention` activates Differential Attention to mitigate attention noise. This requires computing a second set of queries and keys via an additional projection (`q2k2`). 

The attention scores are penalized by subtracting a second attention map, scaled by a set of learnable per-head vectors $\lambda$:

$$\lambda = \exp(\lambda_{q1} \cdot \lambda_{k1}) - \exp(\lambda_{q2} \cdot \lambda_{k2}) + \lambda_{\text{init}}$$

The final attention map is computed as:

$$\text{Attn} = \text{softmax}(Q_1 K_1^\top) - \lambda \cdot \text{softmax}(Q_2 K_2^\top)$$

The output is then normalized using an `RMSNorm` layer (`diff_norm`) and scaled by $(1 - \lambda_{\text{init}})$. This feature is fully supported across both the standard decoding path and the Flash Attention fast path.

---

### Linear Attention

For purely bidirectional contexts (e.g., non-causal encoder architectures or diffusion models), DantinoX supports $O(T)$ Linear Attention (Katharopoulos et al., 2020) via `config.use_linear_attention`. 

Instead of computing the $T \times T$ attention matrix, queries and keys are projected into a positive feature space using an ELU-based mapping:

$$\phi(x) = \text{elu}(x) + 1$$

The key-value product is accumulated once over the sequence length, avoiding the standard causal prefix-sum. 

> **Warning: Bidirectional Only** > Linear attention is strictly ignored if `is_causal=True`. The module will emit a warning and automatically fall back to standard softmax or Flash Attention.

---

### Multi-Head Latent Attention (MLA)

MLA (DeepSeek-V2) compresses keys and values to a low-dimensional latent space
before caching, making the KV cache dramatically smaller.

#### Latent compression

$$\mathbf{c}_{KV} = \mathrm{Norm}(W_{DKV}\,\mathbf{x})$$

The compressed latent $\mathbf{c}_{KV}$ is `down_dim_kv` scalars per token — typically 3 to 20 times smaller than the full KV tensors. Only $\mathbf{c}_{KV}$ is stored in the cache. Full K and V are re-expanded on demand during the forward pass. Queries undergo the same treatment with `down_dim_q`. A separate small RoPE component is appended to preserve relative positional information.

#### Weight absorption at inference

When `config.inference=True`, MLA avoids materialising the full K and V tensors entirely. Because $\mathbf{c}_{KV}$ is used as both the key-side and value-side operand, the up-projection weights are absorbed directly into the `jnp.einsum` computations. 

The DantinoX implementation correctly handles biases during this absorption to remain numerically identical to the explicit training path:

```python
# Inference path: Up-projections folded into einsums
G    = self.n_heads // self.kv_heads
q_up = self.up_q(q).reshape(B, T, self.kv_heads, G, self.head_size)
W_k  = self.up_k.kernel.reshape(self.down_dim_kv, self.kv_heads, self.head_size)

# score[t, s] = q_up[t] · (c_kv[s] W_k + b_k)
qWk  = jnp.einsum("btngh, knh -> btngk", q_up, W_k)
attn = jnp.einsum("btngk, bsk -> bngts", qWk, k) # k is the c_kv cache

if self.up_k.bias is not None:
    b_k  = self.up_k.bias[...].reshape(self.kv_heads, self.head_size)
    attn = attn + jnp.einsum("btngh, nh -> bngt", q_up, b_k)[..., None]
```

A similar absorption happens for the output projection over the values (`W_v`). 

> **Tip: Training vs inference mode** > Set `config.inference=False` during training. Set `config.inference=True` for generation. The weights are identical, but the computation graph changes entirely.

---

### Flash Attention

Flash Attention (Dao et al. 2022) is a hardware-aware algorithm that computes the exact same attention output but avoids materialising the full $O(T^2)$ matrix in HBM, reducing memory complexity to $O(T)$.

In DantinoX, Flash Attention is triggered via `jax.nn.dot_product_attention` when all the following conditions are met:
1. `config.use_flash_attention=True`
2. `use_cache=False` (training/forward pass, not AR decoding)
3. `sliding_window=False`
4. `is_causal=True`
5. `prefix_kv=None` (no diffusion dual-cache active)

It seamlessly supports Differential Attention by computing `q2_fa` and `k2_fa`, calculating the secondary attention map, and blending them before the final projection.

---

### Positional Encoding & Gate Mechanics

**RoPE (Rotary Positional Embeddings):** The base class dynamically computes causal and sliding-window masks on the fly to avoid O(max_context²) memory bloat. RoPE frequencies are precomputed once into an `angle` table. During the forward pass, `_apply_rope_grouped` or `_apply_rope_thd` dynamically slices the precomputed table based on the current `cache_index`, ensuring robust positional alignment during token-by-token generation.

**No-Sink Gating:** If `config.no_sink=True`, DantinoX applies an Attention Sink gating mechanism to the final attention output $y$ before the projection layer:

$$y = y \cdot \sigma(W(x))$$

---

### Static KV-Cache & Prefix Injection

The KV-cache enables efficient autoregressive generation. DantinoX uses strictly typed `NamedTuple` objects (`KVCache` and `MLACache`) to flow through JAX `jit` boundaries and `fori_loop` constructs transparently.

**Pre-allocated buffers:** The cache consists of zero-filled arrays allocated on the first call. XLA requires shapes to be statically known at compile time, preventing dynamic growth.

```python
if kv_cache[0] is None:
    kc = jnp.zeros((B, self.kv_heads, 1, self.max_context, self.head_size), dtype=k.dtype)
    vc = jnp.zeros_like(kc)
```
*Note: If Differential Attention is active, `KVCache` also tracks `k2c` for the secondary key projections.*

**`dynamic_update_slice` for decode:** At each decode step (`T=1`), the new vectors are inserted at `cache_index`. XLA handles this natively without recompilation.

**Dual-Cache Prefix Injection:** DantinoX supports opt-in prefix caching for diffusion architectures. By passing a `prefix_kv` tuple, the model concatenates external keys and values to the sequence axis dynamically:

```python
if prefix_kv is not None:
    pk, pv = prefix_kv
    k = jnp.concatenate([pk, k], axis=3)  # [B, kv_heads, 1, T_pre+T, head_size]
    v = jnp.concatenate([pv, v], axis=3)
```

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
