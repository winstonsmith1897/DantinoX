
## Architecture & Technical Specs


| Feature | Implementation Details |
| :--- | :--- |
| **Attention** | Causal Self-Attention with GQA and optional Sliding Window and gating `no_sink`|
| **Feed-Forward** | Configurable: Dense MLP or Sparse MoE (Top-K Routing) |
| **Positioning** | Rotary Positional Embeddings (RoPE) or Absolute |
| **Memory Opt.** | Gradient checkpointing (`nnx.remat`) & Weight Tying (`lm_head.kernel = wte.embedding.T`)|
| **Inference Opt.**| Autoregressive generation with Static KV-Cache |
| **Regularization**| Attention, residual, and embedding dropout; auxiliary MoE balancing loss `load_balancing_loss` |
| **Distributed** | JAX SPMD (Data / Model / FSDP) - *Future Work* |


## Configuration Reference

DantinoX is entirely driven by a centralized YAML configuration. This design allows you to easily ablate architectural components (like toggling MoE or sliding window attention) without modifying the core JAX codebase.

Below is the annotated `default_config.yaml`:

??? note "Click to expand the full YAML Configuration"
    ```yaml
    model:
      dim: 512                    # Core hidden dimension
      n_heads: 16                 # Number of query heads
      kv_heads: 4                 # Number of key/value heads (set < n_heads for GQA)
      head_size: 32               # Dimensionality of each attention head
      num_blocks: 12              # Number of transformer layers
      max_context: 512            # Maximum sequence length
      weight_tying: true          # Share weights between embedding and LM head
      activation: gelu            # Non-linear activation function
      gradient_checkpointing: true # Enable nnx.remat to reduce VRAM usage
      dropout_rate: 0.15          # Regularization dropout probability

    moe:
      use_moe: true               # Toggle Sparse MoE vs standard Dense FFN
      n_experts: 4                # Total number of routed experts
      top_k_mlp: 2                # Number of experts activated per token
      expansion: 4                # Hidden dimension expansion factor in experts
      alpha_balance: 0.1          # Weight of the auxiliary load-balancing loss

    attention:
      use_rotary_pos: true        # Enable Rotary Positional Embeddings (RoPE)
      trainable_pos: false        # Enable standard learned positional embeddings
      absolute_pos: false         # Enable absolute sinusoidal embeddings
      sliding_window: true        # Restrict attention to a local past context
      context_window: 64          # Size of the local window (if sliding_window: true)
      no_sink: true               # Enable attention gating to stabilize training

    tokenizer:
      tokenizer_type: "char"      # Tokenization strategy (e.g., character-level, BPE)
      vocab_size: 2000            # Maximum vocabulary size
      tokenizer_path: "configs/vocab.json" # Path to save/load vocabulary mapping

    data:
      dataset_source: "huggingface" # Source platform for the training corpus
      dataset_name: "Daniele/dante-corpus" # Dataset identifier
      streaming: true             # Stream data to bypass local RAM constraints

    training:
      lr: 0.0015                  # Peak learning rate
      batch_size: 64              # Global batch size
      grad_accum: 4               # Gradient accumulation steps for large effective batches
      seed: 42                    # RNG seed for reproducibility
      optimizer: "adamw"          # Optimizer algorithm
      epochs: 100                 # Total training epochs
      warmup_steps: 0             # Number of steps for learning rate warmup

    generation:
      use_cache: true             # Enable static KV cache for fast autoregressive decoding
      top_p: null                 # Nucleus sampling threshold (null to disable)
      top_k: null                 # Top-k sampling threshold (null to disable)
      seed: 42                    # RNG seed for generation sampling
      greedy: false               # Toggle greedy decoding vs stochastic sampling
      max_generations: 150        # Maximum number of tokens to generate
      temperature: 1.3            # Sampling temperature (higher = more random)

    logging:
      eval_iters: 20              # Frequency of evaluation and metric logging
      log_file: "training_log.csv" # Path for training metrics output
      summary_file: "model_summary.json" # Path to dump architecture parameter summary
    ```

---


## Quickstart & Installation

```bash
git clone https://github.com/winstonsmith1897/DantinoX.git
cd DantinoX

# 1. Create and activate environment (Conda recommended)
conda create -n dantinox python=3.12 -y
conda activate dantinox

# 2. Install JAX with NVIDIA GPU support, then project dependencies
pip install -U "jax[cuda12]"
pip install -r requirements.txt
```

*(Note: For standard `venv`, use `python -m venv venv && source venv/bin/activate` instead).*

---
## 🔬 Deep Dive: JAX/Flax Implementation

DantinoX incorporates several advanced techniques designed to push the limits of modern LLM architecture. Below is a detailed breakdown of the core components, showcasing the actual JAX/Flax code used in the engine.

### 1. Grouped-Query Attention (GQA) & Rotary Positional Embeddings (RoPE)
The `Attention` module implements GQA to reduce KV-cache memory, alongside RoPE to inject absolute positional information into queries and keys via complex rotations.

**Grouped-Query Attention Projection:**
```python
# From core/model.py - Attention.__init__
self.kv_heads = config.kv_heads if config.kv_heads is not None else self.n_heads

# Single projection for Q, K, and V to optimize memory bandwidth
self.qkv = nnx.Linear(self.dim, 
                      self.dim + 2 * self.kv_heads * self.head_size,
                      use_bias=False, rngs=rngs)
```

**RoPE Frequencies Pre-computation:**
Instead of computing frequencies at every step, DantinoX caches the inverse frequencies matrix $\theta_i = 10000^{-2(i-1)/d}$ during initialization.
```python
def __compute_angle(T:int, C:int) -> jnp.ndarray:
    P = jnp.arange(T)
    W = 1 / (1000 ** (jnp.arange(C//2) / C))
    degree = jnp.einsum('i,j->ij', P, W)[None, None, None, :, :]
    return degree

self.angle: jnp.ndarray = __compute_angle(self.max_context, self.head_size)
```

**Applying the Rotation (Forward Pass):**
During the forward pass, the angles are dynamically sliced to match the current token index, and the rotation is applied mathematically.
```python
def __apply_rotation(self, x: jnp.ndarray, cache_index: int) -> jnp.ndarray:
    T = x.shape[3]
    odd  = x[:, :, :, :, 0::2]
    even = x[:, :, :, :, 1::2]

    angle = jax.lax.dynamic_slice_in_dim(self.angle, start_index=cache_index, slice_size=T, axis=3)
    
    x_odd  = jax.lax.cos(angle) * odd - jax.lax.sin(angle) * even
    x_even = jax.lax.sin(angle) * odd + jax.lax.cos(angle) * even

    return jnp.stack([x_even, x_odd], axis=-1).reshape(x.shape)
```

### 2. Static KV-Caching for XLA Compilation
JAX's XLA compiler requires static array shapes. Dynamic array appending (like `jnp.concatenate`) forces expensive recompilations. DantinoX solves this using `jax.lax.dynamic_update_slice`.

```python
# From core/model.py - Attention.__call__
if use_cache:
    if kv_cache == (None, None):  
        # 1. PREFILL: Pre-allocate the static cache with zeros up to max_context
        k_cache = jnp.zeros((B, self.kv_heads, 1, self.max_context, self.head_size), dtype=k.dtype)
        v_cache = jnp.zeros((B, self.kv_heads, 1, self.max_context, self.head_size), dtype=v.dtype)
        k_cache, v_cache = k_cache.at[:, :, :, :T, :].set(k), v_cache.at[:, :, :, :T, :].set(v)
    else:
        # 2. GENERATION: Surgically insert new tokens at the specific cache_index
        k_cache, v_cache = map(
            lambda x, y, index: jax.lax.dynamic_update_slice(x, y, (0, 0, 0, index, 0)), 
            (kv_cache[0], kv_cache[1]), (k, v), (cache_index, cache_index)
        )
```

### 3. Sliding Window & Attention Gating (`no_sink`)
To handle infinite generation and avoid memory degradation, DantinoX restricts the attention span and prevents the "Attention Sink" phenomenon.

**Sliding Window Mask Initialization:**
```python
if self.sliding_window:
    # Build a banded matrix where values outside the context window are masked
    table = jnp.arange(self.max_context)[:, None] - jnp.arange(self.max_context)[None, :]
    mask  = (table <= config.context_window) & (table >= 0)
    self.window = jnp.where(mask, 0, -1e9)
```

**Applying the Window and Gating in Forward Pass:**
```python
# 1. Apply Sliding Window Mask
if self.sliding_window:
    attn = attn + jax.lax.dynamic_slice_in_dim(operand=self.window,
                                               start_index=cache_index,
                                               slice_size=T,
                                               axis=0)

# Softmax and context projection ...
causal_attn = jax.nn.softmax(attn)
y = causal_attn @ v

# 2. Apply Attention Gating (no_sink)
if self.no_sink:
    # Modulate the attention output with a sigmoid projection of the original input
    y = y * jax.nn.sigmoid(self.W(x))
```

### 4. Sparse Mixture of Experts (MoE) & Load Balancing
The MLP layer can be dynamically replaced by a routed MoE architecture, using Top-K selection and an auxiliary loss to ensure expert utilization.

**Routing and Load Balancing Loss:**
```python
# From core/model.py - MoE.__call__
x_routed = self.router(x)
probs    = jax.nn.softmax(x_routed)
values, indices = jax.lax.top_k(probs, self.top_k_mlp)

# Normalize Top-K probabilities
values = values / jnp.sum(values, axis=-1, keepdims=True)

# Compute Load Balancing Loss
expert_mean_prob = jnp.mean(jnp.reshape(probs, (B*T, self.n_experts)), axis=0)
freq = jnp.mean(jnp.sum(jax.nn.one_hot(indices, self.n_experts), axis=2), axis=(0, 1))
moe_loss = jnp.sum(freq * expert_mean_prob) * self.n_experts
```

**Expert Computation:**
```python
y = jnp.zeros_like(x)
for i in range(self.n_experts):
    mask = (indices == i)
    # Mask out non-selected experts to save compute
    expert_weight = jnp.sum(jnp.where(mask, values, 0), axis=-1, keepdims=True)
    expert_out, _ = self.experts[i](x, deterministic=deterministic)
    y = y + (expert_weight * expert_out)
```

### 5. Gradient Checkpointing (Rematerialization)
To support massive batch sizes and deep networks, DantinoX wraps the Transformer blocks in `nnx.remat`. This discards intermediate activations in the forward pass and recomputes them during the backward pass.

```python
# From core/model.py - Transformer.__call__
def block_fn(block_module, hidden_state, kv_c, det):
    return block_module(hidden_state, use_cache=use_cache, kv_cache=kv_c, 
                        cache_index=cache_index, deterministic=det)

# Rematerialize only if gradient checkpointing is ON and we are NOT in inference mode
if self.gradient_checkpointing and not use_cache:
    checkpointed_block = nnx.remat(lambda bm, hs, kvc: block_fn(bm, hs, kvc, deterministic))
else:
    checkpointed_block = lambda bm, hs, kvc: block_fn(bm, hs, kvc, deterministic)
```