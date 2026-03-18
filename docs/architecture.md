
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

DantinoX incorporates several advanced techniques designed to push the limits of modern LLM architecture. Below is a detailed breakdown of the core components and the mathematical/design philosophy behind them.

### 1. Grouped-Query Attention (GQA) & RoPE
The `Attention` module implements Grouped-Query Attention to drastically reduce the KV-cache size during inference, alongside Rotary Positional Embeddings (RoPE) for robust context extrapolation.

```python
# From core/model.py - Attention.__init__
self.kv_heads = config.kv_heads if config.kv_heads is not None else self.n_heads
self.qkv = nnx.Linear(self.dim, 
                      self.dim + 2 * self.kv_heads * self.head_size,
                      use_bias=False, rngs=rngs)
```

**Why it matters:** Instead of allocating `n_heads` for Keys and Values, DantinoX allocates a smaller `kv_heads` count. The `qkv` projection elegantly handles this by projecting the query to full dimensionality, while shrinking the K and V projections to `kv_heads * head_size`.

For positional awareness, **RoPE** rotates the Query and Key vectors in the complex plane. The frequencies are pre-computed to avoid redundant calculations, implementing the rotary frequency matrix where the angle for position $m$ and feature dimension $i$ is derived from $\theta_i = 10000^{-2(i-1)/d}$.

### 2. Static KV-Caching (The JAX Way)
JAX's XLA compiler requires static array shapes. Dynamic appending forces recompilation at every generation step, destroying performance. DantinoX solves this using `jax.lax.dynamic_update_slice`.

```python
if use_cache:
    if kv_cache == (None, None):  
        # 1. Pre-allocate the maximum possible cache size with zeros
        k_cache = jnp.zeros((B, self.kv_heads, 1, self.max_context, self.head_size), dtype=k.dtype)
        v_cache = jnp.zeros((B, self.kv_heads, 1, self.max_context, self.head_size), dtype=v.dtype)
        k_cache, v_cache = k_cache.at[:, :, :, :T, :].set(k), v_cache.at[:, :, :, :T, :].set(v)
    else:
        # 2. Surgically insert the new token's K and V at the specific cache_index
        k_cache, v_cache = map(
            lambda x, y, index: jax.lax.dynamic_update_slice(x, y, (0, 0, 0, index, 0)), 
            (kv_cache[0], kv_cache[1]), (k, v), (cache_index, cache_index)
        )
```

**How it works:** During the first forward pass (prefill), we initialize a tensor of size `max_context`. In subsequent decoding steps, we update only the slice at `cache_index`. This keeps the memory footprint bounded and XLA highly optimized.

### 3. Sliding Window & Attention Gating (`no_sink`)
To handle infinite generation without memory degradation, DantinoX implements a Sliding Window mask and a custom Attention Gating mechanism.

* **Sliding Window:** Adds a pre-computed banded matrix of `-1e9` to the attention scores, restricting the softmax to attend only to the last `context_window` tokens, reducing complexity from $O(T^2)$ to $O(T \times W)$.
* **Attention Gating (`no_sink`):** The phenomenon of "Attention Sinks" is mitigated here. We apply a learned sigmoid gate $\sigma(W \cdot X)$ to the attention output $Y$, allowing the model to smoothly ignore irrelevant context:
  $$ \text{Output} = Y \odot \sigma(W \cdot X) $$

### 4. Sparse Mixture of Experts (MoE) & Load Balancing
Instead of a monolithic Feed-Forward Network, DantinoX supports a routed MoE architecture to scale parameters without proportionally increasing active compute.

```python
# From core/model.py - MoE.__call__
x_routed = self.router(x)
probs    = jax.nn.softmax(x_routed)
values, indices = jax.lax.top_k(probs, self.top_k_mlp)
```

**The mechanism:**
1. A linear router predicts which experts are best for each specific token.
2. `jax.lax.top_k` selects only the top $k$ experts, masking out the rest.
3. A **Load Balancing Loss** is computed. Without this, the network tends to collapse, routing all tokens to expert 0. The loss forces uniform utilization across the expert pool:
   $$ L_{balance} = N \sum_{i=1}^{N} f_i \cdot P_i $$
   *(Where $N$ is the total number of experts, $f_i$ is the empirical token assignment frequency, and $P_i$ is the mean routing probability).*

### 5. Gradient Checkpointing (Rematerialization)
Training LLMs requires massive VRAM. DantinoX uses Flax's `nnx.remat` to trade a small amount of compute for a massive reduction in memory.

```python
if self.gradient_checkpointing and not use_cache:
    checkpointed_block = nnx.remat(lambda bm, hs, kvc: block_fn(bm, hs, kvc, deterministic))
```

When `gradient_checkpointing` is enabled, the intermediate activations inside the Transformer blocks are discarded after the forward pass. During the backward pass, JAX automatically recomputes them on-the-fly. This enables training much deeper models or using larger batch sizes on a single GPU.