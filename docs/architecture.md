
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