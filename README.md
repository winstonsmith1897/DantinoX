<div align="center">

# 𝔇𝔞𝔫𝔱𝔦𝔫𝔬𝔛

<i>"Ah JAX, vituperio delle genti..."</i>  
<b>(Ah JAX, the shame of the people...)</b>

<br>

A Transformer so **"nano" it barely rhymes**, implemented in **JAX** and **Flax NNX**. Built with **sweat** and **XLA compilation errors**.


<br>

[![JAX](https://img.shields.io/badge/JAX-000000?style=for-the-badge&logo=JAX&logoColor=white)](https://github.com/google/jax)
[![Flax NNX](https://img.shields.io/badge/Flax_NNX-8A2BE2?style=for-the-badge&logo=flax&logoColor=white)](https://github.com/google/flax)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

</div>

<br>

<p align="center">
  <img src="images/dantinox.png" alt="DantinoX Architecture">
</p>

------------------------------------------------------------------------

# 🏛️ Overview: The DantinoX Project

> *"Nel mezzo del cammin di nostra vita / mi ritrovai per una selva oscura, / ché la diritta via era smarrita."*

**DantinoX** is a from-scratch implementation of a modern Large Language Model built natively in **JAX and Flax NNX**. The primary motivation behind this project is educational and exploratory: to understand the internal mechanics of current transformer architectures and to learn how to write efficient JAX code without constantly fighting XLA compilation errors.

To thoroughly understand these constraints, DantinoX implements standard modern Deep Learning components directly from the ground up:

* **Sparse Mixture of Experts (MoE)** with **Load Balancing Loss**
* **Rotary Positional Embeddings (RoPE)**
* **Grouped Query Attention (GQA)**
* **Sliding Window & Attention Gating**
* **Static KV Cache**
* **Weight Tying**
* **Gradient Checkpointing**


### ⚙️ Highly Customizable

Rather than a rigid production artifact, the codebase is designed to be **highly customizable**. The architecture is modular, allowing users to easily toggle between different configurations—such as switching between a standard Dense MLP and Sparse MoE routing—to observe the direct impact on compute requirements and VRAM usage.

The final result is a functional, memory-efficient Transformer. It serves as a practical reference for resolving shape mismatches, managing GPU memory footprint, and successfully taming the XLA compiler.

> *"E quindi uscimmo a riveder le stelle."*

------------------------------------------------------------------------

# Project Structure


    DantinoX/
    ├── core/                   # Core neural network logic
    │   ├── config.py           # Configuration parameters (Config Dataclass)
    │   ├── model.py            # Transformer architecture (Attention, MLP, MoE, Block)
    │   ├── generation.py       # Inference engine & static KV-Cache management
    │   └── __init__.py
    │
    ├── configs/                # YAML configuration files
    │   ├── default_config.yaml # Standard training setup
    │   └── sweep.yaml          # Hyperparameter search config (W&B)
    │
    ├── utils/                  # Utility functions
    │   ├── tokenizer.py        # Tokenizer management (Char-level & Byte-Level BPE)
    │   ├── helpers.py          # Loss functions, batching, sharding logic
    │   └── __init__.py
    │
    ├── runs/                   # Training outputs (weights, logs, saved configs)
    │
    ├── analyze_dataset.py      # Dataset statistical analysis
    ├── train.py                # Training script
    ├── generate.py             # Text generation script
    ├── requirements.txt        # Python dependencies
    └── README.md               # Documentation

## 🛠 Architecture & Technical Specs


| Feature | Implementation Details |
| :--- | :--- |
| **Attention** | Causal Self-Attention with GQA and optional Sliding Window and gating `no_sink`|
| **Feed-Forward** | Configurable: Dense MLP or Sparse MoE (Top-K Routing) |
| **Positioning** | Rotary Positional Embeddings (RoPE) or Absolute |
| **Memory Opt.** | Gradient checkpointing (`nnx.remat`) & Weight Tying (`lm_head.kernel = wte.embedding.T`)|
| **Inference Opt.**| Autoregressive generation with Static KV-Cache |
| **Regularization**| Attention, residual, and embedding dropout; auxiliary MoE balancing loss `load_balancing_loss` |
| **Distributed** | JAX SPMD (Data / Model / FSDP) - *Future Work* |


## ⚙️ Configuration Reference

DantinoX is entirely driven by a centralized YAML configuration. This design allows you to easily ablate architectural components (like toggling MoE or sliding window attention) without modifying the core JAX codebase.

Below is the annotated `default_config.yaml`:

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
  
------------------------------------------------------------------------

# 🚀 Installation

## 1. Clone the Repository

``` bash
git clone https://github.com/your-username/DantinoX.git
cd DantinoX
```

------------------------------------------------------------------------

## 2. Create Virtual Environment

### Using venv

``` bash
python -m venv venv
source venv/bin/activate
```

Windows:

``` bash
venv\Scripts\activate
```

### Using Conda

``` bash
conda create -n dantinox python=3.12
conda activate dantinox
```

------------------------------------------------------------------------

## 3. Install Dependencies

DantinoX relies on **JAX**.

### NVIDIA GPU (recommended)

``` bash
pip install --upgrade "jax[cuda12]"
pip install -r requirements.txt
```

------------------------------------------------------------------------

# 🚄 Training

The training pipeline is optimized with **JAX / Flax NNX**, using:

-   functional state management
-   JIT compilation
-   efficient hardware utilization

------------------------------------------------------------------------

## Basic Usage

Start training using the default configuration:

``` bash
python train.py --config configs/default_config.yaml
```

Override parameters from CLI:

``` bash
python train.py --batch_size 64 --lr 5e-4 --use_moe True
```

------------------------------------------------------------------------

# Training Features

### JIT‑Compiled Training Step

The core update step uses:

    @jax.jit

This fuses:

-   model forward pass
-   loss computation
-   optimizer update

into a single optimized **XLA kernel**.

### Gradient Accumulation

Allows large **effective batch sizes** with limited VRAM.

### MoE Balancing

Automatically applies **balancing loss** to ensure **uniform expert
utilization**.

### Dataset Formatting

Text is preprocessed into structured triplets optimized for **Divine
Comedy training**.

------------------------------------------------------------------------

# Monitoring & Logging

Each run creates a directory:

    runs/run_YYYYMMDD_HHMMSS/

Containing:

-   `config.yaml`
-   `model_summary.json`
-   `training_log.csv`
-   `model_weights.msgpack`

------------------------------------------------------------------------

# Console Output Example

    Step    50/4200 | Train: 4.1204 (Bal: 0.0452) | Val: 4.1560 (Bal: 0.0461) | VRAM: 3.42GB
    Step   100/4200 | Train: 3.8901 (Bal: 0.0421) | Val: 3.9102 (Bal: 0.0415) | VRAM: 3.42GB

------------------------------------------------------------------------

# Metrics Tracked

  Metric            Description
  ----------------- -------------------------------------
  Train Loss        Cross‑Entropy next‑token prediction
  Validation Loss   Validation Cross‑Entropy
  Balancing Loss    MoE expert balancing
  VRAM GB           GPU memory usage
  ms_per_step       Training speed
