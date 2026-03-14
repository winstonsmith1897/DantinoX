::: {align="center"}
# 𝔇𝔞𝔫𝔱𝔦𝔫𝔬𝔛

``` text
██████╗  █████╗ ███╗   ██╗████████╗██╗███╗   ██╗ ██████╗ ██╗  ██╗
██╔══██╗██╔══██╗████╗  ██║╚══██╔══╝██║████╗  ██║██╔═══██╗╚██╗██╔╝
██║  ██║███████║██╔██╗ ██║   ██║   ██║██╔██╗ ██║██║   ██║ ╚███╔╝
██║  ██║██╔══██║██║╚██╗██║   ██║   ██║██║╚██╗██║██║   ██║ ██╔██╗
██████╔╝██║  ██║██║ ╚████║   ██║   ██║██║ ╚████║╚██████╔╝██╔╝ ██╗
╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝  ╚═╝
```

`<i>`{=html}"Ah JAX, vituperio delle genti..."`</i>`{=html}\
`<b>`{=html}(Ah JAX, the shame of the people...)`</b>`{=html}

`<br>`{=html}

A Transformer so **"nano" it barely rhymes**, implemented in **JAX** and
**Flax NNX**.

Built with equal parts **sweat, tears, and XLA compilation errors**.

`<br>`{=html}

[![JAX](https://img.shields.io/badge/JAX-000000?style=for-the-badge&logo=JAX&logoColor=white)](https://github.com/google/jax)
[![Flax
NNX](https://img.shields.io/badge/Flax_NNX-8A2BE2?style=for-the-badge&logo=flax&logoColor=white)](https://github.com/google/flax)
[![Python
3.12+](https://img.shields.io/badge/Python_3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![NVIDIA
GPU](https://img.shields.io/badge/Hardware-NVIDIA_GPU-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/)
[![License:
MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
:::

`<br>`{=html}

![DantinoX Architecture](images/dantinox.png)

---

# Overview: The True Story

Let's be honest: the goal of this project is not to achieve AGI or challenge the Silicon Valley giants. The single, true, desperate purpose of **DantinoX** was just one: **learning how to use JAX without ending up in hell.**

And what better guide than Dante Alighieri to navigate the "dark wood" (*selva oscura*) of `TypeError`s, failed tensor broadcasting, and XLA compilation crashes?

Despite the very humble approach (and the countless hours spent staring at matrix dimensions hoping they would magically align), I decided to get my hands dirty and implement all the trendiest buzzwords in modern Deep Learning from scratch, just to understand how they actually work under the hood:

* **Mixture of Experts (MoE) layers:** Because why have a single, confused Multi-Layer Perceptron when you can have four of them bouncing the responsibility around?
* **Rotary Positional Embeddings (RoPE):** I applied complex rotations to tensors until my own head started spinning.
* **Sliding Window Attention:** To lighten the memory load and only remind the model of its most recent past.
* **Static KV Cache:** Because recalculating the entire universe for every single generated letter seemed a bit excessive.

The final result? A fully functional LLM architecture that is incredibly fast on GPUs and has a memory footprint so small it won't melt your computer. 

Does it always produce pure divine poetry? Let's not exaggerate. But it gets by, and above all, it taught me how to tame the XLA compiler. 

*"And thence we came forth to see again the stars..."*

------------------------------------------------------------------------

# 🏗️ Project Structure

The **DantinoX repository** follows a modular design separating:

-   model architecture
-   utilities
-   data handling
-   training scripts

This structure is optimized for **JAX / Flax NNX workflows**.

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

------------------------------------------------------------------------

# 🧠 Model Architecture

DantinoX implements a **modern Decoder‑only Transformer** optimized for
**JAX / Flax NNX**.

The architecture is **highly configurable** and supports both:

-   **Dense FFN layers**
-   **Sparse Mixture of Experts (MoE)**

------------------------------------------------------------------------

# Core Components

## 1. Hybrid Attention Mechanism (`Attention` class)

The Attention module implements **causal self‑attention** with multiple
advanced capabilities.

### Key Features

-   **Configurable Heads**
    -   Separate `n_heads` (queries)
    -   Separate `kv_heads` (keys/values)
    -   Enables **Grouped Query Attention (GQA)**
-   **Rotary Positional Embeddings (RoPE)**
    -   Implemented through `__apply_rotation`
    -   Enables **relative positional reasoning**
-   **Causal Masking**
    -   Static triangular mask (`self.tril`)
    -   Dynamic slicing based on `cache_index`
-   **Sliding Window Attention**
    -   Enabled via `sliding_window=True`
    -   Restricts attention to `context_window` tokens
-   **Static KV Cache**
    -   Used when `use_cache=True`
    -   Key and Value states stored in:

```{=html}
<!-- -->
```
    k_cache
    v_cache

-   **Attention Gating ("no_sink")**
    -   Optional sigmoid gate (`self.W`)
    -   Reweights attention output for stability

------------------------------------------------------------------------

## 2. Mixture of Experts (MoE) & MLP

The Feed Forward Network can be configured in **two modes**.

### Standard MLP

A traditional two‑layer feedforward network:

-   Linear → Activation → Linear
-   Activation: **GELU (default)**
-   Dropout regularization

### Sparse MoE

Replaces the dense MLP with **expert networks**.

#### Router

A linear layer computes expert probabilities.

#### Top‑K Selection

Only the **Top‑K experts** are activated per token.

Example:

    top_k_mlp = 2

#### Load Balancing Loss

The module computes an auxiliary loss:

    moe_loss

This encourages **balanced expert usage** and prevents **expert
collapse**.

------------------------------------------------------------------------

## 3. Transformer Block (`Block` class)

The main building unit of the model.

Each block contains:

-   **Pre‑LayerNorm**
-   **Attention layer**
-   **MLP / MoE layer**
-   **Residual connections**

### Training Optimization

Supports **gradient checkpointing** using:

    nnx.remat

This recomputes activations during the backward pass to **reduce VRAM
usage**.

------------------------------------------------------------------------

## 4. Full Transformer (`Transformer` class)

The complete model stack.

### Components

-   **Token Embedding Layer** (`wte`)
-   **Stack of Transformer Blocks**
-   **Output LM Head** (`lm_head`)

### Weight Tying

The LM head can reuse the embedding matrix:

    self.lm_head.kernel = self.wte.embedding.T

Benefits:

-   fewer parameters
-   smaller checkpoints
-   reduced VRAM usage

### Positional Encoding Options

Supports:

-   **RoPE**
-   **Absolute sinusoidal encoding**
-   **Trainable positional embeddings**

### Regularization

Includes configurable dropout for:

-   embeddings
-   attention weights
-   residual connections

------------------------------------------------------------------------

# Technical Summary

  Feature               Implementation
  --------------------- ----------------------------------------
  Model Type            Decoder‑only Transformer
  Framework             JAX / Flax NNX
  Normalization         LayerNorm (Pre‑Norm)
  Activation            GELU
  Positioning           RoPE or Absolute
  Inference             Autoregressive with KV Cache
  FFN Types             Dense MLP or Sparse MoE
  MoE Balance           Auxiliary balancing loss
  Regularization        Attention, residual, embedding dropout
  Memory Optimization   Gradient checkpointing, weight tying
  Distributed           JAX SPMD (Data / Model / FSDP)

------------------------------------------------------------------------

# ⚙️ Configuration (`config.yaml`)

DantinoX uses **YAML configuration files** to define:

-   model architecture
-   training parameters
-   optimization strategy

This guarantees **reproducible experiments**.

## Sample Configuration

``` yaml
# Model Architecture
dim: 512
n_heads: 8
n_experts: 4
top_k_mlp: 2
num_blocks: 6
max_context: 512
vocab_size: 2000

# Positional Encoding
use_rotary_pos: true
sliding_window: false
weight_tying: true
use_moe: true

# Training
batch_size: 32
grad_accum: 4
lr: 0.0003
dropout_rate: 0.1
epochs: 10
optimizer: "adamw"

# System
gradient_checkpointing: true
alpha_balance: 0.01
```

------------------------------------------------------------------------

# Parameter Breakdown

## Architecture

### `dim`

Hidden dimension size.

Increasing this improves model capacity but also increases **VRAM
usage**.

### `n_heads`

Number of attention heads.

Allows the model to attend to multiple **representation subspaces**
simultaneously.

### `use_moe`

If enabled, replaces the dense FFN with a **Sparse MoE layer**.

Benefits:

-   higher parameter count
-   same FLOPs per token

### `top_k_mlp`

Number of experts activated per token.

Typical values:

    1 or 2

------------------------------------------------------------------------

## Optimization

### `grad_accum`

Gradient accumulation steps.

Useful when **GPU memory is limited**.

Example:

    effective_batch = batch_size × grad_accum

### `lr`

Peak learning rate.

Training uses:

-   **Cosine decay schedule**
-   **10% warmup phase**

### `alpha_balance`

Coefficient for **MoE load balancing loss**.

Helps prevent **expert collapse**.

------------------------------------------------------------------------

## Efficiency

### `weight_tying`

Reuses the embedding matrix for the LM head.

Advantages:

-   smaller model size
-   reduced VRAM
-   fewer parameters

### `gradient_checkpointing`

Trades extra compute for **lower memory usage** by recomputing
activations during backpropagation.

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
