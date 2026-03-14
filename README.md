# DantinoX 

A high-performance, sub-nano Mixture of Experts (MoE) Transformer implemented in **JAX** and **Flax NNX**. Optimized for efficiency and speed on GPU/TPU devices.



![DantinoX Architecture](images/dantinox.png)

## Overview

**DantinoX** is an efficient, lightweight Transformer model designed to push the boundaries of "Nano" scale architectures. Built from the ground up using **JAX** and **Flax NNX**, it implements advanced modern features like a powerful Mixture of Experts (MoE) layer, Rotary Positional Embeddings (RoPE), and Sliding Window Attention. DantinoX aims to offer a complete, functional, and performant LLM base within a minimal parameter and memory footprint.

## Key Features

* **JAX-Native & Flax NNX**: Leverage the full power of JAX's JIT compilation and hardware acceleration with the new, state-managed Flax NNX API.
* **Mixture of Experts (MoE)**: Efficient gated MLP layer. Replaces the standard MLP with multiple expert networks, routing only the top-k experts per token for faster inference.
* **Rotary Positional Embeddings (RoPE)**: Implements state-of-the-art relative positional encoding for improved performance on long sequences.
* **Sliding Window Attention**: Optimizes memory and compute during attention by attending only to a fixed-size local window of tokens.
* **Sub-Nano Scale**: Meticulously designed to minimize parameter count and memory usage, making it faster to train and run than traditional "Nano" models.
* **KV Cache & Greedy Generation**: High-performance inference with a persistent Key-Value cache and a `generate.py` script featuring throughput metrics.

# 🏗️ Project Structure

The **DantinoX** repository is organized into a modular architecture, separating the model's core logic, data handling, and execution scripts. This structure is designed to leverage the power of **JAX/Flax NNX** for efficient training and inference.

```text
DantinoX/
├── core/                   # Core neural network logic
│   ├── config.py           # Configuration parameters (Config Dataclass)
│   ├── model.py            # Transformer architecture (Attention, MLP, MoE, Block)
│   ├── generation.py       # Inference engine & static KV-Cache management
│   └── __init__.py
├── configs/                # YAML configuration files
│   ├── default_config.yaml # Standard training setup
│   └── sweep.yaml          # Hyperparameter search config (W&B)
├── utils/                  # Utility functions
│   ├── tokenizer.py        # Tokenizer management (Char-level & Byte-Level BPE)
│   ├── helpers.py          # Loss functions, Batching, and Sharding logic
│   └── __init__.py
├── runs/                   # Training outputs (Weights, logs, saved configs)
├── analyze_dataset.py      # Statistical analysis script for the corpus
├── train.py                # Main training execution script
├── generate.py             # Autoregressive text generation script
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

## 🧠 Model Architecture

DantinoX implements a modern **Decoder-only Transformer** optimized for JAX/Flax NNX. It is designed to be highly configurable, supporting both dense and sparse (MoE) configurations.



### Core Components

#### 1. Hybrid Attention Mechanism (`Attention` class)
The Attention module implements a standard causal self-attention mechanism with several advanced features:
* **Configurable Heads:** Separates `n_heads` (for Queries) from `kv_heads` (for Keys and Values) to support architectures like Grouped Query Attention (GQA).
* **Rotary Positional Embeddings (RoPE):** Integrated via the `__apply_rotation` method, allowing for better handling of relative positions and context length extrapolation. 
* **Causal Masking:** Enforced by a static triangular mask (`self.tril`) combined with dynamic slicing based on the current `cache_index`.
* **Sliding Window Attention:** If enabled (`sliding_window=True`), attention is restricted to a fixed-size window (`context_window`) around the current token, reducing computational complexity.
* **Static KV-Cache:** During inference (`use_cache=True`), Key and Value states are stored and dynamically updated in pre-allocated buffers (`k_cache`, `v_cache`), significantly speeding up generation.
* **Attention Gating ("no_sink"):** An optional feature that uses a sigmoid gate (`self.W`) to re-weight the output, potentially helping with stability or initial token attention.

#### 2. Mixture of Experts (MoE) & MLP
The Feed-Forward Network (FFN) can be either a standard MLP or a Sparse MoE layer:
* **MLP:** A standard two-layer linear network with a configurable activation function (defaulting to GELU) and dropout.
* **MoE:** Replaces the dense MLP with a sparse layer.
    * **Routing:** A `router` linear layer computes probabilities for each expert.
    * **Top-K Selection:** Only the `top_k_mlp` experts with the highest probabilities are activated for each token.
    * **Load Balancing Loss:** The module calculates a dedicated loss term (`moe_loss`) based on expert selection frequency and probabilities to ensure all experts are trained evenly and prevent expert collapse. 

#### 3. Transformer Block (`Block` class)
The fundamental building block of the model, which includes:
* **Pre-Layer Normalization:** Normalization (`self.ln1`, `self.ln2`) is applied *before* the Attention and FFN layers for more stable training.
* **Residual Connections:** The outputs of the Attention and FFN layers are added back to the input, facilitating gradient flow in deep networks.
* **Gradient Checkpointing:** When enabled via `nnx.remat`, block activations are recomputed during the backward pass instead of being stored, saving significant VRAM at the cost of some compute.

#### 4. Full Transformer (`Transformer` class)
The complete stack of `num_blocks` layers, including:
* **Embedding Layer (`wte`):** Maps input token IDs to dense vectors.
* **Output Head (`lm_head`):** Maps the final hidden states back to vocabulary logits.
* **Weight Tying:** The kernel of `lm_head` can share weights with the `wte` embedding matrix (`self.lm_head.kernel = self.wte.embedding.T`), reducing the model's total parameter count and memory footprint.
* **Flexible Positional Encodings:** Supports standard RoPE and/or alternative absolute (Fixed Sine/Cosine or Trainable) positional encodings via `wpe`.
* **Dropout:** Includes customizable dropout for embeddings, attention attention weights, and residual paths to prevent overfitting. 

### Technical Summary

| Feature | Implementation |
| :--- | :--- |
| **Model Type** | Decoder-only Transformer |
| **Framework** | JAX / Flax NNX |
| **Normalization** | LayerNorm (Pre-Norm config) |
| **Activation** | GELU (configurable) |
| **Positioning** | RoPE, Absolute (fixed/trainable) |
| **Inference** | Autoregressive with Static KV-Cache |
| **FFN Types** | Dense MLP or Sparse Top-K MoE |
| **MoE Balance** | Dedicated auxiliary balancing loss |
| **Regularization**| Attn, Resid, Embed Dropout; Weight Decay |
| **Memory Opt.** | Gradient Checkpointing (`remat`), Weight Tying |
| **Distributed** | JAX SPMD sharding (Data/Model/FSDP Parallel) |

## ⚙️ Configuration (`config.yaml`)

DantinoX uses YAML files to define the model architecture and training hyperparameters. This ensures experiments are reproducible and easy to modify without touching the source code.

### Sample Configuration

Below is a typical configuration for a medium-sized model:

```yaml
# Model Architecture
dim: 512                # Hidden dimension size
n_heads: 8              # Number of attention heads
n_experts: 4            # Total number of experts (for MoE)
top_k_mlp: 2            # Activated experts per token (for MoE)
num_blocks: 6           # Number of Transformer layers
max_context: 512        # Maximum sequence length
vocab_size: 2000        # Vocabulary size (auto-updated by tokenizer)

# Positional Encoding & Features
use_rotary_pos: true    # Enable Rotary Positional Embeddings (RoPE)
sliding_window: false   # Enable sliding window attention
weight_tying: true      # Share weights between embedding and output head
use_moe: true           # Enable Mixture of Experts instead of dense MLP

# Training Hyperparameters
batch_size: 32          # Global batch size
grad_accum: 4           # Gradient accumulation steps
lr: 0.0003              # Peak learning rate
dropout_rate: 0.1       # Dropout probability
epochs: 10              # Number of training epochs
optimizer: "adamw"      # Optimizer type (adamw or adam)

# System & Checkpointing
gradient_checkpointing: true # Save VRAM by recomputing activations
alpha_balance: 0.01          # Coefficient for MoE balancing loss

### 🔍 Parameter Breakdown

#### 🏗️ Architecture
* **`dim`**: The "width" of the model. Increasing this improves the model's capacity to represent complex patterns but significantly raises VRAM usage.
* **`n_heads`**: Number of parallel attention mechanisms. It allows the model to simultaneously attend to information from different representation subspaces.
* **`use_moe`**: If `true`, the model replaces the standard Feed-Forward Network (FFN) with a **Sparse Mixture of Experts**. This allows for a massive increase in total parameters without increasing the computational cost per token (FLOPs).
* **`top_k_mlp`**: Only used if `use_moe` is active. It defines how many experts are "voted" for by the router for each individual token. Common values are 1 or 2.

#### 📈 Optimization
* **`grad_accum`**: Gradient Accumulation steps. Used to simulate larger batch sizes. If your GPU has limited VRAM, you can decrease `batch_size` and increase `grad_accum` to maintain training stability without crashing.
* **`lr`**: The peak learning rate. The model follows a **Cosine Decay Schedule** with an initial 10% warmup period by default to prevent gradient instability at the start of training.
* **`alpha_balance`**: Specifically for MoE. It controls the penalty for **"Expert Collapse"** (a situation where the router only learns to use one expert, leaving others untrained).

#### 🛠️ Efficiency
* **`weight_tying`**: Significantly reduces the `.msgpack` file size and VRAM usage by reusing the token embedding matrix for the final output predictions (LM Head).
* **`gradient_checkpointing`**: Essential for training deep models on consumer-grade GPUs. It trades a bit of computation time (recomputing activations during the backward pass) for massive memory efficiency.

## 🚀 Installation

Follow these steps to set up the environment and get **DantinoX** running on your local machine or server.

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/DantinoX.git](https://github.com/your-username/DantinoX.git)
cd DantinoX

### 2. Create a Virtual Environment
It is recommended to use `conda` or `venv` to manage dependencies:

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Or using Conda
conda create -n dantinox python=3.12
conda activate dantinox

### 3. Install Dependencies
DantinoX is powered by **JAX**. Depending on your hardware (CPU vs GPU), choose the appropriate installation command:

**For NVIDIA GPU (Highly Recommended):**
```bash
pip install --upgrade "jax[cuda12]"
pip install -r requirements.txt


## 🚄 Training

The training pipeline is optimized using **JAX/Flax NNX**, featuring functional state management and JIT compilation for maximum hardware utilization.

### 1. Basic Usage
To start a training run using the default configuration:
```bash
python train.py --config configs/default_config.yaml

You can also override any configuration parameter directly from the command line:
```bash
python train.py --batch_size 64 --lr 5e-4 --use_moe True

### 2. Training Features
* **JIT-Compiled Steps**: The training loop utilizes `@jax.jit` for the core update step, ensuring that the model logic, optimizer updates, and metric calculations are fused into a single optimized XLA kernel.
* **Gradient Accumulation**: Supports large effective batch sizes on limited VRAM by splitting the global batch into multiple micro-batches via `grad_accum`.
* **Mixture of Experts (MoE) Balancing**: Automatically monitors and applies an auxiliary `balancing_loss` to ensure even expert utilization and prevent routing collapse.
* **Automatic Positional Formatting**: The script pre-processes text (specifically optimized for the Divine Comedy) into formatted triplets before tokenization to preserve poetic structure.

### 3. Monitoring & Logging
Each training session creates a unique directory in `runs/run_YYYYMMDD_HHMMSS/` containing:
* `config.yaml`: A snapshot of the parameters used.
* `model_summary.json`: Estimated VRAM usage and parameter counts.
* `training_log.csv`: Real-time metrics updated every 50 steps.
* `model_weights.msgpack`: The final trained weights in a highly compressed format.

**Console Output Example:**
```text
Step    50/4200 | Train: 4.1204 (Bal: 0.0452) | Val: 4.1560 (Bal: 0.0461) | VRAM: 3.42GB
Step   100/4200 | Train: 3.8901 (Bal: 0.0421) | Val: 3.9102 (Bal: 0.0415) | VRAM: 3.42GB

### 4. Metrics Tracked

| Metric | Description |
| :--- | :--- |
| **Train/Val Loss** | Standard Cross-Entropy loss for next-token prediction. |
| **Balancing Loss** | Auxiliary loss ensuring load balancing across MoE experts. |
| **VRAM GB** | Real-time GPU memory consumption. |
| **ms_per_step** | Temporal efficiency of the training loop. |