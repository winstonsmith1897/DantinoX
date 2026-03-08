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

## Installation

```bash
# Clone the repository
git clone [https://github.com/winstonsmith1897/DantinoX.git](https://github.com/winstonsmith1897/DantinoX.git)
cd DantinoX

# Install all necessary dependencies
pip install -r requirements.txt

## Configuration

DantinoX uses a centralized YAML configuration system. You can fine-tune the architecture and training parameters in `configs/defaults_config.yaml`:

```yaml
# Model Architecture
model:
  dim: 128
  n_heads: 16
  head_size: 8
  num_blocks: 4
  max_context: 256
  
# Mixture of Experts (MoE)
moe:
  use_moe: true
  n_experts: 4            # Total number of experts
  top_k_mlp: 2            # Experts activated per token
  expansion: 4            # MLP dimension multiplier

# Training & Optimization
training:
  lr: 0.0005
  batch_size: 32
  grad_accum: 4           # Gradient accumulation steps
  steps: 5000
  optimizer: "adamw"      # Supports: adamw, adafactor, lion, etc.

# Logging
logging:
  log_file: "training_log.csv"
  summary_file: "model_summary.json"
```

## Usage

### 1. Training
The training script supports both local text files and Hugging Face datasets. It automatically handles the tokenizer initialization, train/validation split, and logs performance metrics.

```bash
# Training on a local text file
python train.py --data_path data/dante.txt

# Overriding configuration parameters via CLI
python train.py --data_path data/math.txt --batch_size 64 --optimizer adafactor
```

During training, metrics are printed every 50 steps and saved to `training_log.csv`.

### 2. Inference & Text Generation
Use the `generate.py` script to test your trained model. It utilizes a persistent **KV Cache** to ensure high-speed, constant-time token generation.

```bash
python generate.py --prompt "Nel mezzo del cammin " --max_new_tokens 50 --greedy
```

## Metrics and Monitoring

DantinoX is built for transparency. Every run generates detailed logs for later analysis and plotting:

* **Model Summary (`model_summary.json`)**: Exported at startup. It contains the total parameter count, estimated VRAM for weights, optimizer states, and peak activation memory.
* **Training Logs (`training_log.csv`)**: Real-time logging of:
    * `train_loss` / `val_loss`: Cross-entropy metrics.
    * `vram_gb`: Actual GPU/TPU memory usage.
    * `ms_per_step`: Latency per training step.

## Architecture Deep Dive

### Sub-Nano Strategy
While "Nano" models usually refer to architectures with 50M+ parameters, **DantinoX** targets the **Sub-Nano** regime (<20M active parameters). By utilizing a **Mixture of Experts (MoE)**, we keep the computational cost per token extremely low while maintaining the representational power of a larger dense model.

### Efficient Attention
By combining **Sliding Window Attention (SWA)** with **Rotary Positional Embeddings (RoPE)**, DantinoX avoids the memory bottleneck of global self-attention. This allows the model to handle sequences effectively without the quadratic memory growth typical of standard Transformers.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
**DantinoX** - *Small in size, Divine in Architecture.*
Created by [winstonsmith1897](https://github.com/winstonsmith1897)