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