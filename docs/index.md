<div align="center">

# DantinoX

*"E quindi uscimmo a riveder le stelle."*

A from-scratch Large Language Model built natively in **JAX** and **Flax NNX**.

[![JAX](https://img.shields.io/badge/JAX-000000?style=for-the-badge&logo=JAX&logoColor=white)](https://github.com/google/jax)
[![Flax NNX](https://img.shields.io/badge/Flax_NNX-8A2BE2?style=for-the-badge&logo=flax&logoColor=white)](https://github.com/google/flax)
[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

</div>

<p align="center">
  <img src="images/dantinox.png" alt="DantinoX logo" width="180">
</p>

---

## Overview

**DantinoX** is a fully self-contained implementation of a modern Transformer, built without framework shortcuts. The primary goal is educational: to understand the internal mechanics of current LLM architectures and to write efficient JAX code that plays well with the XLA compiler.

Every component is implemented from first principles, then validated empirically through large-scale hyperparameter sweeps logged to **Weights & Biases**.

### Key Features

| Component | Implementation |
| :--- | :--- |
| **Attention** | MHA · GQA · Multi-Head Latent Attention (MLA) with weight absorption |
| **Feed-Forward** | Dense MLP or Sparse Mixture of Experts (Top-K routing) |
| **Positional Encoding** | Rotary (RoPE), absolute sinusoidal, or learned |
| **KV Cache** | Static cache with `jax.lax.dynamic_update_slice` — no recompilation |
| **Regularization** | Dropout (attention, residual, embedding) + MoE load-balancing loss |
| **Training** | Gradient checkpointing (`nnx.remat`), gradient accumulation via Optax |
| **Generation** | Greedy, Top-K, Top-P (nucleus) sampling inside `jax.lax.fori_loop` |

---

## Quickstart

```bash
git clone https://github.com/winstonsmith1897/DantinoX.git
cd DantinoX

# Create environment (Conda recommended)
conda create -n dantinox python=3.12 -y && conda activate dantinox

# Install JAX with CUDA 12 support, then project dependencies
pip install -U "jax[cuda12]"
pip install -r requirements.txt

# Train with the default configuration
python train.py --config configs/default_config.yaml

# Generate text from a trained run
python generate.py --run_dir runs/<run_name> --prompt "Nel mezzo del cammin "
```

---

## Project Structure

```text
DantinoX/
├── core/
│   ├── config.py           # Config dataclass — single source of truth
│   ├── model.py            # Transformer, Attention (MHA/GQA/MLA), MoE, Block
│   ├── attention.py        # Attention kernels and KV-cache logic
│   └── generation.py       # Autoregressive inference engine
│
├── configs/
│   ├── default_config.yaml # Standard training setup
│   └── sweep.yaml          # W&B Bayesian sweep configuration
│
├── utils/
│   ├── tokenizer.py        # Character-level and Byte-Level BPE tokenizers
│   └── helpers.py          # Loss functions, batching, sharding utilities
│
├── train.py                # Training entry point
├── generate.py             # Generation entry point
└── requirements.txt
```
