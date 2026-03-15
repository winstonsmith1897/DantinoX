<div align="center">

# 𝔇𝔞𝔫𝔱𝔦𝔫𝔬𝔛

<i>"Ah JAX, vituperio delle genti..."</i>  
<b>(Ah JAX, the shame of the people...)</b>

<br>

A Transformer so **"nano" it barely rhymes**, implemented in **JAX** and **Flax NNX**. Built with **sweat** and **XLA errors**.


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

# Overview: The DantinoX Project

> *"Nel mezzo del cammin di nostra vita mi ritrovai per una selva oscura, ché la diritta via era smarrita."*

**DantinoX** is a from-scratch implementation of a modern Large Language Model built natively in **JAX and Flax NNX**. The primary motivation behind this project is educational and exploratory: to understand the internal mechanics of current transformer architectures and to learn how to write efficient JAX code without constantly fighting XLA compilation errors.

To thoroughly understand these constraints, DantinoX implements standard modern Deep Learning components directly from the ground up:

* **Sparse Mixture of Experts (MoE)** with **Load Balancing Loss**
* **Rotary Positional Embeddings (RoPE)**
* **Grouped Query Attention (GQA)**
* **Sliding Window & Attention Gating**
* **Static KV Cache**
* **Weight Tying**
* **Gradient Checkpointing and Gradient Accumulation**


### Highly Customizable

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
