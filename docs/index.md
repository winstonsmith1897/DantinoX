---
hide:
  - toc
---

<div class="dnx-hero" markdown>

# DantinoX

**A research-grade language model library natively built in JAX and Flax NNX.**

Supports Autoregressive and Masked Diffusion generation across three attention families — MHA, GQA, and MLA — with Fast-dLLM DualCache, Mixture-of-Experts, LoRA fine-tuning, and multi-GPU SPMD sharding, all controlled from a single YAML configuration.

<div class="hero-badges" markdown>
[![JAX](https://img.shields.io/badge/JAX-000000?style=flat-square&logo=JAX&logoColor=white)](https://github.com/google/jax)
[![Flax NNX](https://img.shields.io/badge/Flax_NNX-5E17EB?style=flat-square&logoColor=white)](https://github.com/google/flax)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License MIT](https://img.shields.io/badge/License-MIT-16a34a?style=flat-square)](https://opensource.org/licenses/MIT)
[![Docs](https://readthedocs.org/projects/dantinox/badge/?version=latest&style=flat-square)](https://dantinox.readthedocs.io)
[![W&B](https://img.shields.io/badge/Tracked%20with-W%26B-FFBE00?style=flat-square&logo=weightsandbiases&logoColor=black)](https://wandb.ai)
</div>

[Get Started](architecture.md){ .md-button .md-button--primary }
[API Reference](api.md){ .md-button }
[GitHub](https://github.com/winstonsmith1897/DantinoX){ .md-button }

</div>

## Overview

DantinoX is a from-scratch Transformer implementation designed for research reproducibility and production-grade performance. Every architectural choice — attention mechanism, normalisation, positional encoding, feed-forward network — is expressed as a single field in a typed `Config` dataclass. No subclassing, no source edits required.

The library is structured as a proper Python package (`pip install -e ".[all]"`), not a standalone script, with a `Trainer`, `Generator`, `BenchmarkRunner`, and a full CLI — built on top of JAX's XLA compiler for zero-overhead inference and training.

## Capabilities

| Component | Details |
| :--- | :--- |
| **Attention** | MHA, GQA, MLA with decoupled RoPE and weight absorption; optional Flash Attention |
| **Generation** | Autoregressive (KV-cache) and Masked Diffusion with Fast-dLLM DualCache |
| **Training** | bfloat16, gradient accumulation, gradient clipping, early stopping, LR finder |
| **Fine-tuning** | LoRA with type-level weight freezing via `LoRAParam`; ~0.2 % of parameters trained |
| **Scale** | Multi-GPU data-parallel via JAX SPMD; set `n_devices=N`, XLA handles the rest |
| **Integration** | HuggingFace Hub push/pull, W&B sweep logging, CLI (`dantinox train / generate / ...`) |
| **Quality** | 86 tests, mypy clean, ruff clean, coverage report |

## Installation

```bash
git clone https://github.com/winstonsmith1897/DantinoX.git
cd DantinoX
conda create -n dantinox python=3.12 -y && conda activate dantinox
pip install -U "jax[cuda12]"
pip install -e ".[all]"
```

## Quick Start

```python
from dantinox import Config, Trainer, Generator

config = Config(
    dim=512, n_heads=16, head_size=32, num_blocks=8,
    lr=3e-4, grad_clip=1.0, use_bf16=True,
    norm_type="rmsnorm",
    use_flash_attention=True,
    lr_schedule="wsd",
)

run_dir = Trainer(config).fit("data/corpus.txt")
gen     = Generator(run_dir)
print(gen.generate("In the beginning ", max_new_tokens=200))
```

For batched generation, streaming, LoRA fine-tuning, multi-GPU usage, and the full CLI reference, see the sections below.

## Documentation

| Section | Description |
| :--- | :--- |
| [Architecture](architecture.md) | Attention mechanisms, MLA math, LoRA implementation, multi-GPU sharding, full config reference |
| [Generation Paradigms](paradigms/index.md) | Autoregressive vs. Masked Diffusion, Fast-dLLM DualCache, confidence-aware decoding |
| [Training](training/index.md) | bfloat16, gradient accumulation, LR schedules, early stopping, W&B sweeps, multi-GPU |
| [Inference](inference/index.md) | Single, batch, and streaming generation; KV-cache pipeline; sampling strategies |
| [Benchmarks](benchmarks.md) | MHA vs. GQA vs. MLA — throughput, KV cache size, FLOPs, latency comparison |
| [Ablation Studies](ablation_studies.md) | Optimizer comparison, MoE routing, positional encoding variants, regularisation |
| [API Reference](api.md) | `Trainer`, `Generator`, `LoRALinear`, sharding utilities, `BenchmarkRunner`, Hub |

## Project Structure

```text
DantinoX/
├── dantinox/           # Public API — Trainer, Generator, CLI, BenchmarkRunner, Hub
├── core/               # Internal implementation — model, attention, generation, sharding
├── utils/              # Tokenizer (char / BPE), data utilities
├── configs/            # Default YAML configs, W&B sweep definitions
├── tests/              # 86 pytest unit and integration tests
└── examples/           # Quickstart script, Colab notebook
```

## Citation

```bibtex
@software{dantinox2026,
  author  = {Simoni, Marco},
  title   = {DantinoX: A Research-Grade Transformer Library in {JAX}},
  year    = {2026},
  url     = {https://github.com/winstonsmith1897/DantinoX},
}
```
