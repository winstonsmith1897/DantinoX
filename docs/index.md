# DantinoX

[![JAX](https://img.shields.io/badge/JAX-000000?style=flat-square&logo=JAX&logoColor=white)](https://github.com/google/jax)
[![Flax NNX](https://img.shields.io/badge/Flax_NNX-5E17EB?style=flat-square&logoColor=white)](https://github.com/google/flax)
[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](https://opensource.org/licenses/MIT)
[![ReadTheDocs](https://readthedocs.org/projects/dantinox/badge/?version=latest&style=flat-square)](https://dantinox.readthedocs.io)

DantinoX is a research-grade Transformer library built from scratch in **JAX** and **Flax NNX**. It supports two generation paradigms — Autoregressive and Masked Diffusion — across three attention families (MHA, GQA, MLA), with Fast-dLLM DualCache, Mixture-of-Experts, LoRA fine-tuning, multi-GPU SPMD sharding, and more, all controlled from a single YAML config.

---

## Key Features

- **Two generation paradigms** — Autoregressive and Masked Diffusion Language Models, switchable from config.
- **Three attention families** — Multi-Head Attention (MHA), Grouped-Query Attention (GQA), and Multi-Head Latent Attention (MLA) with decoupled RoPE and weight absorption.
- **Fast-dLLM DualCache** — 2.1× decode speedup for diffusion models via dual KV-cache strategy.
- **JAX-native inference** — static KV cache via `dynamic_update_slice`, `@jax.jit` training loop, optional Flash Attention (`jax.nn.dot_product_attention`).
- **LoRA fine-tuning** — type-level weight freezing via a custom `LoRAParam` variable; only adapter weights are trained (~0.2% of parameters).
- **Multi-GPU SPMD** — data-parallel training via JAX `jax.sharding.Mesh`; set `n_devices=N` in config, XLA handles the rest.
- **Production-ready API** — `Trainer`, `Generator`, `BenchmarkRunner`, CLI, bfloat16, gradient clipping, early stopping, LR finder, HuggingFace Hub integration.
- **Fully tested** — 86 tests, mypy clean, ruff clean, coverage report included.

---

## Installation

```bash
git clone https://github.com/winstonsmith1897/DantinoX.git
cd DantinoX

conda create -n dantinox python=3.12 -y
conda activate dantinox

pip install -U "jax[cuda12]"
pip install -e ".[all]"
```

---

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

gen = Generator(run_dir)
print(gen.generate("In the beginning ", max_new_tokens=200))
```

For the full CLI reference, batched generation, streaming, LoRA fine-tuning, and multi-GPU usage see the [Training](training/index.md) and [Inference](inference/index.md) sections.

---

## Documentation

| Section | Description |
| :--- | :--- |
| [Architecture](architecture.md) | Attention mechanisms, MLA math, LoRA, multi-GPU, full config reference |
| [Generation Paradigms](paradigms/index.md) | Autoregressive vs. Masked Diffusion, Fast-dLLM DualCache, confidence-aware decoding |
| [Training](training/index.md) | bfloat16, gradient accumulation, LR schedules, early stopping, sweeps, multi-GPU |
| [Inference](inference/index.md) | Single, batch, and streaming generation; KV-cache pipeline; sampling strategies |
| [Benchmarks](benchmarks.md) | MHA vs. GQA vs. MLA — throughput, KV cache size, FLOPs, latency |
| [Ablation Studies](ablation_studies.md) | Optimizer comparison, MoE, positional encoding, regularization |
| [API Reference](api.md) | `Trainer`, `Generator`, `LoRALinear`, sharding utilities, `BenchmarkRunner`, Hub |

---

## Project Structure

```text
DantinoX/
├── dantinox/           # Public library API (Trainer, Generator, CLI, Hub)
├── core/               # Internal implementation (model, attention, generation)
├── utils/              # Tokenizer (char / BPE), data helpers
├── configs/            # Default YAML configs and W&B sweep definitions
├── tests/              # 86 pytest unit and integration tests
└── examples/           # Quickstart script and Colab notebook
```

---

## Citation

If you use DantinoX in your research, please cite:

```bibtex
@software{dantinox2026,
  author  = {Simoni, Marco},
  title   = {DantinoX: A Research-Grade Transformer Library in JAX},
  year    = {2026},
  url     = {https://github.com/winstonsmith1897/DantinoX},
}
```
