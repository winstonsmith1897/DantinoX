<div align="center">

# DantinoX

*"Nel mezzo del cammin di nostra vita mi ritrovai per una selva oscura..."*

A research-grade JAX/Flax NNX transformer library for **autoregressive**,
**discrete diffusion**, and **continuous flow-matching** language models.

Three paradigms. One trainer. Zero boilerplate.

<br>

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-Accelerated-000000?style=flat-square&logo=google&logoColor=white)](https://github.com/google/jax)
[![Flax NNX](https://img.shields.io/badge/Flax-NNX-8A2BE2?style=flat-square)](https://github.com/google/flax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/badge/linter-ruff-orange?style=flat-square)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/type--checked-mypy-blue?style=flat-square)](http://mypy-lang.org/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen?style=flat-square)](https://github.com/winstonsmith1897/DantinoX/actions)
[![Documentation](https://readthedocs.org/projects/dantinox/badge/?version=latest&style=flat-square)](https://dantinox.readthedocs.io/en/latest/)

**[Documentation](https://dantinox.readthedocs.io) · [Notebooks](https://dantinox.readthedocs.io/en/latest/notebooks/) · [API Reference](https://dantinox.readthedocs.io/en/latest/api/)**

</div>

---

## Overview

**DantinoX** is a modular, research-focused library for building and training transformer language models in pure JAX. It supports three generation paradigms on the same backbone — autoregressive (AR), masked discrete diffusion (LLaDA), and continuous flow-matching (ELF) — and provides a systematic benchmarking suite for comparing them.

The library ships as an installable Python package with a unified CLI, a three-level programmatic API, typed configuration dataclasses, and a full test suite.

---

## Features

| Layer | What you get |
|:------|:-------------|
| **Attention** | MHA · GQA · MLA (Multi-Latent) · Flash Attention · Sliding Window |
| **Feed-Forward** | Dense MLP (SwiGLU / GELU) · Sparse Mixture-of-Experts (Top-K) |
| **Position** | Rotary (RoPE) · Absolute Sinusoidal · Learned |
| **Paradigms** | Autoregressive · Masked Diffusion (LLaDA) · ELF Continuous Flow-Matching |
| **Training** | Paradigm-agnostic `Trainer` · AdamW / Lion / Muon / Adafactor · WSD / Cosine / Linear LR · Gradient accumulation · Multi-GPU JAX SPMD |
| **Inference** | Static KV-cache · Fast-dLLM DualCache (1.4–2.1× speedup for diffusion) · Streaming |
| **Fine-tuning** | Built-in LoRA (`use_lora=True`) · Auto-frozen base weights · `merge_lora()` |
| **Benchmarking** | `BenchmarkSuite` · Throughput / Latency / Perplexity tasks · CSV + 21 plots |
| **Integration** | HuggingFace Hub push/pull · W&B sweeps · Full CLI · Colab notebooks |

---

## Installation

```bash
pip install dantinox                   # core only
pip install "dantinox[data]"          # + HuggingFace datasets
pip install "dantinox[benchmark]"     # + pandas / matplotlib / scipy
pip install "dantinox[all]"           # everything including dev tools
```

**From source:**

```bash
git clone https://github.com/winstonsmith1897/DantinoX.git
cd DantinoX
conda create -n dantinox python=3.10 -y && conda activate dantinox
make install
```

> **GPU:** after `make install`, run `pip install -U "jax[cuda12]"` for CUDA support.

---

## Quick Start

### One-liner API

```python
import dantinox as dx

# Train an AR model
run_dir = dx.fit("ar", "data/wiki.txt",
                 dim=512, n_heads=8, head_size=64, num_blocks=12,
                 vocab_size=32_000, lr=3e-4, epochs=5)

print(dx.quick_generate(run_dir, "In the beginning"))
```

Switch paradigm by changing the first argument — the trainer, optimizer, and checkpoint logic are identical:

```python
# Masked Diffusion (LLaDA)
run_dir = dx.fit("diffusion", "data/wiki.txt",
                 dim=512, n_heads=8, head_size=64, num_blocks=12,
                 vocab_size=32_000, noise_schedule="cosine",
                 tokenizer_type="bpe", tokenizer_path="t5-base",
                 lr=1e-4, epochs=20)

# ELF — continuous flow-matching in embedding space
run_dir = dx.fit("elf", "data/wiki.txt",
                 embed_dim=512, bottleneck_dim=128,
                 dim=512, n_heads=8, head_size=64, num_blocks=12,
                 vocab_size=32_128, elf_cfg_scale=1.5, lr=1e-4, epochs=30)
```

### Explicit Paradigm API

```python
from core.config import ModelConfig
from dantinox.paradigms.ar import ARParadigm
from dantinox.trainer import Trainer
from dantinox.generator import Generator
from flax import nnx

cfg      = ModelConfig(dim=512, n_heads=8, head_size=64, num_blocks=12, vocab_size=32_000)
paradigm = ARParadigm(cfg)
model    = paradigm.build_model(nnx.Rngs(42))

# Train
run_dir = Trainer(paradigm).fit("data/wiki.txt")

# Generate
gen  = Generator(run_dir)
text = gen.generate("In the beginning", max_new_tokens=200, top_p=0.9)
print(text)
```

### CLI

```bash
# Train (any field from Config can be overridden inline)
dantinox train --config configs/default_config.yaml --data_path wiki.txt

# Override fields on the command line
dantinox train --config configs/default_config.yaml --data_path wiki.txt \
    --model_type diffusion --lr 1e-4 --use_bf16 true --n_devices 4

# Generate with streaming
dantinox generate --run_dir runs/ar_mha_512d \
    --prompt "In the beginning" --stream --top_p 0.9

# Find optimal learning rate
dantinox find-lr --config configs/default_config.yaml --data_path wiki.txt --plot

# Run hyperparameter sweep (W&B)
dantinox sweep --sweep_config configs/sweep.yaml --data_path wiki.txt

# Full inference benchmark suite
dantinox infbench --trained --eval

# Push/pull checkpoints to HuggingFace Hub
dantinox push --run_dir runs/ar_mha_512d --repo my-org/my-model
dantinox pull --repo my-org/my-model --local_dir runs/downloaded

# Generate benchmark plots
dantinox plot --in_csv results/benchmark.csv --out_dir plots/
```

---

## Project Structure

```
DantinoX/
├── core/                        # Neural network primitives
│   ├── config.py                # ModelConfig · TrainingConfig · Config · ELFConfig
│   ├── model.py                 # Transformer · DiffusionTransformer
│   ├── elf.py                   # ELFTransformer (continuous flow-matching)
│   ├── attention.py             # MHA / GQA / MLA + RoPE + KV-cache
│   ├── block.py                 # Transformer block (Attention + FFN + Norm)
│   ├── mlp.py                   # Dense MLP (SwiGLU / GELU)
│   ├── moe.py                   # Sparse MoE with load-balancing loss
│   ├── diffusion.py             # NoiseSchedule · make_noise_schedule
│   ├── lora.py                  # LoRAParam · merge_lora
│   └── generation.py            # generate · diffusion_generate · elf_generate · fast_dllm_generate
│
├── dantinox/                    # Installable library package
│   ├── cli.py                   # 9 subcommands: train/generate/sweep/benchmark/infbench/find-lr/push/pull/plot
│   ├── paradigms/
│   │   ├── ar.py                # ARParadigm
│   │   └── diffusion/
│   │       ├── discrete.py      # DiscreteParadigm (LLaDA)
│   │       └── continuous.py    # ContinuousParadigm (ELF)
│   ├── training/
│   │   ├── trainer.py           # Trainer — JIT loop, checkpointing, multi-GPU
│   │   └── optimizer.py         # build_optimizer · build_schedule
│   ├── benchmarking/            # BenchmarkSuite · BenchmarkTask · ThroughputTask · LatencyTask
│   ├── profiling/               # LatencyTracker · count_flops
│   ├── visualization/           # Visualizer · chart registry
│   └── hub.py                   # push · pull (HuggingFace Hub)
│
├── utils/
│   ├── tokenizer.py             # CharTokenizer · BPETokenizer
│   └── helpers.py               # Loss helpers, batch sampling
│
├── benchmarks/                  # Stand-alone benchmark scripts
│   ├── inference_sweep.py       # Random-model sweep (13 groups)
│   ├── trained_analysis.py      # Throughput on real checkpoints
│   └── generation_quality.py    # Distinct-N, Rep-4, MAUVE
│
├── configs/                     # YAML templates
│   ├── default_config.yaml
│   ├── diffusion_base.yaml
│   └── sweep.yaml
│
├── docs/                        # MkDocs Material documentation
├── tests/                       # Pytest test suite
├── pyproject.toml
└── mkdocs.yml
```

---

## Configuration

All settings are typed dataclasses. The `Config` class is the flat, CLI-compatible form; `ModelConfig` + `TrainingConfig` is the preferred split API for new code.

```python
from core.config import Config

cfg = Config(
    # Architecture
    dim=512, n_heads=8, head_size=64, num_blocks=12,
    vocab_size=32_000, max_context=1024,
    attention_type="gqa", kv_heads=2,
    norm_type="rmsnorm", use_swiglu=True,

    # Paradigm
    model_type="autoregressive",   # "autoregressive" | "diffusion" | "elf"

    # Training
    lr=3e-4, batch_size=64, grad_accum=4,
    optimizer="adamw", lr_schedule="cosine",
    warmup_steps=400, epochs=500,
    use_bf16=True, n_devices=4,
)
```

Key constraint: `dim` must equal `n_heads × head_size`.

```python
Config(dim=512, n_heads=8, head_size=64)   # ✓
Config(dim=512, n_heads=8, head_size=32)   # ✗  ValueError
```

Full field reference: [Configuration Reference](https://dantinox.readthedocs.io/en/latest/configuration/).

---

## Generation Paradigms

### Autoregressive

Token-by-token left-to-right generation with static KV-cache:

```python
from core.generation import generate

tokens = generate(model, prompt_ids, max_generations=256, top_p=0.9, use_cache=True)
```

### Masked Diffusion (LLaDA)

All positions decoded in parallel over iterative unmasking steps:

```python
from core.generation import diffusion_generate, fast_dllm_generate
from core.diffusion import make_noise_schedule

schedule = make_noise_schedule(cfg)

# Standard iterative unmasking
tokens = diffusion_generate(model, prefix, gen_len=128, schedule=schedule,
                            mask_token_id=cfg.mask_token_id)

# Fast-dLLM DualCache — 1.4–2.1× speedup
tokens = fast_dllm_generate(model, prefix, gen_len=256, schedule=schedule,
                             mask_token_id=cfg.mask_token_id,
                             block_size=32, steps_per_block=20)
```

### ELF — Continuous Flow-Matching

Euler ODE from Gaussian noise to clean token embeddings:

```python
from core.generation import elf_generate

tokens = elf_generate(model, gen_len=128, batch_size=4,
                      n_steps=64, cfg_scale=1.5, seed=42)
```

---

## LoRA Fine-Tuning

```python
from core.config import Config
from core.model import Transformer
from core.lora import merge_lora
from flax import nnx

# Enable LoRA — base weights are frozen automatically
cfg   = Config.from_yaml("runs/ar_base/config.yaml")
cfg.use_lora, cfg.lora_rank, cfg.lora_alpha = True, 8, 16.0

model = Transformer(cfg, rngs=nnx.Rngs(42))
# ... load base weights, train adapters ...

merged = merge_lora(model)   # fold adapters into base weights for deployment
```

---

## Benchmarking

```python
from core.config import ModelConfig
from dantinox.paradigms.ar import ARParadigm
from dantinox.benchmarking import BenchmarkSuite
from flax import nnx

cfg      = ModelConfig(dim=512, n_heads=8, head_size=64, num_blocks=12, vocab_size=32_000)
paradigm = ARParadigm(cfg)
model    = paradigm.build_model(nnx.Rngs(0))

report = BenchmarkSuite.default().run(paradigm, model, save_csv="results.csv")
print(report.summary())
```

Or use the full inference sweep via CLI:

```bash
dantinox infbench --groups attention_type scale --n-trials 20
dantinox infbench --trained --eval   # include trained models + quality metrics
```

---

## Development

```bash
make install      # Install package + all dev/doc dependencies (editable)
make test         # Run test suite with coverage report
make lint         # Ruff static analysis
make typecheck    # Mypy type checking
make check        # lint + typecheck + test  (run before every push)
make build        # Build sdist + wheel into dist/
make publish      # Upload to PyPI via twine
make clean        # Remove build artefacts and __pycache__
```

### Running Tests

```bash
JAX_PLATFORM_NAME=cpu python -m pytest tests/ -v
```

Tests run entirely on CPU and cover:

- Forward-pass shapes for MHA, GQA, MLA, MoE, LoRA, Diffusion, ELF
- KV-cache correctness and accumulation
- Weight tying between embedding and LM head
- JIT compilation stability
- `Config` / `ModelConfig` / `ELFConfig` validation and round-trip serialisation

### Code Quality

| Tool | Checks |
|:-----|:-------|
| **ruff** | Style (E/W), imports (I), pyupgrade (UP), bugbear (B), simplify (SIM) |
| **mypy** | Full type annotation coverage across `dantinox/`, `core/`, `utils/` |
| **pytest** | Unit tests, CPU-only, session-scoped fixtures |

---

## Documentation

Full documentation is built with MkDocs Material:

```bash
pip install "dantinox[docs]"
mkdocs serve          # local preview at http://127.0.0.1:8000
mkdocs gh-deploy      # deploy to GitHub Pages
```

Key sections: [Quickstart](https://dantinox.readthedocs.io/en/latest/quickstart/) · [Paradigms](https://dantinox.readthedocs.io/en/latest/paradigms/) · [Configuration](https://dantinox.readthedocs.io/en/latest/configuration/) · [CLI](https://dantinox.readthedocs.io/en/latest/cli/) · [Notebooks](https://dantinox.readthedocs.io/en/latest/notebooks/)

---

## Citation

```bibtex
@software{dantinox2026,
  author  = {Simoni, Marco},
  title   = {DantinoX: A Unified {JAX}/Flax Framework for {AR}, Masked Diffusion, and Flow-Matching Language Models},
  year    = {2026},
  url     = {https://github.com/winstonsmith1897/DantinoX},
}
```

---

## License

MIT — see [LICENSE](LICENSE).
