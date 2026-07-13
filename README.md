<div align="center">

# DantinoX

*"Nel mezzo del cammin di nostra vita mi ritrovai per una selva oscura..."*

A research-grade JAX/Flax NNX transformer library for **autoregressive (AR)**,
**discrete masked diffusion**, and **continuous flow-matching** language models —
one modular Transformer backbone, one trainer, one streaming generator, zero
paradigm-specific boilerplate.

<br>

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-Accelerated-000000?style=flat-square&logo=google&logoColor=white)](https://github.com/google/jax)
[![Flax NNX](https://img.shields.io/badge/Flax-NNX-8A2BE2?style=flat-square)](https://github.com/google/flax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/badge/linter-ruff-orange?style=flat-square)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/type--checked-mypy-blue?style=flat-square)](http://mypy-lang.org/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen?style=flat-square)](https://github.com/winstonsmith1897/DantinoX/actions)
[![Documentation](https://readthedocs.org/projects/dantinox/badge/?version=latest&style=flat-square)](https://dantinox.readthedocs.io/en/latest/)
[![Demo Video](https://img.shields.io/badge/▶_demo-YouTube-FF0000?style=flat-square&logo=youtube&logoColor=white)](https://youtu.be/1u5-AieDzIc)

**[Documentation](https://dantinox.readthedocs.io) · [Demo Video](https://youtu.be/1u5-AieDzIc) · [Notebooks](https://dantinox.readthedocs.io/en/latest/notebooks/) · [API Reference](https://dantinox.readthedocs.io/en/latest/api/)**

</div>

---

## The idea in one snippet

The whole point of DantinoX: **the generation paradigm is a single field.**
Everything else — backbone, weights layout, tokenizer, trainer, generator —
stays byte-for-byte identical.

```python
import dantinox as dx

base = dict(attention="gqa", kv_heads=2, ffn="mlp", use_swiglu=True,
            dim=512, n_heads=8, num_blocks=12, vocab_size=32_128)

cfg_ar   = dx.ModelConfig(paradigm="ar",         **base)   # causal + KV-cache decode
cfg_diff = dx.ModelConfig(paradigm="discrete",   **base)   # LLaDA masked diffusion
cfg_flow = dx.ModelConfig(paradigm="continuous", **base)   # ELF flow-matching

# one Trainer, one Generator — for every paradigm:
run = dx.Trainer(dx.Paradigm(cfg_diff), tcfg).fit("data/wiki.txt")
for chunk in dx.Generator(run).stream("Language models will", n_steps=50):
    print(chunk, end="", flush=True)
```

Attention (MHA/GQA/MLA), FFN (dense/MoE/LatentMoE), positional encoding, norm,
tokenizer, optimizer, LoRA, and DP×TP sharding are **all configuration flags on
the same two dataclasses** — thousands of valid combinations, zero code changes.

---

## Watch the 2-minute demo

A single screencast walks the whole cycle — the paradigm switch, a live training
run, the three inference signatures streaming side by side, zero-execution
profiling, and the benchmark results — all from one API on one GPU.

<div align="center">

[![DantinoX demo video](https://img.youtube.com/vi/1u5-AieDzIc/maxresdefault.jpg)](https://youtu.be/1u5-AieDzIc)

</div>

> Reproduce it locally: `python examples/demo_video.py`

---

## Contents

- [The idea in one snippet](#the-idea-in-one-snippet) · [Demo video](#watch-the-2-minute-demo)
- [Overview](#overview) · [Features](#features) · [Installation](#installation)
- [Quick Start](#quick-start): [one-liner](#one-liner-api) · [paradigm API](#explicit-paradigm-api) · [CLI](#cli)
- [Project Structure](#project-structure) · [Configuration](#configuration)
- [Generation Paradigms](#generation-paradigms): [AR](#autoregressive) · [Diffusion](#masked-diffusion-llada) · [Flow-Matching](#continuous-flow-matching-elf-recipe)
- [LoRA Fine-Tuning](#lora-fine-tuning) · [Benchmarking](#benchmarking) · [Development](#development)
- [Documentation](#documentation) · [Paper](#paper) · [License](#license)

---

## Overview

Comparing autoregressive decoding, masked diffusion, and continuous
flow-matching fairly is hard: each paradigm typically lives in its own
codebase, so any measured difference may reflect tokenizer, initialization,
or training-loop details rather than the paradigm itself. **DantinoX**
removes that confound by separating the model *backbone* (attention,
feed-forward, normalisation, positional encoding) from the *generation
method*. Selecting a paradigm — AR, masked discrete diffusion (LLaDA), or
continuous flow-matching (ELF) — is a single configuration field; the
weights, tokenizer, and training loop stay identical across all three,
enabling controlled cross-paradigm research within one API for training,
streaming inference, and benchmarking.

DantinoX targets **researchers** running controlled paradigm/attention
ablations without editing model code, **educators** who want one readable
codebase covering AR + diffusion + flow-matching, and **practitioners** who
need architectural variants (GQA, MLA, MoE, LoRA) without rewriting the
trainer. It is the only framework we're aware of that unifies all three
generation paradigms on a single JAX/Flax backbone alongside MHA/GQA/MLA
attention, LoRA, multi-GPU scaling, and an integrated benchmarking suite —
see the [framework comparison](https://dantinox.readthedocs.io/en/latest/vs-transformers.html#framework-landscape)
for how this compares to HuggingFace, MaxText, Levanter, OpenLM, torchtune,
Fairseq, xLM, and dLLM.

The library ships as an installable Python package with a unified CLI, a
three-level programmatic API, typed configuration dataclasses, and a full
test suite.

---

## Features

| Layer | What you get |
|:------|:-------------|
| **Attention** | MHA · GQA · MLA (Multi-Head Latent) · Flash Attention · Sliding Window · Gated (attention-sink suppression) · Linear (bidirectional-only, O(T)) · Differential |
| **Feed-Forward** | Dense MLP (SwiGLU / GELU) · Sparse Mixture-of-Experts (Top-K) · LatentMoE (bottleneck-dimension experts) |
| **Position** | Rotary (RoPE, with NTK-aware scaling) · Absolute Sinusoidal · Learned · None |
| **Paradigms** | Autoregressive · Masked Diffusion (LLaDA) · Continuous Flow-Matching (ELF recipe) |
| **Training** | Paradigm-agnostic `Trainer` · AdamW / Lion / Muon / Adafactor / Adam · WSD / Cosine / Linear / Constant LR · Gradient accumulation & clipping · bfloat16 · Multi-GPU JAX SPMD (data + tensor parallel) |
| **Inference** | Static KV-cache · Fast-dLLM DualCache (1.4–2.1× speedup for diffusion) · Euler ODE/SDE + Classifier-Free Guidance for flow-matching · Streaming |
| **Fine-tuning** | Built-in LoRA (`use_lora=True`) · Auto-frozen base weights · `merge_lora()` for zero-overhead deployment |
| **Benchmarking** | `BenchmarkSuite` · Throughput / Latency / Perplexity tasks · `count_flops` (zero-execution FLOPs) · `profile` (latency + MFU) · CSV + plots |
| **Integration** | HuggingFace Hub push/pull · W&B sweeps · 14-subcommand CLI · Colab notebooks |

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
run_dir = dx.fit("discrete", "data/wiki.txt",
                 dim=512, n_heads=8, head_size=64, num_blocks=12,
                 vocab_size=32_000, noise_schedule="cosine",
                 tokenizer_type="bpe", tokenizer_path="t5-base",
                 lr=1e-4, epochs=20)

# Continuous Flow-Matching (ELF recipe) — operates in a frozen T5 embedding space
run_dir = dx.fit("continuous", "data/wiki.txt",
                 embed_dim=768, bottleneck_dim=128,
                 dim=512, n_heads=8, head_size=64, num_blocks=12,
                 flow_cfg_scale=1.5, lr=1e-4, epochs=30)
```

> Unrecognized keyword arguments to `dx.fit()` are silently ignored rather
> than raising an error — double check field names against the
> [Configuration Reference](https://dantinox.readthedocs.io/en/latest/configuration.html)
> if a value doesn't seem to take effect (e.g. the field is `dim`/`flow_cfg_scale`
> on `ModelConfig`, not `model_dim`/`elf_cfg_scale`, which only exist on the
> standalone `FlowMatchingConfig`).

### Explicit Paradigm API

```python
import dantinox as dx
from flax import nnx

cfg      = dx.ModelConfig(paradigm="ar", dim=512, n_heads=8, head_size=64,
                           num_blocks=12, vocab_size=32_000)
paradigm = dx.Paradigm(cfg)
model    = paradigm.build_model(nnx.Rngs(42))

# Train
run_dir = dx.Trainer(paradigm).fit("data/wiki.txt")

# Streaming inference — auto-dispatches to the right paradigm (KV-cache decode
# for AR, reverse diffusion for discrete, ODE/SDE integration for continuous)
gen = dx.Generator(run_dir, seed=42)
for chunk in gen.stream("In the beginning", max_new_tokens=200, top_p=0.9):
    print(chunk, end="", flush=True)
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

# Print FLOPs / parameter count for a checkpoint or config
dantinox profile --run_dir runs/ar_mha_512d

# Evaluate generation quality (Distinct-N, Rep-4, MAUVE)
dantinox eval --run_dir runs/ar_mha_512d

# Full inference benchmark suite
dantinox infbench --trained --eval

# Push/pull checkpoints to HuggingFace Hub
dantinox push --run_dir runs/ar_mha_512d --repo my-org/my-model
dantinox pull --repo my-org/my-model --local_dir runs/downloaded

# Declarative training from a single workflow YAML
dantinox run workflow.yaml

# Export a checkpoint to a StableHLO binary (Python-free inference)
dantinox export runs/ar_mha_512d model.stablehlo

# Generate benchmark plots
dantinox plot --in_csv results/benchmark.csv --out_dir plots/
```

All 14 subcommands (`train`, `generate`, `sweep`, `benchmark`, `find-lr`,
`push`, `pull`, `infbench`, `merge-lora`, `profile`, `run`, `export`, `eval`,
`plot`) are documented in the [CLI Reference](https://dantinox.readthedocs.io/en/latest/cli.html).

---

## Project Structure

Everything importable lives under the installable `dantinox/` package —
there is no separate top-level `core/`/`utils/` package (a couple of empty
stub directories with those names exist at the repo root for legacy reasons
but are unused; the real code is `dantinox/core/`, `dantinox/utils/`, etc.).

```
DantinoX/
├── dantinox/                        # Installable library package
│   ├── core/                        # Neural-network primitives (paradigm-agnostic)
│   │   ├── config.py                # ModelConfig · TrainingConfig · Config · FlowMatchingConfig
│   │   ├── model.py                 # Transformer (DiffusionTransformer is an alias)
│   │   ├── flow.py                  # FlowMatchingTransformer, FlowEmbedder (continuous flow-matching)
│   │   ├── attention.py             # MHA / GQA / MLA + RoPE + KV-cache + Flash/Gated/Linear/Differential
│   │   ├── block.py                 # Block: pre-norm residual (Attention + FFN + Norm)
│   │   ├── mlp.py                   # Dense MLP (SwiGLU / GELU)
│   │   ├── moe.py                   # Sparse MoE + LatentMoE with load-balancing loss
│   │   ├── diffusion.py             # NoiseSchedule · make_noise_schedule · corrupt · masked_cross_entropy
│   │   ├── lora.py                  # LoRALinear · LoRAParam · merge_lora
│   │   ├── generation.py            # generate · diffusion_generate · fast_dllm_generate · flow_generate
│   │   └── sharding.py              # make_mesh · shard_batch — multi-GPU SPMD
│   │
│   ├── paradigms/
│   │   ├── paradigm.py              # Paradigm (unified dispatcher — public API)
│   │   ├── base.py                  # ParadigmBase (ABC): build_model, loss_fn, generate
│   │   ├── ar.py                    # ARParadigm
│   │   ├── embedder.py              # EmbedderParadigm
│   │   └── diffusion/
│   │       ├── discrete.py          # DiscreteParadigm (LLaDA)
│   │       └── continuous.py        # ContinuousParadigm (ELF recipe)
│   │
│   ├── training/
│   │   ├── trainer.py               # Trainer — JIT loop, checkpointing, multi-GPU (this is `dx.Trainer`)
│   │   └── optimizer.py             # build_optimizer · build_schedule (AdamW/Lion/Muon/Adafactor)
│   │
│   ├── benchmarking/                 # BenchmarkSuite · BenchmarkTask · Throughput/Latency/Perplexity tasks
│   ├── profiling/                    # count_flops · LatencyTracker · profile
│   ├── visualization/                 # Visualizer · chart registry
│   ├── utils/                        # Tokenizers (char/BPE/T5), T5 encoder, data pipeline helpers
│   ├── generator.py                  # Generator — paradigm-agnostic streaming inference
│   ├── trainer.py                    # Legacy Config-driven Trainer (backs the `dantinox train` CLI)
│   ├── hub.py                        # push · pull (HuggingFace Hub)
│   └── cli.py                        # 14 subcommands (see CLI section above)
│
├── benchmarks/                       # Stand-alone benchmark/ablation scripts
│   ├── inference_sweep.py            # Random-model sweep (13 groups)
│   ├── trained_analysis.py           # Throughput on real checkpoints
│   └── generation_quality.py         # Distinct-N, Rep-4, MAUVE
│
├── configs/                          # YAML templates
│   ├── default_config.yaml
│   ├── diffusion_base.yaml
│   └── sweep.yaml
│
├── docs/                              # Documentation (built via Sphinx/RTD; docs/index.rst is the toctree)
├── tests/                             # Pytest test suite
├── pyproject.toml
└── mkdocs.yml
```

---

## Configuration

All settings are typed dataclasses. The `Config` class is the flat,
CLI-compatible form; `ModelConfig` + `TrainingConfig` is the preferred split
API for new code (and the one used throughout this README and the paper).

```python
from dantinox.core.config import Config

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

> Note: `attention_type` above is correct for the **legacy** `Config` class.
> The modern `ModelConfig` uses the shorter field name `attention` instead
> (e.g. `dx.ModelConfig(attention="gqa", ...)`) — passing `attention_type=`
> to `ModelConfig` raises `TypeError`. See the
> [Configuration Reference](https://dantinox.readthedocs.io/en/latest/configuration.html)
> for the full field mapping between the two config schemas.

Key constraint: `dim` must equal `n_heads × head_size`.

```python
Config(dim=512, n_heads=8, head_size=64)   # ✓
Config(dim=512, n_heads=8, head_size=32)   # ✗  ValueError
```

Full field reference: [Configuration Reference](https://dantinox.readthedocs.io/en/latest/configuration.html).

---

## Generation Paradigms

### Autoregressive

Token-by-token left-to-right generation with static KV-cache:

```python
from dantinox.core.generation import generate

tokens = generate(model, prompt_ids, max_generations=256, top_p=0.9, use_cache=True)
```

### Masked Diffusion (LLaDA)

All positions decoded in parallel over iterative unmasking steps:

```python
from dantinox.core.generation import diffusion_generate, fast_dllm_generate
from dantinox.core.diffusion import make_noise_schedule

schedule = make_noise_schedule(cfg)

# Standard iterative unmasking
tokens = diffusion_generate(model, prefix, gen_len=128, schedule=schedule,
                            mask_token_id=cfg.mask_token_id)

# Fast-dLLM DualCache — 1.4–2.1× speedup
tokens = fast_dllm_generate(model, prefix, gen_len=256, schedule=schedule,
                             mask_token_id=cfg.mask_token_id,
                             block_size=32, steps_per_block=20)
```

### Continuous Flow-Matching (ELF recipe)

Euler ODE (or SDE, with `gamma > 0`) from Gaussian noise to clean token
embeddings in a frozen T5 encoder's embedding space, with Classifier-Free
Guidance via `cfg_scale`:

```python
from dantinox.core.generation import flow_generate

tokens = flow_generate(model, gen_len=128, batch_size=4,
                       n_steps=64, cfg_scale=1.5, gamma=0.0, seed=42)
```

> `elf_generate` is a deprecated alias of `flow_generate` (same for
> `core.elf` vs `core.flow`, and `ELFConfig`/`ELFTransformer` vs
> `FlowMatchingConfig`/`FlowMatchingTransformer`) — still functional, removed
> in v1.0. Prefix/conditional generation is not yet supported for this
> paradigm; it currently only generates unconditionally from pure noise.

---

## LoRA Fine-Tuning

```python
from dantinox.core.config import Config
from dantinox.core.model import Transformer
from dantinox.core.lora import merge_lora
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
import dantinox as dx
from dantinox.benchmarking import BenchmarkSuite
from flax import nnx

cfg      = dx.ModelConfig(paradigm="ar", dim=512, n_heads=8, head_size=64,
                           num_blocks=12, vocab_size=32_000)
paradigm = dx.Paradigm(cfg)
model    = paradigm.build_model(nnx.Rngs(0))

report = BenchmarkSuite.default().run(paradigm, model, save_csv="results.csv")
print(report.summary())
```

`BenchmarkSuite.default()` sweeps latency, throughput, and energy across
batch sizes and sequence lengths for whichever paradigm you pass in — this
is the same call used to produce the paper's inference-efficiency and
hardware-roofline results across AR, discrete diffusion, and continuous
flow-matching on one shared backbone.

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

- Forward-pass shapes for MHA, GQA, MLA, MoE, LatentMoE, LoRA, Diffusion, Flow-Matching
- KV-cache correctness and accumulation
- Weight tying between embedding and LM head
- JIT compilation stability
- `Config` / `ModelConfig` / `FlowMatchingConfig` validation and round-trip serialisation

### Code Quality

| Tool | Checks |
|:-----|:-------|
| **ruff** | Style (E/W), imports (I), pyupgrade (UP), bugbear (B), simplify (SIM) |
| **mypy** | Full type annotation coverage across `dantinox/` |
| **pytest** | Unit tests, CPU-only, session-scoped fixtures |

---

## Documentation

Full documentation is built with Sphinx and hosted on ReadTheDocs (there is
also an `mkdocs.yml` in the repo for local browsing, but the hosted site at
`dantinox.readthedocs.io` builds from `docs/index.rst`'s `toctree`, not
`mkdocs.yml`'s `nav`):

```bash
pip install "dantinox[docs]"
mkdocs serve          # local preview at http://127.0.0.1:8000
```

Key sections: [Quickstart](https://dantinox.readthedocs.io/en/latest/quickstart.html) · [Paradigms](https://dantinox.readthedocs.io/en/latest/paradigms/) · [Configuration](https://dantinox.readthedocs.io/en/latest/configuration.html) · [CLI](https://dantinox.readthedocs.io/en/latest/cli.html) · [Notebooks](https://dantinox.readthedocs.io/en/latest/notebooks/)

---

## Paper

DantinoX is described in *"DantinoX: A Unified Framework for Multi-Paradigm
Language Modeling"* (Simoni, Fontana, Rossolini, and Saracino). The paper
validates the library with two fully automated evaluations produced entirely
by its own pipeline: **open-ended generation quality** (MAUVE, PPL, lexical
diversity, conditional BLEU) across all nine paradigm × attention
combinations at Small scale (512-d, 12-layer), and **inference efficiency**
(latency, throughput, energy, hardware roofline) across all three paradigms
on a Large backbone (1024-d, 16-layer) on a single A100-40GB GPU. See
[Experiments & Results](https://dantinox.readthedocs.io/en/latest/paper.html)
for the full breakdown.

A short screencast walking through the full train → generate → profile → benchmark
cycle is available as a **[demonstration video ▶](https://youtu.be/1u5-AieDzIc)**
(reproduce it locally with `python examples/demo_video.py`).

```bibtex
@software{dantinox2026,
  author  = {Simoni, Marco and Fontana, Aleksandar and Rossolini, Giulio
             and Saracino, Andrea},
  title   = {{D}antino{X}: A Unified Framework for Multi-Paradigm Language
             Modeling},
  year    = {2026},
  url     = {https://github.com/winstonsmith1897/DantinoX},
}
```

---

## License

MIT — see [LICENSE](LICENSE).
