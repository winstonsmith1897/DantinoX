---
hide:
  - toc
---

<div class="dnx-hero" markdown>

# DantinoX

**A research-grade JAX/Flax NNX transformer library for autoregressive,
discrete diffusion, and continuous flow-matching language models.**

Three paradigms. One trainer. Zero boilerplate.

<div class="hero-badges" markdown>
[![EMNLP 2026](https://img.shields.io/badge/EMNLP-2026%20System%20Demo-blue?style=flat-square)](paper.md)
[![JAX](https://img.shields.io/badge/JAX-000000?style=flat-square&logo=JAX&logoColor=white)](https://github.com/google/jax)
[![Flax NNX](https://img.shields.io/badge/Flax_NNX-5E17EB?style=flat-square&logoColor=white)](https://github.com/google/flax)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License MIT](https://img.shields.io/badge/License-MIT-16a34a?style=flat-square)](https://opensource.org/licenses/MIT)
[![Docs](https://readthedocs.org/projects/dantinox/badge/?version=latest&style=flat-square)](https://dantinox.readthedocs.io)
</div>

[Get Started](quickstart.md){ .md-button .md-button--primary }
[API Reference](api/index.md){ .md-button }
[EMNLP 2026 Paper](paper.md){ .md-button }
[GitHub](https://github.com/winstonsmith1897/DantinoX){ .md-button }

</div>

---

!!! abstract "EMNLP 2026 System Demonstration"
    DantinoX is the official companion codebase for the paper *"DantinoX: A Unified JAX/Flax Framework for Autoregressive and Masked Diffusion Language Models across MHA, GQA, and MLA Attention Variants"*, accepted to the EMNLP 2026 System Demonstrations track. It spans **180 trained checkpoints** across three attention families, ten model sizes, dense and MoE FFNs, and twelve architectural ablations.

    To reproduce all results and paper figures from scratch: `bash scripts/run_full_emnlp.sh`

## In ten lines of code

```python
import dantinox as dx

# Level 1 — ultra low-code
run_dir = dx.fit("ar", "data/wiki.txt",
                 dim=512, n_heads=8, head_size=64, num_blocks=12,
                 vocab_size=32_000, lr=3e-4, epochs=5)

print(dx.quick_generate(run_dir, "In the beginning"))
```

That call builds the model, trains it, saves checkpoints, and returns the run directory.
See [Quickstart](quickstart.md) for the full three-level API.

---

## Capabilities at a glance

| Layer | What you get |
| :--- | :--- |
| **Core** | MHA, GQA, MLA attention · SwiGLU/GELU FFN · Sparse MoE · RMSNorm/LayerNorm · RoPE/NTK/Learned PE · LoRA · Flash Attention |
| **Paradigms** | Autoregressive (AR) · Discrete Diffusion (LLaDA) · Continuous Flow-Matching (ELF) |
| **Training** | Paradigm-agnostic `Trainer` · AdamW/Lion/Muon/Adafactor · WSD/Cosine/Linear schedules · Gradient accumulation · Multi-GPU SPMD |
| **Profiling** | `LatencyTracker` · `count_flops()` · barrier-accurate wall-clock timing |
| **Benchmarking** | `BenchmarkSuite` with plug-in `BenchmarkTask` · CSV export · Throughput/Latency/Perplexity tasks |
| **Visualization** | `Visualizer` chart registry · Training curves · Throughput/Latency/Pareto/Radar charts |
| **Integration** | HuggingFace Hub push/pull · W&B sweeps · Full CLI (`dantinox train / generate / benchmark / plot`) |

---

## Three levels of API

DantinoX is designed for both rapid prototyping and full programmatic control.

=== "Level 1 — One-liner"

    ```python
    import dantinox as dx

    run_dir = dx.fit("ar", "data/wiki.txt",
                     dim=512, n_heads=8, head_size=64,
                     num_blocks=12, vocab_size=32_000)
    ```

=== "Level 2 — Explicit paradigm"

    ```python
    import dantinox as dx

    paradigm = dx.ARParadigm(dx.ModelConfig(
        dim=512, n_heads=8, head_size=64, num_blocks=12, vocab_size=32_000
    ))
    run_dir = dx.Trainer(paradigm, dx.TrainingConfig(lr=3e-4, epochs=5)).fit("data/wiki.txt")
    model   = dx.load(run_dir, paradigm=paradigm)
    tokens  = paradigm.generate(model, prompt, rng)
    ```

=== "Level 3 — Full control"

    ```python
    from core.config import ModelConfig
    from dantinox.paradigms.ar import ARParadigm
    from dantinox.training.trainer import Trainer
    from dantinox.training.optimizer import build_optimizer, build_schedule
    from dantinox.profiling import LatencyTracker, count_flops
    ```

---

## Project structure

```text
DantinoX/
├── core/                    # Neural network primitives (Attention, MLP, MoE, LoRA, …)
├── dantinox/
│   ├── paradigms/           # AR · DiscreteParadigm · ContinuousParadigm (ELF)
│   ├── training/            # Trainer · build_optimizer · build_schedule
│   ├── profiling/           # count_flops · LatencyTracker
│   ├── benchmarking/        # BenchmarkSuite · BenchmarkTask · tasks/
│   ├── visualization/       # Visualizer · Chart · charts/
│   └── cli.py               # dantinox train / generate / sweep / benchmark / plot
├── utils/                   # Tokenizer (char / BPE), batch sampling
├── configs/                 # YAML templates, W&B sweep definitions
└── tests/                   # pytest suite
```

---

## Documentation map

| Section | Description |
| :--- | :--- |
| [Quickstart](quickstart.md) | From install to running model in under 2 minutes |
| [Architecture](architecture.md) | Core layers, Paradigm abstraction, profiling stack |
| [Generation Paradigms](paradigms/index.md) | AR, Discrete Diffusion, ELF, Fast-dLLM, Confidence-Aware |
| [Training](training/index.md) | Trainer internals, optimizers, schedules, multi-GPU |
| [Benchmarks](benchmarks.md) | Throughput, FLOPs, latency — MHA vs GQA vs MLA |
| [API Reference](api/index.md) | Full auto-generated reference for every public symbol |
| [Developer Guide](guides/index.md) | How to add layers, paradigms, benchmark tasks, charts |
| [Contributing](contributing.md) | PR workflow, docstring standards, CI checks |

---

## Citation

```bibtex
@software{dantinox2026,
  author  = {Simoni, Marco},
  title   = {DantinoX: A Unified {JAX}/Flax Framework for {AR} and Masked Diffusion Language Models},
  year    = {2026},
  url     = {https://github.com/winstonsmith1897/DantinoX},
}
```
