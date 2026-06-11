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
[![JAX](https://img.shields.io/badge/JAX-000000?style=flat-square&logo=JAX&logoColor=white)](https://github.com/google/jax)
[![Flax NNX](https://img.shields.io/badge/Flax_NNX-5E17EB?style=flat-square&logoColor=white)](https://github.com/google/flax)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License MIT](https://img.shields.io/badge/License-MIT-16a34a?style=flat-square)](https://opensource.org/licenses/MIT)
[![Docs](https://readthedocs.org/projects/dantinox/badge/?version=latest&style=flat-square)](https://dantinox.readthedocs.io)
</div>

[Get Started](quickstart.md){ .md-button .md-button--primary }
[API Reference](api/index.md){ .md-button }
[Notebooks](notebooks/index.md){ .md-button }
[GitHub](https://github.com/winstonsmith1897/DantinoX){ .md-button }

</div>

---

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

<div class="grid cards" markdown>

-   :material-cpu-64-bit: **Core layers**

    MHA · GQA · MLA · SwiGLU/GELU FFN · Sparse MoE · RMSNorm / LayerNorm · RoPE / NTK / Learned PE · LoRA · Flash Attention

-   :material-source-branch: **Three generation paradigms**

    **Autoregressive** — causal, KV-cached · **Masked Diffusion** (LLaDA) — bidirectional iterative unmasking · **ELF** — continuous flow-matching in embedding space

-   :material-lightning-bolt: **Training**

    Paradigm-agnostic `Trainer` · AdamW / Lion / Muon / Adafactor · WSD / Cosine / Linear schedules · Gradient accumulation · Multi-GPU JAX SPMD

-   :material-speedometer: **Profiling & Benchmarks**

    `LatencyTracker` · `count_flops()` · `BenchmarkSuite` with plug-in tasks · CSV export · Throughput / Latency / Perplexity

-   :material-chart-line: **Visualization**

    `Visualizer` chart registry · Training curves · Throughput / Pareto / Radar charts · 21 auto-generated benchmark figures

-   :material-cloud-sync: **Ecosystem integration**

    HuggingFace Hub push/pull · W&B sweeps · Full CLI · Colab notebooks · HF tokenizers / BPE support

</div>

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

<div class="grid cards" markdown>

-   :material-rocket-launch: **Quickstart**

    From install to trained model in under 2 minutes.

    [Quickstart →](quickstart.md)

-   :material-chef-hat: **Cookbook**

    14 copy-paste recipes: train, generate, LoRA, Hub, benchmarks.

    [Cookbook →](cookbook.md)

-   :material-book-open-variant: **Architecture**

    Core layers, Paradigm abstraction, profiling stack.

    [Architecture →](architecture.md)

-   :material-source-branch: **Generation Paradigms**

    AR · Diffusion (LLaDA) · ELF · Fast-dLLM · Confidence-Aware Decoding.

    [Paradigms →](paradigms/index.md)

-   :material-school: **Training Guide**

    Trainer internals, optimizers, schedules, multi-GPU SPMD.

    [Training →](training/index.md)

-   :material-notebook: **Notebooks**

    5 interactive Colab notebooks — quickstart, diffusion, ELF, benchmarking, LoRA.

    [Notebooks →](notebooks/index.md)

-   :material-file-cog: **Configuration**

    Every `Config` field with type, default, and valid values.

    [Config →](configuration.md)

-   :material-console: **CLI Reference**

    All 9 subcommands with full argument tables.

    [CLI →](cli.md)

-   :material-code-tags: **API Reference**

    Auto-generated docs for every public symbol.

    [API →](api/index.md)

</div>

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
