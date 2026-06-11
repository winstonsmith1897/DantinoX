---
hide:
  - toc
---

<div class="dnx-hero" markdown>

# DantinoX

**A research-grade JAX/Flax NNX library for language model research.**
Three generation paradigms — Autoregressive, Masked Diffusion, and ELF — on the same Transformer architecture, with a single trainer and zero boilerplate.

<div class="hero-badges" markdown>
[![JAX](https://img.shields.io/badge/JAX-000000?style=flat-square&logo=google&logoColor=white)](https://github.com/google/jax)
[![Flax NNX](https://img.shields.io/badge/Flax_NNX-5E17EB?style=flat-square&logoColor=white)](https://github.com/google/flax)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License MIT](https://img.shields.io/badge/License-MIT-16a34a?style=flat-square)](https://opensource.org/licenses/MIT)
[![Docs](https://readthedocs.org/projects/dantinox/badge/?version=latest&style=flat-square)](https://dantinox.readthedocs.io)
</div>

[Get Started →](quickstart.md){ .md-button .md-button--primary }
[API Reference](api/index.md){ .md-button }
[Cookbook](cookbook.md){ .md-button }
[GitHub](https://github.com/winstonsmith1897/DantinoX){ .md-button }

</div>

---

## What is DantinoX?

DantinoX is a research library written in pure JAX for building and training Transformer language models. It was created to answer a simple question: **how do different generation paradigms — autoregressive, masked diffusion, and flow-matching — compare when trained on the same architecture with the same training code?**

The library is designed for three types of users:

- **Researchers** who want to compare AR vs. Diffusion vs. ELF in a reproducible way
- **Students** who want to understand the internal details of a modern Transformer
- **Engineers** who want to experiment with architectural variants (GQA, MLA, MoE, LoRA) without rewriting the trainer from scratch

---

## The three paradigms

<div class="grid cards" markdown>

-   :material-arrow-right-circle: **Autoregressive (AR)**

    The classical paradigm: generates one token at a time, left to right. Each produced token is appended to the context and used to predict the next one.

    **Pros:** Simple, fast at inference with KV-cache, great as a baseline.

    **Cons:** Cannot revise tokens that have already been generated.

    [Learn more →](paradigms/autoregressive.md)

-   :material-blur: **Masked Diffusion (LLaDA)**

    Generates all tokens in parallel, starting from a fully masked sequence and iteratively removing `[MASK]` tokens. Attention is bidirectional — it sees the entire sequence at once.

    **Pros:** More diverse and coherent outputs on certain tasks.

    **Cons:** Requires multiple inference steps (can be accelerated with Fast-dLLM).

    [Learn more →](paradigms/diffusion.md)

-   :material-wave: **ELF — Continuous Flow**

    Operates in the continuous embedding space rather than on discrete tokens. Transforms Gaussian noise into clean embeddings via an Euler ODE solver.

    **Pros:** Experimental paradigm, excellent for flow-matching research.

    **Cons:** More complex to train, requires more data and epochs.

    [Learn more →](paradigms/elf.md)

</div>

---

## What the library includes

<div class="grid cards" markdown>

-   :material-layers: **Complete neural layers**

    MHA, GQA, MLA (Multi-Latent Attention), Flash Attention, Sliding Window Attention, SwiGLU, GELU, Sparse MoE, RMSNorm, LayerNorm, RoPE, NTK-aware RoPE, Sinusoidal, Learned PE.

-   :material-school: **Unified Trainer**

    A single `Trainer` works across all three paradigms. Supports: gradient accumulation, bfloat16, multi-GPU via JAX SPMD, automatic checkpointing, W&B logging, and LR range test.

-   :material-speedometer: **Optimisers and schedules**

    AdamW, Lion, Muon, Adafactor. Schedules: cosine, linear, WSD (warmup-stable-decay). Configurable warmup.

-   :material-lightning-bolt: **Optimised inference**

    Static pre-allocated KV-cache (AR). Fast-dLLM DualCache for Diffusion (1.4–2.1× speedup). Token streaming for AR.

-   :material-tune: **LoRA fine-tuning**

    Built-in LoRA support (`use_lora=True`). Base weights are frozen automatically. Supports adapter merging via `merge_lora()`.

-   :material-chart-bar: **Systematic benchmarking**

    `BenchmarkSuite` with plug-in tasks. Throughput, latency, perplexity. CSV export. 21 auto-generated plots.

-   :material-cloud-sync: **Ecosystem integration**

    HuggingFace Hub push/pull. W&B sweeps. Full CLI with 12 subcommands. Colab notebooks.

-   :material-wrench: **Analysis tools**

    `count_flops()` for theoretical FLOPs. `LatencyTracker` for real measurements. `Visualizer` for charts.

</div>

---

## Three levels of API

The library is designed to be used at different levels of abstraction, from the simplest to the most detailed.

=== "Level 1 — One-liner"

    Best for rapid prototyping. `dx.fit` does everything: builds the model, trains it, saves the checkpoint.

    ```python
    import dantinox as dx

    run_dir = dx.fit("ar", "data/wiki.txt",
                     dim=512, n_heads=8, head_size=64,
                     num_blocks=12, vocab_size=32_000)

    print(dx.quick_generate(run_dir, "In the beginning"))
    ```

=== "Level 2 — Explicit API"

    Separates architecture config, training config, and paradigm. Allows customising each component individually.

    ```python
    import dantinox as dx
    from flax import nnx

    model_cfg = dx.ModelConfig(
        dim=512, n_heads=8, head_size=64, num_blocks=12,
        vocab_size=32_000, attention_type="gqa", kv_heads=2,
    )
    train_cfg = dx.TrainingConfig(lr=3e-4, epochs=5, grad_accum=4)

    paradigm = dx.ARParadigm(model_cfg)
    run_dir  = dx.Trainer(paradigm, train_cfg).fit("data/wiki.txt")
    model    = dx.load(run_dir, paradigm=paradigm)
    tokens   = paradigm.generate(model, prompt_ids, rng=nnx.Rngs(0))
    ```

=== "Level 3 — Full control"

    Direct access to all internal components. Ideal for modifying the training loop or adding custom components.

    ```python
    from core.config import ModelConfig
    from core.model import Transformer
    from dantinox.paradigms.ar import ARParadigm
    from dantinox.training.trainer import Trainer
    from dantinox.training.optimizer import build_optimizer, build_schedule
    from dantinox.profiling import LatencyTracker, count_flops
    from flax import nnx

    cfg      = ModelConfig(dim=512, n_heads=8, head_size=64,
                           num_blocks=12, vocab_size=32_000)
    paradigm = ARParadigm(cfg)
    model    = paradigm.build_model(nnx.Rngs(42))

    tx       = build_optimizer(cfg)
    schedule = build_schedule(cfg)
    # ... custom training loop ...
    ```

---

## Project structure

```text
DantinoX/
│
├── core/                        ← Neural primitives (Attention, FFN, MoE, LoRA, …)
│   ├── config.py                   ModelConfig · TrainingConfig · Config · ELFConfig
│   ├── model.py                    Transformer · DiffusionTransformer
│   ├── elf.py                      ELFTransformer
│   ├── attention.py                MHA / GQA / MLA + RoPE + KV-cache
│   ├── block.py                    TransformerBlock (Attention + FFN + Norm)
│   ├── mlp.py                      Dense MLP (SwiGLU / GELU)
│   ├── moe.py                      Sparse MoE with load-balancing loss
│   ├── diffusion.py                NoiseSchedule · make_noise_schedule
│   ├── lora.py                     LoRAParam · merge_lora
│   └── generation.py               generate · diffusion_generate · elf_generate · fast_dllm_generate
│
├── dantinox/                    ← Installable package
│   ├── cli.py                      12 CLI subcommands
│   ├── generator.py                Generator class (AR, loads checkpoint)
│   ├── paradigms/
│   │   ├── ar.py                   ARParadigm
│   │   └── diffusion/
│   │       ├── discrete.py         DiscreteParadigm (LLaDA)
│   │       └── continuous.py       ContinuousParadigm (ELF)
│   ├── training/
│   │   ├── trainer.py              Trainer — JIT loop, checkpointing, multi-GPU
│   │   └── optimizer.py            build_optimizer · build_schedule
│   ├── benchmarking/               BenchmarkSuite · plug-in tasks
│   ├── profiling/                  LatencyTracker · count_flops
│   ├── visualization/              Visualizer · chart registry
│   └── hub.py                      push · pull to/from HuggingFace Hub
│
├── utils/
│   ├── tokenizer.py                CharTokenizer · BPETokenizer
│   └── helpers.py                  Loss helpers, batch sampling
│
├── benchmarks/                  ← Stand-alone benchmark scripts
├── configs/                     ← YAML templates (default, diffusion, sweep)
├── docs/                        ← MkDocs Material documentation
└── tests/                       ← pytest suite (CPU-only)
```

---

## Citation

```bibtex
@software{dantinox2026,
  author  = {Simoni, Marco},
  title   = {DantinoX: A Unified {JAX}/Flax Framework for {AR},
             Masked Diffusion, and Flow-Matching Language Models},
  year    = {2026},
  url     = {https://github.com/winstonsmith1897/DantinoX},
}
```
