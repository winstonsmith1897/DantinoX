---
hide:
  - toc
  - navigation
---

<div class="home-hero" markdown>

# DantinoX

<p class="hero-tagline">"E quindi uscimmo a riveder le stelle."</p>
<p class="hero-sub">A decoder-only Transformer library — built from scratch in JAX and Flax NNX.</p>

[Get Started](architecture.md){ .md-button .md-button--primary }
[View on GitHub](https://github.com/winstonsmith1897/DantinoX){ .md-button }

<div class="stat-chips" markdown>
<span class="stat-chip">:material-language-python: Python 3.12+</span>
<span class="stat-chip">:material-memory: MLA · GQA · MHA</span>
<span class="stat-chip">:material-lightning-bolt: XLA-Native</span>
<span class="stat-chip">:material-flash: Flash Attention</span>
<span class="stat-chip">:material-chip: bfloat16</span>
<span class="stat-chip">:material-format-list-bulleted-type: RMSNorm · LayerNorm</span>
<span class="stat-chip">:material-hub: HF Hub</span>
<span class="stat-chip">:material-package-variant: pip install</span>
<span class="stat-chip">:material-license: MIT</span>
</div>

<div class="hero-badges" markdown>
[![JAX](https://img.shields.io/badge/JAX-000000?style=flat-square&logo=JAX&logoColor=white)](https://github.com/google/jax)
[![Flax NNX](https://img.shields.io/badge/Flax_NNX-5E17EB?style=flat-square&logoColor=white)](https://github.com/google/flax)
[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](https://opensource.org/licenses/MIT)
[![W&B](https://img.shields.io/badge/Tracked%20with-W%26B-FFBE00?style=flat-square&logo=weightsandbiases&logoColor=black)](https://wandb.ai)
</div>

</div>

<div class="grid cards" markdown>

-   :material-layers-triple-outline: &nbsp;**Three Attention Families**

    ---

    MHA, GQA, and **Multi-Head Latent Attention (MLA)** with decoupled RoPE and full weight absorption — all switchable via a single config flag.

    [:octicons-arrow-right-24: Core Architecture](architecture.md)

-   :material-lightning-bolt-circle: &nbsp;**JAX-Native, XLA-First**

    ---

    Static KV cache via `dynamic_update_slice`, `@jax.jit` training loop, and `jax.lax.fori_loop` decode — zero dynamic shapes, zero recompilation.

    [:octicons-arrow-right-24: Inference & Generation](generation.md)

-   :material-chart-bar: &nbsp;**90+ Runs, Fully Benchmarked**

    ---

    Bayesian sweeps over 20+ hyperparameters logged to W&B. Results visualised in 2D and 3D across throughput, FLOPs, and KV cache size.

    [:octicons-arrow-right-24: Benchmarks](benchmarks.md)

-   :material-package-variant-closed: &nbsp;**Production-Ready Library**

    ---

    `Trainer`, `Generator`, `BenchmarkRunner`, and `Plotter` with bfloat16, gradient clipping, early stopping, checkpoint resume, `from_pretrained`, 4 LR schedules, batch & streaming generation, and HuggingFace Hub push/pull.

    [:octicons-arrow-right-24: API Reference](api.md)

</div>

---

## Quickstart

=== "Python API"

    ```bash
    git clone https://github.com/winstonsmith1897/DantinoX.git
    cd DantinoX

    conda create -n dantinox python=3.12 -y && conda activate dantinox
    pip install -U "jax[cuda12]"
    pip install -e ".[all]"
    ```

    ```python
    from dantinox import Config, Trainer, Generator

    # 1. Train — bfloat16, RMSNorm, Flash Attention, WSD schedule
    config = Config(
        dim=512, n_heads=16, head_size=32, num_blocks=8,
        lr=3e-4, grad_clip=1.0, use_bf16=True,
        norm_type="rmsnorm",         # RMSNorm instead of LayerNorm
        use_flash_attention=True,    # fused scaled-dot-product (JAX ≥ 0.4.25)
        lr_schedule="wsd",           # warmup → stable → cosine decay
        rope_scale_factor=2.0,       # NTK-aware: ~2× effective context window
        patience=5,                  # stop if val loss stalls for 5 evals
    )
    run_dir = Trainer(config).fit("data/corpus.txt")

    # 2. Single-prompt generation
    gen = Generator(run_dir)
    print(gen.generate("Nel mezzo del cammin ", max_new_tokens=200))

    # 3. Batched generation — one forward pass for all prompts
    texts = gen.generate_batch(
        ["Nel mezzo", "Lasciate ogni speranza", "Per me si va"],
        max_new_tokens=100, temperature=0.8,
    )

    # 4. Streaming generation — yield tokens as they are produced
    for chunk in gen.stream("Nel mezzo del cammin ", max_new_tokens=150):
        print(chunk, end="", flush=True)

    # 5. Find the right learning rate before a full run
    lr, lrs, losses = Trainer(config).find_lr("data/corpus.txt", num_steps=100)
    print(f"Suggested LR: {lr:.2e}")

    # 6. Load model directly for custom inference / fine-tuning
    from core import Transformer
    model = Transformer.from_pretrained(run_dir)   # loads config + best weights

    # 7. Push to HuggingFace Hub / pull on another machine
    from dantinox import push, pull
    push(run_dir, "my-org/dantinox-dante", private=False)
    run_dir = pull("my-org/dantinox-dante")
    ```

=== "CLI"

    ```bash
    # Train with bfloat16 and gradient clipping
    dantinox train \
      --config configs/default_config.yaml \
      --data_path data/corpus.txt \
      --use_bf16 True --grad_clip 1.0 --patience 5

    # Resume an interrupted run
    dantinox train --config configs/default_config.yaml \
      --data_path data/corpus.txt \
      --run_dir runs/run_20260101_120000 --resume

    # Find the best learning rate before committing to a long run
    dantinox find-lr \
      --config configs/default_config.yaml \
      --data_path data/corpus.txt \
      --min_lr 1e-6 --max_lr 1e-2 --num_steps 100 --plot

    # Generate text
    dantinox generate \
      --run_dir runs/run_20260101_120000 \
      --prompt "Nel mezzo del cammin " \
      --max_new_tokens 200 --temperature 0.8 --top_k 40

    # Sweep (W&B Bayesian)
    dantinox sweep --sweep_config configs/sweep.yaml --data_path data/corpus.txt

    # Benchmark all runs, then plot
    dantinox benchmark --runs_dir runs --out_csv benchmark_results.csv
    dantinox plot --in_csv benchmark_results.csv --out_dir plots/

    # Share your checkpoint on HuggingFace Hub
    dantinox push --run_dir runs/run_20260101_120000 --repo my-org/dantinox-dante
    dantinox pull --repo my-org/dantinox-dante --local_dir runs/pulled
    ```

---

## Documentation

| | Page | What you'll find |
| :--- | :--- | :--- |
| :material-layers-outline: | [Core Architecture](architecture.md) | Attention types, math, full configuration reference, implementation deep-dives |
| :material-school-outline: | [Training & Sweeps](training.md) | bfloat16, grad clipping, early stopping, resume, LR finder, W&B sweeps |
| :material-play-box-outline: | [Inference & Generation](generation.md) | Single, batch & streaming generation, KV-cache pipeline, sampling strategies |
| :material-chart-scatter-plot: | [Benchmarks](benchmarks.md) | MHA vs GQA vs MLA — throughput, cache size, FLOPs, 3D surfaces |
| :material-microscope: | [Ablation Studies](ablation_studies.md) | Optimizer, MoE, positional encoding, regularization |
| :material-book-open-outline: | [API Reference](api.md) | `Trainer`, `Generator`, `BenchmarkRunner`, `Plotter`, Hub, and core modules |

---

## Project Structure

```text
DantinoX/
├── dantinox/               # Public library API
│   ├── __init__.py         # Top-level imports and __version__
│   ├── trainer.py          # Trainer — training, gradient clipping, early stopping, LR finder
│   ├── generator.py        # Generator — single, batch & streaming generation
│   ├── hub.py              # push() / pull() — HuggingFace Hub integration
│   ├── bench.py            # BenchmarkRunner — throughput / FLOPs benchmarks
│   ├── plotting.py         # Plotter — automated plot generation
│   ├── exceptions.py       # DantinoXError hierarchy
│   └── cli.py              # dantinox CLI (train, generate, find-lr, push, pull, ...)
│
├── core/                   # Internal implementation
│   ├── config.py           # Config dataclass — single source of truth
│   ├── model.py            # Transformer (+ from_pretrained), Block, RMSNorm
│   ├── attention.py        # Attention kernels, Flash Attention, KV-cache logic
│   ├── output.py           # ModelOutput NamedTuple
│   └── generation.py       # Autoregressive inference engine
│
├── utils/
│   ├── tokenizer.py        # CharTokenizer, BPETokenizer, save/load
│   └── helpers.py          # Loss, batching utilities
│
├── configs/
│   ├── default_config.yaml # Standard training setup
│   └── sweep.yaml          # W&B Bayesian sweep configuration
│
├── tests/                  # pytest integration + unit tests
├── examples/
│   └── quickstart.py       # Train → generate end-to-end demo
│
└── pyproject.toml          # pip install -e ".[all]"
```
