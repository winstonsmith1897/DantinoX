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

-   :material-package-variant-closed: &nbsp;**pip-Installable Library**

    ---

    `Trainer`, `Generator`, `BenchmarkRunner`, and `Plotter` classes expose the full experiment lifecycle programmatically. One `dantinox` CLI covers train, generate, sweep, benchmark, and plot.

    [:octicons-arrow-right-24: API Reference](api.md)

</div>

---

## Quickstart

=== "Library (Python API)"

    ```bash
    git clone https://github.com/winstonsmith1897/DantinoX.git
    cd DantinoX

    conda create -n dantinox python=3.12 -y && conda activate dantinox
    pip install -U "jax[cuda12]"
    pip install -e ".[all]"
    ```

    ```python
    from dantinox import Config, Trainer, Generator, BenchmarkRunner
    from dantinox.plotting import Plotter

    # 1. Train
    config = Config.from_yaml("configs/default_config.yaml")
    run_dir = Trainer(config).fit("data/corpus.txt")

    # 2. Generate
    text = Generator(run_dir).generate("Nel mezzo del cammin ")
    print(text)

    # 3. Benchmark all runs, then plot
    df = BenchmarkRunner("runs").run(out_csv="benchmark_results.csv")
    Plotter("benchmark_results.csv").run()
    ```

=== "CLI"

    ```bash
    # Train
    dantinox train --config configs/default_config.yaml --data_path data/corpus.txt

    # Generate
    dantinox generate --run_dir runs/<run_name> --prompt "Nel mezzo del cammin "

    # Sweep (W&B Bayesian)
    dantinox sweep --sweep_config configs/sweep.yaml --data_path data/corpus.txt

    # Benchmark, then generate all plots
    dantinox benchmark --runs_dir runs --out_csv benchmark_results.csv
    dantinox plot --in_csv benchmark_results.csv --out_dir plots/
    ```

=== "Scripts (legacy)"

    ```bash
    python train.py    --config configs/default_config.yaml
    python generate.py --run_dir runs/<run_name> --prompt "Nel mezzo del cammin "
    python benchmark.py
    ```

---

## Documentation

| | Page | What you'll find |
| :--- | :--- | :--- |
| :material-layers-outline: | [Core Architecture](architecture.md) | Attention types, math, full configuration reference, implementation deep-dives |
| :material-school-outline: | [Training & Sweeps](training.md) | Training loop, W&B sweep setup, MLA training notes |
| :material-play-box-outline: | [Inference & Generation](generation.md) | KV-cache pipeline, sampling strategies, MLA inference mode |
| :material-chart-scatter-plot: | [Benchmarks](benchmarks.md) | MHA vs GQA vs MLA — throughput, cache size, FLOPs, 3D surfaces |
| :material-microscope: | [Ablation Studies](ablation_studies.md) | Optimizer, MoE, positional encoding, regularization |
| :material-book-open-outline: | [API Reference](api.md) | `Trainer`, `Generator`, `BenchmarkRunner`, `Plotter`, and core modules |

---

## Project Structure

```text
DantinoX/
├── dantinox/               # Public library API
│   ├── __init__.py         # Top-level imports
│   ├── trainer.py          # Trainer — programmatic training
│   ├── generator.py        # Generator — checkpoint loading + generation
│   ├── bench.py            # BenchmarkRunner — throughput / FLOPs benchmarks
│   ├── plotting.py         # Plotter — automated plot generation
│   └── cli.py              # dantinox CLI entry point
│
├── core/                   # Internal implementation
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
├── plot_insights.py        # Insight figures (Pareto, serving, MLA dial)
├── plot_perf.py            # Performance figures (cache, FLOPs, throughput)
├── plot_3d.py              # 3D surface figures
├── plot_3d_dkv.py          # down_dim_kv sensitivity figures
│
├── pyproject.toml          # pip install -e ".[all]"
└── requirements.txt
```
