# API Reference

Auto-generated from source docstrings via [mkdocstrings](https://mkdocstrings.github.io/).

---

## High-level API

The `dantinox` package exposes four classes that cover the full lifecycle — training, generation, benchmarking, and plotting — without touching internal modules.

### Trainer

::: dantinox.trainer.Trainer
    options:
      show_source: true
      members:
        - __init__
        - fit

---

### Generator

::: dantinox.generator.Generator
    options:
      show_source: true
      members:
        - __init__
        - generate

---

### BenchmarkRunner

::: dantinox.bench.BenchmarkRunner
    options:
      show_source: true
      members:
        - __init__
        - run

---

### Plotter

::: dantinox.plotting.Plotter
    options:
      show_source: true
      members:
        - __init__
        - run

---

## Core modules

Internal implementation. Import directly when you need low-level access.

### Model architecture

Core Transformer components — `Transformer`, `Block`, `Attention` (MHA/GQA/MLA), `MoE`, and `MLP`.

::: core.model
    options:
      members_order: alphabetical
      show_source: true

---

### Configuration

The `Config` dataclass is the single source of truth for all architectural and training hyperparameters.

::: core.config
    options:
      show_root_heading: true

---

### Generation engine

Autoregressive inference with static KV-cache management, `jax.lax.fori_loop` token loop, and sampling strategies (greedy, Top-K, Top-P).

::: core.generation
    options:
      show_source: true
