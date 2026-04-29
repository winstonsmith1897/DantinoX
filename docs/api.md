# API Reference

Technical documentation for DantinoX core modules. These docs are automatically generated from source docstrings via [mkdocstrings](https://mkdocstrings.github.io/).

## Model Architecture

Core Transformer components implemented in Flax NNX — `Transformer`, `Block`, `Attention` (MHA/GQA/MLA), `MoE`, and `MLP`.

::: core.model
    options:
      members_order: alphabetical
      show_source: true

---

## Configuration

The `Config` dataclass is the single source of truth for all architectural and training hyperparameters. Pass it to `Transformer.__init__` and to the training/generation scripts.

::: core.config
    options:
      show_root_heading: true

---

## Generation

Autoregressive inference engine. Handles static KV-cache management, `jax.lax.fori_loop` token generation, and all sampling strategies (greedy, Top-K, Top-P).

::: core.generation
    options:
      show_source: true
