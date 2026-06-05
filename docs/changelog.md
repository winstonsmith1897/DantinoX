---
title: Changelog
---

# Changelog

All notable changes to DantinoX are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
DantinoX uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.0] — 2026-06-05

### Added

- **Masked Diffusion Language Model** — `DiffusionTransformer`, `DiffusionBlock`, `AdaLayerNorm`, cosine/linear/sqrt noise schedules, masked cross-entropy loss (`core/diffusion.py`)
- **Fast-dLLM DualCache** — block-wise denoising with a prefix KV-cache and suffix KV-cache, reducing diffusion decode latency by ~2.1× over the naive sampler
- **Confidence-Aware Decoding** — token-unmasking strategies based on per-position confidence thresholds and linear/exponential factor schedules
- **`model_type` config field** — `"autoregressive"` (default) or `"diffusion"`; a single YAML change switches the full training and inference pipeline
- **`attention_type` config field** — explicit `"mha"` / `"gqa"` / `"mla"` selector; resolves automatically from legacy `mla` and `kv_heads` flags for backward compatibility
- **Three noise schedules** — `"cosine"` (default), `"linear"`, `"sqrt"` configurable via `noise_schedule`
- **Diffusion training docs** — `docs/training/diffusion.md`, `docs/paradigms/diffusion.md`, `docs/paradigms/fast-dllm.md`, `docs/paradigms/confidence.md`
- **Tutorial section** — four step-by-step guides: Training Your First Model, LoRA Fine-Tuning, Masked Diffusion LM, Pushing to HuggingFace Hub

### Changed

- `Transformer.from_pretrained` now accepts both local paths and HuggingFace Hub repository IDs
- `Generator` constructor accepts a `token` argument for private HuggingFace Hub repositories
- `Config.__post_init__` validates `model_type` and `attention_type`; raises `ValueError` on unknown values
- Benchmark suite extended to cover AR vs. Diffusion throughput comparison

### Fixed

- Static KV-cache initialisation now correctly handles `batch_size > 1` in MLA mode
- `dantinox find-lr` no longer overwrites an existing run directory when `--run_dir` is specified

---

## [0.1.0] — 2026-01-15

### Added

- **Core Transformer** (`core/model.py`) — Autoregressive Transformer with MHA, GQA, and Multi-Head Latent Attention (MLA)
- **MLA** — latent KV compression, decoupled RoPE, weight absorption at decode time (DeepSeek-V2 style)
- **Static KV-cache** — `jax.lax.dynamic_update_slice` for O(1) writes; zero recompilation across decode steps
- **Flash Attention** — opt-in `jax.nn.dot_product_attention` fast path for MHA/GQA training (JAX ≥ 0.4.25)
- **Sparse MoE** — top-K router with load-balancing auxiliary loss (`alpha_balance`)
- **LoRA fine-tuning** — `LoRALinear`, `LoRAParam` type-level freezing, `merge_weights()` for deployment
- **NTK-aware RoPE scaling** — `rope_scale_factor` compresses base frequency for long-context extrapolation
- **Sliding Window Attention** — local causal window via `context_window` config field
- **Attention gating** (`no_sink`) — sigmoid gate on attention output to prevent attention sink patterns
- **`Trainer`** — full training loop: bfloat16, gradient accumulation, gradient clipping, early stopping, LR finder, W&B logging, resume from checkpoint
- **Four LR schedules** — `"cosine"`, `"linear"`, `"constant"`, `"wsd"` (warmup → stable → cosine decay)
- **Three optimisers** — AdamW, Adafactor, Lion via Optax
- **`Generator`** — single, batched, and streaming autoregressive generation with top-k, top-p, temperature sampling
- **Multi-GPU SPMD** — data-parallel training via `jax.sharding.Mesh`; set `n_devices=N`
- **HuggingFace Hub integration** — `push()` / `pull()` in `dantinox/hub.py`; `Generator("owner/repo")` direct loading
- **CLI** — `dantinox train`, `generate`, `find-lr`, `sweep`, `benchmark`, `push`, `pull`, `plot`
- **`BenchmarkRunner`** — throughput, FLOPs, KV-cache profiling across attention types
- **CharTokenizer** and **BPETokenizer** — `tokenizer_type: "char"` or `"bpe"` in config
- **Gradient checkpointing** — `nnx.remat` for activation recomputation (disabled automatically during inference)
- **`ModelOutput` NamedTuple** — named access to `logits`, `kv_caches`, `aux_loss`; backward-compatible with positional unpacking
- **86 pytest tests** — unit and integration coverage; mypy clean, ruff clean
- **ReadTheDocs** — MkDocs Material documentation with full API reference, architecture docs, and benchmark results

[0.2.0]: https://github.com/winstonsmith1897/DantinoX/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/winstonsmith1897/DantinoX/releases/tag/v0.1.0
