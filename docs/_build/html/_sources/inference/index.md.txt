---
title: Inference & Generation
---

# Inference & Generation

DantinoX provides generation interfaces for both paradigms.
The high-level `Generator` class handles AR checkpoints;
`DiffusionTransformer.from_pretrained` + `fast_dllm_generate` handles diffusion.

---

## Paradigm Overview

| | AR | Diffusion |
|---|---|---|
| Entry point | `Generator` or `generate()` | `fast_dllm_generate()` |
| Decoding | Token-by-token (left→right) | Block-wise denoising |
| Streaming | ✓ `Generator.stream()` | — |
| Infilling | Requires re-prompting | Native `[MASK]` support |
| Latency @ BS=1 | Low | Medium |
| Throughput @ large BS | Medium | High |

---

## Pages in this Section

| Page | Description |
| :--- | :--- |
| [AR Generation](autoregressive.md) | KV-cache, streaming, sampling strategies (top-p, top-k, temperature) |
| [Diffusion Generation](diffusion.md) | Simple MDLM sampler, Fast-dLLM DualCache, infilling |
| [KV-Cache](kv-cache.md) | Static pre-allocation, MHA vs. GQA vs. MLA memory profiles |
