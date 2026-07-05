---
title: Inference & Generation
---

# Inference & Generation

DantinoX provides generation interfaces for all three paradigms, dispatched
through one `Generator` class. `Generator.stream()` auto-routes to KV-cache
decoding for AR checkpoints, `fast_dllm_generate` (or the simple MDLM
sampler) for discrete diffusion, and Euler ODE/SDE integration
(`flow_generate`) for continuous flow-matching — based on the
`ModelConfig.paradigm` field saved in the checkpoint.

---

## Paradigm Overview

| | AR | Discrete Diffusion | Continuous Flow-Matching |
|---|---|---|---|
| Entry point | `Generator` or `generate()` | `fast_dllm_generate()` | `flow_generate()` |
| Decoding | Token-by-token (left→right) | Block-wise denoising | Euler ODE / SDE integration |
| Streaming | ✓ `Generator.stream()` | ✓ `Generator.stream()` | ✓ `Generator.stream()` (in-place rewrite, not token-by-token) |
| Infilling / conditioning | Requires re-prompting | Native `[MASK]` support | Not yet supported (unconditional only) |
| Cache | Static KV-cache | DualCache (prefix + suffix KV) | None — no discrete state to cache |
| Latency @ BS=1 | Low | Medium | Medium–high (scales with `n_steps`) |
| Throughput @ large BS | Medium | High | High |

---

## Pages in this Section

| Page | Description |
| :--- | :--- |
| [AR Generation](autoregressive.md) | KV-cache, streaming, sampling strategies (top-p, top-k, temperature) |
| [Diffusion Generation](diffusion.md) | Simple MDLM sampler, Fast-dLLM DualCache, infilling |
| [Continuous Flow-Matching Generation](continuous.md) | Euler ODE/SDE sampling, Classifier-Free Guidance, `flow_generate` |
| [KV-Cache](kv-cache.md) | Static pre-allocation, MHA vs. GQA vs. MLA memory profiles |
