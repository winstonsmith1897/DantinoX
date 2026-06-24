---
title: Generation Paradigms
---

# Generation & Retrieval Paradigms

DantinoX implements three paradigms on the same transformer backbone.
Each paradigm can be combined freely with any of the three attention mechanisms
(MHA · GQA · MLA) and the two FFN variants (Dense · MoE).

---

## At a Glance

| | **Autoregressive (AR)** | **Masked Diffusion** | **Retriever (Embedder)** |
|---|---|---|---|
| Directionality | Causal (left → right) | Bidirectional | Bidirectional (recommended) |
| Primary use | Text generation | Text generation | Semantic search / RAG |
| Training loss | Cross-entropy on next token | Masked CE over [MASK] positions | InfoNCE (symmetric) |
| Output | Token logits `[B, T, V]` | Token logits `[B, T, V]` | Sentence vectors `[B, D]` |
| Attention mask | Causal | None (full attention) | None (full attention) |
| Pooling | — | — | mean · last · cls |
| FAISS / ChromaDB | — | — | ✓ drop-in |
| LangChain | — | — | ✓ drop-in |

---

## Paradigm Selection

The paradigm is chosen by a single config field:

```yaml
model:
  model_type: "autoregressive"  # or "diffusion"
```

```python
from dantinox.core.config import Config
from dantinox.core.model import Transformer, DiffusionTransformer

ar_config   = Config(model_type="autoregressive", ...)
diff_config = Config(model_type="diffusion",       ...)

ar_model   = Transformer(ar_config,            rngs=nnx.Rngs(0))
diff_model = DiffusionTransformer(diff_config, rngs=nnx.Rngs(0))
```

---

## Pages in this Section

| Page | Description |
| :--- | :--- |
| [Autoregressive](autoregressive.md) | Causal transformer with KV-cache and streaming generation |
| [Masked Diffusion](diffusion.md) | MDLM-style forward/reverse process, noise schedules, ELBO loss |
| [Fast-dLLM DualCache](fast-dllm.md) | Block-wise inference with prefix + suffix KV caching for diffusion |
| [Confidence-Aware Decoding](confidence.md) | Threshold and factor strategies for parallel token unmasking |
| [AR vs. Diffusion](comparison.md) | Side-by-side quality, efficiency, and use-case analysis |
| [Retriever (Embedder)](retriever.md) | SimCSE / InfoNCE contrastive training for RAG sentence encoders |
