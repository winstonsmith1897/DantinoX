---
title: Generation Paradigms
---

# Generation Paradigms

DantinoX implements two orthogonal generation paradigms on the same transformer backbone.
Each paradigm can be combined freely with any of the three attention mechanisms
(MHA · GQA · MLA) and the two FFN variants (Dense · MoE).

---

## At a Glance

| | **Autoregressive (AR)** | **Masked Diffusion** |
|---|---|---|
| Directionality | Causal (left → right) | Bidirectional |
| Decoding | Sequential token-by-token | Parallel denoising over full sequence |
| Attention mask | Causal | None (full attention) |
| Training loss | Cross-entropy on next token | Masked CE over [MASK] positions |
| KV-cache type | Static KV-cache | DualCache (prefix + suffix) |
| Conditioning | Time-step embedding | None |
| Parallelism | Low (sequential decode) | High (all positions at once) |
| Long-range coherence | ✗ unidirectional | ✓ bidirectional context |
| Infilling | Requires re-prompting | Native |

---

## Paradigm Selection

The paradigm is chosen by a single config field:

```yaml
model:
  model_type: "autoregressive"  # or "diffusion"
```

```python
from core.config import Config
from core.model import Transformer, DiffusionTransformer

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
