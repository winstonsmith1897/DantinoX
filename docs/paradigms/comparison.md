---
title: AR vs Diffusion Comparison
---

# Autoregressive vs Masked Diffusion — Head-to-Head

Both paradigms are trained on the same corpus with the same architecture.
This page summarises the expected tradeoffs based on DantinoX's systematic
benchmark suite across three attention types (MHA · GQA · MLA).

---

## Architecture Differences

| | AR (`Transformer`) | Diffusion (`DiffusionTransformer`) |
|---|---|---|
| Class | `core.model.Transformer` | `core.model.DiffusionTransformer` |
| Block type | `ARBlock` | `DiffusionBlock` |
| Attention mask | Causal | Full (bidirectional) |
| Time conditioning | — | `AdaLayerNorm` (DiT-style) |
| Extra parameters | — | `TimeEmbedding` MLP |
| KV-cache type | Static KV | DualCache (prefix + suffix) |
| Decode step cost | $O(T_{\text{gen}})$ | $O(T_{\text{gen}})$ (block-wise) |
| Total decode cost | $O(T_{\text{gen}}^2)$ | $O(K \cdot B \cdot N_{\text{steps}})$ |

---

## Quality

### Language Modelling (bpb on WikiText-103)

Evaluated with sliding-window evaluation;
AR uses standard CE, Diffusion uses ELBO at a uniform timestep grid.

| Model | AR bpb ↓ | Diffusion ELBO-bpb ↓ |
|---|---|---|
| MHA 256d 12b Dense | — | — |
| GQA 256d 12b Dense | — | — |
| MLA 256d 12b Dense | — | — |

!!! note "Populating this table"
    Run `python benchmarks/perplexity_eval.py` after training to populate these results.
    Output is written to `results/perplexity.csv`. See [Benchmarks](../benchmarks.md) for the full pipeline.

### Long-Range Coherence (LAMBADA accuracy)

LAMBADA tests last-word prediction given a long paragraph. The bidirectional
Diffusion model sees the *full context* including right-of-target tokens,
giving it a structural advantage over causal AR.

Expected finding: Diffusion bpb on LAMBADA ≤ AR bpb at the same parameter
count, with the gap widening for longer contexts.

---

## Efficiency

### Throughput (tok/s, BS=1, 256-token generation)

| | AR | Diffusion (simple) | Diffusion (DualCache) |
|---|---|---|---|
| MHA 256d 12b | — | — | — |
| GQA 256d 12b | — | — | — |
| MLA 256d 12b | — | — | — |

!!! note "Populating this table"
    Run `python benchmarks/trained_analysis.py --runs-dir runs --run-prefix ar_ diff_` to populate throughput results.

### KV-Cache Memory

AR and Diffusion share the same attention mechanism, so KV-cache footprint
is **identical** for the same attention type at inference time.

| Attention | KV-MB @ 512 tok, 12L |
|---|---|
| MHA | 384 KB |
| GQA (×4) | 96 KB |
| MLA | ~23 KB |

For Diffusion with DualCache, the **suffix KV** adds overhead proportional to
the number of remaining MASK blocks.  Averaged across a full generation this
adds ~20–40% to the peak cache size.

---

## When to Use Which

### Use Autoregressive when:

- Latency-critical **streaming** generation (token-by-token output).
- Tasks where **left-to-right** coherence matters (story generation, code completion).
- Simple deployment: no diffusion steps, no noise schedule.
- Integrating with existing AR pipelines and sampling libraries.

### Use Masked Diffusion when:

- **Infilling / editing**: native `[MASK]` tokens make targeted completion trivial.
- **Long-range coherence**: bidirectional context helps with LAMBADA-style tasks.
- **Parallel generation**: all positions decode simultaneously — higher throughput
  at large batch sizes.
- Research into non-autoregressive discrete generation.

---

## Side-by-Side Code

```python
from core.config import Config
from core.model import Transformer, DiffusionTransformer
import flax.nnx as nnx

base = dict(
    dim=256, n_heads=8, head_size=32, num_blocks=12,
    max_context=512, kv_heads=2, vocab_size=100,
)

# ── Autoregressive ────────────────────────────────────────────────────────
ar_cfg   = Config(**base, model_type="autoregressive")
ar_model = Transformer(ar_cfg, rngs=nnx.Rngs(0))

from core.generation import generate
tokens_ar = generate(ar_model, prompt_ids, max_generations=128,
                     top_p=0.9, use_cache=True)

# ── Masked Diffusion ──────────────────────────────────────────────────────
diff_cfg   = Config(**base, model_type="diffusion",
                    diffusion_steps=1000, noise_schedule="cosine")
diff_model = DiffusionTransformer(diff_cfg, rngs=nnx.Rngs(0))

from core.generation import fast_dllm_generate
from core.diffusion import make_noise_schedule

schedule    = make_noise_schedule(diff_cfg)
tokens_diff = fast_dllm_generate(
    diff_model, prompt_ids, gen_len=128,
    schedule=schedule, mask_token_id=0,
    block_size=32, steps_per_block=20,
    confidence_threshold=0.9,
)
```

---

## Training Configuration Differences

The only config changes needed to switch paradigms:

```yaml
# Autoregressive (default_config.yaml)
model:
  model_type: "autoregressive"

# Masked Diffusion (diffusion_base.yaml)
model:
  model_type: "diffusion"
diffusion:
  diffusion_steps: 1000
  noise_schedule: "cosine"
  mask_token_id: 0
  num_sampling_steps: 50
  time_emb_dim: 256
```

All other fields — `dim`, `n_heads`, `num_blocks`, `kv_heads`, `mla`, `use_moe` —
are shared.  This enables **controlled comparisons** where the only variable
is the paradigm.
