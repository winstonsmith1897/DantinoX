---
title: Continuous Flow-Matching Training
---

# Continuous Flow-Matching Training

Set `paradigm: "continuous"` (`ModelConfig`) or `model_type: "elf"` (legacy
`Config`) to train a continuous flow-matching model following the ELF recipe
(Hu et al., 2026). Unlike AR or discrete diffusion, this paradigm trains in
the embedding space of a **frozen T5 encoder**, not directly on token IDs.

---

## Loss Function

Each training example is stochastically routed to one of two branches
(`config.denoiser_prob` controls the split), and the batch-weighted sum of
both branch losses is the final loss (`flow_loss`, ELF Algorithm 1):

**Denoiser branch** — flow-matching MSE with training-time
Classifier-Free-Guidance regression and self-conditioning (ELF Algorithm 3):

$$
z_t = t \cdot x + (1-t)\cdot\varepsilon, \quad t \sim \text{LogitNormal}(p_{\text{mean}}, p_{\text{std}})
$$

Two forward passes compute an unconditioned velocity $v_{\text{no\_sc}}$ and a
self-conditioned velocity $v_{\text{sc}}$ (conditioned on
`stop_gradient` of the first pass's prediction); the CFG regression target is

$$
v_{\text{target}} = v + \left(1 - \frac{1}{w}\right)(v_{\text{sc}} - v_{\text{no\_sc}})
$$

and the loss is `flow_mse_loss(v_pred, v_target)`, with `v_pred` randomly
taken from the self-conditioned or unconditioned pass per
`config.self_cond_prob`.

Every `model(z_t, x_prev, t, w, is_decode)` call returns a
`dantinox.core.output.FlowMatchingOutput` NamedTuple with two fields:
`x_pred` (predicted clean embedding, used to derive the velocity `v` for the
MSE loss above) and `logits` (token logits via the shared unembedding head,
used for the decoder branch's cross-entropy below). `ELFOutput` is a
deprecated alias of the same type.

**Decoder branch** — cross-entropy reconstruction of discrete tokens from a
per-token-corrupted embedding, run in decode mode (ELF Algorithm 4):

$$
\mathcal{L}_{\text{dec}} = \text{CrossEntropy}(\text{logits}, \text{tokens})
$$

At every step, the model runs in **bidirectional** attention and is
conditioned on the timestep $t$, CFG scale $w$, and branch (denoiser/decoder)
via in-context control tokens — not via `AdaLayerNorm` (that mechanism is
used nowhere in DantinoX; see [Discrete Diffusion Training](diffusion.md)
for the equivalent clarification on that paradigm).

---

## Quick Start

```python
import dantinox as dx
from flax import nnx

cfg = dx.ModelConfig(
    paradigm="continuous",   # causal=False set automatically
    embed_dim=768,           # must match the T5 encoder's hidden size
    bottleneck_dim=128,
    dim=512, n_heads=8, head_size=64, num_blocks=12,
    vocab_size=32_128,
    flow_n_steps=64,         # ODE integration steps at inference
    flow_cfg_scale=1.5,      # Classifier-Free Guidance scale
    # NOT elf_n_steps=/elf_cfg_scale= — those are read-only compatibility
    # properties, not constructor arguments.
)

paradigm = dx.Paradigm(cfg)
model    = paradigm.build_model(nnx.Rngs(0))
embedder = paradigm.build_embedder(nnx.Rngs(0))   # frozen T5 encoder
```

Training requires the optional `transformers` dependency for the T5 encoder:
`pip install dantinox[elf]`.

---

## Training loop internals

`ContinuousParadigm` is one of the few paradigms with
`provides_batch_extras = True` — it needs data *besides* raw token IDs
(the T5 embeddings), obtained through two extra `Paradigm` hooks the
`Trainer` calls automatically:

```python
# Trainer._step (simplified)
embeddings = paradigm.prepare_batch(batch)          # frozen T5 forward, outside JIT
loss, metrics = paradigm.loss_fn(model, batch, rng, embeddings=embeddings)
```

- **`prepare_batch(batch)`** runs the frozen T5 encoder (`T5ContextualEncoder`)
  on the raw token batch to produce `[B, T, embed_dim]` embeddings. This runs
  *outside* the JIT-compiled train step so T5's own large graph isn't
  retraced every step.
- **`on_train_start(model, sample_batches)`** is called once before training
  begins: it computes channel-wise mean/std statistics from a handful of
  real T5 outputs and stores them on `model.embedder.emb_mean` /
  `model.embedder.emb_std`, so `model.encode(...)` can normalise every batch
  to a consistent scale during training.
- **`loss_fn(model, batch, rng, embeddings=...)`** raises `ValueError` if
  `embeddings` is `None` — the paradigm cannot compute its loss from raw
  tokens alone.

```python
run_dir = dx.Trainer(paradigm, dx.TrainingConfig(lr=1e-4, epochs=10)).fit("data/wiki.txt")
```

---

## Config Reference

Fields specific to this paradigm (shared architecture fields — `dim`,
`n_heads`, `num_blocks`, attention/norm/FFN toggles — are the same as
`ModelConfig`; see [Configuration Reference](../configuration.md#flowmatchingconfig)
for the complete list):

| Field | Default | Description |
|:------|:-------:|:------------|
| `embed_dim` | `512` | T5 embedding / flow-space dimension. Must match the chosen `t5_model_name` (768 for `t5-base`). |
| `bottleneck_dim` | `128` | Bottleneck between the embedding space and the transformer's hidden dimension. |
| `t5_model_name` | `"t5-base"` | Frozen HuggingFace T5 encoder used to produce embeddings. |
| `flow_n_steps` | `32` | Euler ODE integration steps at inference (not `elf_n_steps` — deprecated read-only alias). |
| `flow_cfg_scale` | `1.5` | Classifier-Free Guidance scale at inference (not `elf_cfg_scale` — deprecated read-only alias). |
| `sde_gamma` | `1.0` | SDE noise re-injection scale during generation (`0.0` = deterministic ODE). |
| `denoiser_prob` | `0.8` | Fraction of training examples routed to the denoiser branch vs. the decoder branch. |
| `self_cond_prob` | `0.5` | Probability of using the self-conditioned velocity as the training target. |
| `cfg_scale_min` / `cfg_scale_max` | `0.5` / `5.0` | Range of CFG scale `w` sampled during training. |
| `denoiser_pmean` / `denoiser_pstd` | `-1.5` / `0.8` | Logit-normal time-sampling params for the denoiser branch. |
| `decoder_pmean` / `decoder_pstd` | `0.8` / `0.8` | Logit-normal per-token corruption sampling for the decoder branch. |

---

## Monitoring Training

`flow_loss` returns both branch losses in its metrics dict
(`den_loss`, `dec_loss`) in addition to the combined scalar loss logged to
`training_log.csv`. A decreasing `den_loss` means the model's velocity field
prediction is improving; a decreasing `dec_loss` means token reconstruction
from noised embeddings is improving — a well-behaved run should see both
decrease together.

!!! note "Not directly comparable to AR/Diffusion loss"
    Flow-matching MSE + CE is not on the same scale as AR next-token
    cross-entropy or diffusion masked ELBO. Use the paper's generation-quality
    metrics (MAUVE, PPL, Distinct-2) — see
    [Comparison — Paper's reported results](../paradigms/comparison.md#papers-reported-results-authoritative)
    — for a fair cross-paradigm comparison, not raw training loss.

---

## Checkpoint Loading and Generation

```python
model = dx.load(run_dir, paradigm=paradigm)
tokens = paradigm.generate(model, max_new_tokens=128, n_steps=64, cfg_scale=1.5,
                            rng=jax.random.PRNGKey(0))
```

See [Continuous Flow-Matching Inference](../inference/continuous.md) for the
full generation API (Euler ODE vs. SDE sampling, streaming, decoding to text).

---

## Current limitation: no prefix conditioning

Training and generation are both **unconditional** — there is no supported
way to condition generation on a prompt/prefix the way AR or discrete
diffusion can. The ELF formulation supports prefix conditioning in
principle, but DantinoX's current implementation does not yet expose it;
this is why conditional-generation metrics (BLEU-4<sub>cond</sub>) are not
reported for this paradigm in the paper's evaluation.
