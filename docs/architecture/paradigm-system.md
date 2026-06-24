# Paradigm System

The Paradigm abstraction is the central design decision in DantinoX. This page explains the motivation, the contract, and how the four built-in paradigms implement it.

---

## Motivation

A language model can be trained with radically different objectives — cross-entropy next-token prediction, masked diffusion, continuous flow-matching, or contrastive embedding. Naively, each objective requires a different trainer, a different data pipeline, and different generation logic. This creates N × M complexity for N paradigms × M architectural choices.

DantinoX breaks this with a single rule: **the `Paradigm` owns the objective; the `Trainer` owns everything else.** The contract between them is just three methods on `ParadigmBase`:

```python
class ParadigmBase(ABC):
    def build_model(self, rngs)  → model
    def loss_fn(self, model, batch, rng) → (loss, metrics)
    def generate(self, model, *args, **kwargs) → tokens
```

The `Trainer` calls `paradigm.build_model()` once at the start of `fit()`, and then calls `paradigm.loss_fn(model, batch, rng)` at every step — nothing more. Generation is never called during training.

---

## Unified entry point

The public API is a single `Paradigm` class. The `paradigm` key in `ModelConfig` selects which implementation is used internally:

```python
import dantinox as dx

# All four paradigms — one class, one config key
p_ar   = dx.Paradigm(dx.ModelConfig(paradigm="ar",         dim=512, n_heads=8, num_blocks=12))
p_disc = dx.Paradigm(dx.ModelConfig(paradigm="discrete",   dim=512, n_heads=8, num_blocks=12))
p_cont = dx.Paradigm(dx.ModelConfig(paradigm="continuous", dim=512, n_heads=8, num_blocks=12,
                                     embed_dim=768, bottleneck_dim=128))
p_emb  = dx.Paradigm(dx.ModelConfig(paradigm="embedder",   dim=512, n_heads=8, num_blocks=12))
```

`causal` is auto-configured: `"ar"` and `"embedder"` set `causal=True`; `"discrete"` and `"continuous"` set `causal=False`. You never need to write `causal=` when using the `paradigm=` key.

---

## The `ParadigmBase` contract in detail

### `build_model(rngs)`

Called once. Returns an NNX module. The Trainer owns the returned model (checkpoints it, shards it, passes it back into `loss_fn`).

```python
def build_model(self, rngs: nnx.Rngs) -> Any:
    return Transformer(self.config, rngs=rngs)
```

### `loss_fn(model, batch, rng)`

The most important method. Three invariants:

1. **`model` is the first argument** — so `nnx.value_and_grad(loss_fn)(model)` works directly.
2. **Returns `(scalar, dict)`** — the dict holds auxiliary scalars (ce_loss, aux_loss, …) for logging.
3. **Pure function** — no side effects, no stored state, identical behaviour across devices.

```python
# Trainer._step (simplified)
def _loss(m):
    return paradigm.loss_fn(m, batch, rng)

(loss, metrics), grads = nnx.value_and_grad(_loss, has_aux=True)(model)
optimizer.update(grads)
```

### `generate(model, *args, **kwargs)`

Called outside training. Each paradigm implements its own decode strategy:

| Paradigm key | Decode strategy |
| :--- | :--- |
| `"ar"` | Causal autoregressive loop with KV cache |
| `"discrete"` | Iterative unmasking (reverse diffusion) |
| `"continuous"` | ODE integration in embedding space |
| `"embedder"` | Pooled representation (no autoregressive generation) |

### `num_parameters(model)`

A shared helper (not abstract). Counts `nnx.Param` leaves. Override in subclasses if the model has non-standard parameter structures (e.g., `ContinuousParadigm` overrides this for `ELFTransformer`).

---

## `"ar"` — Autoregressive

**Objective:** Standard next-token prediction via cross-entropy on shifted targets (teacher-forcing).

```python
p = dx.Paradigm(dx.ModelConfig(
    paradigm="ar",   # causal=True set automatically
    dim=512, n_heads=8, head_size=64, num_blocks=12,
    vocab_size=32_000,
))
```

**`loss_fn` logic:**

```python
x, y = batch[:, :-1], batch[:, 1:]     # shift by one
out  = model(x)                          # causal Transformer forward
ce   = softmax_cross_entropy(out.logits, y).mean()
return ce + out.aux_loss, {"ce_loss": ce, "aux_loss": out.aux_loss}
```

`out.aux_loss` is the MoE load-balancing term (0.0 for dense models).

---

## `"discrete"` — Masked Diffusion (LLaDA)

**Objective:** (1/t)-weighted cross-entropy on masked positions — the LLaDA training objective.

```python
p = dx.Paradigm(dx.ModelConfig(
    paradigm="discrete",   # causal=False set automatically
    dim=512, n_heads=8, head_size=64, num_blocks=12,
    vocab_size=32_000,
    noise_schedule="cosine",
    mask_token_id=4,
))
```

**`loss_fn` logic:**

```python
t   = Uniform(0, 1) per sample           # corruption level
x_t = corrupt(batch, t, schedule, mask)  # mask tokens with prob p(t)
out = model(x_t)                          # bidirectional Transformer
loss = masked_cross_entropy(out.logits, batch, x_t, mask, t, out.aux_loss)
```

The `(1/t)` weighting up-weights loss on lightly-masked inputs (small `t`) where the task is hardest.

**Noise schedules:**

| Schedule | `p_mask(t)` | Notes |
| :--- | :--- | :--- |
| `"linear"` | `t` | Uniform masking rate |
| `"cosine"` | `(1 - cos(πt/2))` | Cosine curve — lighter masking at low t |
| `"sqrt"` | `√t` | Square-root — heavier masking near t=1 |

---

## `"continuous"` — ELF Flow-Matching

**Objective:** Flow-matching MSE in continuous embedding space.

The forward process interpolates between clean embeddings `x` and Gaussian noise `ε`:

$$z_t = t \cdot x + (1 - t) \cdot \varepsilon, \quad t \sim U(0, 1)$$

The model predicts clean `x` from noisy `z_t` (x-prediction). An auxiliary cross-entropy branch reconstructs discrete tokens.

```python
p = dx.Paradigm(dx.ModelConfig(
    paradigm="continuous",   # causal=False set automatically
    embed_dim=768,           # must match T5-base hidden size
    bottleneck_dim=128,
    dim=512, n_heads=8, head_size=64, num_blocks=12,
    vocab_size=32_128,
))
embedder = p.build_embedder()   # frozen T5 encoder
```

**`loss_fn` requires pre-computed embeddings:**

```python
embeddings = embedder(batch)   # [B, T, embed_dim]
loss, metrics = p.loss_fn(model, batch, rng, embeddings=embeddings)
```

!!! warning "Embeddings are mandatory"
    Unlike AR and Discrete, `ContinuousParadigm.loss_fn` raises `ValueError` if `embeddings=None`.
    Pre-compute them via `ELFEmbedder` before the training loop.

---

## `"embedder"` — Sentence Embedder

**Objective:** InfoNCE contrastive loss over positive (anchor, positive) pairs.

```python
p = dx.Paradigm(dx.ModelConfig(
    paradigm="embedder",   # causal=True set automatically
    dim=512, n_heads=8, head_size=64, num_blocks=12,
    embed_pooling="mean",        # "mean" | "last" | "cls" | "auto"
    embed_temperature=0.05,
))
```

Expects batches with `"anchor_ids"` and `"positive_ids"` keys. The InfoNCE loss pulls together similar sentence pairs and pushes apart the other in-batch examples.

---

## Implementing a custom paradigm

Subclass `ParadigmBase` and implement the four abstract methods:

```python
from dantinox.paradigms.base import ParadigmBase

class MyParadigm(ParadigmBase):
    def build_model(self, rngs):
        return MyModel(self.config, rngs=rngs)

    def loss_fn(self, model, batch, rng, **kwargs):
        logits = model(batch["input_ids"])
        loss   = cross_entropy(logits, batch["labels"])
        return loss, {"loss": loss}

    def generate(self, model, *args, **kwargs):
        return greedy_decode(model, *args, **kwargs)

    def num_parameters(self, model):
        return sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param)))

# Use directly with the Trainer
from dantinox import Trainer, TrainingConfig
run_dir = Trainer(MyParadigm(cfg), TrainingConfig(...)).fit("corpus.txt")
```
