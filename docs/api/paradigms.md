# `dantinox.paradigms`

Paradigms define the training objective and generation strategy. The `Trainer` only ever calls `loss_fn` — all paradigm-specific logic is self-contained.

## Overview

A paradigm is a self-contained unit that owns:

1. **Model construction** — `build_model()` returns the JAX/NNX model for this paradigm.
2. **Loss function** — `loss_fn(model, batch)` returns `(loss, metrics)`. The `Trainer` never touches the model directly — it calls this.
3. **Generation** — `generate()` wraps the model-specific decode loop.
4. **Parameter count** — `num_parameters(model)`.

```
┌─────────────┐       loss_fn(model, batch)        ┌────────────┐
│   Trainer   │  ──────────────────────────────►   │  Paradigm  │
│             │  ◄── (loss: float, metrics: dict) ──│            │
└─────────────┘                                     └────────────┘
```

---

## Unified `Paradigm` API

The recommended entry point is the single `Paradigm` class. Pass a `ModelConfig` with a `paradigm` key and the right implementation is selected automatically:

```python
import dantinox as dx

# Autoregressive (causal=True set automatically)
p = dx.Paradigm(dx.ModelConfig(paradigm="ar", dim=512, n_heads=8, num_blocks=12))

# Masked Diffusion / LLaDA (causal=False set automatically)
p = dx.Paradigm(dx.ModelConfig(paradigm="discrete", dim=512, n_heads=8, num_blocks=12,
                                noise_schedule="cosine", mask_token_id=4))

# ELF Continuous Flow-Matching (causal=False set automatically)
p = dx.Paradigm(dx.ModelConfig(paradigm="continuous", dim=256, n_heads=4,
                                embed_dim=768, bottleneck_dim=128, num_blocks=6))

# Sentence Embedder
p = dx.Paradigm(dx.ModelConfig(paradigm="embedder", dim=512, n_heads=8, num_blocks=12,
                                embed_pooling="mean", embed_temperature=0.05))
```

### Paradigm selection table

| `config.paradigm` | Implementation | `causal` auto-set | Model class |
|---|---|---|---|
| `"ar"` | `ARParadigm` | `True` | `Transformer` (causal) |
| `"discrete"` | `DiscreteParadigm` | `False` | `DiffusionTransformer` |
| `"continuous"` | `ContinuousParadigm` | `False` | `ELFTransformer` |
| `"embedder"` | `EmbedderParadigm` | `True` | `Transformer` (pooled) |

When `paradigm=None` (omitted), the implementation is auto-detected from `causal` and `embed_dim` for backward compatibility.

---

## `Paradigm` reference

::: dantinox.paradigms.paradigm.Paradigm
    options:
      show_source: true
      members:
        - __init__
        - build_model
        - loss_fn
        - generate
        - stream
        - type

---

## Base class

`ParadigmBase` is the abstract base class. Subclass it to implement a custom paradigm:

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
```

::: dantinox.paradigms.base.ParadigmBase
    options:
      show_source: true
      members:
        - build_model
        - loss_fn
        - generate
        - num_parameters

---

## Concrete implementations

The implementations below are directly importable for advanced use cases or when subclassing. In normal use, `Paradigm` wraps them transparently.

### Autoregressive

::: dantinox.paradigms.ar.ARParadigm
    options:
      show_source: true
      members:
        - __init__
        - build_model
        - loss_fn
        - generate

---

### Discrete Diffusion (LLaDA)

::: dantinox.paradigms.diffusion.discrete.DiscreteParadigm
    options:
      show_source: true
      members:
        - __init__
        - build_model
        - loss_fn
        - generate

---

### Continuous Flow-Matching (ELF)

::: dantinox.paradigms.diffusion.continuous.ContinuousParadigm
    options:
      show_source: true
      members:
        - __init__
        - build_model
        - build_embedder
        - loss_fn
        - generate
        - num_parameters

---

### Sentence Embedder

::: dantinox.paradigms.embedder.EmbedderParadigm
    options:
      show_source: true
      members:
        - __init__
        - build_model
        - loss_fn
        - generate

---

## Usage examples

### Standard AR training

```python
import dantinox as dx
from flax import nnx

cfg      = dx.ModelConfig(paradigm="ar", dim=256, n_heads=8, head_size=32,
                           num_blocks=6, vocab_size=200)
paradigm = dx.Paradigm(cfg)
model    = paradigm.build_model(nnx.Rngs(42))

print(f"Parameters: {paradigm.num_parameters(model) / 1e6:.2f}M")
print(f"Type: {paradigm.type}")   # → "ar"

# Single train step
import jax.numpy as jnp
batch = {"input_ids": jnp.ones((4, 64), dtype=jnp.int32)}
loss, metrics = paradigm.loss_fn(model, batch, rng=nnx.Rngs(0))
```

### Discrete diffusion

```python
import dantinox as dx

cfg      = dx.ModelConfig(
    paradigm="discrete",          # causal=False auto-configured
    dim=256, n_heads=8, head_size=32,
    num_blocks=6, vocab_size=32000,
    noise_schedule="cosine",
    mask_token_id=4,
)
paradigm = dx.Paradigm(cfg)
model    = paradigm.build_model(nnx.Rngs(42))
```

### Continuous flow-matching (ELF)

```python
import dantinox as dx

cfg = dx.ModelConfig(
    paradigm="continuous",        # causal=False auto-configured
    embed_dim=768,                # must match T5-base hidden size
    bottleneck_dim=128,
    dim=512, n_heads=8, head_size=64,
    num_blocks=6, vocab_size=32128,
)
paradigm = dx.Paradigm(cfg)
model    = paradigm.build_model(nnx.Rngs(42))
embedder = paradigm.build_embedder()   # frozen T5 oracle
```

### Sentence embedder (InfoNCE)

```python
import dantinox as dx

cfg = dx.ModelConfig(
    paradigm="embedder",          # causal=True auto-configured
    dim=512, n_heads=8, head_size=64,
    num_blocks=12,
    embed_pooling="mean",         # "mean" | "last" | "cls" | "auto"
    embed_temperature=0.05,
)
paradigm = dx.Paradigm(cfg)
```

---

## See also

- [Configuration Reference](../configuration.md) — all config fields including `paradigm`, `embed_pooling`, `embed_temperature`
- [Architecture: Paradigm System](../architecture/paradigm-system.md) — conceptual explanation
- [Training API](training.md) — how `Trainer` calls `loss_fn`
