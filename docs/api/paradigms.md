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

### Adding a custom paradigm

Subclass `Paradigm` and implement the four methods:

```python
from dantinox.paradigms.base import Paradigm

class MyParadigm(Paradigm):
    def build_model(self, rngs):
        return MyModel(self.config, rngs=rngs)

    def loss_fn(self, model, batch):
        logits = model(batch["input_ids"])
        loss   = cross_entropy(logits, batch["labels"])
        return loss, {"loss": loss}

    def generate(self, model, prompt, **kwargs):
        return greedy_decode(model, prompt, **kwargs)

    def num_parameters(self, model):
        return sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param)))
```

---

## Paradigm selection

| `config.model_type` | Paradigm class | Model class |
|---|---|---|
| `"autoregressive"` | `ARParadigm` | `Transformer` (causal) |
| `"diffusion"` | `DiscreteParadigm` | `DiffusionTransformer` |
| `"elf"` | `ContinuousParadigm` | `ELFTransformer` |

---

## Base

::: dantinox.paradigms.base.Paradigm
    options:
      show_source: true
      members:
        - build_model
        - loss_fn
        - generate
        - num_parameters

---

## Autoregressive

::: dantinox.paradigms.ar.ARParadigm
    options:
      show_source: true
      members:
        - __init__
        - build_model
        - loss_fn
        - generate

---

## Discrete Diffusion (LLaDA)

::: dantinox.paradigms.diffusion.discrete.DiscreteConfig
    options:
      show_source: true

::: dantinox.paradigms.diffusion.discrete.DiscreteParadigm
    options:
      show_source: true
      members:
        - __init__
        - build_model
        - loss_fn
        - generate

---

## Continuous Flow-Matching (ELF)

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

## Usage examples

### ARParadigm

```python
from dantinox.core.config import ModelConfig
from dantinox.paradigms.ar import ARParadigm
from flax import nnx

cfg      = ModelConfig(dim=256, n_heads=8, head_size=32, num_blocks=6, vocab_size=200)
paradigm = ARParadigm(cfg)
model    = paradigm.build_model(nnx.Rngs(42))

print(f"Parameters: {paradigm.num_parameters(model) / 1e6:.2f}M")

# Single train step
import jax.numpy as jnp
batch  = {"input_ids": jnp.ones((4, 64), dtype=jnp.int32)}
loss, metrics = paradigm.loss_fn(model, batch)
```

### DiscreteParadigm

```python
from dantinox.core.config import Config
from dantinox.paradigms.diffusion.discrete import DiscreteConfig, DiscreteParadigm
from flax import nnx

model_cfg = ModelConfig(dim=256, n_heads=8, head_size=32,
                        num_blocks=6, vocab_size=32000, causal=False)
diff_cfg  = DiscreteConfig(
    diffusion_steps=1000,
    noise_schedule="cosine",
    mask_token_id=32099,
    num_sampling_steps=50,
)
paradigm = DiscreteParadigm(model_cfg, diff_cfg)
model    = paradigm.build_model(nnx.Rngs(42))
```

### ContinuousParadigm (ELF)

```python
from dantinox.core.config import ELFConfig
from dantinox.paradigms.diffusion.continuous import ContinuousParadigm
from flax import nnx

elf_cfg  = ELFConfig(embed_dim=512, bottleneck_dim=128,
                     model_dim=512, n_heads=8, head_size=64,
                     num_blocks=6, vocab_size=32128)
paradigm = ContinuousParadigm(elf_cfg)
model    = paradigm.build_model(nnx.Rngs(42))
embedder = paradigm.build_embedder()   # frozen T5 oracle

print(f"Total params (incl. embedder): {paradigm.num_parameters(model, embedder):,}")
```

---

## See also

- [Configuration Reference](../configuration.md) — all config fields
- [Paradigms overview](../paradigms/index.md) — conceptual explanation of each paradigm
- [Training API](training.md) — how `Trainer` calls `loss_fn`
- [Developer Guide: Custom Paradigm](../guides/new-paradigm.md) — step-by-step guide
