# Paradigm System

The Paradigm abstraction is the central design decision in DantinoX. This page explains the motivation, the contract, and how the three built-in paradigms implement it.

---

## Motivation

A language model can be trained with radically different objectives — cross-entropy next-token prediction, masked diffusion, or continuous flow-matching. Naively, each objective requires a different trainer, a different data pipeline, and different generation logic. This creates N × M complexity for N paradigms × M architectural choices.

DantinoX breaks this with a single rule: **the `Paradigm` owns the objective; the `Trainer` owns everything else.** The contract between them is just three methods on `Paradigm`:

```python
class Paradigm(ABC):
    def build_model(self, rngs)  → model
    def loss_fn(self, model, batch, rng) → (loss, metrics)
    def generate(self, model, prompt, rng, **kwargs) → tokens
```

The `Trainer` calls `paradigm.build_model()` once at the start of `fit()`, and then calls `paradigm.loss_fn(model, batch, rng)` at every step — nothing more. Generation is never called during training.

---

## The `Paradigm` contract in detail

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

### `generate(model, prompt, rng, **kwargs)`

Called outside training. Each paradigm implements its own decode strategy:

| Paradigm | Decode strategy |
| :--- | :--- |
| `ARParadigm` | Causal autoregressive loop with KV cache |
| `DiscreteParadigm` | Iterative unmasking (reverse diffusion) |
| `ContinuousParadigm` | ODE integration in embedding space |

### `num_parameters(model)`

A shared helper (not abstract). Counts `nnx.Param` leaves. Override in subclasses if the model has non-standard parameter structures (e.g., `ContinuousParadigm` overrides this for `ELFTransformer`).

---

## ARParadigm

**Objective:** Standard next-token prediction via cross-entropy on shifted targets (teacher-forcing).

```python
from dantinox.paradigms.ar import ARParadigm
from dantinox.core.config import ModelConfig

paradigm = ARParadigm(ModelConfig(
    dim=512, n_heads=8, head_size=64, num_blocks=12,
    vocab_size=32_000, causal=True
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

**Constraint:** `config.causal=True` is enforced at construction time.

---

## DiscreteParadigm (LLaDA)

**Objective:** (1/t)-weighted cross-entropy on masked positions — the LLaDA training objective.

```python
from dantinox.paradigms.diffusion.discrete import DiscreteParadigm, DiscreteConfig

paradigm = DiscreteParadigm(
    model_config=ModelConfig(dim=512, ..., causal=False),
    diffusion_config=DiscreteConfig(noise_schedule="cosine", mask_token_id=4),
)
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

**Constraint:** `config.causal=False` is enforced.

---

## ContinuousParadigm (ELF)

**Objective:** Flow-matching MSE in continuous embedding space.

The forward process interpolates between clean embeddings `x` and Gaussian noise `ε`:

$$z_t = t \cdot x + (1 - t) \cdot \varepsilon, \quad t \sim U(0, 1)$$

The model predicts clean `x` from noisy `z_t` (x-prediction). An auxiliary cross-entropy branch reconstructs discrete tokens.

```python
from dantinox.paradigms.diffusion.continuous import ContinuousParadigm
from dantinox.core.config import ELFConfig

paradigm  = ContinuousParadigm(ELFConfig(embed_dim=768, model_dim=512, ...))
model     = paradigm.build_model(rngs)
embedder  = paradigm.build_embedder(rngs)  # frozen T5 encoder
```

**`loss_fn` requires pre-computed embeddings:**

```python
embeddings = embedder(batch)   # [B, T, embed_dim]
loss, metrics = paradigm.loss_fn(model, batch, rng, embeddings=embeddings)
```

!!! warning "Embeddings are mandatory"
    Unlike AR and Discrete, `ContinuousParadigm.loss_fn` raises `ValueError` if `embeddings=None`.
    Pre-compute them via `ELFEmbedder` before the training loop.

---

## Implementing a custom paradigm

See [Developer Guide: Custom Paradigm](../guides/new-paradigm.md) for a step-by-step walkthrough.
