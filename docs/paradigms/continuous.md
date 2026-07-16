# Continuous Flow-Matching

Continuous flow-matching is a paradigm that operates in the embedding space of a frozen T5 encoder rather than in token-ID space, following the ELF recipe (Embedded Language Flows; Hu et al., 2026).

---

## Core idea

Instead of corrupting discrete tokens (as in discrete diffusion), this paradigm defines a continuous interpolation between clean embeddings **x** and Gaussian noise **ε**:

$$z_t = t \cdot x + (1 - t) \cdot \varepsilon, \quad t \sim U(0, 1), \quad \varepsilon \sim \mathcal{N}(0, I)$$

The model predicts the clean embedding **x** from the noisy **z_t** (x-prediction formulation). An auxiliary cross-entropy branch reconstructs discrete token IDs from the predicted embeddings.

---

## Architecture

```
Input tokens [B, T]
       │
       ▼
  T5ContextualEncoder (frozen T5, outside JIT)
       │
  raw embeddings [B, T, embed_dim]
       │
       ▼
  FlowEmbedder.encode() — channel-wise normalise using
  running mean/std stats (NOT a T5 forward pass itself)
       │
  normalised embeddings x [B, T, embed_dim]
       │
  ┌────┴────────────────────────────────────┐
  │   z_t = t·x + (1-t)·ε                  │  noise injection
  └────┬────────────────────────────────────┘
       │
  FlowMatchingTransformer (bidirectional)
  with control tokens:
    - [TIME]: timestep t
    - [CFG]:  classifier-free guidance scale
    - [MODE]: denoiser vs. decoder branch
       │
  ┌────┴────────────────────────────────────┐
  │   x̂ = model(z_t)                        │  x-prediction
  └────┬────────────────────────────────────┘
       │
  ┌────┴──────────┐
  │  MSE loss     │  flow-matching objective
  │  CE loss      │  token reconstruction
  └───────────────┘
```

---

## Quick start

```python
import dantinox as dx
from flax import nnx

cfg = dx.ModelConfig(
    paradigm="continuous",  # causal=False set automatically
    embed_dim=768,          # T5 embedding dimension
    bottleneck_dim=128,
    dim=512,                # transformer hidden dim
    n_heads=8,
    head_size=64,
    num_blocks=12,
    vocab_size=32_128,
    flow_n_steps=64,        # ODE integration steps at generation time
    flow_cfg_scale=1.5,     # classifier-free guidance weight
    # NOT elf_n_steps=/elf_cfg_scale= — those are read-only compatibility
    # properties, not constructor arguments.
)

paradigm = dx.Paradigm(cfg)
model    = paradigm.build_model(nnx.Rngs(0))
embedder = paradigm.build_embedder(nnx.Rngs(0))  # frozen T5
```

---

## Training

This paradigm requires pre-computed embeddings passed into `loss_fn`. The embedder is **not** differentiable in the training loop (it is frozen):

```python
from dantinox.training.trainer import Trainer
from dantinox.core.config import TrainingConfig

# Pre-compute embeddings once per batch in a custom trainer loop
# or integrate FlowEmbedder into a custom Paradigm subclass that
# pre-fetches embeddings before calling loss_fn.

trainer = Trainer(paradigm, TrainingConfig(lr=1e-4, epochs=10, optimizer="adamw"))
run_dir = trainer.fit("data/wiki.txt")
```

:::{admonition} Embeddings must be pre-computed
:class: warning

`ContinuousParadigm.loss_fn` raises `ValueError` if `embeddings=None`.
Use `FlowEmbedder` or a custom data pipeline that produces `[B, T, embed_dim]` arrays.
:::

---

## Generation

Generation integrates the learned ODE from t=1 (pure noise) to t=0 (clean embeddings):

```python
import jax

prompt = jnp.array([[1, 2, 3, 4]])   # token IDs
rng    = jax.random.PRNGKey(42)

tokens = paradigm.generate(
    model, prompt, rng,
    n_steps=64,       # ODE steps (more = higher quality, slower)
    cfg_scale=1.5,    # classifier-free guidance weight
)
```

---

## Configuration reference

| `FlowMatchingConfig` field | Default | Description |
| :--- | :--- | :--- |
| `embed_dim` | `512` | T5 embedding dimension |
| `model_dim` | `768` | Transformer hidden dimension |
| `n_heads` | `12` | Attention heads |
| `head_size` | `None` | Head dimension (derived from `model_dim`/`n_heads` if unset) |
| `num_blocks` | `12` | Transformer layers |
| `vocab_size` | `None` | Vocabulary size (must match T5 tokenizer) |
| `flow_n_steps` | `32` | ODE integration steps at inference (`ELFConfig`/`elf_n_steps` are deprecated aliases) |
| `flow_cfg_scale` | `1.5` | Classifier-free guidance weight (`elf_cfg_scale` is a deprecated alias); `1.0` = no CFG |

(`ELFConfig` itself is a deprecated alias for `FlowMatchingConfig` — same
fields, same defaults, still works but emits a `DeprecationWarning`.)

---

## Comparison with Discrete Diffusion

| | Discrete Diffusion (`DiscreteParadigm`) | Continuous Flow-Matching (`ContinuousParadigm`) |
| :--- | :--- | :--- |
| Representation space | Discrete token IDs | Continuous T5 embeddings |
| Corruption | Token masking | Gaussian noise injection |
| Training signal | Cross-entropy on masked tokens | MSE + CE |
| Generation | Iterative unmasking | ODE integration |
| Pre-requisite | None | Frozen T5 encoder |
| Generation speed | Fast (parallel unmask) | Slower (sequential ODE) |
