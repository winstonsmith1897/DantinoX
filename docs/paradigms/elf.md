# ELF — Continuous Flow-Matching

ELF (Embedded Language Flows) is a continuous flow-matching paradigm that operates in the embedding space of a frozen T5 encoder rather than in token-ID space.

---

## Core idea

Instead of corrupting discrete tokens (as in LLaDA), ELF defines a continuous interpolation between clean embeddings **x** and Gaussian noise **ε**:

$$z_t = t \cdot x + (1 - t) \cdot \varepsilon, \quad t \sim U(0, 1), \quad \varepsilon \sim \mathcal{N}(0, I)$$

The model predicts the clean embedding **x** from the noisy **z_t** (x-prediction formulation). An auxiliary cross-entropy branch reconstructs discrete token IDs from the predicted embeddings.

---

## Architecture

```
Input tokens [B, T]
       │
       ▼
  ELFEmbedder (frozen T5)
       │
  embeddings [B, T, embed_dim]
       │
  ┌────┴────────────────────────────────────┐
  │   z_t = t·x + (1-t)·ε                  │  noise injection
  └────┬────────────────────────────────────┘
       │
  ELFTransformer (bidirectional)
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
from dantinox.paradigms.diffusion.continuous import ContinuousParadigm
from core.config import ELFConfig
from flax import nnx

cfg = ELFConfig(
    embed_dim=768,         # T5 embedding dimension
    model_dim=512,         # transformer hidden dim
    n_heads=8,
    head_size=64,
    num_blocks=12,
    vocab_size=32_128,
    elf_n_steps=64,        # ODE integration steps at generation time
    elf_cfg_scale=1.5,     # classifier-free guidance weight
)

paradigm = ContinuousParadigm(cfg)
model    = paradigm.build_model(nnx.Rngs(0))
embedder = paradigm.build_embedder(nnx.Rngs(0))  # frozen T5
```

---

## Training

ELF requires pre-computed embeddings passed into `loss_fn`. The embedder is **not** differentiable in the training loop (it is frozen):

```python
from dantinox.training.trainer import Trainer
from core.config import TrainingConfig

# Pre-compute embeddings once per batch in a custom trainer loop
# or integrate ELFEmbedder into a custom Paradigm subclass that
# pre-fetches embeddings before calling loss_fn.

trainer = Trainer(paradigm, TrainingConfig(lr=1e-4, epochs=10, optimizer="adamw"))
run_dir = trainer.fit("data/wiki.txt")
```

!!! warning "Embeddings must be pre-computed"
    `ContinuousParadigm.loss_fn` raises `ValueError` if `embeddings=None`.
    Use `ELFEmbedder` or a custom data pipeline that produces `[B, T, embed_dim]` arrays.

---

## Generation

ELF generates by integrating the learned ODE from t=1 (pure noise) to t=0 (clean embeddings):

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

| `ELFConfig` field | Default | Description |
| :--- | :--- | :--- |
| `embed_dim` | `768` | T5 embedding dimension |
| `model_dim` | `512` | Transformer hidden dimension |
| `n_heads` | `8` | Attention heads |
| `head_size` | `64` | Head dimension |
| `num_blocks` | `12` | Transformer layers |
| `vocab_size` | `32_128` | Vocabulary size (must match T5 tokenizer) |
| `elf_n_steps` | `64` | ODE integration steps at inference |
| `elf_cfg_scale` | `1.0` | Classifier-free guidance weight (1.0 = no CFG) |

---

## Comparison with Discrete Diffusion

| | LLaDA (`DiscreteParadigm`) | ELF (`ContinuousParadigm`) |
| :--- | :--- | :--- |
| Representation space | Discrete token IDs | Continuous T5 embeddings |
| Corruption | Token masking | Gaussian noise injection |
| Training signal | Cross-entropy on masked tokens | MSE + CE |
| Generation | Iterative unmasking | ODE integration |
| Pre-requisite | None | Frozen T5 encoder |
| Generation speed | Fast (parallel unmask) | Slower (sequential ODE) |
