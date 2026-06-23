# Custom Paradigm

This guide shows how to implement a custom training objective by subclassing `Paradigm`. The example below implements a contrastive (InfoNCE) paradigm for representation learning.

---

## The three-method contract

Every paradigm must implement:

| Method | Purpose | Called by |
| :--- | :--- | :--- |
| `build_model(rngs)` | Construct the NNX model | `Trainer.fit()` — once at startup |
| `loss_fn(model, batch, rng)` | Compute scalar loss + metrics dict | `Trainer._step()` — every step |
| `generate(model, prompt, rng, **kwargs)` | Decode token sequences | User code, `dx.quick_generate()` |

---

## Step 1: Implement the paradigm

```python
# dantinox/paradigms/contrastive.py
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from dantinox.core.config import ModelConfig
from dantinox.core.model import Transformer
from dantinox.paradigms.base import Paradigm


class ContrastiveParadigm(Paradigm):
    """InfoNCE contrastive paradigm for sentence representation learning.

    Trains a bidirectional encoder to maximize mutual information between
    augmented views of the same input (SimCSE-style).

    Args:
        config: Model configuration. ``config.causal`` must be ``False``.
        temperature: InfoNCE temperature τ. Smaller values sharpen
            the contrast; typical range ``[0.01, 0.1]``.
    """

    def __init__(self, config: ModelConfig, temperature: float = 0.05) -> None:
        if config.causal:
            raise ValueError(
                "ContrastiveParadigm requires a bidirectional encoder "
                "(config.causal=False)."
            )
        self.config      = config
        self.temperature = temperature

    # ── Paradigm contract ─────────────────────────────────────────────────────

    def build_model(self, rngs: nnx.Rngs) -> Transformer:
        """Build and return a bidirectional Transformer encoder.

        Args:
            rngs: Flax NNX random state.

        Returns:
            A ``Transformer`` instance with ``causal=False``.
        """
        return Transformer(self.config, rngs=rngs)

    def loss_fn(
        self,
        model: Transformer,
        batch: jnp.ndarray,
        rng: jax.random.KeyArray,
    ) -> tuple[jnp.ndarray, dict]:
        """Compute InfoNCE loss on two dropout-augmented views of the batch.

        Two forward passes are run with different dropout masks (views).
        The loss maximises cosine similarity between views of the same
        sequence and minimises it between different sequences.

        Args:
            model: The NNX encoder.
            batch: Integer token IDs ``[B, T]``.
            rng: JAX random key for dropout augmentation.

        Returns:
            A tuple ``(scalar_loss, metrics)`` where metrics contains
            ``{"infonce_loss": <float>}``.
        """
        rng1, rng2 = jax.random.split(rng)

        # Two stochastic forward passes — different dropout masks = different views
        z1 = _pool(model(batch, deterministic=False, rngs=rng1).logits)   # [B, D]
        z2 = _pool(model(batch, deterministic=False, rngs=rng2).logits)   # [B, D]

        # L2-normalise
        z1 = z1 / (jnp.linalg.norm(z1, axis=-1, keepdims=True) + 1e-8)
        z2 = z2 / (jnp.linalg.norm(z2, axis=-1, keepdims=True) + 1e-8)

        # InfoNCE: B×B cosine similarity matrix
        sim = jnp.dot(z1, z2.T) / self.temperature
        labels = jnp.arange(batch.shape[0])
        loss = (
            _cross_entropy(sim,        labels) +
            _cross_entropy(sim.T,      labels)
        ) / 2.0

        return loss, {"infonce_loss": loss}

    def generate(
        self,
        model: Transformer,
        prompt: jnp.ndarray,
        rng: jax.random.KeyArray,
        **kwargs,
    ) -> jnp.ndarray:
        """Encode a prompt and return the pooled representation.

        Note:
            This paradigm is an encoder — it returns embeddings, not token
            sequences. For downstream generation, attach a decoder head.

        Args:
            model: The NNX encoder.
            prompt: Token IDs ``[B, T]`` or ``[T]``.
            rng: JAX random key (unused; present for interface compatibility).

        Returns:
            Pooled embedding ``[B, D]`` or ``[D]``.
        """
        if prompt.ndim == 1:
            prompt = prompt[None]
        logits = model(prompt, deterministic=True).logits
        return _pool(logits)

    def __repr__(self) -> str:
        return f"ContrastiveParadigm(temperature={self.temperature})"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pool(logits: jnp.ndarray) -> jnp.ndarray:
    """Mean-pool over the sequence dimension."""
    return logits.mean(axis=-2)


def _cross_entropy(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Scalar cross-entropy loss."""
    import optax
    return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
```

---

## Step 2: Register it in the paradigm factory (optional)

To use it with the Level-1 API (`dx.fit("contrastive", ...)`), add it to `dantinox/__init__.py`:

```python
# dantinox/__init__.py
from dantinox.paradigms.contrastive import ContrastiveParadigm

def _build_contrastive(config, kwargs):
    temp = kwargs.pop("temperature", 0.05)
    if config is None:
        from dantinox.core.config import ModelConfig
        config = ModelConfig(**{**kwargs, "causal": False})
    return ContrastiveParadigm(config, temperature=temp)

_PARADIGM_MAP["contrastive"] = _build_contrastive
```

---

## Step 3: Test it

```python
# tests/test_contrastive.py
import jax
import jax.numpy as jnp
from flax import nnx
from dantinox.core.config import ModelConfig
from dantinox.paradigms.contrastive import ContrastiveParadigm

def test_contrastive_loss_shape():
    cfg      = ModelConfig(dim=64, n_heads=4, head_size=16, num_blocks=2,
                           vocab_size=100, causal=False)
    paradigm = ContrastiveParadigm(cfg)
    model    = paradigm.build_model(nnx.Rngs(0))
    batch    = jnp.ones((4, 16), dtype=jnp.int32)
    rng      = jax.random.PRNGKey(0)
    loss, metrics = paradigm.loss_fn(model, batch, rng)
    assert loss.shape == ()
    assert "infonce_loss" in metrics

def test_contrastive_causal_guard():
    import pytest
    cfg = ModelConfig(dim=64, n_heads=4, head_size=16, num_blocks=2,
                      vocab_size=100, causal=True)
    with pytest.raises(ValueError, match="causal=False"):
        ContrastiveParadigm(cfg)
```

---

## Checklist

- [ ] All three `Paradigm` methods implemented with Google docstrings
- [ ] Config validation in `__init__` (raise early, clear message)
- [ ] `__repr__` returns a concise, informative string
- [ ] Tests cover: loss shape, loss is finite, input validation guards
- [ ] Registered in `_PARADIGM_MAP` if Level-1 API access is desired
- [ ] `make doccheck` passes
