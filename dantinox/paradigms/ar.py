from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from dantinox.core.config import ModelConfig
from dantinox.core.generation import generate as _generate
from dantinox.core.model import Transformer
from dantinox.paradigms.base import Paradigm


class ARParadigm(Paradigm):
    """Autoregressive next-token-prediction paradigm.

    Loss: cross-entropy on shifted targets (teacher-forcing).

    Quick-start::

        cfg = ModelConfig(dim=512, n_heads=8, head_size=64, num_blocks=12,
                          vocab_size=32_000, causal=True)
        paradigm = ARParadigm(cfg)
        # hand to Trainer — paradigm.build_model() is called there
    """

    requires_shifted_targets = True

    def __init__(self, config: ModelConfig) -> None:
        if not config.causal:
            raise ValueError(
                "ARParadigm requires a causal model (config.causal=True). "
                "Use DiscreteParadigm or ContinuousParadigm for bidirectional models."
            )
        self.config = config

    # ── Paradigm contract ─────────────────────────────────────────────────────

    def build_model(self, rngs: nnx.Rngs) -> Transformer:
        return Transformer(self.config, rngs=rngs)

    def loss_fn(
        self,
        model: Transformer,
        batch: jnp.ndarray,
        rng: jax.Array,
    ) -> tuple[jnp.ndarray, dict[str, Any]]:
        x, y = batch[:, :-1], batch[:, 1:]
        out  = model(x)
        ce   = optax.softmax_cross_entropy_with_integer_labels(out.logits, y).mean()
        aux  = out.aux_loss
        return ce + aux, {"ce_loss": ce, "aux_loss": aux}

    def generate(
        self,
        model: Transformer,
        prompt: jnp.ndarray,
        rng: jax.Array,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        greedy: bool = False,
        use_cache: bool = True,
    ) -> jnp.ndarray:
        return _generate(
            model,
            prompt,
            max_generations=max_new_tokens,
            greedy=greedy,
            seed=_seed_from(rng),
            use_cache=use_cache,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    def __repr__(self) -> str:
        return f"ARParadigm({self.config!r})"


def _seed_from(rng: jax.Array | None) -> int:
    """Derive a Python int seed from a JAX PRNG key (legacy or typed)."""
    if rng is None:
        return 42
    return int(jax.random.randint(rng, (), 0, 2**31 - 1))
