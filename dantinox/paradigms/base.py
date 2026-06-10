from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import jax
import jax.numpy as jnp


class Paradigm(ABC):
    """Abstract base for all generative paradigms.

    A Paradigm wraps a core model and owns the paradigm-specific logic:
    how to corrupt inputs, compute loss, and generate samples.  The trainer
    calls only ``loss_fn`` and ``generate`` — it never inspects the internals.

    Implementing a new paradigm requires overriding three methods::

        class MyParadigm(Paradigm):
            def build_model(self, rngs): ...
            def loss_fn(self, model, batch, rng): ...
            def generate(self, model, prompt, rng, **kwargs): ...
    """

    @abstractmethod
    def build_model(self, rngs: Any) -> Any:
        """Construct and return the NNX model for this paradigm.

        Called once by the Trainer at the start of ``fit()``.  The returned
        model is then managed (checkpointed, sharded) by the Trainer and
        passed back into ``loss_fn`` / ``generate`` as the first argument.
        """
        ...

    @abstractmethod
    def loss_fn(
        self,
        model: Any,
        batch: jnp.ndarray,
        rng: jax.random.KeyArray,
    ) -> tuple[jnp.ndarray, dict[str, Any]]:
        """Compute the scalar training loss for one batch.

        The model is passed explicitly so ``nnx.value_and_grad`` can
        differentiate through it without the paradigm needing to be an
        NNX module itself.

        Returns:
            (scalar_loss, metrics_dict)  where metrics_dict holds any
            auxiliary scalars (ce_loss, aux_loss, mse_loss, …) for logging.
        """
        ...

    @abstractmethod
    def generate(
        self,
        model: Any,
        prompt: jnp.ndarray,
        rng: jax.random.KeyArray,
        **kwargs: Any,
    ) -> jnp.ndarray:
        """Generate token sequences given a prompt prefix."""
        ...

    # ── Shared helpers ────────────────────────────────────────────────────────

    def num_parameters(self, model: Any) -> int:
        """Count trainable parameters in the model."""
        import jax
        from flax import nnx
        params = nnx.state(model, nnx.Param)
        return sum(x.size for x in jax.tree_util.tree_leaves(params))

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
