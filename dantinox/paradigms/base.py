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

    Two optional hooks let a paradigm participate in the training loop
    without the Trainer knowing its internals:

    * ``on_train_start(model, sample_batches)`` — one-time setup before the
      first step (e.g. ELF computes T5 embedding normalisation statistics).
    * ``prepare_batch(batch)`` — host-side, non-JIT preprocessing of each
      batch; whatever it returns is forwarded to ``loss_fn`` as the
      ``embeddings`` keyword.  Set ``provides_batch_extras = True`` to
      enable it.
    """

    # AR-style paradigms need ``seq_len + 1`` tokens per row so the loss can
    # form (input, shifted-target) pairs; diffusion paradigms consume the
    # batch as-is.
    requires_shifted_targets: bool = False

    # True when ``prepare_batch`` returns per-batch extras that must be
    # computed outside JIT and passed to ``loss_fn(..., embeddings=...)``.
    provides_batch_extras: bool = False

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
        rng: jax.Array,
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
        rng: jax.Array,
        **kwargs: Any,
    ) -> jnp.ndarray:
        """Generate token sequences given a prompt prefix."""
        ...

    # ── Optional training hooks ───────────────────────────────────────────────

    def on_train_start(self, model: Any, sample_batches: list[Any]) -> None:
        """One-time setup before training starts (default: no-op).

        *sample_batches* is a small list of token batches drawn from the
        training set, for paradigms that need data-dependent initialisation.
        """

    def prepare_batch(self, batch: Any) -> Any:
        """Host-side per-batch preprocessing executed outside JIT.

        Only called when ``provides_batch_extras`` is True; the return value
        is forwarded to ``loss_fn`` as the ``embeddings`` keyword argument.
        """
        return None

    # ── Shared helpers ────────────────────────────────────────────────────────

    def num_parameters(self, model: Any) -> int:
        """Count trainable parameters in the model."""
        import jax
        from flax import nnx
        params = nnx.state(model, nnx.Param)
        return sum(x.size for x in jax.tree_util.tree_leaves(params))

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
