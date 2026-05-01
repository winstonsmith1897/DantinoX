from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class ModelOutput(NamedTuple):
    """
    Named return type for ``Transformer.__call__``.

    Supports both attribute access and positional unpacking so existing
    code that destructures the tuple continues to work unchanged::

        # Named (preferred)
        out = model(x, ...)
        loss = cross_entropy(out.logits, targets) + cfg.alpha * out.aux_loss

        # Positional (backward-compatible)
        logits, kv_caches, aux_loss = model(x, ...)
    """

    logits: jnp.ndarray
    """Token logits with shape ``[batch, seq_len, vocab_size]``."""

    kv_caches: tuple
    """Per-layer KV (or compressed-latent) caches; ``None`` entries when ``use_cache=False``."""

    aux_loss: float
    """MoE load-balancing auxiliary loss (``0.0`` for dense models)."""
