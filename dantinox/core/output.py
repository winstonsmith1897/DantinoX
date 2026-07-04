"""NamedTuple return types for model forward passes.

Kept in one leaf module (no internal imports) so any part of the library can
depend on them without cycles: ``ModelOutput`` (Transformer),
``FlowMatchingOutput`` (FlowMatchingTransformer), ``EmbeddingOutput``
(``Transformer.encode_hidden``).
"""
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class ModelOutput(NamedTuple):
    """Return type for ``Transformer.__call__`` (AR and masked-diffusion).

    Supports attribute access and positional unpacking::

        out = model(x, ...)
        loss = cross_entropy(out.logits, targets) + cfg.alpha * out.aux_loss

        logits, kv_caches, aux_loss = model(x, ...)  # positional (backward-compat)
    """

    logits: jnp.ndarray
    """Token logits ``[batch, seq_len, vocab_size]``."""

    kv_caches: tuple
    """Per-layer KV caches: ``attention.KVCache`` tuples for standard
    attention (``k2`` populated only with differential attention) or
    ``attention.MLACache`` for absorbed MLA.  ``None`` entries when
    ``use_cache=False``."""

    aux_loss: float
    """MoE load-balancing auxiliary loss (``0.0`` for dense models).

    Note: this NamedTuple intentionally stays at three fields — call sites
    rely on ``logits, kv_caches, aux_loss = model(x)`` unpacking.  For final
    hidden states use ``Transformer.encode_hidden``."""


class FlowMatchingOutput(NamedTuple):
    """Return type for ``FlowMatchingTransformer.__call__`` (continuous flow-matching).

    The model predicts clean embeddings x̂ (x-prediction) and materialises
    token logits in the same forward pass via the shared unembedding head.

    Usage::

        out = model(z_t, x_prev, t, cfg_scale, is_decode)

        # Denoiser MSE loss
        inv_1mt = 1.0 / jnp.clip(1.0 - t[:, None, None], 1e-6)
        v_pred  = (out.x_pred - z_t) * inv_1mt
        loss    = flow_mse_loss(v_pred, v_target)

        # Decoder CE loss
        loss = flow_ce_loss(out.logits, tokens)

        # ODE velocity step
        v      = (out.x_pred - z) / jnp.clip(1.0 - t, 1e-6)
        z_next = z + dt * v
    """

    x_pred: jnp.ndarray
    """Predicted clean embeddings ``[batch, seq_len, embed_dim]``."""

    logits: jnp.ndarray
    """Token logits ``[batch, seq_len, vocab_size]`` via ``unembed(x_pred)``."""


class EmbeddingOutput(NamedTuple):
    """Return type for ``Transformer.encode_hidden`` — pooled sentence embeddings.

    Usage::

        out = model.encode_hidden(token_ids, pooling="mean", normalize=True)
        # out.embeddings: [B, D]  — ready for cosine-similarity / vector stores
        # out.hidden_states: [B, T, D] — full sequence if you need token-level features
    """

    embeddings: jnp.ndarray
    """Pooled sentence embeddings ``[batch, dim]``."""

    hidden_states: jnp.ndarray
    """Per-token hidden states after the final layer norm ``[batch, seq_len, dim]``."""


# Deprecated ELF-branded alias (removed in v1.0).
ELFOutput = FlowMatchingOutput
