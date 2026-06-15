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
    """Per-layer KV caches.  Each element is ``(k, v)`` for standard attention
    or ``(k, v, k2)`` when differential attention is active.
    ``None`` entries when ``use_cache=False``."""

    aux_loss: float
    """MoE load-balancing auxiliary loss (``0.0`` for dense models)."""


class ELFOutput(NamedTuple):
    """Return type for ``ELFTransformer.__call__`` (continuous flow-matching).

    ELF predicts clean embeddings x̂ (x-prediction) and materialises token
    logits in the same forward pass via the shared unembedding head.

    Usage::

        out = model(z_t, x_prev, t, cfg_scale, is_decode)

        # Denoiser MSE loss
        inv_1mt = 1.0 / jnp.clip(1.0 - t[:, None, None], 1e-6)
        v_pred  = (out.x_pred - z_t) * inv_1mt
        loss    = elf_mse_loss(v_pred, v_target)

        # Decoder CE loss
        loss = elf_ce_loss(out.logits, tokens)

        # ODE velocity step
        v      = (out.x_pred - z) / jnp.clip(1.0 - t, 1e-6)
        z_next = z + dt * v
    """

    x_pred: jnp.ndarray
    """Predicted clean embeddings ``[batch, seq_len, embed_dim]``."""

    logits: jnp.ndarray
    """Token logits ``[batch, seq_len, vocab_size]`` via ``unembed(x_pred)``."""
