from __future__ import annotations

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from .attention import BaseAttention, build_attention
from .config import ModelConfig
from .mlp import MLP
from .moe import MoE


# ── Normalisation layers ───────────────────────────────────────────────────────

class RMSNorm(nnx.Module):
    """Root Mean Square Layer Normalisation (Zhang & Sennrich, 2019)."""

    def __init__(self, dim: int, *, eps: float = 1e-6, rngs: nnx.Rngs) -> None:
        self.scale = nnx.Param(jnp.ones(dim))
        self.eps   = eps

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        rms = jnp.sqrt(jnp.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return (x / rms) * self.scale[...]


def _build_norm(config: ModelConfig, dim: int, rngs: nnx.Rngs) -> nnx.Module:
    if config.norm_type == "rmsnorm":
        return RMSNorm(dim, rngs=rngs)
    return nnx.LayerNorm(dim, rngs=rngs)


# ── Unified Block ──────────────────────────────────────────────────────────────

class Block(nnx.Module):
    """Unified transformer block: pre-norm → attention → FFN → residuals.

    A single class handles both autoregressive (``causal=True``) and
    bidirectional / diffusion (``causal=False``) transformers.  The mode is
    fixed at construction time from ``config.causal``.

    Call signature
    --------------
    For **autoregressive** generation with a KV cache::

        x_out, new_cache, aux = block(x, cache=(k, v), cache_index=i)

    With differential attention the cache is a 3-tuple ``(k, v, k2)``.

    For **training** (AR or diffusion, no cache)::

        x_out, _, aux = block(x)

    For **diffusion inference** with a dual cache::

        x_out, _, aux = block(x, prefix_kv=(pk, pv))
        # or, to also extract KV for caching:
        x_out, _, aux, kv = block(x, return_kv=True)
    """

    def __init__(self, config: ModelConfig, rngs: nnx.Rngs) -> None:
        self.attention: BaseAttention = build_attention(config, rngs)
        self.norm1: nnx.Module        = _build_norm(config, config.dim, rngs)
        self.norm2: nnx.Module        = _build_norm(config, config.dim, rngs)
        self.causal: bool             = config.causal
        self.use_moe: bool            = config.use_moe
        if self.use_moe:
            self.ffn: nnx.Module = MoE(config, rngs)
        else:
            self.ffn = MLP(config, rngs)

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        cache: tuple | None = None,
        cache_index: int = 0,
        prefix_kv: tuple | None = None,
        deterministic: bool = False,
        return_kv: bool = False,
    ) -> tuple:
        """Forward pass.

        Parameters
        ----------
        x:             Hidden state ``[B, T, dim]``.
        cache:         ``(k_cache, v_cache)`` or ``(k_cache, v_cache, k2_cache)``
                       for AR KV-cache generation (3-tuple when differential
                       attention is active).  ``None`` = no cache (training).
        cache_index:   Write position in the AR KV cache.
        prefix_kv:     ``(k, v)`` prefix injected for diffusion dual-cache inference.
        deterministic: Disables dropout when ``True``.
        return_kv:     Also return the pre-attention ``(k, v)`` tensors (for
                       building a dual cache).

        Returns
        -------
        ``(x_out, new_cache, aux_loss)`` or
        ``(x_out, new_cache, aux_loss, kv)`` when ``return_kv=True``.
        """
        x_norm = self.norm1(x)

        kv = self.attention.extract_kv(x_norm, cache_index) if return_kv else None

        x_attn, new_cache = self.attention(
            x_norm,
            use_cache=(cache is not None),
            kv_cache=cache if cache is not None else (None, None),
            cache_index=cache_index,
            deterministic=deterministic,
            is_causal=self.causal,
            prefix_kv=prefix_kv,
        )
        x = x + x_attn

        ff, aux = self.ffn(self.norm2(x), deterministic=deterministic)
        x_out = x + ff

        if return_kv:
            return x_out, new_cache, aux, kv
        return x_out, new_cache, aux


# ── Backward-compatible aliases ────────────────────────────────────────────────
# Old code importing ARBlock or DiffusionBlock continues to work; they are
# identical to Block — the causal flag comes from ModelConfig.

ARBlock       = Block
DiffusionBlock = Block


class AdaLayerNorm(nnx.Module):
    """Adaptive Layer Norm (DiT-style) — kept for compatibility."""

    def __init__(self, dim: int, cond_dim: int, rngs: nnx.Rngs) -> None:
        _zeros = jax.nn.initializers.zeros
        self.norm       = nnx.LayerNorm(dim, use_bias=False, use_scale=False, rngs=rngs)
        self.scale_proj = nnx.Linear(cond_dim, dim, rngs=rngs,
                                     kernel_init=_zeros, bias_init=_zeros)
        self.shift_proj = nnx.Linear(cond_dim, dim, rngs=rngs,
                                     kernel_init=_zeros, bias_init=_zeros)

    def __call__(self, x: jnp.ndarray, cond: jnp.ndarray) -> jnp.ndarray:
        x     = self.norm(x)
        scale = self.scale_proj(cond)[:, None, :]
        shift = self.shift_proj(cond)[:, None, :]
        return x * (1.0 + scale) + shift


def build_block(config: ModelConfig, rngs: nnx.Rngs) -> Block:
    """Factory kept for backward compatibility — identical to ``Block(config, rngs)``."""
    return Block(config, rngs)
