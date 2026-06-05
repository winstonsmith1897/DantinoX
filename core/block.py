from __future__ import annotations

import flax.nnx as nnx
import jax.numpy as jnp

from .attention import BaseAttention, build_attention
from .config import Config
from .mlp import MLP
from .moe import MoE


# ── Normalisation layers ───────────────────────────────────────────────────────

class RMSNorm(nnx.Module):
    """Root Mean Square Layer Normalisation (Zhang & Sennrich, 2019).

    Faster than LayerNorm — no mean subtraction, no bias — with equivalent
    empirical performance on modern LLMs.
    """

    def __init__(self, dim: int, *, eps: float = 1e-6, rngs: nnx.Rngs) -> None:
        self.scale = nnx.Param(jnp.ones(dim))
        self.eps   = eps

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        rms = jnp.sqrt(jnp.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return (x / rms) * self.scale[...]


def _build_norm(config: Config, dim: int, rngs: nnx.Rngs) -> nnx.Module:
    """Return RMSNorm or LayerNorm depending on ``config.norm_type``."""
    if config.norm_type == "rmsnorm":
        return RMSNorm(dim, rngs=rngs)
    return nnx.LayerNorm(dim, rngs=rngs)


class AdaLayerNorm(nnx.Module):
    """Adaptive Layer Norm (DiT-style).

    Standard LayerNorm whose scale and shift are modulated by a conditioning
    vector (e.g., a diffusion time-step embedding):

        AdaLN(x, c) = (1 + scale(c)) * LN(x) + shift(c)

    where scale(c) and shift(c) are learned linear projections of c.
    """

    def __init__(self, dim: int, cond_dim: int, rngs: nnx.Rngs) -> None:
        # No learnable affine inside the norm — scale/shift come from cond.
        self.norm       = nnx.LayerNorm(dim, use_bias=False, use_scale=False, rngs=rngs)
        self.scale_proj = nnx.Linear(cond_dim, dim, rngs=rngs)
        self.shift_proj = nnx.Linear(cond_dim, dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray, cond: jnp.ndarray) -> jnp.ndarray:
        """
        x:    [B, T, dim]
        cond: [B, cond_dim]
        """
        x     = self.norm(x)
        scale = self.scale_proj(cond)[:, None, :]   # [B, 1, dim]
        shift = self.shift_proj(cond)[:, None, :]   # [B, 1, dim]
        return x * (1.0 + scale) + shift


# ── Autoregressive block ───────────────────────────────────────────────────────

class ARBlock(nnx.Module):
    """Standard causal transformer block (pre-norm, residual connections).

    Supports MHA, GQA, and MLA attention (selected via ``config.attention_type``),
    dense MLP or Mixture-of-Experts feed-forward, and optional KV-cache for
    autoregressive generation.
    """

    def __init__(self, config: Config, rngs: nnx.Rngs) -> None:
        self.attention: BaseAttention = build_attention(config, rngs)
        self.ln1: nnx.Module          = _build_norm(config, config.dim, rngs)
        self.ln2: nnx.Module          = _build_norm(config, config.dim, rngs)
        self.use_moe: bool            = config.use_moe
        if self.use_moe:
            self.moe = MoE(config, rngs)
        else:
            self.mlp = MLP(config, rngs)

    def __call__(
        self,
        x: jnp.ndarray,
        use_cache: bool,
        kv_cache: tuple,
        cache_index: int,
        deterministic: bool = False,
    ) -> tuple[jnp.ndarray, tuple, jnp.ndarray | float]:
        x_attn, kv_cache = self.attention(
            self.ln1(x),
            use_cache=use_cache,
            kv_cache=kv_cache,
            cache_index=cache_index,
            deterministic=deterministic,
            is_causal=True,
        )
        x = x + x_attn
        ff, aux = (
            self.moe(self.ln2(x), deterministic=deterministic)
            if self.use_moe
            else self.mlp(self.ln2(x), deterministic=deterministic)
        )
        return x + ff, kv_cache, aux


# ── Diffusion block ────────────────────────────────────────────────────────────

class DiffusionBlock(nnx.Module):
    """Bidirectional transformer block for masked discrete diffusion.

    Differences from ``ARBlock``:
    - Attention is **fully bidirectional** (no causal mask), allowing every
      token to attend to every other token.
    - Layer normalisation is replaced by ``AdaLayerNorm``, which modulates
      scale and shift with the diffusion time-step embedding (DiT-style).
    - An optional ``prefix_kv`` tensor is accepted to enable the dual-cache
      optimisation: the static conditioning prefix is processed once and its
      per-layer KV tensors are concatenated with the noisy sequence's KV,
      avoiding redundant prefix recomputation across denoising steps.
    """

    def __init__(self, config: Config, rngs: nnx.Rngs) -> None:
        self.attention: BaseAttention = build_attention(config, rngs)
        self.ada_ln1 = AdaLayerNorm(config.dim, config.time_emb_dim, rngs=rngs)
        self.ada_ln2 = AdaLayerNorm(config.dim, config.time_emb_dim, rngs=rngs)
        self.use_moe: bool = config.use_moe
        if self.use_moe:
            self.moe = MoE(config, rngs)
        else:
            self.mlp = MLP(config, rngs)

    def __call__(
        self,
        x: jnp.ndarray,
        t_emb: jnp.ndarray,
        prefix_kv: tuple | None = None,
        deterministic: bool = False,
        return_kv: bool = False,
    ) -> tuple:
        """Run one diffusion transformer block.

        Args:
            x:            Hidden state ``[B, T, dim]``.
            t_emb:        Time-step embedding ``[B, time_emb_dim]``.
            prefix_kv:    Optional (k, v) from the dual-cache context
                          (prefix ‖ suffix). ``None`` during training.
            deterministic: Disables dropout when ``True``.
            return_kv:    If ``True``, also return ``(k, v)`` extracted from
                          the AdaLN-normalised hidden state *before* attention.
                          Used by ``compute_block_dual_cache`` to avoid a
                          redundant second AdaLN call.

        Returns:
            ``(x_out, aux_loss)`` normally, or
            ``(x_out, aux_loss, kv)`` when ``return_kv=True``.
        """
        x_norm = self.ada_ln1(x, t_emb)

        # Optionally extract K, V for caching — same x_norm as attention uses
        kv = self.attention.extract_kv(x_norm, cache_index=0) if return_kv else None

        x_attn, _ = self.attention(
            x_norm,
            use_cache=False,
            kv_cache=(None, None),
            cache_index=0,
            deterministic=deterministic,
            is_causal=False,
            prefix_kv=prefix_kv,
        )
        x = x + x_attn
        ff, aux = (
            self.moe(self.ada_ln2(x, t_emb), deterministic=deterministic)
            if self.use_moe
            else self.mlp(self.ada_ln2(x, t_emb), deterministic=deterministic)
        )
        x_out = x + ff
        return (x_out, aux, kv) if return_kv else (x_out, aux)


# ── Factory & backward-compat alias ───────────────────────────────────────────

def build_block(config: Config, rngs: nnx.Rngs) -> ARBlock | DiffusionBlock:
    """Return the block type indicated by ``config.model_type``."""
    if getattr(config, "model_type", "autoregressive") == "diffusion":
        return DiffusionBlock(config, rngs)
    return ARBlock(config, rngs)


# Backward-compatible alias so existing code using `Block` keeps working.
Block = ARBlock
