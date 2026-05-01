import flax.nnx as nnx
import jax.numpy as jnp

from .attention import Attention
from .config import Config
from .mlp import MLP
from .moe import MoE

# ── Normalisation ─────────────────────────────────────────────────────────────

class RMSNorm(nnx.Module):
    """
    Root Mean Square Layer Normalisation (Zhang & Sennrich, 2019).

    Faster than LayerNorm — no mean subtraction, no bias — with identical
    empirical performance on modern LLMs (LLaMA, Mistral, Gemma, …).
    """

    def __init__(self, dim: int, *, eps: float = 1e-6, rngs: nnx.Rngs) -> None:
        self.scale = nnx.Param(jnp.ones(dim))
        self.eps = eps

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        rms = jnp.sqrt(jnp.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return (x / rms) * self.scale[...]


def _build_norm(config: Config, dim: int, rngs: nnx.Rngs) -> nnx.Module:
    """Return a RMSNorm or LayerNorm depending on ``config.norm_type``."""
    if config.norm_type == "rmsnorm":
        return RMSNorm(dim, rngs=rngs)
    return nnx.LayerNorm(dim, rngs=rngs)


# ── Transformer block ─────────────────────────────────────────────────────────

class Block(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs):
        self.attention: Attention        = Attention(config, rngs)
        self.ln1: nnx.Module             = _build_norm(config, config.dim, rngs)
        self.ln2: nnx.Module             = _build_norm(config, config.dim, rngs)
        self.use_moe: bool               = config.use_moe
        if self.use_moe:
            self.moe = MoE(config, rngs)
        else:
            self.mlp = MLP(config, rngs)

    def __call__(self, x: jnp.ndarray,
                 use_cache: bool,
                 kv_cache: tuple,
                 cache_index: int,
                 deterministic: bool = False) -> tuple[jnp.ndarray, tuple, jnp.ndarray | float]:

        x_attn, kv_cache = self.attention(self.ln1(x),
                                          use_cache=use_cache,
                                          kv_cache=kv_cache,
                                          cache_index=cache_index,
                                          deterministic=deterministic)
        x  = x + x_attn
        ff, balancing_loss = (
            self.moe(self.ln2(x), deterministic=deterministic)
            if self.use_moe
            else self.mlp(self.ln2(x), deterministic=deterministic)
        )
        x  = x + ff
        return x, kv_cache, balancing_loss
