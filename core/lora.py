from __future__ import annotations

import math

import flax.nnx as nnx
import jax
import jax.numpy as jnp


class LoRAParam(nnx.Variable):
    """Trainable LoRA variable — distinct type so base nnx.Param weights stay frozen."""
    pass


class LoRALinear(nnx.Module):
    """Drop-in replacement for nnx.Linear with frozen base weight and trainable low-rank delta.

    The effective weight is  W_eff = W_base + (alpha / rank) * A @ B,
    where A is initialised with scaled Gaussian noise and B with zeros,
    so the adapter contributes nothing at initialisation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: int = 8,
        alpha: float = 16.0,
        dropout_rate: float = 0.0,
        use_bias: bool = False,
        rngs: nnx.Rngs,
    ) -> None:
        self.base = nnx.Linear(in_features, out_features, use_bias=use_bias, rngs=rngs)
        self.scale = alpha / rank

        key = rngs.params()
        k_a, k_b = jax.random.split(key)
        self.lora_A = LoRAParam(
            jax.random.normal(k_a, (in_features, rank)) / math.sqrt(in_features)
        )
        self.lora_B = LoRAParam(jnp.zeros((rank, out_features)))

        self.dropout: nnx.Dropout | None = (
            nnx.Dropout(dropout_rate, rngs=rngs) if dropout_rate > 0.0 else None
        )

    def __call__(self, x: jnp.ndarray, deterministic: bool = False) -> jnp.ndarray:
        out = self.base(x)
        delta = x @ self.lora_A[...]
        if self.dropout is not None:
            delta = self.dropout(delta, deterministic=deterministic)
        return out + (delta @ self.lora_B[...]) * self.scale

    def merge_weights(self) -> jnp.ndarray:
        """Return fused kernel W + (alpha/r) * A @ B for export or deployment."""
        return self.base.kernel[...] + self.scale * (self.lora_A[...] @ self.lora_B[...])
