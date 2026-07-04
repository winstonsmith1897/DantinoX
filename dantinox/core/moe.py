"""Sparse Mixture-of-Experts feed-forward network.

``MoE`` routes each token to its top-k experts (softmax router, renormalised
gates) and returns ``(out, aux_loss)`` where *aux_loss* is the
Switch-Transformer load-balancing term consumed by the trainer.  With
``moe_latent=True`` the experts operate in a lower-dimensional latent
(LatentMoE) bracketed by down/up projections.

The dense counterpart lives in mlp.py; block wiring in block.py.
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from .config import Config, ModelConfig
from .mlp import MLP


class _ExpertList(nnx.Module):
    """NNX-compatible container for a fixed-size list of MLP expert modules.

    Stores each expert as a named attribute (``e0``, ``e1``, …).  A plain
    Python list would also be traversed by NNX, but attribute storage gives
    the experts stable *string* keys in checkpoint state dicts — do not
    replace with a list without a checkpoint-migration path.
    """

    def __init__(self, experts: list[MLP]) -> None:
        for i, e in enumerate(experts):
            setattr(self, f"e{i}", e)
        self._n: int = len(experts)

    def __getitem__(self, i: int) -> MLP:
        return getattr(self, f"e{i}")

    def __len__(self) -> int:
        return self._n


class MoE(nnx.Module):
    """Top-k routed Mixture-of-Experts FFN with a load-balancing aux loss.

    Note: every expert is evaluated on every token and masked afterwards
    (dense compute).  This is exact and simple, at O(n_experts) FLOPs of a
    single expert — appropriate at research scale; a dispatch/combine kernel
    would be needed to realise top-k FLOP savings at larger scales.
    """

    def __init__(self, config: ModelConfig | Config, rngs: nnx.Rngs):
        self.n_experts: int         = config.n_experts
        self.latent_dim: int | None = None
        self.moe_latent: bool       = config.moe_latent

        if self.moe_latent:
            self.latent_dim = config.moe_latent_dim
            self.down_proj: nnx.Linear = nnx.Linear(config.dim, self.latent_dim, rngs=rngs)
            self.up_proj: nnx.Linear   = nnx.Linear(self.latent_dim, config.dim, rngs=rngs)

        self.experts: _ExpertList = _ExpertList(
            [MLP(config, rngs, in_dim=self.latent_dim) for _ in range(self.n_experts)]
        )
        self.router: nnx.Linear = nnx.Linear(config.dim, self.n_experts, use_bias=False, rngs=rngs)
        self.top_k_mlp: int     = config.top_k_mlp

    def __call__(self, x: jnp.ndarray, deterministic: bool = False) -> tuple[jnp.ndarray, jnp.ndarray]:
        B, T, _ = x.shape
        x_routed = self.router(x)

        if self.moe_latent:
            x = self.down_proj(x)

        probs    = jax.nn.softmax(x_routed, axis=-1)
        values, indices = jax.lax.top_k(probs, self.top_k_mlp)
        values   = values / jnp.sum(values, axis=-1, keepdims=True)
        y = jnp.zeros_like(x)

        # Switch-Transformer load-balancing loss: n_experts · Σ_e freq_e · P̄_e
        expert_mean_prob = jnp.mean(jnp.reshape(probs, (B * T, self.n_experts)), axis=0)
        freq = jnp.mean(jnp.sum(jax.nn.one_hot(indices, self.n_experts), axis=2), axis=(0, 1))
        moe_loss = jnp.sum(freq * expert_mean_prob) * self.n_experts

        for i in range(self.n_experts):
            mask = (indices == i)
            expert_weight = jnp.sum(jnp.where(mask, values, 0), axis=-1, keepdims=True)
            expert_out, _ = self.experts[i](x, deterministic=deterministic)
            y = y + (expert_weight * expert_out)

        if self.moe_latent:
            y = self.up_proj(y)

        return y, moe_loss

