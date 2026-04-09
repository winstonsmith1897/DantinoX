
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from .config import Config
from .mlp import MLP 


class MoE(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs):
        self.n_experts: int       = config.n_experts
        self.experts: nnx.List    = nnx.List([MLP(config, rngs) for _ in range(self.n_experts)])
        self.router: nnx.Linear   = nnx.Linear(config.dim, self.n_experts, use_bias=False, rngs=rngs)
        self.top_k_mlp: int       = config.top_k_mlp

    def __call__(self, x: jnp.ndarray, deterministic: bool = False) -> tuple[jnp.ndarray, jnp.ndarray]:
        B, T, _ = x.shape
        x_routed = self.router(x)
        probs    = jax.nn.softmax(x_routed)
        values, indices = jax.lax.top_k(probs, self.top_k_mlp)
        values   = values / jnp.sum(values, axis=-1, keepdims=True)
        y = jnp.zeros_like(x)

        expert_mean_prob = jnp.mean(jnp.reshape(probs, (B*T, self.n_experts)), axis=0)
        freq = jnp.mean(jnp.sum(jax.nn.one_hot(indices, self.n_experts), axis=2), axis=(0, 1))
        moe_loss = jnp.sum(freq*expert_mean_prob) * self.n_experts

        for i in range(self.n_experts):
            mask = (indices == i)
            expert_weight = jnp.sum(jnp.where(mask, values, 0), axis=-1, keepdims=True)
            expert_out, _ = self.experts[i](x, deterministic=deterministic)
            y = y + (expert_weight * expert_out)
        return y, moe_loss 
            