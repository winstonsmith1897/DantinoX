import flax.nnx as nnx
import jax
import jax.numpy as jnp

from .config import Config


class Activation(nnx.Module):
    def __init__(self, activation_name: str):
        self.activation_name = activation_name

    def __call__(self, x: jnp.ndarray):
        act_fn = getattr(jax.nn, self.activation_name.lower(), jax.nn.gelu)
        return act_fn(x)

class Swiglu(nnx.Module):
    def __call__(self, x: jnp.ndarray):
        gate, data = jnp.split(x, 2, axis=-1)
        return jax.nn.silu(gate) * data

class MLP(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs):
        intermediate_dim = config.dim * config.expansion
        up_proj_dim = intermediate_dim * 2 if config.use_swiglu else intermediate_dim
        self.up_proj    = nnx.Linear(config.dim, up_proj_dim, rngs=rngs)
        self.down_proj  = nnx.Linear(intermediate_dim, config.dim, rngs=rngs)
        self.activation = Swiglu() if config.use_swiglu else Activation(config.activation)
        self.dropout    = nnx.Dropout(config.dropout_rate, rngs=rngs)
        self.mlp_loss   = 0

    def __call__(self, x: jnp.ndarray, deterministic: bool = False) -> tuple[jnp.ndarray, float]:
        x = self.up_proj(x)
        x = self.activation(x)
        x = self.down_proj(x)
        return self.dropout(x, deterministic=deterministic), self.mlp_loss
