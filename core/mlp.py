import flax.nnx as nnx
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from .config import Config
from .lora import LoRALinear


class Activation(nnx.Module):
    def __init__(self, activation_name: str):
        self.activation_name = activation_name

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        act_fn = getattr(jax.nn, self.activation_name.lower(), jax.nn.gelu)
        return act_fn(x)

class Swiglu(nnx.Module):
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        gate, data = jnp.split(x, 2, axis=-1)
        return jax.nn.silu(gate) * data

class MLP(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs):
        intermediate_dim = config.dim * config.expansion
        up_proj_dim = intermediate_dim * 2 if config.use_swiglu else intermediate_dim

        _use_lora_mlp = getattr(config, "use_lora", False) and getattr(config, "lora_targets", "attention") in ("mlp", "all")
        _lora_kw: dict = dict(rank=getattr(config, "lora_rank", 8), alpha=getattr(config, "lora_alpha", 16.0), dropout_rate=getattr(config, "lora_dropout", 0.0), rngs=rngs)

        self.up_proj: nnx.Linear | LoRALinear = (
            LoRALinear(config.dim, up_proj_dim, **_lora_kw) if _use_lora_mlp
            else nnx.Linear(config.dim, up_proj_dim, rngs=rngs)
        )
        self.down_proj: nnx.Linear | LoRALinear = (
            LoRALinear(intermediate_dim, config.dim, **_lora_kw) if _use_lora_mlp
            else nnx.Linear(intermediate_dim, config.dim, rngs=rngs)
        )
        self.activation = Swiglu() if config.use_swiglu else Activation(config.activation)
        self.dropout    = nnx.Dropout(config.dropout_rate, rngs=rngs)
        self.mlp_loss   = 0
        self.tp_size: int = getattr(config, "tp_size", 1)

    def __call__(self, x: jnp.ndarray, deterministic: bool = False) -> tuple[jnp.ndarray, float]:
        x = self.up_proj(x)
        x = self.activation(x)
        x = self.down_proj(x)
        # All-reduce partial sums from row-parallel down_proj across TP devices.
        # with_sharding_constraint tells XLA the output must be fully replicated,
        # which triggers an all-reduce over the model-parallel axis.
        if self.tp_size > 1:
            x = jax.lax.with_sharding_constraint(x, P(None, None, None))
        return self.dropout(x, deterministic=deterministic), self.mlp_loss
