"""Dense feed-forward network for the transformer block.

``MLP`` is the default FFN: up-projection → activation (SwiGLU by default,
any ``jax.nn`` activation via ``config.activation``) → down-projection →
dropout, with optional LoRA on both projections and a tensor-parallel
all-reduce on the output.  It returns ``(out, 0.0)`` so it is call-compatible
with ``MoE`` (whose second element is the load-balancing loss).

The sparse counterpart lives in moe.py; block wiring in block.py.
"""
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from .config import Config, ModelConfig
from .lora import LoRALinear, call_linear
from .sharding import in_mesh_context


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
    def __init__(self, config: ModelConfig | Config, rngs: nnx.Rngs, in_dim: int | None = None):

        intermediate_dim = config.dim * config.expansion
        up_proj_dim = intermediate_dim * 2 if config.use_swiglu else intermediate_dim

        _use_lora_mlp = config.use_lora and config.lora_targets in ("mlp", "ffn", "all")
        _lora_kw: dict = dict(
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            dropout_rate=config.lora_dropout,
            rngs=rngs,
        )

        input_dim: int = config.dim if in_dim is None else in_dim
        self.up_proj: nnx.Linear | LoRALinear = (
            LoRALinear(input_dim, up_proj_dim, **_lora_kw) if _use_lora_mlp
            else nnx.Linear(input_dim, up_proj_dim, rngs=rngs)
        )
        self.down_proj: nnx.Linear | LoRALinear = (
            LoRALinear(intermediate_dim, input_dim, **_lora_kw) if _use_lora_mlp
            else nnx.Linear(intermediate_dim, input_dim, rngs=rngs)
        )
        self.activation = Swiglu() if config.use_swiglu else Activation(config.activation)
        self.dropout    = nnx.Dropout(config.dropout_rate, rngs=rngs)
        self.tp_size: int = config.tp_size

    def __call__(self, x: jnp.ndarray, deterministic: bool = False) -> tuple[jnp.ndarray, float]:
        x = call_linear(self.up_proj, x, deterministic=deterministic)
        x = self.activation(x)
        x = call_linear(self.down_proj, x, deterministic=deterministic)
        # All-reduce partial sums from row-parallel down_proj across TP devices.
        # with_sharding_constraint tells XLA the output must be fully replicated,
        # which triggers an all-reduce over the model-parallel axis.
        if self.tp_size > 1 and in_mesh_context():
            x = jax.lax.with_sharding_constraint(x, P(None, None, None))
        # Dense MLP has no auxiliary loss; the 0.0 keeps the (out, aux)
        # contract shared with MoE.
        return self.dropout(x, deterministic=deterministic), 0.0
