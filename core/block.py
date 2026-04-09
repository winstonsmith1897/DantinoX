import jax.numpy as jnp
import flax.nnx as nnx
from .config import Config
from .attention import Attention 
from .mlp import MLP             
from .moe import MoE             



class Block(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs):
        self.attention: Attention = Attention(config, rngs)
        self.ln1: nnx.LayerNorm   = nnx.LayerNorm(config.dim, rngs=rngs)
        self.ln2: nnx.LayerNorm   = nnx.LayerNorm(config.dim, rngs=rngs)
        self.use_moe: bool        = config.use_moe
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
        ff, balancing_loss = self.moe(self.ln2(x), deterministic=deterministic) if self.use_moe else self.mlp(self.ln2(x), deterministic=deterministic) 
        x  = x + ff
        return x, kv_cache, balancing_loss