import jax
import jax.numpy as jnp
import flax.nnx as nnx
import math
import yaml
from dataclasses import dataclass, asdict


@dataclass
class Config:
    dim: int = 128
    n_heads: int = 16
    head_size: int = 8
    num_blocks: int = 4
    vocab_size: int = 200
    max_context: int = 110
    use_moe: bool = True
    n_experts: int = 4
    top_k_mlp: int = 2
    expansion: int = 4
    use_rotary_pos: bool = True
    sliding_window: bool = True
    context_window: int = 4
    no_sink: bool = True
    kv_heads: int = None

    def __post_init__(self):
        if self.kv_heads is None:
            self.kv_heads = self.n_heads // 4
        assert self.dim == self.n_heads * self.head_size
        assert self.n_heads % self.kv_heads == 0

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r') as f:
            raw_cfg = yaml.safe_load(f)
        
        flat_cfg = {}
        for section in raw_cfg.values():
            flat_cfg.update(section)
            
        return cls(**flat_cfg)

    def save_yaml(self, path: str):
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f)

class Attention(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs):
        self.max_context:int = config.max_context
        self.head_size:int   = config.head_size
        self.n_heads: int    = config.n_heads
        self.dim: int        = config.dim
        self.kv_heads:int    = config.kv_heads 
        self.qkv: nnx.Linear = nnx.Linear(self.dim, self.dim + 2 * self.kv_heads*self.head_size,
                                       use_bias=False, rngs=rngs)
        self.tril: jnp.ndarray  = jnp.tril(
            jnp.ones((self.max_context, self.max_context), dtype=bool)
        )
        self.o_proj: nnx.Linear = nnx.Linear(self.dim, self.dim, rngs=rngs)
        self.no_sink: bool      = config.no_sink

        self.W: nnx.Linear      = nnx.Linear(self.dim, self.dim, rngs=rngs)

        self.sliding_window: bool = config.sliding_window

        if self.sliding_window:
            table = jnp.arange(self.max_context)[:, None] - jnp.arange(self.max_context)[None, :]
            mask  = (table <= config.context_window)  & (table >= 0)
            self.window = jnp.where(mask, 0, -1e-9)

        self.use_rotary: bool = config.use_rotary_pos
        if self.use_rotary:
            def __compute_angle(T:int, C:int) -> jnp.ndarray:
                P = jnp.arange(T)
                W = 1 / (1000 ** (jnp.arange(C//2) / C))
                degree = jnp.einsum('i,j->ij', P, W)[None, None, None, :, :]
                return degree
            
            self.angle: jnp.ndarray = __compute_angle(self.max_context, self.head_size)

        
    def __apply_rotation(self, x: jnp.ndarray, cache_index: int) -> jnp.ndarray:
        T = x.shape[3]
        odd     = x[:, :, :, :, 0::2]
        even    = x[:, :, :, :, 1::2]

        angle   = jax.lax.dynamic_slice_in_dim(
                self.angle, 
                start_index=cache_index, 
                slice_size=T, 
                axis=3
            )
        x_odd  = jax.lax.cos(angle) * odd - jax.lax.sin(angle) * even
        x_even = jax.lax.sin(angle) * odd + jax.lax.cos(angle) * even

        # y      = jnp.zeros_like(x)
        # y      = y.at[:, :, :, :, 0::2].set(x_odd)
        # y      = y.at[:, :, :, :, 1::2].set(x_even)
        y = jnp.stack([x_even, x_odd], axis=-1).reshape(x.shape)
        return y



    def __call__(self, x: jnp.ndarray, use_cache: bool, kv_cache: tuple, cache_index:int) -> jnp.ndarray:
        B, T, _ = x.shape
        assert T <= self.max_context, "Sequence too Long"

        qkv = self.qkv(x)
        
        q_size  = self.dim
        kv_size = self.kv_heads * self.head_size

        q, k, v = jax.lax.split(
            qkv,
            (q_size, kv_size, kv_size),
            axis=-1
        )

        reshaping = lambda x, n_heads: jnp.reshape(x, (B, T, n_heads, self.head_size))

        q = reshaping(q, self.n_heads).reshape(B, T, self.kv_heads, 
                                               self.n_heads // self.kv_heads, self.head_size)
        
        k, v = map(reshaping, (k, v), (self.kv_heads, self.kv_heads))

        k, v = map(lambda x: jnp.expand_dims(x, axis=3), (k, v))

        permute = lambda x: jnp.transpose(x, (0, 2, 3, 1, 4))

        q, k, v = map(permute, (q, k, v))

        if self.use_rotary:
            q, k = map(self.__apply_rotation, (q, k), (cache_index, cache_index))

        if use_cache:
            if kv_cache == (None, None):  #Prefill Case
                k_cache = jnp.zeros((B, self.kv_heads, 1, self.max_context, self.head_size), dtype=k.dtype)
                v_cache = jnp.zeros((B, self.kv_heads, 1, self.max_context, self.head_size), dtype=v.dtype)
                k_cache, v_cache = k_cache.at[:, :, :, :T, :].set(k), v_cache.at[:, :, :, :T, :].set(v)
            else:
                k_cache, v_cache = map(lambda x, y, index: jax.lax.dynamic_update_slice(x, y, (0, 0, 0, index, 0)), 
                                       (kv_cache[0], kv_cache[1]), (k, v), (cache_index, cache_index))

            kv_cache = (k_cache, v_cache)
            k, v     = k_cache, v_cache

        k = jnp.swapaxes(k, -2, -1)
        attn = q @ k / math.sqrt(self.head_size)
        mask = self.tril[:T, :T] 
        trilled = (~mask) * (-1e9)

        attn = attn + trilled

        if self.sliding_window:
            attn = attn + self.window[:T, :T]
        causal_attn = jax.nn.softmax(attn)

        y = causal_attn @ v

        y = jnp.transpose(y, (0, 3, 1, 2, 4)).reshape(B, T, self.dim)

        if self.no_sink:
            y = y * jax.nn.sigmoid(self.W(x))

        return self.o_proj(y), kv_cache

class Activation(nnx.Module):
    def __call__(self, x: jnp.ndarray):
        return jax.nn.gelu(x)
    
class MLP(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs):
        self.up_proj   = nnx.Linear(config.dim, config.dim*config.expansion, rngs=rngs)
        self.down_proj = nnx.Linear(config.dim * config.expansion, config.dim, rngs=rngs)
        self.mlp       = nnx.Sequential(self.up_proj, Activation(), self.down_proj)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.mlp(x)

class MoE(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs):
        self.n_experts: int       = config.n_experts
        self.experts: list[MLP] = [MLP(config, rngs) for _ in range(self.n_experts)]
        self.router: nnx.Linear   = nnx.Linear(config.dim, self.n_experts, use_bias=False, rngs=rngs)
        self.top_k_mlp: int       = config.top_k_mlp

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x_routed = self.router(x)
        probs    = jax.nn.softmax(x_routed)
        values, indices = jax.lax.top_k(probs, self.top_k_mlp)
        values   = values / jnp.sum(values, axis=-1, keepdims=True)
        y = jnp.zeros_like(x)
        for i in range(self.n_experts):
            mask = (indices == i)
            expert_weight = jnp.sum(jnp.where(mask, values, 0), axis=-1, keepdims=True)
            y = y + (expert_weight * self.experts[i](x))
        return y
            

class Block(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs):
        self.mlp: MLP = MLP(config, rngs)
        self.attention: Attention = Attention(config, rngs)
        self.ln1: nnx.LayerNorm   = nnx.LayerNorm(config.dim, rngs=rngs)
        self.ln2: nnx.LayerNorm   = nnx.LayerNorm(config.dim, rngs=rngs)
        self.use_moe: bool        = config.use_moe
        self.moe: MoE             = MoE(config, rngs)
    def __call__(self, x: jnp.ndarray, use_cache: bool, kv_cache: tuple, cache_index: int) -> jnp.ndarray:
        x_attn, kv_cache = self.attention(self.ln1(x), use_cache=use_cache, kv_cache=kv_cache, cache_index=cache_index)
        x  = x + x_attn
        ff = self.moe(self.ln2(x)) if self.use_moe else self.mlp(self.ln2(x)) 
        x  = x + ff
        return x, kv_cache
    
class Transformer(nnx.Module):
    def __init__(self, config: Config,  rngs: nnx.Rngs):
        self.num_blocks: int     = config.num_blocks
        self.blocks: list[Block] = [Block(config, rngs=rngs) for _ in range(self.num_blocks)]
        self.lm_head: nnx.Linear = nnx.Linear(config.dim, config.vocab_size, rngs=rngs)
        self.wte: nnx.Embed      = nnx.Embed(config.vocab_size, config.dim, rngs=rngs)
        self.trainable_pos: bool = config.trainable_pos
        self.absolute_pos: bool  = config.absolute_pos
        self.max_context: bool   = config.max_context
        if self.trainable_pos:
            self.wpe: nnx.Embed    = nnx.Embed(config.max_context, config.dim, rngs=rngs)
        elif self.absolute_pos:    
            def _build_compute_absolute_pos(T: int, C: int) -> jnp.ndarray:
                pos = jnp.zeros((T, C))
                row = jnp.arange(T)
                col = jnp.arange(0, C, 2)
                k = 1.0 / (10000 ** (col / C))
                
                ratio = jnp.einsum('i,j->ij', row, k)
                pos = pos.at[:, 0::2].set(jnp.sin(ratio))
                pos = pos.at[:, 1::2].set(jnp.cos(ratio))
                
                return jnp.expand_dims(pos, axis=0)

            self.wpe: jnp.ndarray  = _build_compute_absolute_pos(config.max_context, config.dim)

    def __call__(self, 
                 x: jnp.ndarray, 
                 use_cache:bool, 
                 kv_caches: tuple | None, 
                 cache_index: int | None) -> tuple[jnp.ndarray, tuple]:
        B, T = x.shape
        x = self.wte(x)
        if kv_caches is None:
            kv_caches = tuple((None, None) for _ in range(self.num_blocks))
        if self.absolute_pos:
            wpe_slice = jax.lax.dynamic_slice_in_dim(
                self.wpe, 
                start_index=cache_index, 
                slice_size=T, 
                axis=1
            )
            x = x + wpe_slice
        elif self.trainable_pos:
            x = x + self.wpe(jnp.arange(T, dtype=x.dtype))
        new_kv_caches = []
        for i, h in enumerate(self.blocks):
            x, new_kv = h(x, use_cache=use_cache, kv_cache=kv_caches[i], cache_index=cache_index)
            new_kv_caches.append(new_kv)
        return self.lm_head(x), tuple(new_kv_caches)



