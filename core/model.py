import jax
import jax.numpy as jnp
import flax.nnx as nnx
from .config import Config
from .block import Block    


class Transformer(nnx.Module):
    def __init__(self, config: Config,  rngs: nnx.Rngs):
        self.num_blocks: int     = config.num_blocks
        self.blocks: nnx.List    = nnx.List([Block(config, rngs=rngs) for _ in range(self.num_blocks)])
        self.lm_head: nnx.Linear = nnx.Linear(config.dim, config.vocab_size, rngs=rngs)
        self.wte: nnx.Embed      = nnx.Embed(config.vocab_size, config.dim, rngs=rngs)
        self.trainable_pos: bool = config.trainable_pos
        self.absolute_pos: bool  = config.absolute_pos
        self.max_context: bool   = config.max_context
        self.gradient_checkpointing: bool = config.gradient_checkpointing
        self.ln_f: nnx.LayerNorm = nnx.LayerNorm(config.dim, rngs=rngs)
        self.emb_dropout = nnx.Dropout(config.dropout_rate, rngs=rngs)
        self.use_moe: bool        = config.use_moe
        self.alpha_balance: float = config.alpha_balance

        if config.weight_tying:
            self.lm_head.kernel  = self.wte.embedding.T
        if self.trainable_pos:
            self.wpe: nnx.Embed  = nnx.Embed(config.max_context, config.dim, rngs=rngs)
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
                 cache_index: int | None,
                 deterministic: bool = False) -> tuple[jnp.ndarray, tuple]:
        
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

        x = self.emb_dropout(x, deterministic=deterministic)

        def block_fn(block_module, hidden_state, kv_c, det):
            return block_module(
                hidden_state, 
                use_cache=use_cache, 
                kv_cache=kv_c, 
                cache_index=cache_index,
                deterministic=det
            )

        if self.gradient_checkpointing and not use_cache:
            checkpointed_block = nnx.remat(lambda bm, hs, kvc: block_fn(bm, hs, kvc, deterministic))
        else:
            checkpointed_block = lambda bm, hs, kvc: block_fn(bm, hs, kvc, deterministic)

        new_kv_caches = []
        balancing_loss_total = 0
        for i, block in enumerate(self.blocks):
            x, new_kv, balancing_loss = checkpointed_block(block, x, kv_caches[i] if kv_caches else None)
            new_kv_caches.append(new_kv)
            balancing_loss_total += balancing_loss

        x = self.ln_f(x)
        
        return self.lm_head(x), tuple(new_kv_caches), balancing_loss



