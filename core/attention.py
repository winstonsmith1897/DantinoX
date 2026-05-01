import math

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from .config import Config


class Attention(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs):
        self.max_context: int = config.max_context
        self.head_size: int   = config.head_size
        self.n_heads: int     = config.n_heads
        self.dim: int         = config.dim
        self.kv_heads: int    = config.kv_heads if config.kv_heads is not None else self.n_heads
        self.qkv: nnx.Linear  = nnx.Linear(self.dim,
                                            self.dim + 2 * self.kv_heads * self.head_size,
                                            use_bias=False,
                                            rngs=rngs)
        self.tril: jnp.ndarray = jnp.tril(
            jnp.ones((self.max_context, self.max_context), dtype=bool)
        )
        self.o_proj: nnx.Linear = nnx.Linear(self.dim, self.dim, rngs=rngs)
        self.no_sink: bool      = config.no_sink
        self.W: nnx.Linear      = nnx.Linear(self.dim, self.dim, rngs=rngs)

        self.sliding_window: bool = config.sliding_window
        if self.sliding_window:
            table = jnp.arange(self.max_context)[:, None] - jnp.arange(self.max_context)[None, :]
            mask  = (table <= config.context_window) & (table >= 0)
            self.window = jnp.where(mask, 0, -1e9)

        self.use_rotary: bool        = config.use_rotary_pos
        self.use_flash_attention: bool = config.use_flash_attention

        self.attn_dropout:  nnx.Dropout = nnx.Dropout(config.dropout_rate, rngs=rngs)
        self.resid_dropout: nnx.Dropout = nnx.Dropout(config.dropout_rate, rngs=rngs)

        # MLA projections (always allocated; only used when config.mla=True)
        self.down_q:  nnx.Linear = nnx.Linear(config.dim, config.down_dim_q,  rngs=rngs)
        self.down_kv: nnx.Linear = nnx.Linear(config.dim, config.down_dim_kv, rngs=rngs)
        self.up_q:    nnx.Linear = nnx.Linear(config.down_dim_q,  config.head_size * config.n_heads,   rngs=rngs)
        self.up_k:    nnx.Linear = nnx.Linear(config.down_dim_kv, config.head_size * config.kv_heads,  rngs=rngs)
        self.up_v:    nnx.Linear = nnx.Linear(config.down_dim_kv, config.head_size * config.kv_heads,  rngs=rngs)

        self.down_dim_q:  int = config.down_dim_q
        self.down_dim_kv: int = config.down_dim_kv
        self.mla:         bool = config.mla
        self.inference:   bool = config.inference
        self.rope_dim:    int  = config.rope_dim if hasattr(config, "rope_dim") else self.head_size // 2

        if self.mla:
            self.q_pe:   nnx.Linear  = nnx.Linear(config.dim, self.rope_dim, rngs=rngs)
            self.k_pe:   nnx.Linear  = nnx.Linear(config.dim, self.rope_dim, rngs=rngs)
            self.norm_q  = nnx.LayerNorm(config.down_dim_q,  rngs=rngs)
            self.norm_kv = nnx.LayerNorm(config.down_dim_kv, rngs=rngs)

        # RoPE frequency table — scaled by rope_scale_factor for NTK-aware extension
        rope_dim_size      = self.head_size if not self.mla else self.rope_dim
        self._rope_scale   = getattr(config, "rope_scale_factor", 1.0)
        self.angle: jnp.ndarray = self._compute_angle(self.max_context, rope_dim_size)

    # ── RoPE helpers ──────────────────────────────────────────────────────────

    def _compute_angle(self, T: int, C: int) -> jnp.ndarray:
        """Precompute RoPE angles [1, 1, 1, T, C//2].

        When ``rope_scale_factor > 1`` the base frequency is compressed
        (NTK-aware scaling), allowing the model to generalise to contexts
        longer than ``max_context`` without fine-tuning.
        """
        P        = jnp.arange(T, dtype=jnp.float32)
        base     = 10000.0 * self._rope_scale
        inv_freq = 1.0 / (base ** (jnp.arange(0, C, 2, dtype=jnp.float32) / C))
        degree   = jnp.einsum("i,j->ij", P, inv_freq)
        return degree[None, None, None, :, :]   # [1, 1, 1, T, C//2]

    def _apply_rope_grouped(self, x: jnp.ndarray, cache_index: int) -> jnp.ndarray:
        """Apply RoPE to a tensor in grouped [B, H, G, T, D] layout (cache path)."""
        T     = x.shape[3]
        angle = jax.lax.dynamic_slice_in_dim(self.angle, cache_index, T, axis=3)
        cos_a, sin_a = jax.lax.cos(angle), jax.lax.sin(angle)
        out = jnp.empty_like(x)
        out = out.at[..., 0::2].set(x[..., 0::2] * cos_a - x[..., 1::2] * sin_a)
        out = out.at[..., 1::2].set(x[..., 0::2] * sin_a + x[..., 1::2] * cos_a)
        return out

    # kept as an alias so MLA code that calls __apply_rotation still works
    __apply_rotation = _apply_rope_grouped

    def _apply_rope_thd(self, x: jnp.ndarray, cache_index: int) -> jnp.ndarray:
        """Apply RoPE to a tensor in [B, T, H, D] layout (Flash Attention path)."""
        T     = x.shape[1]                           # static under JIT
        angle = jax.lax.dynamic_slice_in_dim(
            self.angle[0, 0, 0],                     # [T_max, D//2]
            start_index=cache_index, slice_size=T, axis=0,
        )                                             # [T, D//2]
        angle = angle[None, :, None, :]               # [1, T, 1, D//2]
        cos_a, sin_a = jnp.cos(angle), jnp.sin(angle)
        out = jnp.empty_like(x)
        out = out.at[..., 0::2].set(x[..., 0::2] * cos_a - x[..., 1::2] * sin_a)
        out = out.at[..., 1::2].set(x[..., 0::2] * sin_a + x[..., 1::2] * cos_a)
        return out

    # ── Cache helpers ─────────────────────────────────────────────────────────

    def _compute_cache(self, kv_cache, cache_index, B, T, k=None, v=None, c_kv=None, k_rope=None):
        if self.mla is False:
            if kv_cache == (None, None):
                k_cache = jnp.zeros((B, self.kv_heads, 1, self.max_context, self.head_size), dtype=k.dtype)
                v_cache = jnp.zeros((B, self.kv_heads, 1, self.max_context, self.head_size), dtype=v.dtype)
                k_cache = k_cache.at[:, :, :, :T, :].set(k)
                v_cache = v_cache.at[:, :, :, :T, :].set(v)
            else:
                k_cache, v_cache = map(
                    lambda x, y, index: jax.lax.dynamic_update_slice(x, y, (0, 0, 0, index, 0)),
                    (kv_cache[0], kv_cache[1]), (k, v), (cache_index, cache_index)
                )
            kv_cache     = (k_cache, v_cache)
            k, v         = k_cache, v_cache
            k_rope_cache = None
        else:
            if kv_cache == (None, None):
                c_cache      = jnp.zeros((B, self.max_context, self.down_dim_kv), dtype=c_kv.dtype)
                c_cache      = c_cache.at[:, :T, :].set(c_kv)
                k_rope_cache = jnp.zeros((B, 1, 1, self.max_context, self.rope_dim), dtype=c_kv.dtype)
                k_rope_cache = k_rope_cache.at[:, :, :, :T, :].set(k_rope)
            else:
                c_cache      = jax.lax.dynamic_update_slice(kv_cache[0], c_kv, (0, cache_index, 0))
                k_rope_cache = jax.lax.dynamic_update_slice(kv_cache[1], k_rope, (0, 0, 0, cache_index, 0))
            kv_cache = (c_cache, k_rope_cache)
            k, v     = c_cache, c_cache
        return kv_cache, k_rope_cache, k, v

    # ── Head reshape (grouped layout for cache path) ──────────────────────────

    def reshape_head(self, B, T, q, k, v):
        def _reshaping(x, n_heads):
            return jnp.reshape(x, (B, T, n_heads, self.head_size))

        def _permute(x):
            return jnp.transpose(x, (0, 2, 3, 1, 4))

        q = _reshaping(q, self.n_heads).reshape(
            B, T, self.kv_heads, self.n_heads // self.kv_heads, self.head_size
        )
        k, v = map(_reshaping, (k, v), (self.kv_heads, self.kv_heads))
        k, v = map(lambda x: jnp.expand_dims(x, axis=3), (k, v))
        q, k, v = map(_permute, (q, k, v))
        return q, k, v

    def _compute_attention(self, use_cache, kv_cache, cache_index, B, T, q, k, v):
        """Manual (non-Flash) attention for the cache/sliding-window path."""
        q, k, v = self.reshape_head(B, T, q, k, v)
        if self.use_rotary:
            q, k = map(self._apply_rope_grouped, (q, k), (cache_index, cache_index))
        if use_cache:
            kv_cache, _, k, v = self._compute_cache(kv_cache, cache_index, B, T, k, v)
        k   = jnp.swapaxes(k, -2, -1)
        attn = q @ k / math.sqrt(self.head_size)
        return kv_cache, q, k, v, attn

    # ── Forward ───────────────────────────────────────────────────────────────

    def __call__(
        self,
        x: jnp.ndarray,
        use_cache: bool,
        kv_cache: tuple,
        cache_index: int,
        deterministic: bool = False,
    ) -> tuple[jnp.ndarray, tuple]:

        B, T, _ = x.shape
        assert self.max_context >= T, "Sequence too long"

        # ── Flash Attention fast path (MHA/GQA, training only) ────────────────
        # Conditions: not MLA, no KV-cache (training), no sliding window.
        # Uses jax.nn.dot_product_attention for fused scaled-dot-product with
        # a causal mask. Requires JAX >= 0.4.25.
        if self.mla is False and not use_cache and not self.sliding_window and self.use_flash_attention:
            q_size  = self.dim
            kv_size = self.kv_heads * self.head_size
            qkv     = self.qkv(x)
            q, k, v = jax.lax.split(qkv, (q_size, kv_size, kv_size), axis=-1)

            # [B, T, H, D] layout expected by dot_product_attention
            q_fa = q.reshape(B, T, self.n_heads,  self.head_size)
            k_fa = k.reshape(B, T, self.kv_heads, self.head_size)
            v_fa = v.reshape(B, T, self.kv_heads, self.head_size)

            if self.use_rotary:
                q_fa = self._apply_rope_thd(q_fa, cache_index)
                k_fa = self._apply_rope_thd(k_fa, cache_index)

            # GQA: broadcast k/v to full head count for JAX < 0.4.31 compat
            if self.kv_heads < self.n_heads:
                g    = self.n_heads // self.kv_heads
                k_fa = jnp.repeat(k_fa, g, axis=2)
                v_fa = jnp.repeat(v_fa, g, axis=2)

            y = jax.nn.dot_product_attention(q_fa, k_fa, v_fa, is_causal=True)
            # y: [B, T, n_heads, head_size] → [B, T, dim]
            y = y.reshape(B, T, self.dim)

            if self.no_sink:
                y = y * jax.nn.sigmoid(self.W(x))
            out = self.o_proj(y)
            out = self.resid_dropout(out, deterministic=deterministic)
            return out, kv_cache

        # ── MLA path ──────────────────────────────────────────────────────────
        if self.mla is True:
            q    = self.down_q(x)
            c_kv = self.down_kv(x)
            q    = self.norm_q(q)
            c_kv = self.norm_kv(c_kv)

            if self.use_rotary:
                q_rope = self.q_pe(x)[:, None, None, :, :]
                k_rope = self.k_pe(x)[:, None, None, :, :]
                q_rope, k_rope = map(
                    self._apply_rope_grouped, (q_rope, k_rope), (cache_index, cache_index)
                )

            if self.inference is False:
                q = self.up_q(q)
                k, v = map(lambda f, vec: f(vec), (self.up_k, self.up_v), (c_kv, c_kv))
                q, k, v = self.reshape_head(B, T, q, k, v)

                q_rope_ext, k_rope_ext = map(
                    lambda x, m: jnp.broadcast_to(x, m.shape[:-1] + (self.rope_dim,)),
                    (q_rope, k_rope), (q, k)
                )
                q_full = jnp.concatenate([q, q_rope_ext], axis=-1)
                k_full = jnp.concatenate([k, k_rope_ext], axis=-1)
                k_full = jnp.swapaxes(k_full, -2, -1)
                attn   = q_full @ k_full / math.sqrt(self.head_size + self.rope_dim)
            else:
                if use_cache:
                    kv_cache, k_rope, k, v = self._compute_cache(
                        kv_cache, cache_index, B, T, c_kv=c_kv, k_rope=k_rope
                    )
                else:
                    k = c_kv
                    v = c_kv

                q_proj    = self.up_q.kernel.reshape(
                    self.down_dim_q, self.kv_heads, self.n_heads // self.kv_heads, self.head_size
                )
                k_proj    = self.up_k.kernel.reshape(self.down_dim_kv, self.kv_heads, self.head_size)
                attn_proj = jnp.einsum("qngh, knh -> ngqk", q_proj, k_proj)
                attn_proj = jnp.einsum("btq, ngqk -> btngk", q, attn_proj)
                attn      = jnp.einsum("btngk, bsk -> bngts", attn_proj, k)
                attn_rope = q_rope @ jnp.swapaxes(k_rope, -2, -1)
                attn      = (attn + attn_rope) / math.sqrt(self.head_size + self.rope_dim)

        # ── Manual MHA/GQA path (cache or sliding window) ─────────────────────
        elif self.mla is False:
            q_size  = self.dim
            kv_size = self.kv_heads * self.head_size
            qkv     = self.qkv(x)
            q, k, v = jax.lax.split(qkv, (q_size, kv_size, kv_size), axis=-1)
            kv_cache, q, k, v, attn = self._compute_attention(
                use_cache, kv_cache, cache_index, B, T, q, k, v
            )

        # ── Causal mask + softmax ─────────────────────────────────────────────
        S    = attn.shape[-1]
        mask = jax.lax.dynamic_slice(self.tril, (cache_index, 0), (T, S))
        attn = attn + jnp.where(mask, 0.0, -1e9)

        if self.sliding_window:
            window_mask = jax.lax.dynamic_slice(self.window, (cache_index, 0), (T, S))
            attn = attn + window_mask

        causal_attn = jax.nn.softmax(attn)
        causal_attn = self.attn_dropout(causal_attn, deterministic=deterministic)

        # ── Output projection ─────────────────────────────────────────────────
        if self.inference is True and self.mla is True:
            L   = jnp.einsum("bngts, bsd -> bngtd", causal_attn, v)
            W_v = self.up_v.kernel.reshape(self.down_dim_kv, self.kv_heads, self.head_size)
            if self.no_sink:
                y_heads = jnp.einsum("bngtd, dnh -> bngth", L, W_v)
                y       = jnp.transpose(y_heads, (0, 3, 1, 2, 4)).reshape(B, T, self.dim)
                y       = y * jax.nn.sigmoid(self.W(x))
                out     = self.o_proj(y)
            else:
                W_o  = self.o_proj.kernel.reshape(
                    self.kv_heads, self.n_heads // self.kv_heads, self.head_size, self.dim
                )
                W_vo = jnp.einsum("dnh, nghc -> dngc", W_v, W_o)
                out  = jnp.einsum("bngtd, dngc -> btc", L, W_vo)
        else:
            y = causal_attn @ v
            y = jnp.transpose(y, (0, 3, 1, 2, 4)).reshape(B, T, self.dim)
            if self.no_sink:
                y = y * jax.nn.sigmoid(self.W(x))
            out = self.o_proj(y)

        out = self.resid_dropout(out, deterministic=deterministic)
        return out, kv_cache
