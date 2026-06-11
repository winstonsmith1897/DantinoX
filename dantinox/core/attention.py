from __future__ import annotations

import math

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from .config import Config
from .lora import LoRALinear


# ── Abstract base ─────────────────────────────────────────────────────────────

class BaseAttention(nnx.Module):
    """Abstract base for all attention variants.

    Provides shared infrastructure: RoPE positional encoding, causal-mask
    helpers, no-sink gating, and dropout.  Concrete subclasses must override
    ``__call__``; calling the base implementation raises ``NotImplementedError``.

    Dual-cache support (for diffusion inference) is opt-in: subclasses that can
    produce prefix KV tensors override ``extract_kv``; the default returns None.
    """

    def __init__(self, config: Config, rngs: nnx.Rngs) -> None:
        self.max_context    = config.max_context
        self.head_size      = config.head_size
        self.n_heads        = config.n_heads
        self.dim            = config.dim
        self.kv_heads       = config.kv_heads if config.kv_heads is not None else config.n_heads
        self.no_sink        = config.no_sink
        self.use_rotary     = config.use_rotary_pos
        self.use_flash      = config.use_flash_attention
        self.sliding_window = config.sliding_window
        self.inference      = config.inference
        self._rope_scale    = getattr(config, "rope_scale_factor", 1.0)

        self.attn_dropout  = nnx.Dropout(config.dropout_rate, rngs=rngs)
        self.resid_dropout = nnx.Dropout(config.dropout_rate, rngs=rngs)
        self.tp_size: int  = getattr(config, "tp_size", 1)

        # Output projection (with optional LoRA)
        _use_lora = (
            getattr(config, "use_lora", False)
            and getattr(config, "lora_targets", "attention") in ("attention", "all")
        )
        _lk = dict(
            rank=getattr(config, "lora_rank", 8),
            alpha=getattr(config, "lora_alpha", 16.0),
            dropout_rate=getattr(config, "lora_dropout", 0.0),
            rngs=rngs,
        )
        self.o_proj: nnx.Linear | LoRALinear = (
            LoRALinear(self.dim, self.dim, **_lk)
            if _use_lora else nnx.Linear(self.dim, self.dim, rngs=rngs)
        )

        # No-sink gating (Attention Sink, optional)
        if self.no_sink:
            self.W = nnx.Linear(self.dim, self.dim, rngs=rngs)

        # Causal mask buffer
        self.tril = jnp.tril(jnp.ones((self.max_context, self.max_context), dtype=bool))

        if self.sliding_window:
            table = (
                jnp.arange(self.max_context)[:, None]
                - jnp.arange(self.max_context)[None, :]
            )
            mask        = (table <= config.context_window) & (table >= 0)
            self.window = jnp.where(mask, 0.0, -1e9)

        # RoPE frequency table (angle[1,1,1,T,C//2]); subclasses may override.
        self.angle = self._compute_angle(self.max_context, self.head_size)

    # ── RoPE helpers ──────────────────────────────────────────────────────────

    def _compute_angle(self, T: int, C: int) -> jnp.ndarray:
        """Precompute RoPE frequency table [1, 1, 1, T, C//2]."""
        P        = jnp.arange(T, dtype=jnp.float32)
        base     = 10_000.0 * self._rope_scale
        inv_freq = 1.0 / (base ** (jnp.arange(0, C, 2, dtype=jnp.float32) / C))
        degree   = jnp.einsum("i,j->ij", P, inv_freq)
        return degree[None, None, None, :, :]

    def _apply_rope_grouped(self, x: jnp.ndarray, cache_index: int) -> jnp.ndarray:
        """Apply RoPE to a [B, H, G, T, D] grouped-head tensor."""
        T     = x.shape[3]
        angle = jax.lax.dynamic_slice_in_dim(self.angle, cache_index, T, axis=3)
        cos_a, sin_a = jax.lax.cos(angle), jax.lax.sin(angle)
        out = jnp.empty_like(x)
        out = out.at[..., 0::2].set(x[..., 0::2] * cos_a - x[..., 1::2] * sin_a)
        out = out.at[..., 1::2].set(x[..., 0::2] * sin_a + x[..., 1::2] * cos_a)
        return out

    def _apply_rope_thd(self, x: jnp.ndarray, cache_index: int) -> jnp.ndarray:
        """Apply RoPE to a [B, T, H, D] Flash-Attention tensor."""
        T     = x.shape[1]
        angle = jax.lax.dynamic_slice_in_dim(
            self.angle[0, 0, 0], start_index=cache_index, slice_size=T, axis=0,
        )
        angle = angle[None, :, None, :]
        cos_a, sin_a = jnp.cos(angle), jnp.sin(angle)
        out = jnp.empty_like(x)
        out = out.at[..., 0::2].set(x[..., 0::2] * cos_a - x[..., 1::2] * sin_a)
        out = out.at[..., 1::2].set(x[..., 0::2] * sin_a + x[..., 1::2] * cos_a)
        return out

    # ── Shape utilities ───────────────────────────────────────────────────────

    def reshape_head(
        self,
        B: int,
        T: int,
        q: jnp.ndarray,
        k: jnp.ndarray,
        v: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Reshape flat QKV vectors into grouped [B, kv_heads, G, T, head_size]."""
        def _reshape(x: jnp.ndarray, h: int) -> jnp.ndarray:
            return jnp.reshape(x, (B, T, h, self.head_size))

        def _perm(x: jnp.ndarray) -> jnp.ndarray:
            return jnp.transpose(x, (0, 2, 3, 1, 4))

        q = _reshape(q, self.n_heads).reshape(
            B, T, self.kv_heads, self.n_heads // self.kv_heads, self.head_size
        )
        k, v = map(_reshape, (k, v), (self.kv_heads, self.kv_heads))
        k, v = map(lambda z: jnp.expand_dims(z, axis=3), (k, v))
        q, k, v = map(_perm, (q, k, v))
        return q, k, v

    def _apply_attn_mask(
        self,
        attn: jnp.ndarray,
        cache_index: int,
        T: int,
        S: int,
        is_causal: bool,
    ) -> jnp.ndarray:
        if is_causal:
            mask = jax.lax.dynamic_slice(self.tril, (cache_index, 0), (T, S))
            attn = attn + jnp.where(mask, 0.0, -1e9)
        if self.sliding_window:
            wm   = jax.lax.dynamic_slice(self.window, (cache_index, 0), (T, S))
            attn = attn + wm
        return attn

    def _apply_gate(self, y: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        return y * jax.nn.sigmoid(self.W(x)) if self.no_sink else y

    # ── Dual-cache interface (opt-in) ─────────────────────────────────────────

    def extract_kv(
        self, x: jnp.ndarray, cache_index: int
    ) -> tuple[jnp.ndarray, jnp.ndarray] | None:
        """Return (k, v) tensors for prefix caching; ``None`` if unsupported."""
        return None

    # ── Forward pass (must override) ──────────────────────────────────────────

    def __call__(
        self,
        x: jnp.ndarray,
        use_cache: bool,
        kv_cache: tuple,
        cache_index: int,
        deterministic: bool = False,
        is_causal: bool = True,
        prefix_kv: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    ) -> tuple[jnp.ndarray, tuple]:
        raise NotImplementedError(f"{type(self).__name__} must implement __call__")


# ── Standard attention (MHA and GQA share the same forward pass) ──────────────

class _StandardAttention(BaseAttention):
    """Shared forward implementation for MHA and GQA.

    The distinction between the two variants is purely in ``kv_heads``:
    - MHA: ``kv_heads == n_heads``
    - GQA: ``kv_heads < n_heads`` (multiple query heads per KV head)
    Both support LoRA, Flash Attention, KV-cache, and dual-cache prefix injection.
    """

    def __init__(self, config: Config, rngs: nnx.Rngs) -> None:
        super().__init__(config, rngs)
        _use_lora = (
            getattr(config, "use_lora", False)
            and getattr(config, "lora_targets", "attention") in ("attention", "all")
        )
        _lk = dict(
            rank=getattr(config, "lora_rank", 8),
            alpha=getattr(config, "lora_alpha", 16.0),
            dropout_rate=getattr(config, "lora_dropout", 0.0),
            rngs=rngs,
        )
        qkv_out = self.dim + 2 * self.kv_heads * self.head_size
        self.qkv: nnx.Linear | LoRALinear = (
            LoRALinear(self.dim, qkv_out, use_bias=False, **_lk)
            if _use_lora else nnx.Linear(self.dim, qkv_out, use_bias=False, rngs=rngs)
        )

    # ── KV-cache update ───────────────────────────────────────────────────────

    def _update_kv_cache(
        self,
        kv_cache: tuple,
        cache_index: int,
        B: int,
        T: int,
        k: jnp.ndarray,
        v: jnp.ndarray,
    ) -> tuple[tuple, jnp.ndarray, jnp.ndarray]:
        if kv_cache[0] is None:
            kc = jnp.zeros((B, self.kv_heads, 1, self.max_context, self.head_size), dtype=k.dtype)
            vc = jnp.zeros((B, self.kv_heads, 1, self.max_context, self.head_size), dtype=v.dtype)
            kc = kc.at[:, :, :, :T, :].set(k)
            vc = vc.at[:, :, :, :T, :].set(v)
        else:
            kc, vc = map(
                lambda arr, new, idx: jax.lax.dynamic_update_slice(
                    arr, new, (0, 0, 0, idx, 0)
                ),
                (kv_cache[0], kv_cache[1]),
                (k, v),
                (cache_index, cache_index),
            )
        return (kc, vc), kc, vc

    # ── Dual-cache: extract prefix KV ─────────────────────────────────────────

    def extract_kv(
        self, x: jnp.ndarray, cache_index: int
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute and return (k, v) in grouped layout [B, kv_heads, 1, T, head_size]."""
        B, T, _ = x.shape
        q_size  = self.dim
        kv_size = self.kv_heads * self.head_size
        q, k, v = jax.lax.split(self.qkv(x), (q_size, kv_size, kv_size), axis=-1)
        q, k, v = self.reshape_head(B, T, q, k, v)
        if self.use_rotary:
            k = self._apply_rope_grouped(k, cache_index)
        return k, v

    # ── Forward pass ──────────────────────────────────────────────────────────

    def __call__(
        self,
        x: jnp.ndarray,
        use_cache: bool,
        kv_cache: tuple,
        cache_index: int,
        deterministic: bool = False,
        is_causal: bool = True,
        prefix_kv: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    ) -> tuple[jnp.ndarray, tuple]:
        B, T, _ = x.shape
        q_size  = self.dim
        kv_size = self.kv_heads * self.head_size
        q, k, v = jax.lax.split(self.qkv(x), (q_size, kv_size, kv_size), axis=-1)

        # Flash Attention fast path (causal training, no cache, no prefix injection)
        if (
            not use_cache
            and not self.sliding_window
            and self.use_flash
            and is_causal
            and prefix_kv is None
        ):
            q_fa = q.reshape(B, T, self.n_heads,  self.head_size)
            k_fa = k.reshape(B, T, self.kv_heads, self.head_size)
            v_fa = v.reshape(B, T, self.kv_heads, self.head_size)
            if self.use_rotary:
                q_fa = self._apply_rope_thd(q_fa, cache_index)
                k_fa = self._apply_rope_thd(k_fa, cache_index)
            if self.kv_heads < self.n_heads:
                g    = self.n_heads // self.kv_heads
                k_fa = jnp.repeat(k_fa, g, axis=2)
                v_fa = jnp.repeat(v_fa, g, axis=2)
            y   = jax.nn.dot_product_attention(q_fa, k_fa, v_fa, is_causal=True)
            y   = y.reshape(B, T, self.dim)
            y   = self._apply_gate(y, x)
            out = self.o_proj(y)
            # All-reduce partial sums from row-parallel o_proj across TP devices.
            if self.tp_size > 1:
                out = jax.lax.with_sharding_constraint(out, P(None, None, None))
            return self.resid_dropout(out, deterministic=deterministic), kv_cache

        # General path (cache / sliding window / bidirectional diffusion)
        q, k, v = self.reshape_head(B, T, q, k, v)
        if self.use_rotary:
            q = self._apply_rope_grouped(q, cache_index)
            k = self._apply_rope_grouped(k, cache_index)
        if use_cache:
            kv_cache, k, v = self._update_kv_cache(kv_cache, cache_index, B, T, k, v)

        # Dual-cache prefix injection: prepend prefix K/V along the sequence axis
        if prefix_kv is not None:
            pk, pv = prefix_kv
            k = jnp.concatenate([pk, k], axis=3)  # [B, kv_heads, 1, T_pre+T, head_size]
            v = jnp.concatenate([pv, v], axis=3)

        attn = q @ jnp.swapaxes(k, -2, -1) / math.sqrt(self.head_size)
        S    = attn.shape[-1]
        attn = self._apply_attn_mask(attn, cache_index, T, S, is_causal)
        attn = jax.nn.softmax(attn)
        attn = self.attn_dropout(attn, deterministic=deterministic)

        y   = attn @ v
        y   = jnp.transpose(y, (0, 3, 1, 2, 4)).reshape(B, T, self.dim)
        y   = self._apply_gate(y, x)
        out = self.o_proj(y)
        if self.tp_size > 1:
            out = jax.lax.with_sharding_constraint(out, P(None, None, None))
        return self.resid_dropout(out, deterministic=deterministic), kv_cache


class MHAAttention(_StandardAttention):
    """Multi-Head Attention — every head has its own key/value projection."""


class GQAAttention(_StandardAttention):
    """Grouped-Query Attention — multiple query heads share each key/value head."""


# ── Multi-Head Latent Attention ───────────────────────────────────────────────

class MLAAttention(BaseAttention):
    """Multi-Head Latent Attention (DeepSeek-V2 style).

    Keys and values are first projected to a low-dimensional latent (down_kv),
    then re-expanded as needed, dramatically shrinking KV-cache memory at
    inference time.  Queries undergo an analogous compression (down_q).

    At inference time (``config.inference=True``) the up-projection matrices
    for K and V are absorbed into the attention computation via einsum, so the
    cache stores the compact latent rather than the full-sized KV tensors.
    """

    def __init__(self, config: Config, rngs: nnx.Rngs) -> None:
        super().__init__(config, rngs)
        self.rope_dim    = config.rope_dim if hasattr(config, "rope_dim") else self.head_size // 2
        self.down_dim_q  = config.down_dim_q
        self.down_dim_kv = config.down_dim_kv

        self.down_q  = nnx.Linear(config.dim, config.down_dim_q,  rngs=rngs)
        self.down_kv = nnx.Linear(config.dim, config.down_dim_kv, rngs=rngs)
        self.up_q    = nnx.Linear(config.down_dim_q,  self.head_size * self.n_heads,  rngs=rngs)
        self.up_k    = nnx.Linear(config.down_dim_kv, self.head_size * self.kv_heads, rngs=rngs)
        self.up_v    = nnx.Linear(config.down_dim_kv, self.head_size * self.kv_heads, rngs=rngs)

        self.q_pe   = nnx.Linear(config.dim, self.rope_dim, rngs=rngs)
        self.k_pe   = nnx.Linear(config.dim, self.rope_dim, rngs=rngs)
        self.norm_q  = nnx.LayerNorm(config.down_dim_q,  rngs=rngs)
        self.norm_kv = nnx.LayerNorm(config.down_dim_kv, rngs=rngs)

        # MLA uses a smaller rope_dim; override the angle table from BaseAttention.
        self.angle = self._compute_angle(self.max_context, self.rope_dim)

    # ── MLA KV-cache update (stores compact latent, not full KV) ─────────────

    def _update_mla_cache(
        self,
        kv_cache: tuple,
        cache_index: int,
        B: int,
        T: int,
        c_kv: jnp.ndarray,
        k_rope: jnp.ndarray,
    ) -> tuple[tuple, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Returns (new_kv_cache, k_rope_cache, c_cache, c_cache).

        The last two values are both the latent ``c_cache``: in the absorbed
        MLA attention computation the same compressed latent is used as both
        the key-side operand (k) and the value-side operand (v).
        """
        if kv_cache[0] is None:
            c_cache = jnp.zeros((B, self.max_context, self.down_dim_kv), dtype=c_kv.dtype)
            r_cache = jnp.zeros((B, 1, 1, self.max_context, self.rope_dim), dtype=c_kv.dtype)
            c_cache = c_cache.at[:, :T, :].set(c_kv)
            r_cache = r_cache.at[:, :, :, :T, :].set(k_rope)
        else:
            c_cache = jax.lax.dynamic_update_slice(kv_cache[0], c_kv, (0, cache_index, 0))
            r_cache = jax.lax.dynamic_update_slice(
                kv_cache[1], k_rope, (0, 0, 0, cache_index, 0)
            )
        return (c_cache, r_cache), r_cache, c_cache, c_cache

    # ── Forward pass ──────────────────────────────────────────────────────────

    def __call__(
        self,
        x: jnp.ndarray,
        use_cache: bool,
        kv_cache: tuple,
        cache_index: int,
        deterministic: bool = False,
        is_causal: bool = True,
        prefix_kv: tuple | None = None,
    ) -> tuple[jnp.ndarray, tuple]:
        B, T, _ = x.shape
        q    = self.norm_q(self.down_q(x))
        c_kv = self.norm_kv(self.down_kv(x))

        if self.use_rotary:
            q_rope = self.q_pe(x)[:, None, None, :, :]
            k_rope = self.k_pe(x)[:, None, None, :, :]
            q_rope = self._apply_rope_grouped(q_rope, cache_index)
            k_rope = self._apply_rope_grouped(k_rope, cache_index)

        if not self.inference:
            # Training path: explicit up-projection
            q_up = self.up_q(q)
            k_up = self.up_k(c_kv)
            v_up = self.up_v(c_kv)
            q_up, k_up, v_up = self.reshape_head(B, T, q_up, k_up, v_up)
            q_rope_b = jnp.broadcast_to(q_rope, q_up.shape[:-1] + (self.rope_dim,))
            k_rope_b = jnp.broadcast_to(k_rope, k_up.shape[:-1] + (self.rope_dim,))
            q_full   = jnp.concatenate([q_up, q_rope_b], axis=-1)
            k_full   = jnp.concatenate([k_up, k_rope_b], axis=-1)
            attn     = q_full @ jnp.swapaxes(k_full, -2, -1) / math.sqrt(
                self.head_size + self.rope_dim
            )
            v = v_up
        else:
            # Inference path: absorbed projection (latent cache)
            if use_cache:
                kv_cache, k_rope, k, v = self._update_mla_cache(
                    kv_cache, cache_index, B, T, c_kv, k_rope
                )
            else:
                k = v = c_kv
            q_proj    = self.up_q.kernel.reshape(
                self.down_dim_q, self.kv_heads, self.n_heads // self.kv_heads, self.head_size
            )
            k_proj    = self.up_k.kernel.reshape(self.down_dim_kv, self.kv_heads, self.head_size)
            attn_proj = jnp.einsum("qngh, knh -> ngqk", q_proj, k_proj)
            attn_proj = jnp.einsum("btq, ngqk -> btngk", q, attn_proj)
            attn      = jnp.einsum("btngk, bsk -> bngts", attn_proj, k)
            attn_rope = q_rope @ jnp.swapaxes(k_rope, -2, -1)
            attn      = (attn + attn_rope) / math.sqrt(self.head_size + self.rope_dim)

        S    = attn.shape[-1]
        attn = self._apply_attn_mask(attn, cache_index, T, S, is_causal)
        attn = jax.nn.softmax(attn)
        attn = self.attn_dropout(attn, deterministic=deterministic)

        if self.inference:
            L   = jnp.einsum("bngts, bsd -> bngtd", attn, v)
            W_v = self.up_v.kernel.reshape(self.down_dim_kv, self.kv_heads, self.head_size)
            if self.no_sink:
                y_heads = jnp.einsum("bngtd, dnh -> bngth", L, W_v)
                y   = jnp.transpose(y_heads, (0, 3, 1, 2, 4)).reshape(B, T, self.dim)
                y   = self._apply_gate(y, x)
                out = self.o_proj(y)
            else:
                W_o  = self.o_proj.kernel.reshape(  # type: ignore[union-attr]
                    self.kv_heads, self.n_heads // self.kv_heads, self.head_size, self.dim
                )
                W_vo = jnp.einsum("dnh, nghc -> dngc", W_v, W_o)
                out  = jnp.einsum("bngtd, dngc -> btc", L, W_vo)
        else:
            y   = attn @ v
            y   = jnp.transpose(y, (0, 3, 1, 2, 4)).reshape(B, T, self.dim)
            y   = self._apply_gate(y, x)
            out = self.o_proj(y)

        return self.resid_dropout(out, deterministic=deterministic), kv_cache


# ── Factory ───────────────────────────────────────────────────────────────────

def build_attention(config: Config, rngs: nnx.Rngs) -> BaseAttention:
    """Return the attention variant specified by ``config.attention_type``.

    ``"auto"`` falls back to the legacy ``config.mla`` / ``config.kv_heads``
    flags so existing configs continue to work without modification.
    """
    t = getattr(config, "attention_type", "auto")
    if t == "mla":
        return MLAAttention(config, rngs)
    if t == "gqa":
        return GQAAttention(config, rngs)
    if t == "mha":
        return MHAAttention(config, rngs)
    # "auto" fallback
    if getattr(config, "mla", False):
        return MLAAttention(config, rngs)
    if (config.kv_heads or config.n_heads) < config.n_heads:
        return GQAAttention(config, rngs)
    return MHAAttention(config, rngs)


# Backward-compatible shim for code that does `from dantinox.core.attention import Attention`
def Attention(config: Config, rngs: nnx.Rngs) -> BaseAttention:
    """Deprecated — use ``build_attention()`` or a concrete attention class."""
    return build_attention(config, rngs)
