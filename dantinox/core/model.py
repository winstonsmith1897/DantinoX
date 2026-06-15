from __future__ import annotations

import contextlib

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from .block import Block, RMSNorm, _build_norm
from .config import Config, ModelConfig
from .diffusion import DualCache
from .output import ModelOutput


def _to_model_config(config: ModelConfig | Config) -> ModelConfig:
    """Accept either the new ModelConfig or the legacy monolithic Config."""
    if isinstance(config, ModelConfig) and not isinstance(config, Config):
        return config
    return config.to_model_config()  # type: ignore[attr-defined]


# ── Unified Transformer ────────────────────────────────────────────────────────

class Transformer(nnx.Module, pytree=False):
    """Composable transformer for autoregressive and masked-diffusion modelling.

    A single class replaces the old ``Transformer`` (AR) and
    ``DiffusionTransformer`` (diffusion).  The ``causal`` flag in
    ``ModelConfig`` drives the behaviour:

    - ``causal=True``  — standard causal AR transformer with KV-cache support.
    - ``causal=False`` — bidirectional LLaDA-style transformer with optional
      dual-cache inference helpers.

    Quick-start
    -----------
    **String-based** (serialisable to YAML, recommended for experiments)::

        config = ModelConfig(
            dim=512, n_heads=16, head_size=32, num_blocks=12,
            attention="gqa", kv_heads=4, causal=False,
            vocab_size=tok.vocab_size,
        )
        model = Transformer(config, rngs=nnx.Rngs(42))

    **Class-based builder** (most explicit, zero magic strings)::

        from dantinox.core.attention import GQAAttention
        model = Transformer.build(
            dim=512, n_heads=16, head_size=32, num_blocks=12,
            attention=GQAAttention, kv_heads=4, causal=False,
            vocab_size=tok.vocab_size, max_context=512,
            rngs=nnx.Rngs(42),
        )

    **Legacy Config** (trainer/CLI unchanged)::

        model = Transformer(config, rngs=nnx.Rngs(42))   # Config auto-converted
    """

    def __init__(self, config: ModelConfig | Config, rngs: nnx.Rngs) -> None:
        cfg = _to_model_config(config)

        self.num_blocks: int              = cfg.num_blocks
        self.blocks: list[Block]          = [Block(cfg, rngs=rngs) for _ in range(cfg.num_blocks)]
        self.embed: nnx.Embed             = nnx.Embed(cfg.vocab_size, cfg.dim, rngs=rngs)
        self.norm_f: nnx.Module           = _build_norm(cfg, cfg.dim, rngs)
        self.emb_dropout                  = nnx.Dropout(cfg.dropout_rate, rngs=rngs)
        self.weight_tying: bool           = cfg.weight_tying
        self.causal: bool                 = cfg.causal
        self.max_context: int             = cfg.max_context
        self.gradient_checkpointing: bool = cfg.gradient_checkpointing
        self.use_moe: bool                = cfg.use_moe
        self.moe_balance_coeff: float     = cfg.moe_balance_coeff
        self.pos_encoding: str            = cfg.pos_encoding

        if cfg.weight_tying:
            self.head: nnx.Linear | None = None
        else:
            self.head = nnx.Linear(cfg.dim, cfg.vocab_size, rngs=rngs)

        if cfg.pos_encoding == "learned":
            self.wpe: nnx.Embed = nnx.Embed(cfg.max_context, cfg.dim, rngs=rngs)
        elif cfg.pos_encoding == "absolute":
            self.wpe: jnp.ndarray = self._build_sinusoidal(cfg.max_context, cfg.dim)  # type: ignore[assignment]

    # ── Positional encoding helpers ────────────────────────────────────────────

    @staticmethod
    def _build_sinusoidal(T: int, C: int) -> jnp.ndarray:
        row = jnp.arange(T)
        col = jnp.arange(0, C, 2)
        k   = 1.0 / (10000 ** (col / C))
        ratio = jnp.einsum("i,j->ij", row, k)
        pos   = jnp.zeros((T, C))
        pos   = pos.at[:, 0::2].set(jnp.sin(ratio))
        pos   = pos.at[:, 1::2].set(jnp.cos(ratio))
        return jnp.expand_dims(pos, axis=0)   # [1, T, C]

    def _add_pos(self, x: jnp.ndarray, cache_index: int) -> jnp.ndarray:
        T = x.shape[1]
        if self.pos_encoding == "learned":
            return x + self.wpe(jnp.arange(T, dtype=jnp.int32))
        if self.pos_encoding == "absolute":
            wpe_slice = jax.lax.dynamic_slice_in_dim(
                self.wpe, start_index=cache_index, slice_size=T, axis=1  # type: ignore[arg-type]
            )
            return x + wpe_slice
        return x  # "rotary" and "none": positional info is in attention, not added here

    # ── Backward-compat properties ────────────────────────────────────────────

    @property
    def alpha_balance(self) -> float:
        return self.moe_balance_coeff

    @property
    def lm_head(self):
        return self.head

    @property
    def wte(self):
        return self.embed

    # ── Forward pass ──────────────────────────────────────────────────────────

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        caches: tuple | None = None,
        cache_index: int = 0,
        dual_cache: DualCache | None = None,
        deterministic: bool = False,
    ) -> ModelOutput:
        """Run the transformer forward pass.

        Parameters
        ----------
        x:             Token IDs ``[B, T]``.
        caches:        Per-layer KV cache for AR generation.  Each element is
                       ``(k_cache, v_cache)`` for standard attention, or
                       ``(k_cache, v_cache, k2_cache)`` when differential
                       attention is active.  Pass
                       ``tuple((None, None) for _ in range(model.num_blocks))``
                       to initialise a fresh cache on the first token step.
                       ``None`` (default) disables caching entirely (training).
        cache_index:   Write position for the AR KV cache.
        dual_cache:    Bidirectional prefix KV cache for diffusion inference.
        deterministic: Disables dropout.

        Returns
        -------
        ``ModelOutput(logits, kv_caches, aux_loss)``
        """
        use_cache = (caches is not None)

        h = self.embed(x)
        h = self._add_pos(h, cache_index)
        h = self.emb_dropout(h, deterministic=deterministic)

        # Block-level caches: (None, None) sentinel = "create cache on first step".
        # After the first step each block returns (kc, vc) or (kc, vc, k2c)
        # for differential attention; subsequent calls pass those tuples back.
        block_caches: tuple = (
            caches if use_cache
            else tuple((None, None) for _ in range(self.num_blocks))
        )
        prefix_kvs: tuple = (
            dual_cache.prefix_kvs if dual_cache is not None
            else (None,) * self.num_blocks
        )

        use_remat = (
            self.gradient_checkpointing
            and not deterministic
            and not use_cache
            and dual_cache is None
        )

        if use_remat:
            def _block_fn(block: object, hs: jnp.ndarray) -> tuple:
                return block(hs, deterministic=False)  # type: ignore[call-arg, operator]
            _checkpointed = nnx.remat(_block_fn)

        new_caches: list      = []
        balancing_loss: float = 0.0

        for i, block in enumerate(self.blocks):
            if use_remat:
                h, new_c, aux = _checkpointed(block, h)  # type: ignore[possibly-undefined]
            else:
                h, new_c, aux = block(
                    h,
                    cache=block_caches[i] if use_cache else None,
                    cache_index=cache_index,
                    prefix_kv=prefix_kvs[i],
                    deterministic=deterministic,
                )
            new_caches.append(new_c)
            balancing_loss += aux

        h = self.norm_f(h)
        logits = (
            h @ self.embed.embedding[...].T
            if self.weight_tying
            else self.head(h)  # type: ignore[misc]
        )

        return ModelOutput(
            logits=logits,
            kv_caches=tuple(new_caches),
            aux_loss=balancing_loss,
        )

    # ── Diffusion-specific inference methods ──────────────────────────────────
    # Valid only when causal=False (bidirectional transformer).

    def compute_prefix_cache(self, prefix: jnp.ndarray) -> DualCache:
        """Process a static conditioning prefix once and cache per-layer KV."""
        h = self.embed(prefix)
        h = self.emb_dropout(h, deterministic=True)

        prefix_kvs: list = []
        for block in self.blocks:
            h, _, _, kv = block(h, deterministic=True, return_kv=True)
            prefix_kvs.append(kv)

        return DualCache(prefix_kvs=tuple(prefix_kvs))

    def compute_block_dual_cache(
        self,
        x_full: jnp.ndarray,
        block_start: int,
        block_end: int,
    ) -> DualCache:
        """Run a full forward pass and split KV into prefix and suffix parts."""
        h = self.embed(x_full)
        h = self.emb_dropout(h, deterministic=True)

        prefix_kvs: list = []
        suffix_kvs: list = []

        for block in self.blocks:
            h, _, _, kv = block(h, deterministic=True, return_kv=True)
            if kv is not None:
                k_full, v_full = kv
                prefix_kvs.append((k_full[:, :, :, :block_start, :],
                                   v_full[:, :, :, :block_start, :]))
                suffix_kvs.append((k_full[:, :, :, block_end:, :],
                                   v_full[:, :, :, block_end:, :]))
            else:
                prefix_kvs.append(None)
                suffix_kvs.append(None)

        return DualCache(prefix_kvs=tuple(prefix_kvs), suffix_kvs=tuple(suffix_kvs))

    def decode_block(
        self,
        x_block: jnp.ndarray,
        dual_cache: DualCache,
        block_start: int | jax.Array,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Denoise a single block using the dual KV cache; returns logits."""
        h      = self.embed(x_block)
        h      = self.emb_dropout(h, deterministic=deterministic)
        offset = jnp.asarray(block_start, dtype=jnp.int32)

        for i, block in enumerate(self.blocks):
            p_kv = dual_cache.prefix_kvs[i]
            s_kv = dual_cache.suffix_kvs[i] if dual_cache.suffix_kvs is not None else None

            if p_kv is not None and s_kv is not None:
                ctx: tuple | None = (
                    jnp.concatenate([p_kv[0], s_kv[0]], axis=3),
                    jnp.concatenate([p_kv[1], s_kv[1]], axis=3),
                )
            elif p_kv is not None:
                ctx = p_kv
            elif s_kv is not None:
                ctx = s_kv
            else:
                ctx = None

            x_norm = block.norm1(h)
            x_attn, _ = block.attention(
                x_norm,
                use_cache=False,
                kv_cache=(None, None),
                cache_index=offset,
                deterministic=deterministic,
                is_causal=False,
                prefix_kv=ctx,
            )
            h = h + x_attn
            ff, _ = block.ffn(block.norm2(h), deterministic=deterministic)
            h = h + ff

        h = self.norm_f(h)
        return (
            h @ self.embed.embedding[...].T
            if self.weight_tying
            else self.head(h)  # type: ignore[misc]
        )

    # ── Class-based builder ────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        *,
        dim: int,
        n_heads: int,
        head_size: int,
        num_blocks: int,
        vocab_size: int,
        max_context: int,
        rngs: nnx.Rngs,
        attention: type | str = "mha",
        ffn: type | str = "mlp",
        norm: type | str = "rmsnorm",
        causal: bool = True,
        **kwargs: object,
    ) -> Transformer:
        """Build a Transformer by passing component classes (or canonical strings).

        Component classes are resolved to their canonical string names before
        creating the ``ModelConfig``, so the config remains serialisable.

        Example::

            from dantinox.core.attention import GQAAttention
            model = Transformer.build(
                dim=512, n_heads=16, head_size=32, num_blocks=12,
                vocab_size=tok.vocab_size, max_context=512,
                attention=GQAAttention, kv_heads=4,
                causal=False,           # bidirectional diffusion model
                rngs=nnx.Rngs(42),
            )
        """
        from .attention import GQAAttention, MHAAttention, MLAAttention
        from .mlp import MLP as _MLP
        from .moe import MoE as _MoE

        _attn_cls_map: dict = {MHAAttention: "mha", GQAAttention: "gqa", MLAAttention: "mla"}
        _ffn_cls_map:  dict = {_MLP: "mlp", _MoE: "moe"}
        _norm_cls_map: dict = {RMSNorm: "rmsnorm", nnx.LayerNorm: "layernorm"}

        attn_str = _attn_cls_map.get(attention, attention)   # type: ignore[arg-type]
        ffn_str  = _ffn_cls_map.get(ffn, ffn)                # type: ignore[arg-type]
        norm_str = _norm_cls_map.get(norm, norm)              # type: ignore[arg-type]

        config = ModelConfig(
            dim=dim, n_heads=n_heads, head_size=head_size, num_blocks=num_blocks,
            vocab_size=vocab_size, max_context=max_context,
            attention=attn_str,   # type: ignore[arg-type]
            ffn=ffn_str,          # type: ignore[arg-type]
            norm=norm_str,        # type: ignore[arg-type]
            causal=causal,
            **kwargs,             # type: ignore[arg-type]
        )
        return cls(config, rngs=rngs)

    # ── Pretrained checkpoint loader ──────────────────────────────────────────

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: str,
        rngs: nnx.Rngs | None = None,
        *,
        best: bool = True,
        token: str | None = None,
        revision: str | None = None,
    ) -> Transformer:
        """Load a trained Transformer from a local directory or HuggingFace Hub."""
        import os

        import msgpack

        from dantinox.hub import resolve_checkpoint  # type: ignore[import]

        run_dir = resolve_checkpoint(path_or_repo, token=token, revision=revision)

        if rngs is None:
            rngs = nnx.Rngs(0)

        # Try new ModelConfig first, fall back to legacy Config
        config_path = os.path.join(run_dir, "config.yaml")
        try:
            config: ModelConfig | Config = ModelConfig.from_yaml(config_path)
        except Exception:
            config = Config.from_yaml(config_path)

        model = cls(config, rngs=rngs)

        weights_path = os.path.join(run_dir, "best_model_weights.msgpack")
        if not best or not os.path.exists(weights_path):
            weights_path = os.path.join(run_dir, "model_weights.msgpack")

        _ext_hook: object = None
        with contextlib.suppress(ImportError, AttributeError):
            from flax.serialization import _msgpack_ext_unpack  # type: ignore[attr-defined]
            _ext_hook = _msgpack_ext_unpack

        with open(weights_path, "rb") as f:
            state_dict = msgpack.unpackb(f.read(), ext_hook=_ext_hook, strict_map_key=False)
        nnx.update(model, state_dict)
        return model


# ── Backward-compatible alias ─────────────────────────────────────────────────

DiffusionTransformer = Transformer
