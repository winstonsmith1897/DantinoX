"""The unified Transformer backbone — one model class for three paradigms.

``Transformer`` implements embed → L × Block → norm → (tied or separate)
vocabulary head.  The ``causal`` flag from the config selects autoregressive
(causal mask + KV cache) versus bidirectional (masked diffusion) behaviour —
the weights, code, and training loop are otherwise identical, which is what
makes cross-paradigm comparisons controlled.

Structure of the class:

* ``_forward_hidden`` — the single backbone pass shared by ``__call__`` and
  ``encode_hidden`` (so the two can never diverge); handles KV caches,
  dual-cache prefix injection, and gradient checkpointing.
* ``__call__`` — backbone + vocabulary head → ``ModelOutput``.
* ``encode_hidden`` — pooled sentence embeddings for retrieval / RAG.
* ``compute_prefix_cache`` / ``compute_block_dual_cache`` / ``decode_block``
  — Fast-dLLM dual-cache helpers for block-wise diffusion inference.
* ``build`` / ``from_pretrained`` — class-based construction and checkpoint
  loading (via core/checkpoint.py).

Decoding loops live in generation.py; the flow-matching model in flow.py.
"""
from __future__ import annotations

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from .block import Block, RMSNorm, _build_norm
from .config import Config, ModelConfig
from .diffusion import DualCache
from .output import EmbeddingOutput, ModelOutput


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
        if cfg.vocab_size is None:
            raise ValueError(
                "ModelConfig.vocab_size is not set. "
                "When using Trainer.fit() it is inferred automatically from the tokenizer. "
                "For direct model building pass vocab_size=<n> to ModelConfig."
            )


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
            self.wpe = self._build_sinusoidal(cfg.max_context, cfg.dim)  # type: ignore[assignment,no-redef]

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
    def lm_head(self) -> nnx.Linear | None:
        return self.head

    @property
    def wte(self) -> nnx.Embed:
        return self.embed

    # ── Forward pass ──────────────────────────────────────────────────────────

    def _forward_hidden(
        self,
        x: jnp.ndarray,
        *,
        caches: tuple | None = None,
        cache_index: int = 0,
        dual_cache: DualCache | None = None,
        deterministic: bool = False,
    ) -> tuple[jnp.ndarray, tuple, float]:
        """Shared backbone pass: embed → blocks → norm_f.

        Single implementation used by ``__call__`` and ``encode_hidden`` so the
        two paths can never diverge.  Returns ``(hidden, kv_caches, aux_loss)``.
        """
        use_cache = (caches is not None)

        h = self.embed(x)
        h = self._add_pos(h, cache_index)
        h = self.emb_dropout(h, deterministic=deterministic)

        # Block-level caches: (None, None) sentinel = "create cache on first step".
        # After the first step each block returns a KVCache / MLACache;
        # subsequent calls pass those back in.
        if caches is not None:
            block_caches: tuple = caches
        else:
            block_caches = tuple((None, None) for _ in range(self.num_blocks))
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

        return self.norm_f(h), tuple(new_caches), balancing_loss

    def _unembed(self, h: jnp.ndarray) -> jnp.ndarray:
        """Project hidden states to vocabulary logits (tied or separate head)."""
        if self.weight_tying:
            return h @ self.embed.embedding[...].T
        return self.head(h)  # type: ignore[misc]

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
        caches:        Per-layer KV cache for AR generation
                       (``attention.KVCache`` / ``attention.MLACache``).  Pass
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
        h, new_caches, balancing_loss = self._forward_hidden(
            x,
            caches=caches,
            cache_index=cache_index,
            dual_cache=dual_cache,
            deterministic=deterministic,
        )
        return ModelOutput(
            logits=self._unembed(h),
            kv_caches=new_caches,
            aux_loss=balancing_loss,
        )

    # ── Embedding / RAG interface ─────────────────────────────────────────────

    def encode_hidden(
        self,
        x: jnp.ndarray,
        *,
        pooling: str = "auto",
        normalize: bool = True,
        attention_mask: jnp.ndarray | None = None,
        deterministic: bool = True,
    ) -> EmbeddingOutput:
        """Extract pooled sentence embeddings for retrieval / RAG.

        Runs the full transformer backbone (embed → blocks → norm_f) and pools
        the per-token hidden states to a fixed-size vector ``[B, D]``.

        Parameters
        ----------
        x:              Token IDs ``[B, T]``.
        pooling:        ``"auto"`` selects ``"last"`` for causal models and
                        ``"mean"`` for bidirectional ones.  Other options:
                        ``"mean"`` — average over the sequence (mask-aware),
                        ``"last"`` — hidden state of the final non-padding token,
                        ``"cls"``  — hidden state of the first token.
        normalize:      L2-normalize the output (required for cosine similarity).
        attention_mask: Boolean mask ``[B, T]`` (``True`` = keep).  When
                        ``None`` all positions are treated as valid.
        deterministic:  Disable dropout (``True`` at inference, ``False``
                        during contrastive training for SimCSE augmentation).

        Returns
        -------
        ``EmbeddingOutput(embeddings, hidden_states)``

        Example::

            ids = tokenizer.encode_batch(texts)           # [B, T]
            out = model.encode_hidden(ids, normalize=True)
            # out.embeddings: [B, D] — ready for FAISS / ChromaDB
        """
        resolved = pooling
        if pooling == "auto":
            resolved = "last" if self.causal else "mean"

        # ── backbone (same code path as __call__, minus the logits head) ──────
        h, _, _ = self._forward_hidden(x, deterministic=deterministic)   # [B, T, D]

        # ── pooling ───────────────────────────────────────────────────────────
        if resolved == "cls":
            pooled = h[:, 0, :]

        elif resolved == "last":
            if attention_mask is not None:
                # index of the last valid token per sample
                last_idx = attention_mask.sum(axis=1) - 1          # [B]
                last_idx = jnp.clip(last_idx, 0, h.shape[1] - 1)
                pooled = h[jnp.arange(h.shape[0]), last_idx]      # [B, D]
            else:
                pooled = h[:, -1, :]

        else:  # "mean"
            if attention_mask is not None:
                mask = attention_mask[:, :, None].astype(jnp.float32)  # [B, T, 1]
                pooled = (h * mask).sum(axis=1) / mask.sum(axis=1).clip(1e-9)
            else:
                pooled = h.mean(axis=1)

        if normalize:
            pooled = pooled / jnp.linalg.norm(pooled, axis=-1, keepdims=True).clip(1e-12)

        return EmbeddingOutput(embeddings=pooled, hidden_states=h)

    # ── Diffusion-specific inference methods ──────────────────────────────────
    # Valid only when causal=False (bidirectional transformer).

    def compute_prefix_cache(self, prefix: jnp.ndarray) -> DualCache:
        """Process a static conditioning prefix once and cache per-layer KV."""
        h = self.embed(prefix)
        h = self._add_pos(h, 0)
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
        h = self._add_pos(h, 0)
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

            h = block.decode_with_context(h, ctx, offset, deterministic=deterministic)

        return self._unembed(self.norm_f(h))

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
        from dantinox.hub import resolve_checkpoint  # type: ignore[import]

        from .checkpoint import find_weights_file, load_config, restore_model

        run_dir = resolve_checkpoint(path_or_repo, token=token, revision=revision)

        if rngs is None:
            rngs = nnx.Rngs(0)

        model = cls(load_config(run_dir), rngs=rngs)
        restore_model(model, find_weights_file(run_dir, best=best))
        return model


# ── Backward-compatible alias ─────────────────────────────────────────────────

DiffusionTransformer = Transformer
