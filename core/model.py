from __future__ import annotations

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from .block import ARBlock, DiffusionBlock, _build_norm
from .config import Config
from .diffusion import DualCache, TimeEmbedding
from .output import ModelOutput


# ── Autoregressive Transformer ─────────────────────────────────────────────────

class Transformer(nnx.Module, pytree=False):
    """Causal (autoregressive) transformer with MHA / GQA / MLA attention.

    Supports KV-cache for efficient autoregressive generation, optional
    Mixture-of-Experts feed-forward, gradient checkpointing, and weight tying.
    """

    def __init__(self, config: Config, rngs: nnx.Rngs):
        self.num_blocks: int     = config.num_blocks
        self.blocks: list        = [ARBlock(config, rngs=rngs) for _ in range(self.num_blocks)]
        self.wte: nnx.Embed      = nnx.Embed(config.vocab_size, config.dim, rngs=rngs)
        self.weight_tying: bool  = config.weight_tying
        self.trainable_pos: bool = config.trainable_pos
        self.absolute_pos: bool  = config.absolute_pos
        self.max_context: int    = config.max_context
        self.gradient_checkpointing: bool = config.gradient_checkpointing
        self.ln_f: nnx.Module    = _build_norm(config, config.dim, rngs)
        self.emb_dropout         = nnx.Dropout(config.dropout_rate, rngs=rngs)
        self.use_moe: bool        = config.use_moe
        self.alpha_balance: float = config.alpha_balance

        if config.weight_tying:
            self.lm_head: nnx.Linear | None = None
        else:
            self.lm_head = nnx.Linear(config.dim, config.vocab_size, rngs=rngs)

        if self.trainable_pos:
            self.wpe: nnx.Embed = nnx.Embed(config.max_context, config.dim, rngs=rngs)
        elif self.absolute_pos:
            def _build_compute_absolute_pos(T: int, C: int) -> jnp.ndarray:
                pos = jnp.zeros((T, C))
                row = jnp.arange(T)
                col = jnp.arange(0, C, 2)
                k   = 1.0 / (10000 ** (col / C))
                ratio = jnp.einsum("i,j->ij", row, k)
                pos = pos.at[:, 0::2].set(jnp.sin(ratio))
                pos = pos.at[:, 1::2].set(jnp.cos(ratio))
                return jnp.expand_dims(pos, axis=0)

            self.wpe: jnp.ndarray = _build_compute_absolute_pos(  # type: ignore[assignment, no-redef]
                config.max_context, config.dim
            )

    def __call__(
        self,
        x: jnp.ndarray,
        use_cache: bool,
        kv_caches: tuple | None,
        cache_index: int | None,
        deterministic: bool = False,
    ) -> ModelOutput:
        B, T = x.shape
        x = self.wte(x)
        if kv_caches is None:
            kv_caches = tuple((None, None) for _ in range(self.num_blocks))
        if self.absolute_pos:
            wpe_slice = jax.lax.dynamic_slice_in_dim(
                self.wpe,  # type: ignore[arg-type]
                start_index=cache_index,  # type: ignore[arg-type]
                slice_size=T,
                axis=1,
            )
            x = x + wpe_slice
        elif self.trainable_pos:
            x = x + self.wpe(jnp.arange(T, dtype=x.dtype))

        x = self.emb_dropout(x, deterministic=deterministic)

        def block_fn(
            block_module: object, hidden_state: jnp.ndarray, kv_c: object, det: bool
        ) -> tuple:
            return block_module(  # type: ignore[call-arg, operator]
                hidden_state,
                use_cache=use_cache,
                kv_cache=kv_c,
                cache_index=cache_index,
                deterministic=det,
            )

        def _apply_block(bm: object, hs: jnp.ndarray, kvc: object) -> tuple:
            return block_fn(bm, hs, kvc, deterministic)

        if self.gradient_checkpointing and not use_cache:
            checkpointed_block = nnx.remat(_apply_block)
        else:
            checkpointed_block = _apply_block

        new_kv_caches         = []
        balancing_loss_total  = 0.0
        for i, block in enumerate(self.blocks):
            x, new_kv, balancing_loss = checkpointed_block(
                block, x, kv_caches[i] if kv_caches else None
            )
            new_kv_caches.append(new_kv)
            balancing_loss_total += balancing_loss

        x      = self.ln_f(x)
        logits = (
            x @ self.wte.embedding[...].T
            if self.weight_tying
            else self.lm_head(x)  # type: ignore[union-attr, misc]
        )

        return ModelOutput(
            logits=logits,
            kv_caches=tuple(new_kv_caches),
            aux_loss=balancing_loss_total,
        )

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
        """Load a trained Transformer from a local directory or HuggingFace Hub.

        Parameters
        ----------
        path_or_repo:
            Local path produced by ``Trainer.fit()`` **or** a Hub repo ID such
            as ``"my-org/dantinox-dante"``.  The checkpoint is downloaded
            automatically when a Hub ID is given.
        rngs:
            PRNG state for initialisation. Defaults to ``nnx.Rngs(0)``.
        best:
            When ``True`` (default), loads ``best_model_weights.msgpack``
            if it exists, otherwise falls back to ``model_weights.msgpack``.
        token:
            HuggingFace access token for private repositories.
        revision:
            Branch, tag, or commit SHA to download from the Hub.
        """
        import contextlib
        import os

        import msgpack

        from dantinox.hub import resolve_checkpoint  # type: ignore[import]

        run_dir = resolve_checkpoint(path_or_repo, token=token, revision=revision)

        if rngs is None:
            rngs = nnx.Rngs(0)

        config = Config.from_yaml(os.path.join(run_dir, "config.yaml"))
        model  = cls(config, rngs=rngs)

        weights_path = os.path.join(run_dir, "best_model_weights.msgpack")
        if not best or not os.path.exists(weights_path):
            weights_path = os.path.join(run_dir, "model_weights.msgpack")

        with open(weights_path, "rb") as f:
            raw = f.read()

        _ext_hook: object = None
        with contextlib.suppress(ImportError):
            from flax.serialization import _msgpack_ext_unpack  # type: ignore[attr-defined]
            _ext_hook = _msgpack_ext_unpack

        state_dict = msgpack.unpackb(raw, ext_hook=_ext_hook, strict_map_key=False)
        nnx.update(model, state_dict)
        return model


# ── Diffusion Transformer ──────────────────────────────────────────────────────

class DiffusionTransformer(nnx.Module, pytree=False):
    """Bidirectional transformer for masked discrete diffusion (MDLM-style).

    Architecture differences from ``Transformer``
    ---------------------------------------------
    - **Bidirectional attention**: no causal mask; every token attends to every
      other token.
    - **Time-step conditioning**: each block uses ``AdaLayerNorm`` to modulate
      normalisation parameters based on a learned time-step embedding.
    - **Dual-cache inference**: ``compute_prefix_cache`` processes a static
      conditioning prefix once and returns per-layer KV tensors that are
      concatenated with the noisy sequence's KV at each denoising step.

    Training
    --------
    Pass noisy token IDs ``x_t`` (produced by ``diffusion.corrupt``) and
    integer timesteps ``t`` to ``__call__``.  Compute loss with
    ``diffusion.masked_cross_entropy``.

    Inference
    ---------
    Use ``diffusion_generate`` from ``core.generation`` for iterative
    reverse-diffusion sampling with dual-cache acceleration.
    """

    def __init__(self, config: Config, rngs: nnx.Rngs) -> None:
        self.num_blocks: int = config.num_blocks
        self.diff_blocks: list = [
            DiffusionBlock(config, rngs=rngs) for _ in range(self.num_blocks)
        ]
        self.wte         = nnx.Embed(config.vocab_size, config.dim, rngs=rngs)
        self.weight_tying: bool = config.weight_tying
        self.gradient_checkpointing: bool = config.gradient_checkpointing
        self.ln_f        = _build_norm(config, config.dim, rngs)
        self.emb_dropout = nnx.Dropout(config.dropout_rate, rngs=rngs)
        self.use_moe: bool       = config.use_moe
        self.alpha_balance: float = config.alpha_balance
        self.max_context: int    = config.max_context

        # Time-step embedding: sinusoidal → 2-layer MLP
        self.time_emb = TimeEmbedding(config.dim, config.time_emb_dim, rngs=rngs)

        if config.weight_tying:
            self.lm_head: nnx.Linear | None = None
        else:
            self.lm_head = nnx.Linear(config.dim, config.vocab_size, rngs=rngs)

    # ── Fast-dLLM block-wise dual cache ──────────────────────────────────────

    def compute_block_dual_cache(
        self,
        x_full: jnp.ndarray,
        t: jnp.ndarray,
        block_start: int,
        block_end: int,
    ) -> DualCache:
        """Run a full forward pass and split KV into prefix and suffix parts.

        This implements the DualCache initialisation / refresh from Fast-dLLM
        (Algorithm 1, line 2 and line 19).

        The full sequence ``x_full = [prompt | ... | block_k | ... | suffix]``
        is passed through the transformer.  At each transformer layer the KV
        tensors are sliced into:

          - ``prefix_kvs[i]``:  positions ``0 … block_start-1``  (prompt)
          - ``suffix_kvs[i]``:  positions ``block_end … T-1``    (remaining MASK blocks)

        The slice for the current block (``block_start … block_end-1``) is
        **discarded** here — it will be recomputed fresh at every inner step
        inside ``decode_block``.

        Call this method once before decoding block k, and again (refresh) after
        block k is finished and before starting block k+1.

        Args:
            x_full:      Full token sequence ``[B, T_total]``.
            t:           Per-sample timestep ``[B]``.
            block_start: Absolute token index of the current block's first token.
            block_end:   Absolute token index one past the current block's last token.

        Returns:
            ``DualCache(prefix_kvs, suffix_kvs)`` with per-layer (k, v) slices.
        """
        B = x_full.shape[0]
        x = self.wte(x_full)
        x = self.emb_dropout(x, deterministic=True)
        t_emb = self.time_emb(t)

        prefix_kvs: list = []
        suffix_kvs: list = []

        for block in self.diff_blocks:
            # Single pass: run the block and capture KV (no double AdaLN)
            x, _, kv = block(x, t_emb, prefix_kv=None,
                             deterministic=True, return_kv=True)

            if kv is not None:
                k_full, v_full = kv  # [B, kv_heads, 1, T_total, head_size]
                prefix_kvs.append((k_full[:, :, :, :block_start, :],
                                   v_full[:, :, :, :block_start, :]))
                suffix_kvs.append((k_full[:, :, :, block_end:, :],
                                   v_full[:, :, :, block_end:, :]))
            else:
                prefix_kvs.append(None)
                suffix_kvs.append(None)

        return DualCache(
            prefix_kvs=tuple(prefix_kvs),
            suffix_kvs=tuple(suffix_kvs),
        )

    def decode_block(
        self,
        x_block: jnp.ndarray,
        t: jnp.ndarray,
        dual_cache: DualCache,
        block_start: int | jax.Array,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Run the denoising model on a single block with the dual KV cache.

        This is the inner-loop operation of Fast-dLLM block-wise decoding
        (Algorithm 1, line 6 — ``use_DualCache`` branch).

        Only the current block's tokens ``x[s:e]`` are processed.  Their
        queries attend to:
          1. Fresh KV computed from ``x_block`` (positions ``block_start … block_end-1``).
          2. Cached prefix KV (prompt positions ``0 … block_start-1``).
          3. Cached suffix KV (remaining MASK positions ``block_end … T-1``).

        RoPE is applied at the correct absolute offset (``cache_index=block_start``)
        so that position encodings are consistent with those stored in the cache.

        Args:
            x_block:     Current block tokens ``[B, block_size]``.
            t:           Per-sample timestep ``[B]``.
            dual_cache:  ``DualCache`` from ``compute_block_dual_cache``.
            block_start: Absolute start index of the block (for RoPE offset).
            deterministic: Disables dropout when ``True``.

        Returns:
            Logits ``[B, block_size, vocab_size]`` for the current block.
        """
        B, block_size = x_block.shape
        x     = self.wte(x_block)
        x     = self.emb_dropout(x, deterministic=deterministic)
        t_emb = self.time_emb(t)

        # Convert block_start to a JAX scalar so dynamic_slice_in_dim stays dynamic
        # across different block positions without triggering recompilation.
        offset = jnp.asarray(block_start, dtype=jnp.int32)

        for i, block in enumerate(self.diff_blocks):
            # ── Build context KV: prefix ‖ suffix ──────────────────────────
            p_kv = dual_cache.prefix_kvs[i]   # (k_prefix, v_prefix) or None
            s_kv = dual_cache.suffix_kvs[i]   # (k_suffix, v_suffix) or None

            if p_kv is not None and s_kv is not None:
                # Concatenate prefix and suffix along the sequence axis (axis=3)
                context_kv: tuple | None = (
                    jnp.concatenate([p_kv[0], s_kv[0]], axis=3),
                    jnp.concatenate([p_kv[1], s_kv[1]], axis=3),
                )
            elif p_kv is not None:
                context_kv = p_kv
            elif s_kv is not None:
                context_kv = s_kv
            else:
                context_kv = None

            # ── AdaLN norm ─────────────────────────────────────────────────
            x_norm = block.ada_ln1(x, t_emb)

            # ── Bidirectional attention with correct RoPE offset ────────────
            # cache_index=offset so Q gets RoPE for positions block_start..block_end-1.
            # prefix_kv=context_kv prepends [prefix_K|suffix_K] to the fresh block K.
            # is_causal=False: fully bidirectional.
            x_attn, _ = block.attention(
                x_norm,
                use_cache=False,
                kv_cache=(None, None),
                cache_index=offset,
                deterministic=deterministic,
                is_causal=False,
                prefix_kv=context_kv,
            )
            x = x + x_attn

            # ── Feed-forward ───────────────────────────────────────────────
            ff, _ = (
                block.moe(block.ada_ln2(x, t_emb), deterministic=deterministic)
                if block.use_moe
                else block.mlp(block.ada_ln2(x, t_emb), deterministic=deterministic)
            )
            x = x + ff

        x      = self.ln_f(x)
        logits = (
            x @ self.wte.embedding[...].T
            if self.weight_tying
            else self.lm_head(x)  # type: ignore[union-attr, misc]
        )
        return logits

    # ── Simple prefix-only cache (kept for backward compat) ──────────────────

    def compute_prefix_cache(self, prefix: jnp.ndarray) -> DualCache:
        """Process a static conditioning prefix once and cache per-layer KV.

        The returned ``DualCache`` should be passed to every subsequent call of
        ``__call__`` during reverse diffusion.  This amortises the cost of
        encoding the prefix across all denoising steps.

        .. note::
            This is an *approximate* optimisation for fully-bidirectional
            models: the cached prefix KV is computed from the prefix alone
            (without seeing the noisy sequence), so cross-token interactions
            between prefix and noisy tokens are not captured in the cache.
            Empirically this approximation works well when the prefix is long
            relative to the generated sequence.

        Args:
            prefix: Conditioning token IDs, shape ``[B, T_prefix]``.

        Returns:
            A ``DualCache`` whose ``prefix_kvs`` is a tuple of per-layer
            ``(k, v)`` tensors (or ``None`` for blocks whose attention variant
            does not support prefix injection, e.g. MLA).
        """
        B = prefix.shape[0]
        x = self.wte(prefix)
        x = self.emb_dropout(x, deterministic=True)

        # Use t=0 (no noise) for the prefix — it is always a clean sequence.
        t_zero = jnp.zeros((B,), dtype=jnp.int32)
        t_emb  = self.time_emb(t_zero)

        prefix_kvs = []
        for block in self.diff_blocks:
            # Single pass: run the block and capture KV simultaneously
            x, _, kv = block(x, t_emb, prefix_kv=None,
                             deterministic=True, return_kv=True)
            prefix_kvs.append(kv)

        return DualCache(prefix_kvs=tuple(prefix_kvs))

    # ── Forward pass ──────────────────────────────────────────────────────────

    def __call__(
        self,
        x_t: jnp.ndarray,
        t: jnp.ndarray,
        dual_cache: DualCache | None = None,
        deterministic: bool = False,
    ) -> ModelOutput:
        """Run the denoising network.

        Args:
            x_t:         Noisy token IDs, shape ``[B, T]``.
            t:           Per-sample diffusion timestep (integer), shape ``[B]``.
            dual_cache:  Optional ``DualCache`` from ``compute_prefix_cache``.
                         When provided each block prepends the cached prefix KV
                         to its attention context.
            deterministic: Disables dropout when ``True``.

        Returns:
            ``ModelOutput`` with ``logits`` of shape ``[B, T, vocab_size]``
            representing the predicted clean-token distribution p(x_0 | x_t, t).
        """
        B, T = x_t.shape
        x    = self.wte(x_t)
        x    = self.emb_dropout(x, deterministic=deterministic)
        t_emb = self.time_emb(t)  # [B, time_emb_dim]

        # Resolve per-block prefix KV (None during training)
        prefix_kvs: tuple | list = (
            dual_cache.prefix_kvs if dual_cache is not None
            else (None,) * self.num_blocks
        )

        # Gradient checkpointing during training (no dual_cache, deterministic=False)
        use_remat = self.gradient_checkpointing and not deterministic and dual_cache is None

        if use_remat:
            # Capture t_emb as a closure variable; deterministic is always False here.
            def _block_fn(bm: object, hs: jnp.ndarray) -> tuple:
                return bm(hs, t_emb, prefix_kv=None, deterministic=False)  # type: ignore[call-arg, operator]
            _checkpointed = nnx.remat(_block_fn)

        balancing_loss = 0.0
        for i, block in enumerate(self.diff_blocks):
            if use_remat:
                x, aux = _checkpointed(block, x)  # type: ignore[possibly-undefined]
            else:
                x, aux = block(
                    x, t_emb, prefix_kv=prefix_kvs[i], deterministic=deterministic
                )
            balancing_loss += aux

        x      = self.ln_f(x)
        logits = (
            x @ self.wte.embedding[...].T
            if self.weight_tying
            else self.lm_head(x)  # type: ignore[union-attr, misc]
        )

        return ModelOutput(
            logits=logits,
            kv_caches=(),           # diffusion does not use AR KV-cache
            aux_loss=balancing_loss,
        )

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: str,
        rngs: nnx.Rngs | None = None,
        *,
        best: bool = True,
        token: str | None = None,
        revision: str | None = None,
    ) -> DiffusionTransformer:
        """Load a trained DiffusionTransformer from a local directory or HuggingFace Hub."""
        import contextlib
        import os

        import msgpack

        from dantinox.hub import resolve_checkpoint  # type: ignore[import]

        run_dir = resolve_checkpoint(path_or_repo, token=token, revision=revision)

        if rngs is None:
            rngs = nnx.Rngs(0)

        config = Config.from_yaml(os.path.join(run_dir, "config.yaml"))
        model  = cls(config, rngs=rngs)

        weights_path = os.path.join(run_dir, "best_model_weights.msgpack")
        if not best or not os.path.exists(weights_path):
            weights_path = os.path.join(run_dir, "model_weights.msgpack")

        with open(weights_path, "rb") as f:
            raw = f.read()

        _ext_hook: object = None
        with contextlib.suppress(ImportError):
            from flax.serialization import _msgpack_ext_unpack  # type: ignore[attr-defined]
            _ext_hook = _msgpack_ext_unpack

        state_dict = msgpack.unpackb(raw, ext_hook=_ext_hook, strict_map_key=False)
        nnx.update(model, state_dict)
        return model
