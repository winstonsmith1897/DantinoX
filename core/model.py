from __future__ import annotations

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from .block import Block, _build_norm
from .config import Config
from .output import ModelOutput


class Transformer(nnx.Module, pytree=False):
    def __init__(self, config: Config, rngs: nnx.Rngs):
        self.num_blocks: int     = config.num_blocks
        self.blocks: list        = [Block(config, rngs=rngs) for _ in range(self.num_blocks)]
        self.lm_head: nnx.Linear = nnx.Linear(config.dim, config.vocab_size, rngs=rngs)
        self.wte: nnx.Embed      = nnx.Embed(config.vocab_size, config.dim, rngs=rngs)
        self.trainable_pos: bool = config.trainable_pos
        self.absolute_pos: bool  = config.absolute_pos
        self.max_context: int    = config.max_context
        self.gradient_checkpointing: bool = config.gradient_checkpointing
        self.ln_f: nnx.Module    = _build_norm(config, config.dim, rngs)
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

            self.wpe: jnp.ndarray = _build_compute_absolute_pos(config.max_context, config.dim)  # type: ignore[assignment, no-redef]

    def __call__(self,
                 x: jnp.ndarray,
                 use_cache: bool,
                 kv_caches: tuple | None,
                 cache_index: int | None,
                 deterministic: bool = False) -> ModelOutput:

        B, T = x.shape
        x = self.wte(x)
        if kv_caches is None:
            kv_caches = tuple((None, None) for _ in range(self.num_blocks))
        if self.absolute_pos:
            wpe_slice = jax.lax.dynamic_slice_in_dim(
                self.wpe,  # type: ignore[arg-type]
                start_index=cache_index,  # type: ignore[arg-type]
                slice_size=T,
                axis=1
            )
            x = x + wpe_slice
        elif self.trainable_pos:
            x = x + self.wpe(jnp.arange(T, dtype=x.dtype))

        x = self.emb_dropout(x, deterministic=deterministic)

        def block_fn(block_module: object, hidden_state: jnp.ndarray, kv_c: object, det: bool) -> tuple:
            return block_module(  # type: ignore[call-arg, operator]
                hidden_state,
                use_cache=use_cache,
                kv_cache=kv_c,
                cache_index=cache_index,
                deterministic=det
            )

        def _apply_block(bm: object, hs: jnp.ndarray, kvc: object) -> tuple:
            return block_fn(bm, hs, kvc, deterministic)

        if self.gradient_checkpointing and not use_cache:
            checkpointed_block = nnx.remat(_apply_block)
        else:
            checkpointed_block = _apply_block

        new_kv_caches = []
        balancing_loss_total = 0.0
        for i, block in enumerate(self.blocks):
            x, new_kv, balancing_loss = checkpointed_block(block, x, kv_caches[i] if kv_caches else None)
            new_kv_caches.append(new_kv)
            balancing_loss_total += balancing_loss

        x = self.ln_f(x)

        return ModelOutput(
            logits=self.lm_head(x),
            kv_caches=tuple(new_kv_caches),
            aux_loss=balancing_loss_total,
        )

    @classmethod
    def from_pretrained(
        cls,
        run_dir: str,
        rngs: nnx.Rngs | None = None,
        *,
        best: bool = True,
    ) -> Transformer:
        """Load a trained Transformer from a run directory.

        Parameters
        ----------
        run_dir:
            Path produced by ``Trainer.fit()`` — must contain ``config.yaml``
            and either ``best_model_weights.msgpack`` or ``model_weights.msgpack``.
        rngs:
            PRNG state for initialisation. Defaults to ``nnx.Rngs(0)``.
        best:
            When ``True`` (default), loads ``best_model_weights.msgpack``
            if it exists, otherwise falls back to ``model_weights.msgpack``.
        """
        import contextlib
        import os

        import msgpack

        if rngs is None:
            rngs = nnx.Rngs(0)

        config = Config.from_yaml(os.path.join(run_dir, "config.yaml"))
        model = cls(config, rngs=rngs)

        weights_path = os.path.join(run_dir, "best_model_weights.msgpack")
        if not best or not os.path.exists(weights_path):
            weights_path = os.path.join(run_dir, "model_weights.msgpack")

        with open(weights_path, "rb") as f:
            raw = f.read()

        # Use the same private hook that trainer.py uses for consistency.
        _ext_hook: object = None
        with contextlib.suppress(ImportError):
            from flax.serialization import _msgpack_ext_unpack  # type: ignore[attr-defined]
            _ext_hook = _msgpack_ext_unpack

        state_dict = msgpack.unpackb(raw, ext_hook=_ext_hook, strict_map_key=False)
        nnx.update(model, state_dict)
        return model
