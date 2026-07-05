from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from dantinox.core.config import FlowMatchingConfig, ModelConfig
from dantinox.core.flow import FlowEmbedder, FlowMatchingTransformer, flow_loss
from dantinox.core.generation import (
    flow_generate as _flow_generate,
)
from dantinox.core.generation import (
    stream_flow_generate as _stream_flow_generate,
)
from dantinox.paradigms.base import ParadigmBase


def _elf_config_from_model_config(m: ModelConfig) -> FlowMatchingConfig:
    """Build an FlowMatchingConfig from a ModelConfig that has embed_dim > 0."""
    return FlowMatchingConfig(
        embed_dim=m.embed_dim,
        bottleneck_dim=m.bottleneck_dim,
        model_dim=m.dim,
        n_heads=m.n_heads,
        head_size=m.head_size,
        num_blocks=m.num_blocks,
        vocab_size=m.vocab_size,
        max_seq_len=m.max_context,
        pos_encoding=m.pos_encoding,
        norm=m.norm,
        dropout=m.dropout,
        gradient_checkpointing=m.gradient_checkpointing,
        attention=m.attention,
        kv_heads=m.kv_heads,
        down_dim_q=m.down_dim_q,
        down_dim_kv=m.down_dim_kv,
        rope_dim=m.rope_dim,
        flow_n_steps=m.flow_n_steps,
        flow_cfg_scale=m.flow_cfg_scale,
        sde_gamma=m.sde_gamma,
        t5_model_name=m.t5_model_name,
    )


class ContinuousParadigm(ParadigmBase):
    """ELF (Embedded Language Flows) continuous flow-matching paradigm.

    The forward process is ``z_t = t·x + (1−t)·ε`` where t ∈ [0,1],
    ε ~ N(0,I), and the model predicts the clean embedding x (x-prediction).

    Architecture: a bidirectional transformer operating in a continuous
    embedding space, conditioned on in-context control tokens for timestep,
    CFG scale, and operating mode (denoiser vs. decoder branch).

    Training requires a frozen T5 contextual encoder (``transformers``
    package, ``pip install dantinox[elf]``).  The encoder runs outside JIT;
    the Trainer obtains per-batch embeddings through ``prepare_batch`` and
    initialises the embedding normalisation statistics via
    ``on_train_start``.

    Quick-start::

        cfg      = dx.ModelConfig(dim=512, n_heads=8, num_blocks=12,
                                  embed_dim=768, bottleneck_dim=128, causal=False)
        paradigm = ContinuousParadigm(cfg)

    A raw ``FlowMatchingConfig`` is also accepted for Level-3 control over training
    hyper-parameters (denoiser schedules, CFG bounds, etc.).
    """

    provides_batch_extras = True

    def __init__(self, config: ModelConfig | FlowMatchingConfig) -> None:
        if isinstance(config, ModelConfig):
            if config.embed_dim == 0:
                raise ValueError(
                    "ContinuousParadigm requires embed_dim > 0 in ModelConfig. "
                    "Set embed_dim to match your T5 encoder (e.g. 768 for t5-base)."
                )
            config = _elf_config_from_model_config(config)
        self.config = config
        self._t5_encoder: Any = None

    # ── Paradigm contract ─────────────────────────────────────────────────────

    def build_model(self, rngs: nnx.Rngs) -> FlowMatchingTransformer:
        return FlowMatchingTransformer(self.config, rngs=rngs)

    def build_embedder(self, rngs: nnx.Rngs) -> FlowEmbedder:
        """Build the frozen T5 embedder used to project tokens to flow space."""
        return FlowEmbedder(self.config, rngs=rngs)

    def loss_fn(
        self,
        model: FlowMatchingTransformer,
        batch: jnp.ndarray,
        rng: jax.Array,
        embeddings: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, dict[str, Any]]:
        """Compute the flow-matching training loss.

        Args:
            model      : FlowMatchingTransformer NNX module.
            batch      : Integer token IDs ``[B, T]`` (targets for CE branch).
            rng        : JAX random key.
            embeddings : Raw T5 contextual embeddings ``[B, T, embed_dim]``
                         from ``prepare_batch``; normalised here via
                         ``model.encode`` before the flow-matching loss.

        Returns:
            (scalar_loss, metrics_dict)
        """
        if embeddings is None:
            raise ValueError(
                "ContinuousParadigm.loss_fn requires 'embeddings' — "
                "pre-compute them via prepare_batch() / FlowEmbedder before "
                "calling loss_fn."
            )
        normed = model.encode(embeddings)
        loss, metrics = flow_loss(model, normed, batch, rng, self.config)
        return loss, metrics

    # ── Training hooks ────────────────────────────────────────────────────────

    def on_train_start(self, model: FlowMatchingTransformer, sample_batches: list[Any]) -> None:
        """Initialise the embedder's normalisation stats from real T5 outputs."""
        encoder = self._encoder()
        token_batches = [jnp.asarray(b) for b in sample_batches]
        emb_mean, emb_std = encoder.compute_norm_stats(token_batches)
        model.embedder.emb_mean.value = emb_mean
        model.embedder.emb_std.value = emb_std

    def prepare_batch(self, batch: Any) -> jnp.ndarray:
        """Run the frozen T5 encoder (outside JIT) → embeddings ``[B, T, E]``."""
        return self._encoder().encode(jnp.asarray(batch))

    def _encoder(self) -> Any:
        if self._t5_encoder is None:
            from dantinox.utils.t5_encoder import T5ContextualEncoder
            self._t5_encoder = T5ContextualEncoder(self.config.t5_model_name)
        return self._t5_encoder

    # ── Generation ────────────────────────────────────────────────────────────

    def generate(
        self,
        model: FlowMatchingTransformer,
        prompt: jnp.ndarray | None = None,
        rng: jax.Array | None = None,
        max_new_tokens: int | None = None,
        n_steps: int | None = None,
        cfg_scale: float | None = None,
        gamma: float | None = None,
        batch_size: int | None = None,
        seed: int | None = None,
    ) -> jnp.ndarray:
        """Flow-matching generates unconditionally from Gaussian noise.

        *prompt* only provides the batch size / sequence length defaults
        (``max_new_tokens`` overrides its length); its token contents are unused.
        ``batch_size`` and ``seed`` can be passed directly as an alternative
        to providing a *prompt* and *rng*.
        """
        from dantinox.paradigms.ar import _seed_from
        steps  = n_steps   or getattr(self.config, "flow_n_steps", 64)
        cfg_w  = cfg_scale or getattr(self.config, "flow_cfg_scale", 1.0)
        sde_g  = gamma if gamma is not None else getattr(self.config, "sde_gamma", 0.0)
        length = max_new_tokens or (prompt.shape[1] if prompt is not None and prompt.ndim == 2
                                    else self.config.max_seq_len)
        if batch_size is None:
            batch_size = prompt.shape[0] if prompt is not None and prompt.ndim == 2 else 1
        if seed is None:
            seed = _seed_from(rng) if rng is not None else 42
        return _flow_generate(
            model,
            gen_len=length,
            batch_size=batch_size,
            n_steps=steps,
            cfg_scale=cfg_w,
            gamma=sde_g,
            seed=seed,
        )

    def stream(
        self,
        model: FlowMatchingTransformer,
        prompt: jnp.ndarray | None = None,
        rng: jax.Array | None = None,
        max_new_tokens: int | None = None,
        n_steps: int | None = None,
        cfg_scale: float | None = None,
        gamma: float | None = None,
    ):
        """Like ``generate`` but yields ``(step, total, tokens)`` after each ODE step."""
        from dantinox.paradigms.ar import _seed_from
        steps  = n_steps   or getattr(self.config, "flow_n_steps", 64)
        cfg_w  = cfg_scale or getattr(self.config, "flow_cfg_scale", 1.0)
        sde_g  = gamma if gamma is not None else getattr(self.config, "sde_gamma", 0.0)
        length = max_new_tokens or (prompt.shape[1] if prompt is not None and prompt.ndim == 2
                                    else self.config.max_seq_len)
        batch  = prompt.shape[0] if prompt is not None and prompt.ndim == 2 else 1
        seed   = _seed_from(rng) if rng is not None else 42
        yield from _stream_flow_generate(
            model, gen_len=length, batch_size=batch,
            n_steps=steps, cfg_scale=cfg_w, gamma=sde_g, seed=seed,
        )

    def num_parameters(self, model: FlowMatchingTransformer) -> int:
        import jax as _jax
        from flax import nnx as _nnx
        params = _nnx.state(model, _nnx.Param)
        return sum(x.size for x in _jax.tree_util.tree_leaves(params))

    def __repr__(self) -> str:
        c = self.config
        return (
            f"ContinuousParadigm(embed={c.embed_dim}, "
            f"dim={c.model_dim}, blocks={c.num_blocks})"
        )
