from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from dantinox.core.config import ELFConfig
from dantinox.core.elf import ELFEmbedder, ELFTransformer, elf_loss
from dantinox.core.generation import elf_generate as _elf_generate
from dantinox.paradigms.base import Paradigm


class ContinuousParadigm(Paradigm):
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

        cfg      = ELFConfig(embed_dim=768, model_dim=512, n_heads=8,
                             head_size=64, num_blocks=12, vocab_size=32_128)
        paradigm = ContinuousParadigm(cfg)
    """

    provides_batch_extras = True

    def __init__(self, config: ELFConfig) -> None:
        self.config = config
        self._t5_encoder: Any = None

    # ── Paradigm contract ─────────────────────────────────────────────────────

    def build_model(self, rngs: nnx.Rngs) -> ELFTransformer:
        return ELFTransformer(self.config, rngs=rngs)

    def build_embedder(self, rngs: nnx.Rngs) -> ELFEmbedder:
        """Build the frozen T5 embedder used to project tokens to flow space."""
        return ELFEmbedder(self.config, rngs=rngs)

    def loss_fn(
        self,
        model: ELFTransformer,
        batch: jnp.ndarray,
        rng: jax.Array,
        embeddings: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, dict[str, Any]]:
        """Compute ELF training loss.

        Args:
            model      : ELFTransformer NNX module.
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
                "pre-compute them via prepare_batch() / ELFEmbedder before "
                "calling loss_fn."
            )
        normed = model.encode(embeddings)
        loss, metrics = elf_loss(model, normed, batch, rng, self.config)
        return loss, metrics

    # ── Training hooks ────────────────────────────────────────────────────────

    def on_train_start(self, model: ELFTransformer, sample_batches: list[Any]) -> None:
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
        model: ELFTransformer,
        prompt: jnp.ndarray,
        rng: jax.Array,
        gen_len: int | None = None,
        n_steps: int | None = None,
        cfg_scale: float | None = None,
        gamma: float | None = None,
    ) -> jnp.ndarray:
        """ELF generates unconditionally from Gaussian noise.

        *prompt* only provides the batch size / sequence length defaults
        (``gen_len`` overrides its length); its token contents are unused.
        """
        from dantinox.paradigms.ar import _seed_from
        steps  = n_steps   or getattr(self.config, "elf_n_steps", 64)
        cfg_w  = cfg_scale or getattr(self.config, "elf_cfg_scale", 1.0)
        sde_g  = gamma if gamma is not None else getattr(self.config, "sde_gamma", 0.0)
        length = gen_len or (prompt.shape[1] if prompt is not None and prompt.ndim == 2
                             else self.config.max_seq_len)
        batch  = prompt.shape[0] if prompt is not None and prompt.ndim == 2 else 1
        return _elf_generate(
            model,
            gen_len=length,
            batch_size=batch,
            n_steps=steps,
            cfg_scale=cfg_w,
            gamma=sde_g,
            seed=_seed_from(rng),
        )

    def num_parameters(self, model: ELFTransformer) -> int:
        from flax import nnx as _nnx
        import jax as _jax
        params = _nnx.state(model, _nnx.Param)
        return sum(x.size for x in _jax.tree_util.tree_leaves(params))

    def __repr__(self) -> str:
        c = self.config
        return (
            f"ContinuousParadigm(embed={c.embed_dim}, "
            f"dim={c.model_dim}, blocks={c.num_blocks})"
        )
