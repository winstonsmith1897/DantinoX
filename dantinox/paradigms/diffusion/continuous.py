from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from core.config import ELFConfig
from core.elf import ELFEmbedder, ELFTransformer, elf_loss
from core.generation import elf_generate as _elf_generate
from dantinox.paradigms.base import Paradigm


class ContinuousParadigm(Paradigm):
    """ELF (Embedded Language Flows) continuous flow-matching paradigm.

    The forward process is ``z_t = t·x + (1−t)·ε`` where t ∈ [0,1],
    ε ~ N(0,I), and the model predicts the clean embedding x (x-prediction).

    Architecture: a bidirectional transformer operating in a continuous
    embedding space, conditioned on in-context control tokens for timestep,
    CFG scale, and operating mode (denoiser vs. decoder branch).

    Quick-start::

        cfg      = ELFConfig(embed_dim=768, model_dim=512, n_heads=8,
                             head_size=64, num_blocks=12, vocab_size=32_128)
        paradigm = ContinuousParadigm(cfg)
    """

    def __init__(self, config: ELFConfig) -> None:
        self.config = config

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
        rng: jax.random.KeyArray,
        embeddings: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, dict[str, Any]]:
        """Compute ELF training loss.

        Args:
            model      : ELFTransformer NNX module.
            batch      : Integer token IDs ``[B, T]`` (targets for CE branch).
            rng        : JAX random key.
            embeddings : Pre-computed T5 embeddings ``[B, T, embed_dim]``.
                         Must be provided; obtain from ``ELFEmbedder``.

        Returns:
            (scalar_loss, metrics_dict)
        """
        if embeddings is None:
            raise ValueError(
                "ContinuousParadigm.loss_fn requires 'embeddings' — "
                "pre-compute them via ELFEmbedder before calling loss_fn."
            )
        loss, metrics = elf_loss(model, embeddings, batch, rng, self.config)
        return loss, metrics

    def generate(
        self,
        model: ELFTransformer,
        prompt: jnp.ndarray,
        rng: jax.random.KeyArray,
        n_steps: int | None = None,
        cfg_scale: float | None = None,
    ) -> jnp.ndarray:
        steps     = n_steps   or getattr(self.config, "elf_n_steps",   64)
        cfg_w     = cfg_scale or getattr(self.config, "elf_cfg_scale",  1.0)
        return _elf_generate(
            model,
            prompt,
            rng,
            n_steps=steps,
            cfg_scale=cfg_w,
            config=self.config,
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
