"""EmbedderParadigm — contrastive (SimCSE-style) training with the stock Trainer.

How it works
------------
The standard ``Trainer`` samples random windows of tokens ``[B, T]`` from a
flat corpus and calls ``paradigm.loss_fn(model, batch, rng)``.

``EmbedderParadigm`` runs **two forward passes** through the model with
*dropout enabled* and different random masks (NNX advances the RNG between
calls). The two resulting embeddings act as anchor/positive in the InfoNCE
objective, giving the model the SimCSE training signal without requiring any
labelled pairs.

For supervised pairs see ``EmbedderTrainer.fit_pairs()``.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from dantinox.core.config import ModelConfig
from dantinox.core.model import Transformer
from dantinox.paradigms.base import ParadigmBase


# ── InfoNCE loss ──────────────────────────────────────────────────────────────

def info_nce_loss(
    anchor: jnp.ndarray,
    positive: jnp.ndarray,
    temperature: float = 0.05,
) -> jnp.ndarray:
    """Symmetric InfoNCE / NT-Xent loss for in-batch negatives.

    Parameters
    ----------
    anchor:      L2-normalised embeddings ``[B, D]``.
    positive:    L2-normalised embeddings ``[B, D]``.
    temperature: Logit scale (lower = sharper distribution; 0.05 is a common default).

    Returns
    -------
    Scalar loss (mean over both directions).
    """
    B = anchor.shape[0]
    # Cosine similarities — already == dot product for normalised vectors
    sim = anchor @ positive.T / temperature   # [B, B]
    labels = jnp.arange(B)
    loss_a = optax.softmax_cross_entropy_with_integer_labels(sim,     labels).mean()
    loss_p = optax.softmax_cross_entropy_with_integer_labels(sim.T,   labels).mean()
    return (loss_a + loss_p) * 0.5


# ── Paradigm ──────────────────────────────────────────────────────────────────

class EmbedderParadigm(ParadigmBase):
    """Contrastive embedding paradigm (SimCSE unsupervised).

    Works directly with ``Trainer`` — no labelled pairs required.  Each
    ``[B, T]`` token batch is encoded **twice** with different dropout masks;
    the two views are treated as anchor/positive and trained with InfoNCE.

    Quick-start::

        import dantinox as dx

        cfg = dx.ModelConfig(
            dim=256, n_heads=4, head_size=64, num_blocks=4,
            vocab_size=32_000, causal=False,   # bidirectional → best embeddings
        )
        paradigm = dx.EmbedderParadigm(cfg)

        # train with the stock Trainer — any flat text corpus works
        run_dir = dx.train(paradigm, "data/corpus.txt", lr=3e-4, epochs=5)

        # use as encoder
        embedder = dx.Embedder.from_run(run_dir)
        vecs = embedder.embed(["hello world", "foo bar"])

    Parameters
    ----------
    config:      Model architecture config (prefer ``causal=False`` for better
                 bidirectional representations).
    pooling:     ``"auto"`` | ``"mean"`` | ``"last"`` | ``"cls"``.
    temperature: InfoNCE temperature (default 0.05).
    """

    requires_shifted_targets: bool = False

    def __init__(
        self,
        config: ModelConfig,
        *,
        pooling: str = "auto",
        temperature: float = 0.05,
    ) -> None:
        if config.dropout == 0.0:
            import warnings
            warnings.warn(
                "EmbedderParadigm: config.dropout=0.0 — SimCSE augmentation requires "
                "dropout > 0 (e.g. dropout=0.1) to produce distinct anchor/positive views. "
                "Training will still run but the two forward passes will be identical.",
                UserWarning,
                stacklevel=2,
            )
        self.config      = config
        self.pooling     = pooling
        self.temperature = temperature

    # ── Paradigm contract ─────────────────────────────────────────────────────

    def build_model(self, rngs: nnx.Rngs) -> Transformer:
        return Transformer(self.config, rngs=rngs)

    def loss_fn(
        self,
        model: Transformer,
        batch: jnp.ndarray,
        rng: jax.Array,
    ) -> tuple[jnp.ndarray, dict[str, Any]]:
        """SimCSE: encode the same batch twice with dropout → InfoNCE."""
        # Two forward passes — NNX advances the dropout RNG between calls,
        # giving different masks without any explicit augmentation.
        out_a = model.encode_hidden(
            batch, pooling=self.pooling, normalize=True, deterministic=False
        )
        out_p = model.encode_hidden(
            batch, pooling=self.pooling, normalize=True, deterministic=False
        )
        loss = info_nce_loss(out_a.embeddings, out_p.embeddings, self.temperature)
        return loss, {"contrastive_loss": loss}

    def generate(
        self,
        model: Transformer,
        prompt: jnp.ndarray,
        rng: jax.Array,
        **kwargs: Any,
    ) -> jnp.ndarray:
        """For embedders ``generate`` returns pooled embeddings of *prompt*."""
        out = model.encode_hidden(prompt, pooling=self.pooling, normalize=True)
        return out.embeddings

    def __repr__(self) -> str:
        return (
            f"EmbedderParadigm(pooling={self.pooling!r}, "
            f"temperature={self.temperature})"
        )
