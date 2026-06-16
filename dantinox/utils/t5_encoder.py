"""Contextual T5 encoder for ELF — runs outside JIT, never updated during training."""
from __future__ import annotations

import jax
import jax.numpy as jnp


class T5ContextualEncoder:
    """Full T5 encoder forward pass → contextual embeddings [B, L, hidden_dim].

    Unlike a token embedding lookup, this runs the complete T5 encoder stack so
    each token's representation reflects its context (ELF §3.1, 'Pretrained
    encoder' ablation, Gen.PPL ~40 vs ~70 for lookup-only).

    Not a JAX module — lives outside @nnx.jit and is never updated during training.
    """

    def __init__(self, model_name: str = "t5-base") -> None:
        # Try the direct submodule path first — avoids transformers' is_flax_available()
        # cache hit (e.g. in Colab after a fresh pip install without runtime restart).
        try:
            from transformers.models.t5.modeling_flax_t5 import FlaxT5EncoderModel
        except ImportError:
            from transformers import FlaxT5EncoderModel
        try:
            self._model = FlaxT5EncoderModel.from_pretrained(model_name)
        except Exception:
            # Flax weights absent — convert from cached PyTorch checkpoint
            self._model = FlaxT5EncoderModel.from_pretrained(model_name, from_pt=True)
        self._params = jax.device_put(self._model.params)
        self.hidden_dim: int = self._model.config.d_model

    def encode(self, token_ids: jnp.ndarray) -> jnp.ndarray:
        """token_ids [B, L] int → contextual embeddings [B, L, hidden_dim]."""
        attention_mask = jnp.ones_like(token_ids, dtype=jnp.int32)
        out = self._model(
            input_ids=token_ids,
            attention_mask=attention_mask,
            params=self._params,
            train=False,
        )
        return out.last_hidden_state  # [B, L, hidden_dim]

    def compute_norm_stats(
        self, token_batches: list[jnp.ndarray]
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute channel-wise mean/std over a list of token ID batches."""
        embs = jnp.concatenate([self.encode(x) for x in token_batches], axis=0)
        flat = embs.reshape(-1, embs.shape[-1])  # [(N*B*L), E]
        return flat.mean(axis=0), flat.std(axis=0)
