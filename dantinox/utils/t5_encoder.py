"""Contextual T5 encoder for ELF — runs outside JIT, never updated during training."""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp


def _load_flax_t5(model_name: str):
    """Load T5 encoder using the Flax backend. Returns (model, params, hidden_dim)."""
    # Try the direct submodule path first — avoids transformers' is_flax_available()
    # cache miss when JAX/Flax was installed in the same session without restart.
    try:
        from transformers.models.t5.modeling_flax_t5 import FlaxT5EncoderModel
    except ImportError:
        from transformers import FlaxT5EncoderModel  # type: ignore[no-redef]

    try:
        model = FlaxT5EncoderModel.from_pretrained(model_name)
    except Exception:
        model = FlaxT5EncoderModel.from_pretrained(model_name, from_pt=True)

    params = jax.device_put(model.params)
    hidden_dim: int = model.config.d_model
    return model, params, hidden_dim


def _load_pt_t5(model_name: str):
    """Load T5 encoder using the PyTorch backend. Returns (model, None, hidden_dim)."""
    import torch
    from transformers import T5EncoderModel

    model = T5EncoderModel.from_pretrained(model_name)
    model.eval()
    hidden_dim: int = model.config.d_model
    return model, None, hidden_dim


class T5ContextualEncoder:
    """Full T5 encoder forward pass → contextual embeddings [B, L, hidden_dim].

    Unlike a token embedding lookup, this runs the complete T5 encoder stack so
    each token's representation reflects its context (ELF §3.1, 'Pretrained
    encoder' ablation, Gen.PPL ~40 vs ~70 for lookup-only).

    Not a JAX module — lives outside @nnx.jit and is never updated during training.
    Tries Flax T5 first; falls back to PyTorch T5 (always available in Colab).
    """

    def __init__(self, model_name: str = "t5-base") -> None:
        try:
            self._model, self._params, self.hidden_dim = _load_flax_t5(model_name)
            self._backend = "flax"
        except Exception:
            self._model, self._params, self.hidden_dim = _load_pt_t5(model_name)
            self._backend = "torch"

    def encode(self, token_ids: jnp.ndarray) -> jnp.ndarray:
        """token_ids [B, L] int → contextual embeddings [B, L, hidden_dim]."""
        if self._backend == "flax":
            attention_mask = jnp.ones_like(token_ids, dtype=jnp.int32)
            out = self._model(
                input_ids=token_ids,
                attention_mask=attention_mask,
                params=self._params,
                train=False,
            )
            return out.last_hidden_state  # [B, L, hidden_dim]
        else:
            import torch
            ids_np = np.array(token_ids)
            ids_torch = torch.from_numpy(ids_np)
            mask_torch = torch.ones_like(ids_torch)
            with torch.no_grad():
                out = self._model(input_ids=ids_torch, attention_mask=mask_torch)
            return jnp.array(out.last_hidden_state.numpy())

    def compute_norm_stats(
        self, token_batches: list[jnp.ndarray]
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute channel-wise mean/std over a list of token ID batches."""
        embs = jnp.concatenate([self.encode(x) for x in token_batches], axis=0)
        flat = embs.reshape(-1, embs.shape[-1])  # [(N*B*L), E]
        return flat.mean(axis=0), flat.std(axis=0)
