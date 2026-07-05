"""Contextual T5 encoder for continuous flow-matching — runs outside JIT, never updated during training."""
from __future__ import annotations

import importlib
import importlib.util
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def _import_flax_t5() -> Any:
    """Import FlaxT5EncoderModel avoiding transformers' lazy-loader gate.

    transformers/__init__.py only re-exports Flax classes when is_flax_available()
    returned True at module load time.  In Colab, flax may be installed *after*
    transformers was first imported (same Python session, no runtime restart), so
    the gate stays False even though flax is now on disk.

    We bypass the gate entirely by importing the submodule directly; the submodule
    itself imports flax unconditionally so it works as long as flax is installed.
    """
    spec = importlib.util.find_spec("transformers.models.t5.modeling_flax_t5")
    if spec is None:
        raise ImportError("transformers not installed or too old")
    mod = importlib.import_module("transformers.models.t5.modeling_flax_t5")
    return mod.FlaxT5EncoderModel


def _import_pt_t5() -> Any:
    """Import T5EncoderModel (PyTorch) via the submodule path."""
    spec = importlib.util.find_spec("transformers.models.t5.modeling_t5")
    if spec is None:
        raise ImportError("transformers not installed or too old")
    mod = importlib.import_module("transformers.models.t5.modeling_t5")
    return mod.T5EncoderModel


class T5ContextualEncoder:
    """Full T5 encoder forward pass → contextual embeddings [B, L, hidden_dim].

    Tries Flax T5 first (no extra deps beyond flax); falls back to PyTorch T5
    (always available in Colab / any standard ML environment).

    Not a JAX module — lives outside @nnx.jit and is never updated during training.
    """

    def __init__(self, model_name: str = "t5-base") -> None:
        flax_exc: Exception | None = None
        try:
            FlaxT5EncoderModel = _import_flax_t5()
            try:
                model = FlaxT5EncoderModel.from_pretrained(model_name)
            except Exception:
                # from_pt=True converts PyTorch weights; only try if flax weights
                # are unavailable and PyTorch is healthy (no torchvision conflict).
                model = FlaxT5EncoderModel.from_pretrained(model_name, from_pt=True)
            self._model = model
            self._params = jax.device_put(model.params)
            self.hidden_dim: int = model.config.d_model
            self._backend = "flax"
            return
        except Exception as e:
            flax_exc = e

        # PyTorch fallback — importing torch can fail on systems where torchvision
        # was compiled against a different PyTorch version.  Catch that and re-raise
        # with a clear message rather than a cryptic C++ operator error.
        try:
            T5EncoderModel = _import_pt_t5()
            model = T5EncoderModel.from_pretrained(model_name)
            model.eval()
            self._model = model
            self._params = None
            self.hidden_dim = model.config.d_model
            self._backend = "torch"
        except (RuntimeError, OSError) as pt_exc:
            if "torchvision" in str(pt_exc) or "operator" in str(pt_exc):
                raise RuntimeError(
                    f"T5ContextualEncoder: Flax T5 failed ({flax_exc}) and the "
                    f"PyTorch fallback hit a torchvision version conflict.\n"
                    f"Fix: pip install torchvision --upgrade  OR  "
                    f"pip install transformers[flax] to use the Flax-only path."
                ) from pt_exc
            raise

    def encode(self, token_ids: jnp.ndarray) -> jnp.ndarray:
        """token_ids [B, L] int → contextual embeddings [B, L, hidden_dim]."""
        if self._backend == "flax":
            mask = jnp.ones_like(token_ids, dtype=jnp.int32)
            out  = self._model(
                input_ids=token_ids,
                attention_mask=mask,
                params=self._params,
                train=False,
            )
            return out.last_hidden_state
        else:
            import torch
            ids_np  = np.array(token_ids, dtype=np.int64)
            ids_t   = torch.from_numpy(ids_np)
            mask_t  = torch.ones_like(ids_t)
            with torch.no_grad():
                out = self._model(input_ids=ids_t, attention_mask=mask_t)
            return jnp.array(out.last_hidden_state.detach().numpy())

    def compute_norm_stats(
        self, token_batches: list[jnp.ndarray]
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute channel-wise mean/std over a list of token ID batches."""
        embs = jnp.concatenate([self.encode(x) for x in token_batches], axis=0)
        flat = embs.reshape(-1, embs.shape[-1])
        return flat.mean(axis=0), flat.std(axis=0)
