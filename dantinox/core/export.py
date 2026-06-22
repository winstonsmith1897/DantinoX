"""StableHLO export of DantinoX NNX models via ``jax.export``.

Serialises a trained checkpoint into a self-contained StableHLO binary that
embeds the computation graph *and* the model weights.  The resulting file can
be executed by any StableHLO-compatible runtime (XLA, IREE, …) without Python
or the DantinoX library.

Requirements
------------
* JAX ≥ 0.4.26  (``jax.export`` API)
* JAX ≥ 0.4.28  (``Exported.serialize()``)

Quick-start::

    from dantinox.core.export import export_to_stablehlo

    path = export_to_stablehlo(
        checkpoint_path="runs/20240101_120000",
        output_path="exports/my_model.stablehlo",
    )

Load the exported binary later (no Python model code needed)::

    import jax.export
    with open(path, "rb") as f:
        exported = jax.export.deserialize(f.read())
    logits = exported.call(x)          # x: int32[batch, seq_len]
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

log = logging.getLogger(__name__)


def export_to_stablehlo(
    checkpoint_path: str,
    output_path: str,
    *,
    batch_size: int = 1,
    seq_len: int | None = None,
    seed: int = 42,
) -> str:
    """Compile and serialise a DantinoX model to a StableHLO binary.

    Loads the model from *checkpoint_path*, closes over its weights so they
    are embedded in the exported graph, and writes the serialised bytes to
    *output_path*.

    The exported file is self-contained: it contains both the computation and
    the trained weights, so inference requires no Python or DantinoX code.

    Args:
        checkpoint_path: Run directory containing ``config.yaml`` and a
            ``checkpoint_best.msgpack`` (or legacy ``best_model_weights.msgpack``).
        output_path: Destination path for the serialised binary.  The parent
            directory is created if it does not exist.
        batch_size: Batch size baked into the exported graph.
        seq_len: Sequence length.  Defaults to ``cfg.max_context`` from the
            config file.
        seed: PRNG seed for parameter initialisation before weight loading.

    Returns:
        Absolute path to the written binary.

    Raises:
        FileNotFoundError: ``config.yaml`` or a weights file is missing.
        ImportError: JAX is older than 0.4.26.
        RuntimeError: XLA tracing or serialisation fails.

    Example::

        path = export_to_stablehlo(
            "runs/20240101_120000",
            "exports/my_model.stablehlo",
            batch_size=4,
        )
        print(f"Written {os.path.getsize(path) / 1e6:.1f} MB → {path}")
    """
    if not hasattr(jax, "export"):
        raise ImportError(
            "jax.export requires JAX ≥ 0.4.26. "
            "Upgrade with:  pip install -U jax jaxlib"
        )

    from dantinox.core.config import Config

    # ── Load config + model ───────────────────────────────────────────────────
    config_path = os.path.join(checkpoint_path, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"config.yaml not found in {checkpoint_path!r}"
        )
    cfg = Config.from_yaml(config_path)
    T = seq_len if seq_len is not None else cfg.max_context
    log.info("Exporting %s  batch=%d  seq=%d", cfg, batch_size, T)

    model, weights_path = _load_model_for_export(cfg, checkpoint_path, seed)
    log.info("Weights loaded from %s", weights_path)

    # ── Build a pure, weight-closed-over forward function ─────────────────────
    # Split into a static graph definition (Python object, JAX-opaque) and
    # a pytree of weight arrays.  Closing over `state` inside the jit boundary
    # causes XLA to embed the weights as constants in the compiled graph.
    graphdef, state = nnx.split(model)

    @jax.jit
    def _forward(x: jax.Array) -> jax.Array:
        """Token IDs → logits.  Weights are constants baked into the graph."""
        m = nnx.merge(graphdef, state)
        output = m(x, deterministic=True)
        # Unified models return ModelOutput with .logits; handle both.
        return output.logits if hasattr(output, "logits") else output

    # ── Export ────────────────────────────────────────────────────────────────
    x_abstract = jax.ShapeDtypeStruct((batch_size, T), jnp.int32)
    log.info("Tracing forward pass (batch=%d, seq=%d) …", batch_size, T)

    exported: Any = jax.export.export(_forward)(x_abstract)
    log.info("Trace complete.  Serialising to %s …", output_path)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    serialized: bytes = _serialize(exported, output_path)
    with open(output_path, "wb") as fh:
        fh.write(serialized)

    size_mb = len(serialized) / 1_048_576
    log.info("Export complete: %s  (%.1f MB)", output_path, size_mb)
    print(f"StableHLO export → {output_path}  ({size_mb:.1f} MB)")
    return os.path.abspath(output_path)


# ── Private helpers ───────────────────────────────────────────────────────────


def _load_model_for_export(cfg: Any, run_dir: str, seed: int) -> tuple[Any, str]:
    """Instantiate the NNX model and restore weights from the best checkpoint."""
    import msgpack
    from flax.serialization import _msgpack_ext_unpack

    _CANDIDATES = (
        "checkpoint_best.msgpack",
        "checkpoint_latest.msgpack",
        "best_model_weights.msgpack",
        "model_weights.msgpack",
    )
    for fname in _CANDIDATES:
        wp = os.path.join(run_dir, fname)
        if os.path.exists(wp):
            weights_path = wp
            break
    else:
        raise FileNotFoundError(
            f"No weights file found in {run_dir!r}. "
            f"Tried: {_CANDIDATES}"
        )

    rngs = nnx.Rngs(seed)
    model_type = cfg.model_type

    if model_type == "elf":
        from dantinox.core.elf import ELFTransformer
        model = ELFTransformer(cfg.to_elf_config(), rngs=rngs)
    else:
        # Unified Transformer handles both AR (causal=True) and diffusion.
        from dantinox.core.model import Transformer
        model = Transformer(cfg.to_model_config(), rngs=rngs)

    with open(weights_path, "rb") as fh:
        raw = msgpack.unpackb(
            fh.read(), ext_hook=_msgpack_ext_unpack, strict_map_key=False
        )

    if isinstance(raw, dict) and "model" in raw and "opt" in raw:
        raw = raw["model"]

    state = nnx.state(model, nnx.Not(nnx.RngState))
    state.replace_by_pure_dict(raw)
    nnx.update(model, state)
    return model, weights_path


def _serialize(exported: Any, output_path: str) -> bytes:
    """Serialise *exported* to bytes, with a graceful fallback for older JAX."""
    # Exported.serialize() was added in JAX 0.4.28 and requires the
    # `flatbuffers` package.  Try it first; fall back to MLIR bytecode when
    # the method is absent (JAX < 0.4.28) or when `flatbuffers` is missing.
    if hasattr(exported, "serialize"):
        try:
            return exported.serialize()
        except ModuleNotFoundError as exc:
            if "flatbuffers" not in str(exc):
                raise
            log.warning(
                "flatbuffers is not installed — cannot use "
                "jax.export.Exported.serialize().  "
                "Falling back to raw MLIR bytecode.  "
                "Install with: pip install flatbuffers"
            )

    log.warning(
        "jax.export.Exported.serialize() not available (JAX < 0.4.28 or "
        "flatbuffers missing).  Saving raw MLIR bytecode instead.  "
        "Upgrade JAX and install flatbuffers for the portable StableHLO format."
    )
    mlir_module = exported.mlir_module()
    return mlir_module.operation.get_asm(binary=True)
