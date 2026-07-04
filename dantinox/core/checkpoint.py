"""Single checkpoint-loading path for DantinoX run directories.

Shared by ``Transformer.from_pretrained``, ``dantinox.core.pipeline`` and
``dantinox.core.export`` so that filename resolution, config-format detection,
msgpack decoding, and weight restoration behave identically everywhere.

A run directory contains:

* ``config.yaml`` — either a legacy monolithic ``Config`` (has ``model_type``),
  an ``FlowMatchingConfig`` (has ``model_dim``), or a split ``ModelConfig``.
* a weights file — model-only (flat state dict) or a full train-state
  checkpoint that wraps the weights under ``{"model": ..., "opt": ...}``.
"""

from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger(__name__)

#: Weight filenames tried in priority order (new-style first, legacy last).
WEIGHT_FILENAMES: tuple[str, ...] = (
    "checkpoint_best.msgpack",
    "checkpoint_latest.msgpack",
    "best_model_weights.msgpack",
    "model_weights.msgpack",
)


def find_weights_file(run_dir: str, *, best: bool = True) -> str:
    """Return the path of the first existing weights file in *run_dir*.

    Args:
        run_dir: Run directory to search.
        best:    When ``False``, skip ``checkpoint_best.msgpack`` and prefer
                 the latest checkpoint instead.

    Raises:
        FileNotFoundError: No known weights file exists in *run_dir*.
    """
    for fname in WEIGHT_FILENAMES:
        if not best and fname == "checkpoint_best.msgpack":
            continue
        path = os.path.join(run_dir, fname)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"No weights file found in {run_dir!r}. Expected one of: {WEIGHT_FILENAMES}"
    )


def _msgpack_ext_hook() -> Any:
    # flax's msgpack ext hook is private API; isolate the import here so a
    # flax upgrade breaks exactly one line, with a graceful fallback.
    try:
        from flax.serialization import _msgpack_ext_unpack  # type: ignore[attr-defined]
        return _msgpack_ext_unpack
    except (ImportError, AttributeError):  # pragma: no cover - future flax
        log.warning(
            "flax.serialization._msgpack_ext_unpack unavailable; "
            "loading msgpack without the flax extension hook."
        )
        return None


def load_raw_state(weights_path: str) -> dict:
    """Decode a msgpack weights file into a flat state dict.

    Full train-state checkpoints (``{"model": ..., "opt": ...}``) are
    unwrapped to the model weights automatically.
    """
    import msgpack

    with open(weights_path, "rb") as f:
        raw = msgpack.unpackb(f.read(), ext_hook=_msgpack_ext_hook(), strict_map_key=False)
    if isinstance(raw, dict) and "model" in raw and "opt" in raw:
        raw = raw["model"]
    return raw


def load_config(run_dir: str):
    """Read ``config.yaml`` and detect its format from its keys.

    Returns a ``Config`` (legacy monolithic, detected by ``model_type``),
    an ``FlowMatchingConfig`` (detected by ``model_dim``), or a ``ModelConfig``.
    """
    import yaml

    from .config import Config, FlowMatchingConfig, ModelConfig

    path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"config.yaml not found in {run_dir!r}. Is this a valid DantinoX run directory?"
        )
    with open(path) as f:
        raw = yaml.safe_load(f)
    # Allow one level of section nesting, mirroring *.from_yaml.
    flat: dict[str, Any] = {}
    for v in raw.values():
        if isinstance(v, dict):
            flat.update(v)
    d = flat if flat else raw

    if "model_type" in d:
        return Config.from_dict(d)
    if "model_dim" in d:
        return FlowMatchingConfig.from_dict(d)
    return ModelConfig.from_dict(d)


def model_kind(cfg: Any) -> str:
    """Normalise any config type to ``'autoregressive' | 'diffusion' | 'elf'``."""
    from .config import Config, FlowMatchingConfig, ModelConfig

    if isinstance(cfg, FlowMatchingConfig):
        return "elf"
    if isinstance(cfg, Config):
        return cfg.model_type
    if isinstance(cfg, ModelConfig):
        if cfg.paradigm == "continuous":
            return "elf"
        if cfg.paradigm == "discrete":
            return "diffusion"
        if cfg.paradigm in ("ar", "embedder"):
            return "autoregressive"
        return "autoregressive" if cfg.causal else "diffusion"
    raise TypeError(f"Unsupported config type: {type(cfg).__name__}")


def build_model(cfg: Any, rngs) -> Any:
    """Instantiate the model class matching *cfg* (uninitialised weights)."""
    from .config import Config, FlowMatchingConfig

    if model_kind(cfg) == "elf":
        from .flow import FlowMatchingTransformer

        if isinstance(cfg, FlowMatchingConfig):
            return FlowMatchingTransformer(cfg, rngs=rngs)
        if isinstance(cfg, Config):
            return FlowMatchingTransformer(cfg.to_flow_config(), rngs=rngs)
        raise ValueError(
            "A continuous-paradigm ModelConfig cannot build an FlowMatchingTransformer "
            "directly; the run directory must contain an FlowMatchingConfig or legacy "
            "Config yaml."
        )
    from .model import Transformer

    return Transformer(cfg, rngs=rngs)


def restore_model(model: Any, weights_path: str) -> None:
    """Load *weights_path* into *model* in place (RNG state excluded)."""
    from flax import nnx

    raw = load_raw_state(weights_path)
    state = nnx.state(model, nnx.Not(nnx.RngState))
    state.replace_by_pure_dict(raw)
    nnx.update(model, state)


def load_model(run_dir: str, *, seed: int = 42, best: bool = True) -> tuple[Any, Any, str]:
    """Build the right model from a run directory and restore its weights.

    Returns ``(model, config, weights_path)``.
    """
    from flax import nnx

    cfg = load_config(run_dir)
    weights_path = find_weights_file(run_dir, best=best)
    model = build_model(cfg, nnx.Rngs(seed))
    restore_model(model, weights_path)
    return model, cfg, weights_path
