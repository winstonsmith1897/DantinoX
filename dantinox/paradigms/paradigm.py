"""Unified ``Paradigm`` factory — the single public entry point for all paradigms.

The paradigm is selected via the ``paradigm`` field in ``ModelConfig``:

* ``"ar"``         — autoregressive next-token prediction
* ``"discrete"``   — LLaDA-style masked-token diffusion
* ``"continuous"`` — continuous flow-matching, ELF recipe (requires ``embed_dim > 0``)
* ``"embedder"``   — SimCSE contrastive embedding

When ``paradigm`` is not set, the selection falls back to inspecting
``causal`` and ``embed_dim`` (backward-compatible with earlier code).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from dantinox.core.config import ModelConfig
from dantinox.paradigms.base import ParadigmBase


def _select_impl(config: ModelConfig) -> ParadigmBase:
    p = config.paradigm
    if p is None:
        if config.causal:
            p = "ar"
        elif config.embed_dim > 0:
            p = "continuous"
        else:
            p = "discrete"

    if p == "ar":
        from dantinox.paradigms.ar import ARParadigm
        return ARParadigm(config)
    if p == "discrete":
        from dantinox.paradigms.diffusion.discrete import DiscreteParadigm
        return DiscreteParadigm(config)
    if p == "continuous":
        from dantinox.paradigms.diffusion.continuous import ContinuousParadigm
        return ContinuousParadigm(config)
    if p == "embedder":
        from dantinox.paradigms.embedder import EmbedderParadigm
        return EmbedderParadigm(
            config,
            pooling=config.embed_pooling,
            temperature=config.embed_temperature,
        )
    raise ValueError(
        f"Unknown paradigm {p!r}. "
        "Set ModelConfig(paradigm=...) to one of: 'ar', 'discrete', 'continuous', 'embedder'."
    )


class Paradigm(ParadigmBase):
    """Unified paradigm that routes to the right implementation via ``ModelConfig.paradigm``.

    Quick-start::

        import dantinox as dx

        # Autoregressive
        p = dx.Paradigm(dx.ModelConfig(paradigm="ar", dim=512, n_heads=8, num_blocks=12))

        # Discrete diffusion (LLaDA)
        p = dx.Paradigm(dx.ModelConfig(paradigm="discrete", dim=512, n_heads=8, num_blocks=12,
                                        noise_schedule="cosine"))

        # Continuous flow-matching
        p = dx.Paradigm(dx.ModelConfig(paradigm="continuous", dim=512, n_heads=8, num_blocks=12,
                                        embed_dim=768))

        # Contrastive embedder (SimCSE)
        p = dx.Paradigm(dx.ModelConfig(paradigm="embedder", dim=512, n_heads=8, num_blocks=12,
                                        dropout=0.1))

        run_dir = dx.Trainer(p, dx.TrainingConfig(lr=3e-4, epochs=5)).fit("corpus.txt")

    Setting ``paradigm`` in ``ModelConfig`` also auto-configures ``causal``, so
    you never need to write ``causal=False`` for discrete/continuous.
    """

    def __init__(self, config: ModelConfig) -> None:
        # Use object.__setattr__ to bypass any __setattr__ overrides
        object.__setattr__(self, '_impl', _select_impl(config))

    # ── ParadigmBase abstract method implementations ───────────────────────────

    def build_model(self, rngs: Any) -> Any:
        return self._impl.build_model(rngs)

    def loss_fn(self, model: Any, batch: Any, rng: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        return self._impl.loss_fn(model, batch, rng, **kwargs)

    def generate(self, model: Any, *args: Any, **kwargs: Any) -> Any:
        return self._impl.generate(model, *args, **kwargs)

    # ── ParadigmBase class-attribute overrides ────────────────────────────────
    # These are class attributes on ParadigmBase, so __getattr__ is never
    # triggered for them — Python finds them via normal MRO lookup before
    # reaching __getattr__.  We must override them explicitly as properties so
    # they delegate to the inner implementation.

    @property
    def provides_batch_extras(self) -> bool:  # type: ignore[override]
        return self._impl.provides_batch_extras

    @property
    def diffusion_config(self) -> Any:
        return self._impl.diffusion_config

    @property
    def requires_shifted_targets(self) -> bool:  # type: ignore[override]
        return self._impl.requires_shifted_targets

    # ── ParadigmBase hook overrides ───────────────────────────────────────────
    # Same issue: on_train_start / prepare_batch are concrete (non-abstract)
    # methods on ParadigmBase, so __getattr__ never routes to the inner impl.

    def on_train_start(self, model: Any, sample_batches: list) -> None:
        return self._impl.on_train_start(model, sample_batches)

    def prepare_batch(self, batch: Any) -> Any:
        return self._impl.prepare_batch(batch)

    # ── Streaming (discrete + continuous only) ────────────────────────────────

    def stream(self, model: Any, *args: Any, **kwargs: Any) -> Iterator[Any]:
        """Yield ``(step, total, tokens)`` after each generation step.

        Only available for ``"discrete"`` and ``"continuous"`` paradigms.
        """
        if not hasattr(self._impl, 'stream'):
            raise NotImplementedError(
                f"Streaming is not supported for paradigm={self.type!r}. "
                "Use generate() instead."
            )
        yield from self._impl.stream(model, *args, **kwargs)

    # ── Forwarded attributes and hooks ────────────────────────────────────────

    def __getattr__(self, name: str) -> Any:
        # Forward any attribute not found on Paradigm itself to the inner impl.
        # This covers: config, requires_shifted_targets, provides_batch_extras,
        # on_train_start, prepare_batch, num_parameters, build_embedder, …
        try:
            impl = object.__getattribute__(self, '_impl')
        except AttributeError:
            # Suppress the confusing "_impl not found" context; the caller
            # asked for `name`, not `_impl`.
            raise AttributeError(name) from None
        return getattr(impl, name)

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def type(self) -> str:
        """Return the active paradigm type: ``"ar"``, ``"discrete"``, ``"continuous"``, or ``"embedder"``."""
        from dantinox.paradigms.ar import ARParadigm
        from dantinox.paradigms.diffusion.continuous import ContinuousParadigm
        from dantinox.paradigms.diffusion.discrete import DiscreteParadigm
        from dantinox.paradigms.embedder import EmbedderParadigm
        _map: dict[type, str] = {
            ARParadigm: "ar",
            DiscreteParadigm: "discrete",
            ContinuousParadigm: "continuous",
            EmbedderParadigm: "embedder",
        }
        return _map.get(type(self._impl), "unknown")

    def __repr__(self) -> str:
        return f"Paradigm(type={self.type!r}, impl={self._impl!r})"
