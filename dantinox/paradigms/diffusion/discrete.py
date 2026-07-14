from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from dantinox.core.config import ModelConfig
from dantinox.core.diffusion import (
    NoiseSchedule,
    corrupt,
    make_noise_schedule,
    masked_cross_entropy,
)
from dantinox.core.generation import (
    diffusion_generate as _diffusion_generate,
)
from dantinox.core.generation import (
    fast_dllm_generate as _fast_dllm_generate,
)
from dantinox.core.generation import (
    stream_diffusion_generate as _stream_diffusion_generate,
)
from dantinox.core.generation import (
    stream_fast_dllm_generate as _stream_fast_dllm_generate,
)

# _fast_dllm_generate and _stream_fast_dllm_generate are used by generate/stream
# when block_size is provided.
from dantinox.core.model import Transformer
from dantinox.paradigms.base import ParadigmBase


@dataclass
class DiscreteConfig:
    """Hyper-parameters specific to discrete masked-diffusion (LLaDA-style).

    Fields
    ------
    noise_schedule : "linear" | "cosine" | "sqrt"
        Masking rate schedule p_mask(t).
    mask_token_id : int
        Vocabulary index reserved for the ``[MASK]`` token.
    """

    noise_schedule: str = "linear"
    mask_token_id: int = -1  # sentinel: auto-synced from the tokenizer by Trainer.fit()

    def __post_init__(self) -> None:
        if self.noise_schedule not in ("linear", "cosine", "sqrt"):
            raise ValueError(
                f"noise_schedule must be 'linear', 'cosine', or 'sqrt'; "
                f"got {self.noise_schedule!r}"
            )


class DiscreteParadigm(ParadigmBase):
    """LLaDA-style masked-token diffusion paradigm.

    Training objective: (1/t)-weighted cross-entropy on masked positions.
    Corruption: randomly mask tokens with probability p_mask(t) where
    t ~ Uniform(0, 1) per sample.

    Quick-start::

        cfg      = ModelConfig(dim=512, n_heads=8, num_blocks=12,
                               causal=False, noise_schedule="cosine")
        # mask_token_id auto-detected from the tokenizer at Trainer.fit() time
        paradigm = DiscreteParadigm(cfg)

    A ``DiscreteConfig`` is still accepted as a second positional argument
    for backward compatibility.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        diffusion_config: DiscreteConfig | None = None,
    ) -> None:
        if model_config.causal:
            raise ValueError(
                "DiscreteParadigm requires a bidirectional model (config.causal=False)."
            )
        self.model_config     = model_config
        self.config           = model_config  # alias for _paradigm_config() in Trainer
        self._diffusion_config = diffusion_config
        if diffusion_config is not None:
            self._noise_schedule = diffusion_config.noise_schedule
        else:
            self._noise_schedule = model_config.noise_schedule
        self._schedule: NoiseSchedule = make_noise_schedule(self._noise_schedule)

    @property
    def _mask_token_id(self) -> int:
        # Read live from model_config rather than caching at construction
        # time: Trainer.fit() auto-syncs config.mask_token_id from the
        # tokenizer's real mask position *after* the paradigm is built (the
        # user constructs `Paradigm(config)` before calling `.fit()`), so a
        # value cached in __init__ would silently keep the stale default
        # (mask_token_id=4) instead of the tokenizer's actual [MASK] id.
        if self._diffusion_config is not None:
            return self._diffusion_config.mask_token_id
        return self.model_config.mask_token_id

    def _require_mask_id(self) -> int:
        """Return the mask token id, rejecting the unset ``-1`` sentinel.

        ``mask_token_id`` defaults to ``-1`` and is auto-synced from the
        tokenizer inside ``Trainer.fit()``.  Generating with the sentinel
        would silently corrupt every sequence, so fail loudly instead.
        """
        mask_id = self._mask_token_id
        if mask_id < 0:
            raise ValueError(
                "mask_token_id is not set (still the -1 sentinel). It is "
                "auto-detected from the tokenizer by Trainer.fit(); when "
                "building a paradigm manually for inference, pass "
                "mask_token_id explicitly in ModelConfig (e.g. "
                "tokenizer.mask_token_id)."
            )
        return mask_id

    @property
    def diffusion_config(self) -> Any:
        """Return a lightweight namespace with noise_schedule and mask_token_id."""
        from types import SimpleNamespace
        return SimpleNamespace(
            noise_schedule=self._noise_schedule,
            mask_token_id=self._mask_token_id,
        )

    # ── Paradigm contract ─────────────────────────────────────────────────────

    def build_model(self, rngs: nnx.Rngs) -> Transformer:
        return Transformer(self.model_config, rngs=rngs)

    def loss_fn(
        self,
        model: Transformer,
        batch: jnp.ndarray,
        rng: jax.Array,
        embeddings: jnp.ndarray | None = None,  # unused: discrete diffusion has no batch extras
    ) -> tuple[jnp.ndarray, dict[str, Any]]:
        mask_id = self._mask_token_id
        rng_t, rng_corrupt = jax.random.split(rng)

        B = batch.shape[0]
        t   = jax.random.uniform(rng_t, (B,), minval=0.0, maxval=1.0)
        x_t = corrupt(batch, t, rng_corrupt, self._schedule, mask_id)
        out = model(x_t)
        loss = masked_cross_entropy(
            out.logits, batch, x_t, mask_id, t, out.aux_loss
        )
        return loss, {"loss": loss, "aux_loss": out.aux_loss}

    def generate(  # type: ignore[override]
        self,
        model: Transformer,
        prompt: jnp.ndarray,
        rng: jax.Array,
        max_new_tokens: int = 256,
        n_steps: int = 50,
        temperature: float = 1.0,
        decoding_strategy: str = "sample",
        confidence_threshold: float = 0.9,
        factor: float = 1.5,
        block_size: int | None = None,
        steps_per_block: int = 50,
        use_dual_cache: bool = True,
        refresh_interval: int | None = None,
        verbose: bool = False,
    ) -> jnp.ndarray:
        """Run reverse diffusion and return the generated token IDs.

        Args:
            decoding_strategy: ``"sample"`` (default), ``"greedy"``,
                ``"confidence"``, or ``"factor"``.
                When ``block_size`` is set only ``"threshold"``/``"confidence"``
                and ``"factor"`` are meaningful (confidence-based strategies).
            confidence_threshold: τ for the ``"confidence"``/``"threshold"`` strategy.
            factor:               f for the ``"factor"`` strategy.
            block_size:           If set, use block-wise generation (Fast-dLLM style).
                                  ``None`` (default) uses global denoising.
            steps_per_block:      Inner denoising steps per block (block mode only).
            use_dual_cache:       Enable prefix+suffix DualCache (block mode only).
            refresh_interval:     Recompute suffix cache every N steps (block mode only).
            verbose:              Print a per-step unmasking trace — tokens revealed,
                                  masks left, average confidence (global mode only).
        """
        from dantinox.paradigms.ar import _seed_from
        seed = _seed_from(rng)
        if block_size is not None:
            return _fast_dllm_generate(
                model, prompt,
                gen_len=max_new_tokens,
                schedule=self._schedule,
                mask_token_id=self._require_mask_id(),
                block_size=block_size,
                steps_per_block=steps_per_block,
                decoding_strategy=decoding_strategy,
                confidence_threshold=confidence_threshold,
                factor=factor,
                use_dual_cache=use_dual_cache,
                refresh_interval=refresh_interval,
                seed=seed,
            )
        return _diffusion_generate(
            model, prompt,
            gen_len=max_new_tokens,
            schedule=self._schedule,
            mask_token_id=self._require_mask_id(),
            seed=seed,
            num_sampling_steps=n_steps,
            temperature=temperature,
            decoding_strategy=decoding_strategy,
            confidence_threshold=confidence_threshold,
            factor=factor,
            verbose=verbose,
        )

    def stream(
        self,
        model: Transformer,
        prompt: jnp.ndarray,
        rng: jax.Array,
        max_new_tokens: int = 256,
        n_steps: int = 50,
        temperature: float = 1.0,
        decoding_strategy: str = "sample",
        confidence_threshold: float = 0.9,
        factor: float = 1.5,
        block_size: int | None = None,
        steps_per_block: int = 50,
        use_dual_cache: bool = True,
        refresh_interval: int | None = None,
        verbose: bool = False,
    ) -> Iterator[tuple[int, int, jnp.ndarray]]:
        """Yields ``(step, total, x_gen)`` after each denoising step.

        Args:
            decoding_strategy: ``"sample"`` (default), ``"greedy"``,
                ``"confidence"``, or ``"factor"``.
                When ``block_size`` is set only ``"threshold"``/``"confidence"``
                and ``"factor"`` are meaningful.
            confidence_threshold: τ for the ``"confidence"``/``"threshold"`` strategy.
            factor:               f for the ``"factor"`` strategy.
            block_size:           If set, use block-wise generation (Fast-dLLM style).
                                  ``None`` (default) uses global denoising.
            steps_per_block:      Inner denoising steps per block (block mode only).
            use_dual_cache:       Enable prefix+suffix DualCache (block mode only).
            refresh_interval:     Recompute suffix cache every N steps (block mode only).
            verbose:              Print a per-step unmasking trace — tokens revealed,
                                  masks left, average confidence (global mode only).
        """
        from dantinox.paradigms.ar import _seed_from
        seed = _seed_from(rng)
        if block_size is not None:
            yield from _stream_fast_dllm_generate(
                model, prompt,
                gen_len=max_new_tokens,
                schedule=self._schedule,
                mask_token_id=self._require_mask_id(),
                block_size=block_size,
                steps_per_block=steps_per_block,
                decoding_strategy=decoding_strategy,
                confidence_threshold=confidence_threshold,
                factor=factor,
                use_dual_cache=use_dual_cache,
                refresh_interval=refresh_interval,
                seed=seed,
            )
        else:
            yield from _stream_diffusion_generate(
                model, prompt,
                gen_len=max_new_tokens,
                schedule=self._schedule,
                mask_token_id=self._require_mask_id(),
                seed=seed,
                num_sampling_steps=n_steps,
                temperature=temperature,
                decoding_strategy=decoding_strategy,
                confidence_threshold=confidence_threshold,
                factor=factor,
                verbose=verbose,
            )

    def __repr__(self) -> str:
        return (
            f"DiscreteParadigm("
            f"schedule={self._noise_schedule!r}, "
            f"mask_id={self._mask_token_id})"
        )
