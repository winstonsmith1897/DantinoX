from __future__ import annotations

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
    stream_diffusion_generate as _stream_diffusion_generate,
)
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
    mask_token_id: int = 4

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
                               causal=False, noise_schedule="cosine", mask_token_id=4)
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
        self.model_config = model_config
        self.config       = model_config  # alias for _paradigm_config() in Trainer
        if diffusion_config is not None:
            self._noise_schedule = diffusion_config.noise_schedule
            self._mask_token_id  = diffusion_config.mask_token_id
        else:
            self._noise_schedule = model_config.noise_schedule
            self._mask_token_id  = model_config.mask_token_id
        self._schedule: NoiseSchedule = make_noise_schedule(self._noise_schedule)

    # ── Paradigm contract ─────────────────────────────────────────────────────

    def build_model(self, rngs: nnx.Rngs) -> Transformer:
        return Transformer(self.model_config, rngs=rngs)

    def loss_fn(
        self,
        model: Transformer,
        batch: jnp.ndarray,
        rng: jax.Array,
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

    def generate(
        self,
        model: Transformer,
        prompt: jnp.ndarray,
        rng: jax.Array,
        max_new_tokens: int = 256,
        n_steps: int = 50,
        temperature: float = 1.0,
    ) -> jnp.ndarray:
        from dantinox.paradigms.ar import _seed_from
        return _diffusion_generate(
            model,
            prompt,
            gen_len=max_new_tokens,
            schedule=self._schedule,
            mask_token_id=self._mask_token_id,
            seed=_seed_from(rng),
            num_sampling_steps=n_steps,
            temperature=temperature,
        )

    def stream(
        self,
        model: Transformer,
        prompt: jnp.ndarray,
        rng: jax.Array,
        max_new_tokens: int = 256,
        n_steps: int = 50,
        temperature: float = 1.0,
    ):
        """Like ``generate`` but yields ``(step, total, x_t)`` after each denoising step."""
        from dantinox.paradigms.ar import _seed_from
        yield from _stream_diffusion_generate(
            model,
            prompt,
            gen_len=max_new_tokens,
            schedule=self._schedule,
            mask_token_id=self._mask_token_id,
            seed=_seed_from(rng),
            num_sampling_steps=n_steps,
            temperature=temperature,
        )

    def __repr__(self) -> str:
        return (
            f"DiscreteParadigm("
            f"schedule={self._noise_schedule!r}, "
            f"mask_id={self._mask_token_id})"
        )
