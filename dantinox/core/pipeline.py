"""Smart one-call inference pipeline for DantinoX models.

Resolves a model from a local run directory or HuggingFace Hub repo, builds
the tokenizer and NNX model, and dispatches to the correct generation backend
based on ``cfg.model_type``.
"""

from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger(__name__)

_SUPPORTED_TASKS: frozenset[str] = frozenset({"text-generation"})


def pipeline(
    task: str,
    model_id_or_path: str,
    *,
    prompt: str = "",
    max_new_tokens: int = 200,
    seed: int = 42,
    temperature: float = 1.0,
    greedy: bool = False,
    **kwargs: Any,
) -> str:
    """Load a DantinoX checkpoint and generate text in a single call.

    Resolves *model_id_or_path* (local directory or HuggingFace Hub repo ID),
    reads ``config.yaml`` to determine the model type, instantiates the
    tokenizer and NNX model, and calls the appropriate generation function.

    Args:
        task: Task name.  Currently only ``"text-generation"`` is supported.
        model_id_or_path: Local run directory **or** HuggingFace Hub repo ID
            (e.g. ``"my-org/my-dantinox-model"``).
        prompt: Input text prompt (AR models).  Ignored for diffusion/flow-matching.
        max_new_tokens: Number of new tokens to generate.
        seed: PRNG seed for sampling.
        temperature: Sampling temperature (AR models only).
        greedy: Use greedy decoding (AR only; overrides temperature).
        **kwargs: Extra options forwarded to the generation function:
            ``top_k``, ``top_p``, ``use_cache`` (AR);
            ``n_steps``, ``block_size``, ``confidence_threshold`` (diffusion);
            ``n_steps``, ``cfg_scale`` (flow-matching).

    Returns:
        Decoded generated text string.

    Raises:
        ValueError: For unsupported tasks or unknown model types.
        FileNotFoundError: If ``config.yaml`` or a weights file is missing.

    Example::

        from dantinox.core.pipeline import pipeline

        text = pipeline(
            "text-generation",
            "runs/20240101_120000",
            prompt="Once upon a time",
            max_new_tokens=150,
        )
        print(text)
    """
    if task not in _SUPPORTED_TASKS:
        raise ValueError(
            f"Unsupported task {task!r}. "
            f"Supported: {sorted(_SUPPORTED_TASKS)}"
        )

    from dantinox.core.checkpoint import load_config, model_kind
    from dantinox.hub import resolve_checkpoint
    from dantinox.utils.tokenizer import load_tokenizer_from_file

    local_dir = resolve_checkpoint(model_id_or_path)
    log.info("Resolved checkpoint: %s → %s", model_id_or_path, local_dir)

    cfg = load_config(local_dir)
    kind = model_kind(cfg)
    log.info("Config: %s  model_kind=%s", cfg, kind)

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    # Declared once as Any: falls back to a transformers AutoTokenizer (a
    # different, unrelated class) when no native tokenizer.json is present.
    tokenizer: Any
    tok_path = os.path.join(local_dir, "tokenizer.json")
    if os.path.exists(tok_path):
        tokenizer = load_tokenizer_from_file(tok_path)
        log.info("Tokenizer loaded from %s", tok_path)
    else:
        log.warning(
            "tokenizer.json not found in %s — falling back to T5 tokenizer", local_dir
        )
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("t5-base")

    # ── Model ─────────────────────────────────────────────────────────────────
    from dantinox.core.checkpoint import load_model

    model, _, weights_path = load_model(local_dir, seed=seed)
    log.info("Weights restored from %s", weights_path)

    # ── Generation ────────────────────────────────────────────────────────────
    if kind == "autoregressive":
        return _ar_generate(
            model, tokenizer, prompt, max_new_tokens,
            seed=seed, temperature=temperature, greedy=greedy, **kwargs,
        )
    if kind == "diffusion":
        return _diffusion_generate(
            model, tokenizer, cfg, max_new_tokens, seed=seed,
            temperature=temperature, **kwargs,
        )
    if kind == "elf":
        return _flow_generate(model, tokenizer, cfg, max_new_tokens, seed=seed, **kwargs)

    raise ValueError(
        f"Unknown model kind {kind!r} for {local_dir}. "
        "Expected 'autoregressive', 'diffusion', or 'elf'."
    )


# ── Private helpers ───────────────────────────────────────────────────────────


def _decode(tokenizer: Any, ids: list[int]) -> str:
    """Decode token IDs with either a HuggingFace or DantinoX tokenizer."""
    try:
        return tokenizer.decode(ids, skip_special_tokens=True)
    except TypeError:
        return tokenizer.decode(ids)


def _ar_generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    *,
    seed: int,
    temperature: float,
    greedy: bool,
    **kwargs: Any,
) -> str:
    import jax.numpy as jnp

    from dantinox.core.generation import generate

    prompt_ids = tokenizer.encode(prompt)
    x = jnp.asarray([prompt_ids], dtype=jnp.int32)
    out = generate(
        model, x, max_new_tokens,
        greedy=greedy,
        seed=seed,
        temperature=temperature,
        use_cache=kwargs.get("use_cache", True),
        top_k=kwargs.get("top_k"),
        top_p=kwargs.get("top_p"),
    )
    new_ids = out[0].tolist()[len(prompt_ids):]
    return prompt + _decode(tokenizer, new_ids)


def _diffusion_generate(
    model: Any,
    tokenizer: Any,
    cfg: Any,
    max_new_tokens: int,
    *,
    seed: int,
    temperature: float,
    **kwargs: Any,
) -> str:
    import jax.numpy as jnp

    from dantinox.core.diffusion import make_noise_schedule
    from dantinox.core.generation import fast_dllm_generate

    schedule = make_noise_schedule(cfg)
    prefix = jnp.zeros((1, 0), dtype=jnp.int32)
    out = fast_dllm_generate(
        model, prefix, max_new_tokens, schedule,
        mask_token_id=cfg.mask_token_id,
        block_size=kwargs.get("block_size", 32),
        # ModelConfig has no num_sampling_steps field; only legacy Config does.
        steps_per_block=kwargs.get("n_steps", getattr(cfg, "num_sampling_steps", 50)),
        confidence_threshold=kwargs.get("confidence_threshold", 0.9),
        seed=seed,
    )
    return _decode(tokenizer, out[0].tolist())


def _flow_generate(
    model: Any,
    tokenizer: Any,
    cfg: Any,
    max_new_tokens: int,
    *,
    seed: int,
    **kwargs: Any,
) -> str:
    from dantinox.core.generation import flow_generate

    out = flow_generate(
        model,
        gen_len=max_new_tokens,
        batch_size=1,
        n_steps=kwargs.get("n_steps", cfg.flow_n_steps),
        cfg_scale=kwargs.get("cfg_scale", cfg.flow_cfg_scale),
        seed=seed,
    )
    return _decode(tokenizer, out[0].tolist())
