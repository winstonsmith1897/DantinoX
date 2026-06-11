from __future__ import annotations

import logging
from typing import Any

import jax
import jax.numpy as jnp

from dantinox.benchmarking.base import BenchmarkConfig, BenchmarkResult, BenchmarkTask

log = logging.getLogger(__name__)


class PerplexityTask(BenchmarkTask):
    """Compute perplexity on a held-out text corpus via the paradigm's loss.

    Works with any paradigm that implements ``loss_fn(model, batch, rng)``
    returning a cross-entropy-based scalar.  For AR models this is the
    standard next-token perplexity; for diffusion models it is the
    ELBO-weighted surrogate (still a valid quality proxy).

    Args:
        data_source : Path to a plain-text file used as the evaluation corpus.
                      If ``None``, the task skips and returns ``NaN``.

    Metrics produced
    ----------------
    ``perplexity``  : exp(mean cross-entropy loss over eval batches).
    ``eval_loss``   : raw mean cross-entropy.
    """

    name = "perplexity"

    def __init__(self, data_source: str | None = None) -> None:
        self.data_source = data_source

    def run(
        self,
        paradigm: Any,
        model: Any,
        config: BenchmarkConfig,
        rng: jax.Array,
    ) -> BenchmarkResult:
        if self.data_source is None:
            log.warning(
                "[perplexity] no data_source — skipping. "
                "Pass PerplexityTask('data/val.txt') to evaluate."
            )
            return BenchmarkResult(
                task=self.name,
                metrics={"perplexity": float("nan"), "eval_loss": float("nan")},
            )

        tokens     = _load_tokens(self.data_source)
        seq_len    = config.eval_seq_len
        bs         = config.eval_batch_size
        n_batches  = config.eval_batches
        vocab_size = _infer_vocab(paradigm)

        if len(tokens) < seq_len + 1:
            log.warning("[perplexity] corpus too short (%d tokens)", len(tokens))
            return BenchmarkResult(
                task=self.name,
                metrics={"perplexity": float("nan"), "eval_loss": float("nan")},
            )

        total_loss = 0.0
        for i in range(n_batches):
            rng, rng_b = jax.random.split(rng)
            batch      = _sample_batch(tokens, bs, seq_len, rng_b)
            loss, _    = paradigm.loss_fn(model, batch, rng_b)
            total_loss += float(loss)

        mean_loss  = total_loss / n_batches
        perplexity = float(jnp.exp(mean_loss))

        return BenchmarkResult(
            task=self.name,
            metrics={"perplexity": perplexity, "eval_loss": mean_loss},
        )

    def __repr__(self) -> str:
        return f"PerplexityTask(data_source={self.data_source!r})"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_tokens(path: str) -> list[int]:
    """Minimal character-level tokenization for eval purposes."""
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        return [ord(c) % 256 for c in text]
    except FileNotFoundError:
        log.error("[perplexity] file not found: %s", path)
        return []


def _sample_batch(
    tokens: list[int],
    bs: int,
    seq_len: int,
    rng: jax.Array,
) -> jnp.ndarray:
    max_start = max(len(tokens) - seq_len - 1, 1)
    starts    = jax.random.randint(rng, (bs,), 0, max_start)
    rows      = [tokens[s : s + seq_len + 1] for s in starts.tolist()]
    return jnp.array(rows, dtype=jnp.int32)


def _infer_vocab(paradigm: Any) -> int:
    for attr in ("config", "model_config"):
        cfg = getattr(paradigm, attr, None)
        if cfg is not None:
            vs = getattr(cfg, "vocab_size", None)
            if vs is not None:
                return int(vs)
    return 32_000
