from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from dantinox.benchmarking.base import BenchmarkConfig, BenchmarkResult, BenchmarkTask
from dantinox.profiling.tracker import LatencyTracker


class LatencyTask(BenchmarkTask):
    """Measure prefill and (for AR models) decode latency.

    Prefill is defined as a single full-sequence forward pass.
    Decode is defined as a single next-token generation step, applicable
    only to autoregressive paradigms.

    Metrics produced
    ----------------
    ``prefill_mean_ms``   : mean prefill latency in ms (prompt_len = seq_lens[-1]).
    ``prefill_p99_ms``    : 99th-percentile prefill latency.
    ``decode_mean_ms``    : mean single-step decode latency in ms (AR only).
    ``decode_p99_ms``     : 99th-percentile decode latency (AR only).
    ``decode_tps``        : 1 / decode_mean_s  (AR only).
    """

    name = "latency"

    def run(
        self,
        paradigm: Any,
        model: Any,
        config: BenchmarkConfig,
        rng: jax.random.KeyArray,
    ) -> BenchmarkResult:
        from dantinox.paradigms.ar import ARParadigm

        vocab_size  = _infer_vocab(paradigm)
        prompt_len  = config.seq_lens[-1]
        rng, rng_x  = jax.random.split(rng)
        prompt      = jax.random.randint(rng_x, (1, prompt_len), 0, vocab_size)

        metrics: dict[str, float] = {}

        # ── Prefill latency ───────────────────────────────────────────────────
        prefill_tracker = LatencyTracker()
        _warmup_fn(lambda: model(prompt), config.n_warmup)

        for _ in range(config.n_measure):
            with prefill_tracker.measure(n_tokens=prompt_len):
                jax.block_until_ready(model(prompt))

        pr = prefill_tracker.result()
        metrics["prefill_mean_ms"] = pr.latency_mean_ms
        metrics["prefill_p99_ms"]  = pr.latency_p99_ms

        # ── Decode latency (AR only) ──────────────────────────────────────────
        if isinstance(paradigm, ARParadigm):
            rng, rng_tok = jax.random.split(rng)
            tok            = jax.random.randint(rng_tok, (1, 1), 0, vocab_size)
            decode_tracker = LatencyTracker()
            _warmup_fn(lambda: model(tok), config.n_warmup)

            for _ in range(config.n_measure):
                with decode_tracker.measure(n_tokens=1):
                    jax.block_until_ready(model(tok))

            dr = decode_tracker.result()
            metrics["decode_mean_ms"] = dr.latency_mean_ms
            metrics["decode_p99_ms"]  = dr.latency_p99_ms
            metrics["decode_tps"]     = dr.throughput_tps

        return BenchmarkResult(
            task=self.name,
            metrics=metrics,
            profiling=prefill_tracker.result(),
        )


def _infer_vocab(paradigm: Any) -> int:
    for attr in ("config", "model_config"):
        cfg = getattr(paradigm, attr, None)
        if cfg is not None:
            vs = getattr(cfg, "vocab_size", None)
            if vs is not None:
                return int(vs)
    return 32_000


def _warmup_fn(fn, n: int) -> None:
    for _ in range(n):
        jax.block_until_ready(fn())
