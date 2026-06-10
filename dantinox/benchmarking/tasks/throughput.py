from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from dantinox.benchmarking.base import BenchmarkConfig, BenchmarkResult, BenchmarkTask
from dantinox.profiling.counter import count_flops
from dantinox.profiling.tracker import LatencyTracker


class ThroughputTask(BenchmarkTask):
    """Measure decode throughput (tokens/s) across sequence lengths and batch sizes.

    Uses the profiling :class:`~dantinox.profiling.tracker.LatencyTracker`
    to record wall-clock latency and derive tokens/s.  Both a seq-len sweep
    (batch=1) and a batch-size sweep (fixed seq_len) are reported.

    Metrics produced
    ----------------
    ``tps_seq{L}``   : tokens/s at sequence length L, batch=1.
    ``tps_bs{B}``    : tokens/s at batch size B, seq_len=config.seq_lens[0].
    ``peak_tps``     : highest measured throughput across all configurations.
    """

    name = "throughput"

    def run(
        self,
        paradigm: Any,
        model: Any,
        config: BenchmarkConfig,
        rng: jax.random.KeyArray,
    ) -> BenchmarkResult:
        vocab_size = _infer_vocab(paradigm)
        metrics: dict[str, float] = {}
        tracker = LatencyTracker()

        # ── Sequence-length sweep  (BS = 1) ───────────────────────────────────
        for seq_len in config.seq_lens:
            rng, rng_x = jax.random.split(rng)
            x = jax.random.randint(rng_x, (1, seq_len), 0, vocab_size)
            _warmup(model, x, config.n_warmup)
            tracker.reset()

            for _ in range(config.n_measure):
                with tracker.measure(n_tokens=seq_len):
                    jax.block_until_ready(model(x))

            metrics[f"tps_seq{seq_len}"] = tracker.result().throughput_tps

        # ── Batch-size sweep  (seq = seq_lens[0]) ─────────────────────────────
        fixed_seq = config.seq_lens[0]
        for bs in config.batch_sizes:
            rng, rng_x = jax.random.split(rng)
            x = jax.random.randint(rng_x, (bs, fixed_seq), 0, vocab_size)
            try:
                _warmup(model, x, config.n_warmup)
            except Exception:
                break  # OOM — stop batch sweep here
            tracker.reset()

            try:
                for _ in range(config.n_measure):
                    with tracker.measure(n_tokens=bs * fixed_seq):
                        jax.block_until_ready(model(x))
                metrics[f"tps_bs{bs}"] = tracker.result().throughput_tps
            except Exception:
                break

        # ── Aggregate ─────────────────────────────────────────────────────────
        tps_values = [v for v in metrics.values() if v == v]  # drop NaN
        metrics["peak_tps"] = max(tps_values) if tps_values else float("nan")

        # ── FLOPs (analytical) ────────────────────────────────────────────────
        flops = None
        try:
            cfg = getattr(paradigm, "config", None) or getattr(paradigm, "model_config", None)
            if cfg is not None:
                flops = count_flops(cfg, seq_len=config.seq_lens[-1], batch_size=1)
        except Exception:
            pass

        return BenchmarkResult(
            task=self.name,
            metrics=metrics,
            profiling=tracker.result(),
            flops=flops,
        )


def _infer_vocab(paradigm: Any) -> int:
    for attr in ("config", "model_config"):
        cfg = getattr(paradigm, attr, None)
        if cfg is not None:
            vs = getattr(cfg, "vocab_size", None)
            if vs is not None:
                return int(vs)
    return 32_000


def _warmup(model: Any, x: jnp.ndarray, n: int) -> None:
    for _ in range(n):
        jax.block_until_ready(model(x))
