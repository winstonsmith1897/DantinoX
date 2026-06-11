from __future__ import annotations

import contextlib
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Iterator


@dataclass
class ProfilingResult:
    """Aggregated latency and throughput statistics.

    ``flops`` is filled in by :func:`dantinox.profile` with the analytical
    FLOPs breakdown when a model config is provided.
    """

    latency_mean_ms: float
    latency_p50_ms: float
    latency_p99_ms: float
    throughput_tps: float
    n_samples: int
    total_tokens: int
    flops: Any | None = None

    def __str__(self) -> str:
        out = (
            f"Profiling ({self.n_samples} samples, {self.total_tokens:,} tokens):\n"
            f"  latency mean : {self.latency_mean_ms:.2f} ms\n"
            f"  latency p50  : {self.latency_p50_ms:.2f} ms\n"
            f"  latency p99  : {self.latency_p99_ms:.2f} ms\n"
            f"  throughput   : {self.throughput_tps:,.0f} tokens/s"
        )
        if self.flops is not None:
            out += f"\n  flops        : {self.flops}"
        return out


class LatencyTracker:
    """Accumulates per-call latency measurements and computes statistics.

    Usage::

        tracker = LatencyTracker()

        with tracker.measure(n_tokens=256):
            output = model(x)

        print(tracker.result())
    """

    def __init__(self, window: int = 10_000) -> None:
        self._elapsed_s: deque[float] = deque(maxlen=window)
        self._tokens: deque[int] = deque(maxlen=window)

    # ── Recording ─────────────────────────────────────────────────────────────

    def record(self, elapsed_s: float, n_tokens: int) -> None:
        self._elapsed_s.append(elapsed_s)
        self._tokens.append(n_tokens)

    @contextlib.contextmanager
    def measure(self, n_tokens: int) -> Iterator[None]:
        """Context manager that times one call and records the sample.

        Example::

            with tracker.measure(n_tokens=seq_len * batch_size):
                logits = model(x)
        """
        _jax_barrier()
        t0 = time.perf_counter()
        yield
        _jax_barrier()
        self.record(time.perf_counter() - t0, n_tokens)

    # ── Statistics ────────────────────────────────────────────────────────────

    def result(self) -> ProfilingResult:
        n = len(self._elapsed_s)
        if n == 0:
            return ProfilingResult(0.0, 0.0, 0.0, 0.0, 0, 0)

        times_ms = sorted(t * 1_000 for t in self._elapsed_s)
        total_time_s = sum(self._elapsed_s)
        total_tokens = sum(self._tokens)

        mean_ms = sum(times_ms) / n
        p50_ms  = times_ms[int(0.50 * n)]
        p99_ms  = times_ms[min(int(0.99 * n), n - 1)]
        tps     = total_tokens / total_time_s if total_time_s > 0 else 0.0

        return ProfilingResult(
            latency_mean_ms=mean_ms,
            latency_p50_ms=p50_ms,
            latency_p99_ms=p99_ms,
            throughput_tps=tps,
            n_samples=n,
            total_tokens=total_tokens,
        )

    def reset(self) -> None:
        self._elapsed_s.clear()
        self._tokens.clear()

    def __len__(self) -> int:
        return len(self._elapsed_s)


# ── Functional helpers ─────────────────────────────────────────────────────────


def profile_fn(
    fn: Callable[..., Any],
    tracker: LatencyTracker,
    n_tokens: int,
) -> Callable[..., Any]:
    """Return a wrapped version of *fn* that records one sample per call.

    Example::

        fast_generate = profile_fn(model.generate, tracker, n_tokens=256)
        output = fast_generate(prompt, rng)  # latency recorded automatically
    """

    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        with tracker.measure(n_tokens):
            return fn(*args, **kwargs)

    return _wrapper


# ── Internal ──────────────────────────────────────────────────────────────────


def _jax_barrier() -> None:
    """Block until all pending JAX operations complete."""
    try:
        import jax
        jax.effects_barrier()
    except Exception:
        pass
