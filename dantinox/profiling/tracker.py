"""Per-metric profiling API for DantinoX.

Each metric is a self-contained class with a ``measure()`` call.
All metrics accept lists for ``seq_lens`` / ``batch_sizes`` to run
a 2-D sweep and populate a ``sweep`` field in the result:

    lat = LatencyMetric(n_warmup=5, n_measure=50)
    r = lat.measure(fn, n_tokens=256)           # single point
    r = lat.measure_sweep(                       # 2-D grid
            get_batch_fn, model_fn,
            batch_sizes=[1, 4, 16], seq_lens=[64, 256])
    print(r.grid)   # list of {"batch_size":…,"seq_len":…,"mean_ms":…}

    thr = ThroughputMetric(batch_sizes=[1,4,16,64], seq_lens=[64,128,256])
    r = thr.measure(get_batch_fn, model_fn)     # full 2-D grid
    print(r.grid)   # list of {"batch_size":…,"seq_len":…,"tps":…}

    eng = EnergyMetric()
    r = eng.measure(fn, n_tokens=256)

    flops = FLOPsMetric()
    r = flops.measure(config, seq_len=256, batch_size=1, elapsed_s=0.1)

    ppl = PerplexityMetric(data=ids,
                           seq_lens=[64, 128, 256],
                           batch_sizes=[1, 4, 8])
    r = ppl.measure(loss_fn, rng)
    print(r.sweep)  # list of dicts — one per (seq_len, batch_size) pair

    ent = EntropyMetric(data=ids, seq_lens=[128, 256], batch_sizes=[2, 4])
    r = ent.measure(logit_fn, rng)
    print(r.sweep)

Backward-compatible names: ``LatencyTracker``, ``ProfilingResult``, ``profile_fn``.
"""
from __future__ import annotations

import contextlib
import math
import statistics
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

# ── Shared helpers ────────────────────────────────────────────────────────────


def _jax_barrier() -> None:
    """Block until all pending JAX operations complete."""
    try:
        import jax
        jax.effects_barrier()
    except Exception:
        pass


def _warmup(fn: Callable[[], Any], n: int) -> None:
    for _ in range(n):
        fn()
    _jax_barrier()


def _as_list(v: int | list[int]) -> list[int]:
    return [v] if isinstance(v, int) else list(v)


# ══════════════════════════════════════════════════════════════════════════════
# LATENCY
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class LatencyResult:
    """Per-call latency statistics, optionally with a 2-D sweep grid."""

    mean_ms: float
    p50_ms:  float
    p95_ms:  float
    p99_ms:  float
    n_samples:    int
    total_tokens: int
    # Populated by measure_sweep(); each entry:
    # {"batch_size": int, "seq_len": int, "mean_ms": float,
    #  "p50_ms": float, "p95_ms": float, "p99_ms": float}
    grid: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, float]:
        return {
            "lat_mean_ms": self.mean_ms,
            "lat_p50_ms":  self.p50_ms,
            "lat_p95_ms":  self.p95_ms,
            "lat_p99_ms":  self.p99_ms,
        }

    def __str__(self) -> str:
        s = (
            f"Latency ({self.n_samples} samples, {self.total_tokens:,} tok):\n"
            f"  mean={self.mean_ms:.2f} ms  p50={self.p50_ms:.2f} ms"
            f"  p95={self.p95_ms:.2f} ms  p99={self.p99_ms:.2f} ms"
        )
        if self.grid:
            s += f"\n  grid: {len(self.grid)} (bs, sl) points"
        return s


class LatencyMetric:
    """Measures per-request latency.

    Single-point mode::

        lat = LatencyMetric(n_warmup=5, n_measure=100)
        r = lat.measure(lambda: model(x), n_tokens=batch_size * seq_len)

    2-D sweep mode::

        r = lat.measure_sweep(
                get_batch_fn=lambda bs, sl: jnp.ones((bs, sl), jnp.int32),
                model_fn=lambda x: jax.block_until_ready(model(x)),
                batch_sizes=[1, 4, 16],
                seq_lens=[64, 128, 256],
            )
        # r.grid — list of dicts with keys batch_size, seq_len, mean_ms, …
    """

    def __init__(self, n_warmup: int = 5, n_measure: int = 50) -> None:
        self.n_warmup  = n_warmup
        self.n_measure = n_measure

    def measure(self, fn: Callable[[], Any], n_tokens: int) -> LatencyResult:
        """Time *fn* for *n_measure* calls after *n_warmup* warm-up calls."""
        _warmup(fn, self.n_warmup)
        times_s: list[float] = []
        for _ in range(self.n_measure):
            _jax_barrier()
            t0 = time.perf_counter()
            fn()
            _jax_barrier()
            times_s.append(time.perf_counter() - t0)
        return _latency_result_from_times(times_s, n_tokens * self.n_measure)

    def measure_sweep(
        self,
        get_batch_fn: Callable[[int, int], Any],
        model_fn:     Callable[[Any], Any],
        batch_sizes:  int | list[int] = 1,
        seq_lens:     int | list[int] = 256,
    ) -> LatencyResult:
        """Measure latency at every (batch_size, seq_len) combination.

        Returns a :class:`LatencyResult` where the aggregate fields reflect
        the (bs=1, first seq_len) point and ``grid`` contains all points.
        """
        bss = _as_list(batch_sizes)
        sls = _as_list(seq_lens)
        grid: list[dict[str, Any]] = []
        all_times_s: list[float]  = []

        for bs in bss:
            for sl in sls:
                try:
                    x = get_batch_fn(bs, sl)
                    # Bind `x` as a default arg: _warmup calls this immediately
                    # (not deferred), but this keeps it correct even if that
                    # ever changes, instead of relying on late-binding closure
                    # semantics over the loop variable.
                    _warmup(lambda x=x: model_fn(x), self.n_warmup)
                    times_s: list[float] = []
                    for _ in range(self.n_measure):
                        _jax_barrier()
                        t0 = time.perf_counter()
                        model_fn(x)
                        _jax_barrier()
                        times_s.append(time.perf_counter() - t0)
                    r = _latency_result_from_times(times_s, bs * sl)
                    grid.append({
                        "batch_size": bs, "seq_len": sl,
                        "mean_ms": r.mean_ms, "p50_ms": r.p50_ms,
                        "p95_ms": r.p95_ms,  "p99_ms": r.p99_ms,
                    })
                    all_times_s.extend(times_s)
                except Exception:
                    break  # OOM — skip larger configs

        agg = _latency_result_from_times(all_times_s, sum(
            e["batch_size"] * e["seq_len"] for e in grid
        )) if all_times_s else LatencyResult(
            float("nan"), float("nan"), float("nan"), float("nan"), 0, 0
        )
        agg.grid = grid
        return agg


def _latency_result_from_times(times_s: list[float], total_tokens: int) -> LatencyResult:
    if not times_s:
        return LatencyResult(float("nan"), float("nan"), float("nan"),
                             float("nan"), 0, total_tokens)
    ms = sorted(t * 1_000 for t in times_s)
    n  = len(ms)
    return LatencyResult(
        mean_ms=sum(ms) / n,
        p50_ms=ms[n // 2],
        p95_ms=ms[min(int(0.95 * n), n - 1)],
        p99_ms=ms[min(int(0.99 * n), n - 1)],
        n_samples=n,
        total_tokens=total_tokens,
    )


# ══════════════════════════════════════════════════════════════════════════════
# THROUGHPUT
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ThroughputResult:
    """Tokens-per-second across a 2-D (batch_size × seq_len) grid.

    ``grid`` contains one entry per (batch_size, seq_len) point::

        {"batch_size": 4, "seq_len": 256, "tps": 45_000.0}

    ``by_batch`` and ``by_seq`` are convenience 1-D views at the fixed
    axis from the constructor (backward compatible).
    """

    peak_tps: float
    by_batch: dict[int, float]   # batch_size → tok/s  (seq_len fixed)
    by_seq:   dict[int, float]   # seq_len   → tok/s  (batch_size=1)
    seq_len:  int                # reference seq_len for the batch sweep
    # Full 2-D grid — populated when both batch_sizes and seq_lens are lists
    grid: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, float]:
        d: dict[str, float] = {"peak_tps": self.peak_tps}
        d.update({f"tps_bs{bs}": v for bs, v in self.by_batch.items()})
        d.update({f"tps_sl{sl}": v for sl, v in self.by_seq.items()})
        return d

    def __str__(self) -> str:
        s = f"Throughput (peak={self.peak_tps:,.0f} tok/s)"
        if self.grid:
            s += f"  grid: {len(self.grid)} points"
        return s


class ThroughputMetric:
    """Measures tokens/s over a 2-D (batch_size × seq_len) grid.

    When both ``batch_sizes`` and ``seq_lens`` are lists the metric runs a
    **full grid** (all combinations).  Results are stored in ``grid``::

        thr = ThroughputMetric(
                  batch_sizes=[1, 4, 16, 64],
                  seq_lens=[64, 128, 256, 512])
        r = thr.measure(get_batch_fn, model_fn)
        # r.grid — 16 entries, one per (bs, sl) pair
        # r.by_batch — tok/s at seq_len=seq_lens[0], varying bs
        # r.by_seq   — tok/s at batch_size=1, varying sl
    """

    def __init__(
        self,
        n_warmup:    int              = 5,
        n_measure:   int              = 20,
        batch_sizes: int | list[int]  = None,
        seq_lens:    int | list[int]  = None,
    ) -> None:
        self.n_warmup    = n_warmup
        self.n_measure   = n_measure
        self.batch_sizes = _as_list(batch_sizes or [1, 4, 16, 64, 128, 256])
        self.seq_lens    = _as_list(seq_lens    or [64, 128, 256, 512])

    def measure(
        self,
        get_batch_fn: Callable[[int, int], Any],
        model_fn:     Callable[[Any], Any],
        seq_len:      int | None = None,
    ) -> ThroughputResult:
        """Run the full 2-D grid and return :class:`ThroughputResult`.

        Args:
            get_batch_fn : ``(batch_size, seq_len) → input``.
            model_fn     : ``(input) → output``.
            seq_len      : Override the reference seq_len for ``by_batch``.
                           Defaults to ``seq_lens[0]``.
        """
        ref_sl   = seq_len or self.seq_lens[0]
        grid:     list[dict[str, Any]] = []
        by_batch: dict[int, float]     = {}
        by_seq:   dict[int, float]     = {}

        for bs in self.batch_sizes:
            for sl in self.seq_lens:
                tps = self._measure_one(get_batch_fn, model_fn, bs, sl)
                if tps is None:
                    break  # OOM — skip larger batches at this seq_len
                grid.append({"batch_size": bs, "seq_len": sl, "tps": tps})
                if sl == ref_sl:
                    by_batch[bs] = tps
                if bs == 1:
                    by_seq[sl] = tps

        all_tps = [e["tps"] for e in grid]
        return ThroughputResult(
            peak_tps=max(all_tps) if all_tps else float("nan"),
            by_batch=by_batch,
            by_seq=by_seq,
            seq_len=ref_sl,
            grid=grid,
        )

    def _measure_one(
        self,
        get_batch_fn: Callable[[int, int], Any],
        model_fn:     Callable[[Any], Any],
        bs: int, sl: int,
    ) -> float | None:
        try:
            x = get_batch_fn(bs, sl)
            _warmup(lambda: model_fn(x), self.n_warmup)
            times_s: list[float] = []
            for _ in range(self.n_measure):
                _jax_barrier()
                t0 = time.perf_counter()
                model_fn(x)
                _jax_barrier()
                times_s.append(time.perf_counter() - t0)
            return (bs * sl * len(times_s)) / sum(times_s)
        except Exception:
            return None


# ══════════════════════════════════════════════════════════════════════════════
# ENERGY
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class EnergyResult:
    """GPU energy consumption per token."""

    j_per_tok:    float
    watts_active: float
    watts_idle:   float
    n_runs:       int

    def to_dict(self) -> dict[str, float]:
        return {
            "energy_j_per_tok": self.j_per_tok,
            "power_active_w":   self.watts_active,
            "power_idle_w":     self.watts_idle,
        }

    def __str__(self) -> str:
        return (
            f"Energy: {self.j_per_tok*1e3:.3f} mJ/tok  "
            f"(active={self.watts_active:.1f}W  idle={self.watts_idle:.1f}W)"
        )


class _PowerSampler:
    def __init__(self, device_idx: int = 0, interval_s: float = 0.025) -> None:
        import pynvml
        pynvml.nvmlInit()
        self._handle   = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
        self._interval = interval_s
        self._running  = False
        self._ts: list[float] = []
        self._w:  list[float] = []

    def start(self) -> None:
        self._running = True
        self._ts.clear()
        self._w.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._thread.join()

    def _loop(self) -> None:
        import pynvml
        while self._running:
            mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)
            self._ts.append(time.perf_counter())
            self._w.append(mw / 1_000.0)
            time.sleep(self._interval)

    def joules(self) -> float:
        import numpy as np
        if len(self._ts) < 2:
            return 0.0
        trap = getattr(np, "trapezoid", np.trapz)
        return float(trap(self._w, self._ts))

    def mean_watts(self) -> float:
        return sum(self._w) / len(self._w) if self._w else 0.0

    def elapsed_s(self) -> float:
        return (self._ts[-1] - self._ts[0]) if len(self._ts) >= 2 else 0.0

    def idle_watts(self, window_s: float = 0.5) -> float:
        self.start()
        time.sleep(window_s)
        self.stop()
        return self.mean_watts()


class EnergyMetric:
    """Measures GPU energy per token via NVML power sampling (requires pynvml).

    Example::

        eng = EnergyMetric(device_idx=0)
        r = eng.measure(lambda: model(x), n_tokens=batch_size * seq_len)
        print(r.j_per_tok * 1e3, "mJ/tok")
    """

    def __init__(self, device_idx: int = 0, min_window_s: float = 1.5) -> None:
        self.device_idx   = device_idx
        self.min_window_s = min_window_s

    def measure(self, fn: Callable[[], Any], n_tokens: int) -> EnergyResult:
        _nan = EnergyResult(float("nan"), float("nan"), float("nan"), 0)
        try:
            ps = _PowerSampler(self.device_idx)
        except Exception:
            return _nan

        idle_w = ps.idle_watts(window_s=0.5)
        fn()
        _jax_barrier()  # warmup

        ps.start()
        n_runs = 0
        t_start = time.perf_counter()
        while time.perf_counter() - t_start < self.min_window_s:
            fn()
            _jax_barrier()
            n_runs += 1
        ps.stop()

        gross_j = ps.joules()
        dt      = ps.elapsed_s()
        net_j   = max(gross_j - idle_w * dt, 0.0)
        j_per_tok = (
            net_j / (n_runs * n_tokens) if n_runs > 0 and n_tokens > 0
            else float("nan")
        )
        return EnergyResult(
            j_per_tok=j_per_tok,
            watts_active=ps.mean_watts(),
            watts_idle=idle_w,
            n_runs=n_runs,
        )


# ══════════════════════════════════════════════════════════════════════════════
# FLOPS
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class FLOPsResult:
    """Analytical FLOPs breakdown plus optional hardware efficiency."""

    total_gflops:     float
    attention_gflops: float
    ffn_gflops:       float
    embedding_gflops: float
    efficiency_pct:   float

    def to_dict(self) -> dict[str, float]:
        return {
            "flops_total_g":        self.total_gflops,
            "flops_attn_g":         self.attention_gflops,
            "flops_ffn_g":          self.ffn_gflops,
            "flops_efficiency_pct": self.efficiency_pct,
        }

    def __str__(self) -> str:
        eff = f"{self.efficiency_pct:.2f}%" if not math.isnan(self.efficiency_pct) else "N/A"
        return (
            f"FLOPs: {self.total_gflops:.2f} GFLOPs  "
            f"(attn={self.attention_gflops:.2f}  ffn={self.ffn_gflops:.2f})  "
            f"efficiency={eff}"
        )


class FLOPsMetric:
    """Computes analytical FLOPs and optional hardware efficiency.

    Example::

        r = FLOPsMetric(gpu_peak_tflops=312.0).measure(
                config, seq_len=256, batch_size=1, elapsed_s=0.042)
    """

    def __init__(self, gpu_peak_tflops: float = 312.0) -> None:
        self.gpu_peak_tflops = gpu_peak_tflops

    def measure(
        self,
        config:    Any,
        seq_len:   int,
        batch_size: int = 1,
        elapsed_s:  float | None = None,
    ) -> FLOPsResult:
        from dantinox.profiling.counter import count_flops
        bd = count_flops(config, seq_len=seq_len, batch_size=batch_size)
        efficiency_pct = float("nan")
        if elapsed_s is not None and elapsed_s > 0:
            efficiency_pct = (bd.total / elapsed_s / 1e12) / self.gpu_peak_tflops * 100
        return FLOPsResult(
            total_gflops=bd.total / 1e9,
            attention_gflops=bd.attention / 1e9,
            ffn_gflops=bd.ffn / 1e9,
            embedding_gflops=bd.embedding / 1e9,
            efficiency_pct=efficiency_pct,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PERPLEXITY
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class PerplexityResult:
    """Perplexity evaluated over a (seq_lens × batch_sizes) grid.

    ``sweep`` contains one dict per (seq_len, batch_size) combination::

        {"seq_len": 256, "batch_size": 4,
         "perplexity": 12.3, "bpb": 3.4, "eval_loss": 2.5}

    The top-level fields aggregate over the whole sweep (mean loss).
    When only a single (seq_len, batch_size) is used, ``sweep`` is empty.
    """

    perplexity: float
    bpb:        float
    eval_loss:  float
    n_batches:  int
    sweep: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, float]:
        return {"perplexity": self.perplexity, "bpb": self.bpb, "eval_loss": self.eval_loss}

    def __str__(self) -> str:
        s = (f"Perplexity: {self.perplexity:.3f}  "
             f"bpb={self.bpb:.4f}  loss={self.eval_loss:.4f}")
        if self.sweep:
            s += f"  ({len(self.sweep)} grid points)"
        return s


class PerplexityMetric:
    """Estimates perplexity over a (seq_lens × batch_sizes) sweep.

    ``seq_lens`` and ``batch_sizes`` accept a single int **or** a list.
    When lists are given, the metric evaluates every combination and
    stores the results in ``PerplexityResult.sweep``.

    Example — single point (backward compatible)::

        ppl = PerplexityMetric(data=ids, seq_len=256, batch_size=4)
        r = ppl.measure(loss_fn, rng)
        print(r.perplexity)

    Example — 2-D sweep::

        ppl = PerplexityMetric(
                  data=ids,
                  seq_lens=[64, 128, 256, 512],
                  batch_sizes=[1, 4, 8])
        r = ppl.measure(loss_fn, rng)
        print(r.sweep)   # 12 entries
    """

    def __init__(
        self,
        data:        Any,
        seq_lens:    int | list[int] = 256,
        batch_sizes: int | list[int] = 4,
        n_batches:   int             = 50,
        # backward-compat aliases
        seq_len:     int | None = None,
        batch_size:  int | None = None,
    ) -> None:
        self.data        = data
        self.seq_lens    = _as_list(seq_len  if seq_len  is not None else seq_lens)
        self.batch_sizes = _as_list(batch_size if batch_size is not None else batch_sizes)
        self.n_batches   = n_batches

    # backward compat
    @property
    def seq_len(self) -> int:
        return self.seq_lens[0]

    @property
    def batch_size(self) -> int:
        return self.batch_sizes[0]

    def measure(
        self,
        loss_fn: Callable,   # (batch [B, T+1], rng) → (loss, aux)
        rng:     Any,
    ) -> PerplexityResult:
        import jax

        sweep: list[dict[str, Any]] = []
        all_losses: list[float]     = []

        for sl in self.seq_lens:
            for bs in self.batch_sizes:
                rng, rng_eval = jax.random.split(rng)
                losses = self._eval_loss(loss_fn, rng_eval, sl, bs)
                mean_l = sum(losses) / len(losses)
                ppl    = math.exp(min(mean_l, 88.0))
                bpb    = mean_l / math.log(2)
                all_losses.extend(losses)
                sweep.append({
                    "seq_len":    sl,
                    "batch_size": bs,
                    "perplexity": ppl,
                    "bpb":        bpb,
                    "eval_loss":  mean_l,
                })

        mean_loss  = sum(all_losses) / len(all_losses) if all_losses else float("nan")
        perplexity = math.exp(min(mean_loss, 88.0))
        bpb        = mean_loss / math.log(2)
        multi      = len(self.seq_lens) * len(self.batch_sizes) > 1

        return PerplexityResult(
            perplexity=perplexity,
            bpb=bpb,
            eval_loss=mean_loss,
            n_batches=self.n_batches,
            sweep=sweep if multi else [],
        )

    def _eval_loss(self, loss_fn, rng, sl: int, bs: int) -> list[float]:
        import jax
        import jax.numpy as jnp

        n      = len(self.data)
        losses = []
        for _ in range(self.n_batches):
            rng, rng_b = jax.random.split(rng)
            max_start  = max(n - sl - 1, 1)
            starts     = jax.random.randint(rng_b, (bs,), 0, max_start)
            rows       = [self.data[int(s): int(s) + sl + 1] for s in starts.tolist()]
            batch      = jnp.array(rows, dtype=jnp.int32)
            loss, _    = loss_fn(batch, rng_b)
            losses.append(float(loss))
        return losses


# ══════════════════════════════════════════════════════════════════════════════
# ENTROPY
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class EntropyResult:
    """Token-level entropy statistics, optionally over a 2-D sweep grid.

    ``sweep`` contains one dict per (seq_len, batch_size) combination::

        {"seq_len": 256, "batch_size": 4,
         "mean_entropy": 3.1, "std_entropy": 0.4, "mean_top1_prob": 0.12}
    """

    mean_entropy:   float
    std_entropy:    float
    mean_top1_prob: float
    n_tokens:       int
    sweep: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, float]:
        return {
            "entropy_mean_nats": self.mean_entropy,
            "entropy_std":       self.std_entropy,
            "top1_prob_mean":    self.mean_top1_prob,
        }

    def __str__(self) -> str:
        s = (f"Entropy: mean={self.mean_entropy:.3f} nats  "
             f"std={self.std_entropy:.3f}  top1={self.mean_top1_prob:.3f}")
        if self.sweep:
            s += f"  ({len(self.sweep)} grid points)"
        return s


class EntropyMetric:
    """Computes per-token output entropy over a (seq_lens × batch_sizes) sweep.

    Works with any model returning ``[B, T, V]`` logits for a batch of token ids.

    Example — 2-D sweep::

        ent = EntropyMetric(
                  data=ids,
                  seq_lens=[128, 256, 512],
                  batch_sizes=[1, 4])
        r = ent.measure(lambda x: model(x).logits, rng)
        print(r.sweep)   # 6 entries
    """

    def __init__(
        self,
        data:        Any,
        seq_lens:    int | list[int] = 256,
        batch_sizes: int | list[int] = 4,
        n_batches:   int             = 20,
        # backward-compat aliases
        seq_len:     int | None = None,
        batch_size:  int | None = None,
    ) -> None:
        self.data        = data
        self.seq_lens    = _as_list(seq_len  if seq_len  is not None else seq_lens)
        self.batch_sizes = _as_list(batch_size if batch_size is not None else batch_sizes)
        self.n_batches   = n_batches

    @property
    def seq_len(self) -> int:
        return self.seq_lens[0]

    @property
    def batch_size(self) -> int:
        return self.batch_sizes[0]

    def measure(
        self,
        logit_fn: Callable,   # (batch [B, T]) → [B, T, V]
        rng:      Any,
    ) -> EntropyResult:
        import jax

        sweep: list[dict[str, Any]] = []
        all_entropy: list[float]    = []
        all_top1:    list[float]    = []

        for sl in self.seq_lens:
            for bs in self.batch_sizes:
                rng, rng_eval = jax.random.split(rng)
                entropies, top1s = self._eval_entropy(logit_fn, rng_eval, sl, bs)
                me  = sum(entropies) / len(entropies)
                std = statistics.stdev(entropies) if len(entropies) > 1 else 0.0
                mt1 = sum(top1s) / len(top1s)
                all_entropy.extend(entropies)
                all_top1.extend(top1s)
                sweep.append({
                    "seq_len":        sl,
                    "batch_size":     bs,
                    "mean_entropy":   me,
                    "std_entropy":    std,
                    "mean_top1_prob": mt1,
                })

        me_agg  = sum(all_entropy) / len(all_entropy) if all_entropy else float("nan")
        std_agg = statistics.stdev(all_entropy) if len(all_entropy) > 1 else 0.0
        mt1_agg = sum(all_top1) / len(all_top1) if all_top1 else float("nan")
        multi   = len(self.seq_lens) * len(self.batch_sizes) > 1

        return EntropyResult(
            mean_entropy=me_agg,
            std_entropy=std_agg,
            mean_top1_prob=mt1_agg,
            n_tokens=len(all_entropy),
            sweep=sweep if multi else [],
        )

    def _eval_entropy(self, logit_fn, rng, sl: int, bs: int):
        import jax
        import jax.numpy as jnp

        n   = len(self.data)
        ent_all: list[float] = []
        t1_all:  list[float] = []

        for _ in range(self.n_batches):
            rng, rng_b = jax.random.split(rng)
            starts = jax.random.randint(rng_b, (bs,), 0, max(n - sl, 1))
            rows   = [self.data[int(s): int(s) + sl] for s in starts.tolist()]
            batch  = jnp.array(rows, dtype=jnp.int32)
            logits = logit_fn(batch)
            probs  = jax.nn.softmax(logits, axis=-1)
            log_p  = jnp.log(probs + 1e-10)
            ent    = -(probs * log_p).sum(-1)
            top1   = probs.max(-1)
            ent_all.extend(ent.ravel().tolist())
            t1_all.extend(top1.ravel().tolist())

        return ent_all, t1_all


# ══════════════════════════════════════════════════════════════════════════════
# BACKWARD-COMPATIBLE API
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ProfilingResult:
    """Legacy result type. Prefer ``LatencyResult`` for new code."""

    latency_mean_ms: float
    latency_p50_ms:  float
    latency_p99_ms:  float
    throughput_tps:  float
    n_samples:       int
    total_tokens:    int
    flops:           Any | None = None

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
    """Legacy tracker. Prefer ``LatencyMetric`` for new code."""

    def __init__(self, window: int = 10_000) -> None:
        self._elapsed_s: deque[float] = deque(maxlen=window)
        self._tokens:    deque[int]   = deque(maxlen=window)

    def record(self, elapsed_s: float, n_tokens: int) -> None:
        self._elapsed_s.append(elapsed_s)
        self._tokens.append(n_tokens)

    @contextlib.contextmanager
    def measure(self, n_tokens: int) -> Iterator[None]:
        _jax_barrier()
        t0 = time.perf_counter()
        yield
        _jax_barrier()
        self.record(time.perf_counter() - t0, n_tokens)

    def result(self) -> ProfilingResult:
        n = len(self._elapsed_s)
        if n == 0:
            return ProfilingResult(0.0, 0.0, 0.0, 0.0, 0, 0)
        ms = sorted(t * 1_000 for t in self._elapsed_s)
        tt = sum(self._elapsed_s)
        tk = sum(self._tokens)
        return ProfilingResult(
            latency_mean_ms=sum(ms) / n,
            latency_p50_ms=ms[int(0.50 * n)],
            latency_p99_ms=ms[min(int(0.99 * n), n - 1)],
            throughput_tps=tk / tt if tt > 0 else 0.0,
            n_samples=n,
            total_tokens=tk,
        )

    def reset(self) -> None:
        self._elapsed_s.clear()
        self._tokens.clear()

    def __len__(self) -> int:
        return len(self._elapsed_s)


def profile_fn(
    fn: Callable[..., Any],
    tracker: LatencyTracker,
    n_tokens: int,
) -> Callable[..., Any]:
    """Legacy helper. Wraps *fn* to record one sample per call."""
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        with tracker.measure(n_tokens):
            return fn(*args, **kwargs)
    return _wrapper
