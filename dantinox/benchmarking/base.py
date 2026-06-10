from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar

import yaml

from dantinox.profiling.counter import FLOPsBreakdown
from dantinox.profiling.tracker import ProfilingResult


# ── Result types ──────────────────────────────────────────────────────────────


@dataclass
class BenchmarkResult:
    """Outcome of a single :class:`BenchmarkTask` run.

    *metrics* holds the task-specific scalars (e.g. perplexity, accuracy,
    tok/s).  *profiling* and *flops* are populated automatically by tasks
    that measure hardware efficiency.
    """

    task: str
    metrics: dict[str, float]
    profiling: ProfilingResult | None = None
    flops: FLOPsBreakdown | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"task": self.task, **self.metrics}
        if self.profiling is not None:
            d["latency_mean_ms"] = self.profiling.latency_mean_ms
            d["latency_p99_ms"]  = self.profiling.latency_p99_ms
            d["throughput_tps"]  = self.profiling.throughput_tps
        if self.flops is not None:
            d["flops_total_g"]   = self.flops.total / 1e9
            d["flops_attn_g"]    = self.flops.attention / 1e9
            d["flops_ffn_g"]     = self.flops.ffn / 1e9
        d.update(self.meta)
        return d


@dataclass
class SuiteReport:
    """Aggregated output of a :class:`~dantinox.benchmarking.suite.BenchmarkSuite` run.

    Attributes
    ----------
    results       : One :class:`BenchmarkResult` per task that was executed.
    model_meta    : Static model info (n_params, paradigm name, config repr…).
    total_time_s  : Wall-clock seconds for the entire suite.
    """

    results: list[BenchmarkResult]
    model_meta: dict[str, Any] = field(default_factory=dict)
    total_time_s: float = 0.0

    # ── Conversion ────────────────────────────────────────────────────────────

    def to_dataframe(self):
        """Return a ``pandas.DataFrame`` with one row per task."""
        import pandas as pd
        return pd.DataFrame([r.to_dict() for r in self.results])

    def save(self, path: str) -> None:
        """Save the suite results to *path* (CSV)."""
        self.to_dataframe().to_csv(path, index=False)

    # ── Display ───────────────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = [
            f"BenchmarkSuite  —  {len(self.results)} task(s)  "
            f"({self.total_time_s:.1f} s total)"
        ]
        if self.model_meta:
            lines.append(
                "  model: "
                + ", ".join(f"{k}={v}" for k, v in self.model_meta.items())
            )
        for r in self.results:
            metric_str = "  ".join(
                f"{k}={v:.4f}" for k, v in r.metrics.items()
            )
            lines.append(f"  [{r.task}]  {metric_str}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"SuiteReport(tasks={[r.task for r in self.results]}, "
            f"time={self.total_time_s:.1f}s)"
        )


# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class BenchmarkConfig:
    """Hardware and evaluation settings for a benchmark suite.

    All fields have sensible defaults so a zero-config run is possible::

        suite.run(model, paradigm)  # uses BenchmarkConfig defaults
    """

    # ── Throughput / latency sweep ────────────────────────────────────────────
    seq_lens: list[int]    = field(default_factory=lambda: [64, 128, 256, 512])
    batch_sizes: list[int] = field(default_factory=lambda: [1, 4, 16, 64, 128])
    n_warmup: int  = 5
    n_measure: int = 20

    # ── Perplexity evaluation ─────────────────────────────────────────────────
    eval_batches: int   = 50
    eval_seq_len: int   = 256
    eval_batch_size: int = 4

    # ── Output ────────────────────────────────────────────────────────────────
    output_dir: str = "benchmark_results"
    seed: int       = 0

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BenchmarkConfig:
        from dataclasses import fields
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})

    @classmethod
    def from_yaml(cls, path: str) -> BenchmarkConfig:
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    def save_yaml(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f)


# ── Task ABC ──────────────────────────────────────────────────────────────────


class BenchmarkTask(ABC):
    """Abstract base for all benchmark tasks.

    Implementing a new task requires only one override::

        class MyTask(BenchmarkTask):
            name = "my_task"

            def run(self, paradigm, model, config, rng):
                score = evaluate_something(model)
                return BenchmarkResult(task=self.name, metrics={"score": score})

    The task is then plug-and-play with any :class:`BenchmarkSuite`::

        suite = BenchmarkSuite([MyTask(), ThroughputTask()])
    """

    name: ClassVar[str]

    @abstractmethod
    def run(
        self,
        paradigm: Any,
        model: Any,
        config: BenchmarkConfig,
        rng: Any,
    ) -> BenchmarkResult:
        """Execute the task and return a :class:`BenchmarkResult`.

        Args:
            paradigm : The :class:`~dantinox.paradigms.Paradigm` wrapping the model.
            model    : The NNX model object (Transformer or ELFTransformer).
            config   : Suite-level benchmark configuration.
            rng      : JAX random key.
        """
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
