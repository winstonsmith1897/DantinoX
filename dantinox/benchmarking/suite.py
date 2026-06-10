from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import jax

from dantinox.benchmarking.base import BenchmarkConfig, BenchmarkResult, BenchmarkTask, SuiteReport

log = logging.getLogger(__name__)


class BenchmarkSuite:
    """Runs a collection of :class:`BenchmarkTask` instances against a model.

    The suite is the single entry-point for evaluation.  It owns no model
    logic — it delegates entirely to each task's ``run()`` method.

    Quick-start::

        from dantinox.benchmarking import BenchmarkSuite

        # Default suite: Throughput + Latency + Perplexity
        report = BenchmarkSuite.default().run(paradigm, model)
        print(report.summary())
        report.save("results.csv")

    Custom suite::

        suite = BenchmarkSuite(
            tasks=[ThroughputTask(), PerplexityTask("data/val.txt")],
            config=BenchmarkConfig(seq_lens=[128, 512], n_measure=50),
        )
        report = suite.run(paradigm, model)
    """

    def __init__(
        self,
        tasks: list[BenchmarkTask],
        config: BenchmarkConfig | None = None,
    ) -> None:
        if not tasks:
            raise ValueError("BenchmarkSuite requires at least one task.")
        self.tasks  = tasks
        self.config = config or BenchmarkConfig()

    # ── Execution ─────────────────────────────────────────────────────────────

    def run(
        self,
        paradigm: Any,
        model: Any,
        *,
        save_csv: str | None = None,
    ) -> SuiteReport:
        """Run all tasks sequentially and return a :class:`SuiteReport`.

        Args:
            paradigm  : Any :class:`~dantinox.paradigms.Paradigm` instance.
            model     : The NNX model returned by ``paradigm.build_model()``.
            save_csv  : If provided, write the report to this CSV path.

        Returns:
            A :class:`SuiteReport` aggregating every task's outcome.
        """
        rng          = jax.random.PRNGKey(self.config.seed)
        model_meta   = _extract_model_meta(paradigm, model)
        results: list[BenchmarkResult] = []
        suite_start  = time.perf_counter()

        for task in self.tasks:
            rng, rng_task = jax.random.split(rng)
            log.info("[benchmark] running task '%s'…", task.name)
            t0 = time.perf_counter()
            try:
                result = task.run(paradigm, model, self.config, rng_task)
                elapsed = time.perf_counter() - t0
                log.info(
                    "[benchmark] '%s' done in %.2f s — %s",
                    task.name,
                    elapsed,
                    "  ".join(f"{k}={v:.4f}" for k, v in result.metrics.items()),
                )
                results.append(result)
            except Exception as exc:
                log.error("[benchmark] task '%s' failed: %s", task.name, exc)

        report = SuiteReport(
            results=results,
            model_meta=model_meta,
            total_time_s=time.perf_counter() - suite_start,
        )

        if save_csv is not None:
            Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
            report.save(save_csv)
            log.info("[benchmark] results saved to %s", save_csv)

        return report

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def default(cls, config: BenchmarkConfig | None = None) -> BenchmarkSuite:
        """Return a suite with the three standard tasks: Throughput + Latency + Perplexity.

        Perplexity is only meaningful when a data source is attached at runtime;
        if no data is available the task will log a warning and return NaN.
        """
        from dantinox.benchmarking.tasks import LatencyTask, PerplexityTask, ThroughputTask

        return cls(
            tasks=[ThroughputTask(), LatencyTask(), PerplexityTask()],
            config=config,
        )

    @classmethod
    def throughput_only(cls, config: BenchmarkConfig | None = None) -> BenchmarkSuite:
        """Return a minimal suite for quick hardware throughput checks."""
        from dantinox.benchmarking.tasks import ThroughputTask
        return cls(tasks=[ThroughputTask()], config=config)

    # ── Representation ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        task_names = [t.name for t in self.tasks]
        return f"BenchmarkSuite(tasks={task_names})"


# ── Internal helpers ──────────────────────────────────────────────────────────


def _extract_model_meta(paradigm: Any, model: Any) -> dict[str, Any]:
    meta: dict[str, Any] = {"paradigm": type(paradigm).__name__}
    try:
        meta["n_params"] = paradigm.num_parameters(model)
    except Exception:
        pass
    try:
        cfg = getattr(paradigm, "config", None) or getattr(paradigm, "model_config", None)
        if cfg is not None:
            meta["dim"]        = getattr(cfg, "dim", None)
            meta["num_blocks"] = getattr(cfg, "num_blocks", None)
            meta["n_heads"]    = getattr(cfg, "n_heads", None)
    except Exception:
        pass
    return {k: v for k, v in meta.items() if v is not None}
