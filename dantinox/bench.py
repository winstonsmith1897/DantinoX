"""Backward-compatible shim — new code should use dantinox.benchmarking directly.

    >>> from dantinox.benchmarking import BenchmarkSuite
    >>> report = BenchmarkSuite.default().run(paradigm, model)
"""
from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from typing import Any

from dantinox.benchmarking import (
    BenchmarkConfig,
    BenchmarkSuite,
    LatencyTask,
    PerplexityTask,
    ThroughputTask,
)
from dantinox.exceptions import BenchmarkError

log = logging.getLogger(__name__)


class BenchmarkRunner:
    """Legacy runner — wraps :class:`~dantinox.benchmarking.BenchmarkSuite`.

    For new code, use ``BenchmarkSuite`` directly.

    Example::

        # Legacy API (still works)
        runner = BenchmarkRunner("runs")
        df     = runner.run(out_csv="results.csv")

        # New API
        suite  = BenchmarkSuite.default()
        report = suite.run(paradigm, model, save_csv="results.csv")
    """

    def __init__(
        self,
        runs_dir: str = "runs",
        *,
        seq_lens: Sequence[int] | None = None,
        batch_sizes: Sequence[int] | None = None,
    ) -> None:
        warnings.warn(
            "BenchmarkRunner is deprecated. Use dantinox.benchmarking.BenchmarkSuite instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.runs_dir = runs_dir
        self._config  = BenchmarkConfig(
            seq_lens    = list(seq_lens)    if seq_lens    else [64, 128, 256, 512],
            batch_sizes = list(batch_sizes) if batch_sizes else [1, 4, 16, 64, 128, 256],
        )

    def __repr__(self) -> str:
        return f"BenchmarkRunner(runs_dir={self.runs_dir!r})"

    def run(
        self,
        run_names: Sequence[str] | None = None,
        *,
        out_csv: str | None = None,
    ) -> Any:
        """Run throughput benchmarks and return a DataFrame.

        Loads each run directory individually via the legacy checkpoint loader.
        """
        import os
        import traceback
        import pandas as pd
        from core.config import Config
        from core.model import Transformer
        from flax import nnx
        from dantinox.paradigms.ar import ARParadigm

        if not os.path.isdir(self.runs_dir):
            raise BenchmarkError(f"Runs directory not found: {self.runs_dir}")

        if run_names is None:
            run_names = [
                d for d in os.listdir(self.runs_dir)
                if os.path.isdir(os.path.join(self.runs_dir, d))
            ]

        rows: list[dict] = []
        for name in run_names:
            path = os.path.join(self.runs_dir, name)
            log.info("Benchmarking legacy run: %s", name)
            try:
                model   = Transformer.from_pretrained(path, rngs=nnx.Rngs(42))
                cfg     = getattr(model, "config", None)
                if cfg is None:
                    raise BenchmarkError(f"Cannot read config from {name}")
                paradigm = ARParadigm(cfg.to_model_config() if hasattr(cfg, "to_model_config") else cfg)
                suite    = BenchmarkSuite(
                    tasks=[ThroughputTask(), LatencyTask()],
                    config=self._config,
                )
                report = suite.run(paradigm, model)
                row    = {"run": name, **report.to_dataframe().to_dict("records")[0]}
                rows.append(row)
            except BenchmarkError as exc:
                log.error("  Skipped %s: %s", name, exc)
            except Exception as exc:
                log.error("  Unexpected error for %s: %s\n%s", name, exc, traceback.format_exc())

        df = pd.DataFrame(rows)
        if out_csv:
            df.to_csv(out_csv, index=False)
            log.info("Saved benchmark results to %s", out_csv)
        return df
