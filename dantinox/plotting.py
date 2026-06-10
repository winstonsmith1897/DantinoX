"""Backward-compatible shim — new code should use dantinox.visualization directly.

    >>> from dantinox.visualization import Visualizer
    >>> Visualizer().render(report, out_dir="plots")
"""
from __future__ import annotations

import logging
import warnings

from dantinox.exceptions import PlotError
from dantinox.visualization import RenderConfig, Visualizer

log = logging.getLogger(__name__)

ALL_GROUPS = ["perf", "insights", "3d", "3d_dkv"]

# Map old group names → new chart names for backward compat
_GROUP_TO_CHARTS: dict[str, list[str]] = {
    "perf":     ["throughput", "throughput_batch", "latency"],
    "insights": ["pareto"],
    "3d":       ["pareto"],
    "3d_dkv":   ["throughput"],
}


class Plotter:
    """Legacy plotter — wraps :class:`~dantinox.visualization.Visualizer`.

    For new code, use ``Visualizer`` directly.

    Example::

        # Legacy API (still works)
        Plotter("benchmark_results.csv").run()

        # New API
        import pandas as pd
        df = pd.read_csv("benchmark_results.csv")
        Visualizer().render(df, out_dir="plots")
    """

    def __init__(
        self,
        in_csv: str = "benchmark_results.csv",
        out_dir: str = "plots",
        *,
        batch_csv: str | None = None,
    ) -> None:
        warnings.warn(
            "Plotter is deprecated. Use dantinox.visualization.Visualizer instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.in_csv    = in_csv
        self.out_dir   = out_dir
        self.batch_csv = batch_csv

    def __repr__(self) -> str:
        return f"Plotter(in_csv={self.in_csv!r}, out_dir={self.out_dir!r})"

    def run(self, groups: list[str] | None = None) -> dict[str, list[str]]:
        """Generate plots from the benchmark CSV.

        Translates legacy group names to the new chart registry and delegates
        to :class:`~dantinox.visualization.Visualizer`.
        """
        import os
        import pandas as pd

        if not os.path.exists(self.in_csv):
            raise PlotError(
                f"Benchmark CSV not found: {self.in_csv}\n"
                "Run BenchmarkSuite.run(save_csv='benchmark_results.csv') first."
            )

        selected = list(groups) if groups else ALL_GROUPS
        unknown  = [g for g in selected if g not in _GROUP_TO_CHARTS]
        if unknown:
            raise PlotError(
                f"Unknown plot group(s): {unknown}. Valid groups: {ALL_GROUPS}"
            )

        chart_names: list[str] = []
        for g in selected:
            chart_names.extend(_GROUP_TO_CHARTS[g])
        chart_names = list(dict.fromkeys(chart_names))  # deduplicate, preserve order

        df      = pd.read_csv(self.in_csv)
        viz     = Visualizer()
        paths   = viz.render(df, charts=chart_names, out_dir=self.out_dir)

        return {g: [c for c in _GROUP_TO_CHARTS[g] if c in paths] for g in selected}
