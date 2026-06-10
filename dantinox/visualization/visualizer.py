from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from dantinox.visualization.base import Chart, RenderConfig

log = logging.getLogger(__name__)


class Visualizer:
    """Central chart registry and rendering orchestrator.

    The Visualizer maintains a class-level registry of :class:`Chart`
    subclasses.  Built-in charts are registered automatically on import.
    User charts are registered with the :meth:`register` decorator.

    Quick-start::

        from dantinox.visualization import Visualizer

        viz = Visualizer()
        viz.render(report, out_dir="plots")           # all charts
        viz.render(report, charts=["pareto", "radar"])# specific charts

    Adding a custom chart::

        from dantinox.visualization import Visualizer
        from dantinox.visualization.base import Chart

        @Visualizer.register
        class MyChart(Chart):
            name    = "my_chart"
            accepts = object

            def _render_mpl(self, data, config, fig, ax):
                ax.bar([1, 2], [3, 4])

        viz = Visualizer()
        viz.render(data, charts=["my_chart"])
    """

    _registry: dict[str, type[Chart]] = {}

    def __init__(self, config: RenderConfig | None = None) -> None:
        self.config = config or RenderConfig()

    # ── Registry ──────────────────────────────────────────────────────────────

    @classmethod
    def register(cls, chart_cls: type[Chart]) -> type[Chart]:
        """Register a :class:`Chart` subclass.  Can be used as a decorator."""
        cls._registry[chart_cls.name] = chart_cls
        log.debug("Registered chart: %s", chart_cls.name)
        return chart_cls

    @classmethod
    def available_charts(cls) -> list[str]:
        """Return the names of all registered charts."""
        return sorted(cls._registry)

    # ── Rendering ─────────────────────────────────────────────────────────────

    def render(
        self,
        data: Any,
        *,
        charts: list[str] | None = None,
        out_dir: str = "plots",
        config: RenderConfig | None = None,
    ) -> dict[str, Path]:
        """Render one or more charts and write them to *out_dir*.

        Args:
            data    : Any data understood by the registered charts
                      (e.g. :class:`~dantinox.benchmarking.base.SuiteReport`,
                      ``pandas.DataFrame``, or a CSV path).
            charts  : Names of charts to render.  Renders all registered
                      charts when omitted.
            out_dir : Directory for output files (created if absent).
            config  : Override the instance-level :class:`RenderConfig`.

        Returns:
            ``{chart_name: output_path}`` for every chart that succeeded.
        """
        cfg      = config or self.config
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        selected = charts or list(self._registry)
        unknown  = [c for c in selected if c not in self._registry]
        if unknown:
            raise ValueError(
                f"Unknown charts: {unknown}. "
                f"Available: {self.available_charts()}"
            )

        results: dict[str, Path] = {}
        for name in selected:
            chart_cls = self._registry[name]
            chart     = chart_cls() if _is_default_constructible(chart_cls) else None
            if chart is None:
                log.warning(
                    "Chart %r requires constructor arguments — "
                    "instantiate it manually and call chart.render() directly.",
                    name,
                )
                continue

            file_name = f"{name}.{cfg.fmt}"
            file_path = out_path / file_name
            try:
                chart.render(data, cfg, file_path)
                results[name] = file_path
                log.info("[visualizer] %s → %s", name, file_path)
            except Exception as exc:
                log.error("[visualizer] %s failed: %s", name, exc)

        return results

    def __repr__(self) -> str:
        return (
            f"Visualizer(charts={self.available_charts()}, "
            f"backend={self.config.backend!r})"
        )


# ── Internal helpers ──────────────────────────────────────────────────────────


def _is_default_constructible(cls: type) -> bool:
    """True if the class can be constructed with no arguments."""
    import inspect
    try:
        sig    = inspect.signature(cls.__init__)
        params = [
            p for p in sig.parameters.values()
            if p.name != "self" and p.default is inspect.Parameter.empty
        ]
        return len(params) == 0
    except (ValueError, TypeError):
        return False


# ── Auto-register built-in charts ────────────────────────────────────────────
# Importing this module triggers registration of all bundled chart types.

def _register_builtins() -> None:
    from dantinox.visualization.charts.training   import TrainingCurveChart
    from dantinox.visualization.charts.throughput import ThroughputChart, ThroughputBatchChart
    from dantinox.visualization.charts.latency    import LatencyChart
    from dantinox.visualization.charts.radar      import RadarChart
    from dantinox.visualization.charts.pareto     import ParetoChart

    for cls in (
        TrainingCurveChart,
        ThroughputChart,
        ThroughputBatchChart,
        LatencyChart,
        ParetoChart,
        # RadarChart requires constructor args (optional metrics list) so it
        # is registered but the auto-render path skips it unless instantiated.
        RadarChart,
    ):
        Visualizer.register(cls)


_register_builtins()
