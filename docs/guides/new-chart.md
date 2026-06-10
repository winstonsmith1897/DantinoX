# Custom Chart

Charts are registered with a class-level decorator and auto-discovered by `Visualizer`. Adding a new chart is a two-step process: implement the `Chart` subclass, then apply `@Visualizer.register`.

---

## Step 1: Implement the chart

```python
# dantinox/visualization/charts/loss_histogram.py
from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pandas as pd

from dantinox.visualization.base import Chart, RenderConfig
from dantinox.visualization.style import apply_style


class LossHistogramChart(Chart):
    """Distribution of final per-step loss values across training runs.

    Renders a histogram (or KDE) of ``data["loss"]`` values so that
    training instabilities (spikes, heavy tails) are immediately visible.

    Attributes:
        name: Registry key — ``"loss_histogram"``.
        accepts: Expects a ``pd.DataFrame`` with a ``"loss"`` column.
        bins: Number of histogram bins.
    """

    name:    ClassVar[str]  = "loss_histogram"
    accepts: ClassVar[type] = pd.DataFrame

    def __init__(self, bins: int = 50) -> None:
        """Initialize the chart.

        Args:
            bins: Number of histogram bins. Default ``50``.
        """
        self.bins = bins

    def _render_mpl(
        self,
        data: pd.DataFrame,
        config: RenderConfig,
        fig,
        ax,
    ) -> None:
        """Render the loss histogram using matplotlib.

        Args:
            data: DataFrame with a ``"loss"`` column.
            config: Render configuration (style, dpi, …).
            fig: Matplotlib figure.
            ax: Matplotlib axes.
        """
        apply_style(config.style)
        if "loss" not in data.columns:
            ax.text(0.5, 0.5, "No 'loss' column found",
                    ha="center", va="center", transform=ax.transAxes)
            return

        ax.hist(data["loss"].dropna(), bins=self.bins,
                color="#3A86FF", edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Loss", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title("Training Loss Distribution", fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
```

---

## Step 2: Register it

Apply `@Visualizer.register` and import the class in `dantinox/visualization/charts/__init__.py`:

```python
# dantinox/visualization/charts/__init__.py
from dantinox.visualization.visualizer import Visualizer

from dantinox.visualization.charts.training       import TrainingCurveChart
from dantinox.visualization.charts.throughput     import ThroughputChart, ThroughputBatchChart
from dantinox.visualization.charts.latency        import LatencyChart
from dantinox.visualization.charts.pareto         import ParetoChart
from dantinox.visualization.charts.radar          import RadarChart
from dantinox.visualization.charts.loss_histogram import LossHistogramChart   # ← new

# Auto-register all default-constructible charts
for _cls in [TrainingCurveChart, ThroughputChart, ThroughputBatchChart,
             LatencyChart, ParetoChart, LossHistogramChart]:         # ← add here
    Visualizer.register(_cls)
# RadarChart requires constructor args — register but skip auto-render
Visualizer.register(RadarChart)
```

---

## Step 3: Use it

```python
import pandas as pd
from dantinox.visualization import Visualizer

df = pd.read_csv("training_log.csv")
Visualizer().render(df, charts=["loss_histogram"], out_dir="plots/")
```

Or via CLI:

```bash
dantinox plot --in_csv training_log.csv --out_dir plots/ --groups perf
```

---

## Optional: implement the Plotly backend

Override `_render_plotly` for interactive HTML output:

```python
def _render_plotly(self, data: pd.DataFrame, config: RenderConfig):
    import plotly.express as px
    return px.histogram(data, x="loss", nbins=self.bins,
                        title="Training Loss Distribution")
```

Users activate it via `RenderConfig(backend="plotly", fmt="html")`.

---

## Checklist

- [ ] `name: ClassVar[str]` — unique, snake_case
- [ ] `accepts: ClassVar[type]` — the data type `render()` passes to `_render_mpl`
- [ ] `__init__` has a Google docstring (needed for `interrogate`)
- [ ] `_render_mpl` has a Google docstring
- [ ] Registered in `charts/__init__.py`
- [ ] Unit test verifying the chart produces a file at the expected path
- [ ] `make doccheck` passes
