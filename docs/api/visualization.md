# `dantinox.visualization`

The visualization module uses a class-level registry. Charts are registered with `@Visualizer.register` and auto-discovered at import time. No configuration required to use built-in charts.

---

## Visualizer

::: dantinox.visualization.visualizer.Visualizer
    options:
      show_source: true
      members:
        - register
        - render

---

## Base types

::: dantinox.visualization.base.RenderConfig
    options:
      show_source: true

::: dantinox.visualization.base.Chart
    options:
      show_source: true
      members:
        - render
        - _render_mpl
        - _render_plotly

---

## Built-in charts

::: dantinox.visualization.charts.training.TrainingCurveChart
    options:
      show_source: true
      heading_level: 3

::: dantinox.visualization.charts.throughput.ThroughputChart
    options:
      show_source: true
      heading_level: 3

::: dantinox.visualization.charts.throughput.ThroughputBatchChart
    options:
      show_source: true
      heading_level: 3

::: dantinox.visualization.charts.latency.LatencyChart
    options:
      show_source: true
      heading_level: 3

::: dantinox.visualization.charts.pareto.ParetoChart
    options:
      show_source: true
      heading_level: 3

::: dantinox.visualization.charts.radar.RadarChart
    options:
      show_source: true
      heading_level: 3

---

## Quick reference

```python
import pandas as pd
from dantinox.visualization import Visualizer, RenderConfig

df = pd.read_csv("benchmark_results.csv")

# Render all registered default-constructible charts
Visualizer().render(df, out_dir="plots/")

# Specific charts with custom config
cfg = RenderConfig(backend="matplotlib", fmt="pdf", style="publication", dpi=300)
Visualizer().render(df, charts=["throughput", "pareto"], out_dir="paper_plots/", config=cfg)

# Radar chart (requires explicit instantiation — not auto-rendered)
from dantinox.visualization import RadarChart, Visualizer
radar = RadarChart(metrics=["peak_tps", "perplexity", "prefill_mean_ms"])
Visualizer(extra_charts=[radar]).render(df, charts=["radar"], out_dir="plots/")
```

### Style presets

| `style` | Use case |
| :--- | :--- |
| `"publication"` | LaTeX-compatible, high-DPI, serif fonts |
| `"dark"` | Presentations, dark-mode slides |
| `"minimal"` | Lightweight, no gridlines |

### Chart registry

```python
from dantinox.visualization import Visualizer

# List all registered chart names
print(list(Visualizer._registry.keys()))
# ['training_curve', 'throughput', 'throughput_batch', 'latency', 'pareto', 'radar']
```
