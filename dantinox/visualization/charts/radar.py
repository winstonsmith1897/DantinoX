from __future__ import annotations

from typing import Any

import numpy as np

from dantinox.visualization.base import Chart, RenderConfig
from dantinox.visualization.style import get_palette


class RadarChart(Chart):
    """Spider / radar chart for multi-task benchmark comparison.

    Each spoke represents one benchmark metric; each series represents one
    model or configuration.  Values are automatically normalised to [0, 1].

    Accepts
    -------
    ``SuiteReport`` or ``pandas.DataFrame``
        DataFrame where each row is a model and each column is a metric.
        Specify which columns to show via ``metrics`` constructor argument.

    Args:
        metrics   : Column names to include as spokes.  Defaults to all
                    numeric columns.
        model_col : Column used to label series (default: ``"run"``).

    Example::

        RadarChart(metrics=["perplexity", "throughput_tps", "prefill_mean_ms"])
    """

    name    = "radar"
    accepts = object

    def __init__(
        self,
        metrics: list[str] | None = None,
        model_col: str = "run",
    ) -> None:
        self.metrics   = metrics
        self.model_col = model_col

    def render(self, data: Any, config: RenderConfig, out_path: Any) -> Any:
        import matplotlib.pyplot as plt
        from dantinox.visualization.style import apply_style

        apply_style(config.style)
        fig = plt.figure(figsize=(config.width, config.height))
        ax  = fig.add_subplot(111, polar=True)
        self._render_mpl(data, config, fig, ax)
        out_path = __import__("pathlib").Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=config.dpi, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def _render_mpl(self, data: Any, config: RenderConfig, fig: Any, ax: Any) -> None:
        import pandas as pd
        df     = _to_df(data)
        colors = get_palette(config.style)

        # Select numeric columns
        num_cols = df.select_dtypes(include="number").columns.tolist()
        spokes   = self.metrics if self.metrics else num_cols
        spokes   = [s for s in spokes if s in df.columns]
        if not spokes:
            ax.text(0, 0, "No numeric columns found", ha="center")
            return

        n         = len(spokes)
        angles    = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        angles   += angles[:1]  # close the polygon

        # Normalise each spoke to [0, 1] across all rows
        norm_df = df[spokes].copy().astype(float)
        for col in spokes:
            col_min, col_max = norm_df[col].min(), norm_df[col].max()
            if col_max > col_min:
                norm_df[col] = (norm_df[col] - col_min) / (col_max - col_min)
            else:
                norm_df[col] = 0.5

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(spokes, size=9)
        ax.set_ylim(0, 1)

        label_col = self.model_col if self.model_col in df.columns else None
        for i, (_, row) in enumerate(norm_df.iterrows()):
            vals   = row[spokes].tolist() + row[spokes[:1]].tolist()
            label  = str(df[label_col].iloc[i]) if label_col else f"Model {i}"
            color  = colors[i % len(colors)]
            ax.plot(angles, vals, color=color, linewidth=1.8, label=label)
            ax.fill(angles, vals, color=color, alpha=0.12)

        ax.set_title("Multi-task benchmark", y=1.08)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))


def _to_df(data: Any):
    import pandas as pd
    from dantinox.benchmarking.base import SuiteReport
    if isinstance(data, SuiteReport):
        return data.to_dataframe()
    if isinstance(data, str):
        return pd.read_csv(data)
    return data
