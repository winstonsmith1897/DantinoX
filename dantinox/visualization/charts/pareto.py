from __future__ import annotations

from typing import Any

from dantinox.visualization.base import Chart, RenderConfig
from dantinox.visualization.style import TYPE_COLORS, get_palette


class ParetoChart(Chart):
    """Quality vs efficiency Pareto chart.

    Plots perplexity (y-axis, lower is better) against throughput or a
    model-size metric (x-axis, higher is better), revealing the Pareto
    frontier of quality-vs-efficiency trade-offs.

    Args:
        x_col    : Column name for the efficiency axis (default: ``"throughput_tps"``).
        y_col    : Column name for the quality axis (default: ``"perplexity"``).
        label_col: Column used to annotate each point (default: ``"run"``).
        size_col : Column used to size markers (e.g. ``"n_params"``). Optional.

    Example::

        ParetoChart(x_col="throughput_tps", y_col="perplexity")
    """

    name    = "pareto"
    accepts = object

    def __init__(
        self,
        x_col: str = "throughput_tps",
        y_col: str = "perplexity",
        label_col: str = "run",
        size_col: str | None = None,
    ) -> None:
        self.x_col     = x_col
        self.y_col     = y_col
        self.label_col = label_col
        self.size_col  = size_col

    def _render_mpl(self, data: Any, config: RenderConfig, fig: Any, ax: Any) -> None:
        import numpy as np
        df     = _to_df(data)
        colors = get_palette(config.style)

        for col in (self.x_col, self.y_col):
            if col not in df.columns:
                ax.text(0.5, 0.5, f"Missing column: {col!r}",
                        ha="center", va="center", transform=ax.transAxes)
                return

        group_col = "type" if "type" in df.columns else None
        sizes = (
            _scale_sizes(df[self.size_col])
            if self.size_col and self.size_col in df.columns
            else [80] * len(df)
        )

        if group_col:
            for i, (grp, sub) in enumerate(df.groupby(group_col)):
                idx   = sub.index
                color = TYPE_COLORS.get(str(grp), colors[i % len(colors)])
                ax.scatter(
                    sub[self.x_col], sub[self.y_col],
                    s=[sizes[j] for j in idx],
                    color=color, label=str(grp), zorder=3, alpha=0.85,
                )
        else:
            ax.scatter(
                df[self.x_col], df[self.y_col],
                s=sizes, color=colors[0], zorder=3, alpha=0.85,
            )

        # Annotate points
        if self.label_col in df.columns:
            for _, row in df.iterrows():
                ax.annotate(
                    str(row[self.label_col]),
                    xy=(row[self.x_col], row[self.y_col]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=7, alpha=0.75,
                )

        # Pareto frontier (lower ppl + higher tps)
        _draw_pareto_2d(ax, df[self.x_col].values, df[self.y_col].values)

        ax.set_xlabel(self.x_col.replace("_", " ").title())
        ax.set_ylabel(self.y_col.replace("_", " ").title())
        ax.set_title("Quality–efficiency Pareto chart")
        if group_col:
            ax.legend(title="Model type")
        if self.size_col:
            ax.text(
                0.02, 0.98, f"Marker size ∝ {self.size_col}",
                transform=ax.transAxes, fontsize=7, va="top", alpha=0.6,
            )


def _draw_pareto_2d(ax: Any, x: Any, y: Any) -> None:
    import numpy as np
    """Draw the Pareto frontier for (max x, min y) optimum."""
    pts = sorted(zip(x, y), key=lambda p: p[0])
    frontier: list[tuple] = []
    best_y = float("inf")
    for xi, yi in pts:
        if yi < best_y:
            frontier.append((xi, yi))
            best_y = yi
    if len(frontier) > 1:
        xs, ys = zip(*frontier)
        ax.plot(xs, ys, "k--", alpha=0.3, linewidth=1.2, zorder=2)


def _scale_sizes(col: Any) -> list[float]:
    import numpy as np
    v = col.astype(float).values
    v_min, v_max = v.min(), v.max()
    if v_max > v_min:
        return (30 + 200 * (v - v_min) / (v_max - v_min)).tolist()
    return [80.0] * len(v)


def _to_df(data: Any):
    import pandas as pd
    from dantinox.benchmarking.base import SuiteReport
    if isinstance(data, SuiteReport):
        return data.to_dataframe()
    if isinstance(data, str):
        return pd.read_csv(data)
    return data
