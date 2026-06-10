from __future__ import annotations

from typing import Any

from dantinox.visualization.base import Chart, RenderConfig
from dantinox.visualization.style import TYPE_COLORS, get_palette


class LatencyChart(Chart):
    """Throughput (tok/s) vs latency (ms) scatter — the efficiency frontier.

    Points above and to the left are Pareto-dominant.  A dashed frontier
    curve is drawn automatically.

    Accepts
    -------
    ``SuiteReport`` or ``pandas.DataFrame``
        Must contain ``latency_mean_ms`` and ``throughput_tps`` columns.
        Optional ``type`` column used for colour-coding.
    """

    name    = "latency"
    accepts = object

    def _render_mpl(self, data: Any, config: RenderConfig, fig: Any, ax: Any) -> None:
        import numpy as np
        df     = _to_df(data)
        colors = get_palette(config.style)

        required = {"latency_mean_ms", "throughput_tps"}
        missing  = required - set(df.columns)
        if missing:
            ax.text(0.5, 0.5, f"Missing columns: {missing}",
                    ha="center", va="center", transform=ax.transAxes)
            return

        group_col = "type" if "type" in df.columns else None

        if group_col:
            for i, (grp, sub) in enumerate(df.groupby(group_col)):
                color = TYPE_COLORS.get(str(grp), colors[i % len(colors)])
                ax.scatter(
                    sub["latency_mean_ms"], sub["throughput_tps"],
                    color=color, label=str(grp), s=80, zorder=3,
                )
                if "latency_p99_ms" in df.columns:
                    for _, row in sub.iterrows():
                        ax.plot(
                            [row["latency_mean_ms"], row["latency_p99_ms"]],
                            [row["throughput_tps"], row["throughput_tps"]],
                            color=color, alpha=0.4, linewidth=1,
                        )
        else:
            ax.scatter(
                df["latency_mean_ms"], df["throughput_tps"],
                color=colors[0], s=80, zorder=3,
            )

        _draw_pareto_frontier(ax, df["latency_mean_ms"], df["throughput_tps"])

        ax.set_xlabel("Latency — mean (ms)")
        ax.set_ylabel("Throughput (tokens/s)")
        ax.set_title("Latency vs throughput")
        if group_col:
            ax.legend(title="Model type")


def _draw_pareto_frontier(ax: Any, latency: Any, throughput: Any) -> None:
    import numpy as np
    points = sorted(zip(latency, throughput))
    frontier: list[tuple] = []
    best_tps = -float("inf")
    for lat, tps in points:
        if tps > best_tps:
            frontier.append((lat, tps))
            best_tps = tps
    if len(frontier) > 1:
        xs, ys = zip(*frontier)
        ax.plot(xs, ys, "k--", alpha=0.35, linewidth=1.2, label="Pareto frontier")


def _to_df(data: Any):
    import pandas as pd
    from dantinox.benchmarking.base import SuiteReport
    if isinstance(data, SuiteReport):
        return data.to_dataframe()
    if isinstance(data, str):
        return pd.read_csv(data)
    return data
