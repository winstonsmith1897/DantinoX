from __future__ import annotations

from typing import Any

import numpy as np

from dantinox.visualization.base import Chart, RenderConfig
from dantinox.visualization.style import TYPE_COLORS, get_palette


class ThroughputChart(Chart):
    """Tokens/s vs sequence length — one line per attention type or model.

    Accepts
    -------
    ``SuiteReport`` or ``pandas.DataFrame``
        A DataFrame must contain columns ``tps_seq{L}`` for each sequence length
        and optionally a ``type`` column for colour-coding.
    """

    name    = "throughput"
    accepts = object

    def _render_mpl(self, data: Any, config: RenderConfig, fig: Any, ax: Any) -> None:
        df     = _to_df(data)
        colors = get_palette(config.style)

        seq_cols  = [c for c in df.columns if c.startswith("tps_seq")]
        seq_lens  = sorted(int(c.replace("tps_seq", "")) for c in seq_cols)
        group_col = "type" if "type" in df.columns else None

        if group_col and df[group_col].nunique() > 1:
            for i, (grp, sub) in enumerate(df.groupby(group_col)):
                color = TYPE_COLORS.get(str(grp), colors[i % len(colors)])
                vals  = [sub[f"tps_seq{s}"].mean() for s in seq_lens]
                ax.plot(seq_lens, vals, marker="o", label=str(grp), color=color)
        else:
            vals = [df[f"tps_seq{s}"].mean() for s in seq_lens]
            ax.plot(seq_lens, vals, marker="o", color=colors[0], label="Throughput")

        ax.set_xlabel("Sequence length (tokens)")
        ax.set_ylabel("Throughput (tokens/s)")
        ax.set_title("Throughput vs sequence length")
        ax.legend(title="Model type")
        ax.set_xticks(seq_lens)


class ThroughputBatchChart(Chart):
    """Tokens/s vs batch size — shows how well the model scales with parallelism.

    Accepts
    -------
    ``SuiteReport`` or ``pandas.DataFrame``
        Must contain columns ``tps_bs{B}`` for each batch size.
    """

    name    = "throughput_batch"
    accepts = object

    def _render_mpl(self, data: Any, config: RenderConfig, fig: Any, ax: Any) -> None:
        df     = _to_df(data)
        colors = get_palette(config.style)

        bs_cols = [c for c in df.columns if c.startswith("tps_bs")]
        batches = sorted(int(c.replace("tps_bs", "")) for c in bs_cols)

        group_col = "type" if "type" in df.columns else None
        if group_col and df[group_col].nunique() > 1:
            for i, (grp, sub) in enumerate(df.groupby(group_col)):
                color = TYPE_COLORS.get(str(grp), colors[i % len(colors)])
                vals  = [sub[f"tps_bs{b}"].mean() for b in batches]
                ax.plot(batches, vals, marker="s", label=str(grp), color=color)
        else:
            vals = [df[f"tps_bs{b}"].mean() for b in batches]
            ax.plot(batches, vals, marker="s", color=colors[0])

        ax.set_xlabel("Batch size")
        ax.set_ylabel("Throughput (tokens/s)")
        ax.set_title("Throughput vs batch size")
        ax.set_xscale("log", base=2)
        ax.set_xticks(batches)
        ax.get_xaxis().set_major_formatter(__import__("matplotlib").ticker.ScalarFormatter())
        if group_col:
            ax.legend(title="Model type")


def _to_df(data: Any):
    import pandas as pd
    from dantinox.benchmarking.base import SuiteReport
    if isinstance(data, SuiteReport):
        return data.to_dataframe()
    if isinstance(data, str):
        return pd.read_csv(data)
    return data
