"""Profiling plots: interactive 3D surfaces (Plotly) and 2D benchmark figures (matplotlib).

Quick start — 3D interactive::

    from dantinox.profiling import plot_3d_compare, plot_3d_from_df

    # From a sweep DataFrame (no RunProfile objects needed):
    fig = plot_3d_from_df(batch_df, z="tps")
    fig.show()

    # From MultiRunReport:
    fig = plot_3d_compare(report, metric="tps", mode="surface")
    fig.show()

Quick start — 2D matplotlib figures::

    from dantinox.profiling import (
        plot_scale, plot_batch, plot_steps, plot_dtype, plot_composite,
    )

    fig, axes = plot_scale(scale_df)          # throughput + speedup vs model size
    fig, axes = plot_batch(batch_df)           # throughput vs batch_size
    fig, axes = plot_steps(steps_df)           # throughput + latency vs diffusion steps
    fig, ax   = plot_dtype(dtype_df)           # fp32 vs bf16 bar chart
    fig, axes = plot_composite(scale_df, batch_df, steps_df)  # 2×2 summary panel

DataFrame column contracts
--------------------------
- **scale_df**  : ``paradigm, params_m, tok_s`` (+ optional ``mfu_pct, step_ms``)
- **batch_df**  : ``paradigm, batch_size, seq_len, tps, e2e_ms``
- **steps_df**  : ``paradigm, steps, tps, e2e_ms``
- **dtype_df**  : ``paradigm, dtype, tps``

Supported 3D metric aliases (``metric`` arg)
--------------------------------------------
    tps / throughput  → grid key "tps"
    latency / mean_ms → grid key "mean_ms"
    p50 / p95 / p99   → grid keys "p50_ms", "p95_ms", "p99_ms"
    perplexity / ppl  → sweep key "perplexity"
    entropy           → sweep key "mean_entropy"
    bpb               → sweep key "bpb"
"""
from __future__ import annotations

import math
from typing import Any

# ── Metric alias resolution ───────────────────────────────────────────────────

_METRIC_ALIASES: list[tuple[list[str], str, str]] = [
    # (triggers,                grid_key,    source)
    (["tps", "throughput"],      "tps",       "throughput_grid"),
    (["mean_ms", "latency"],     "mean_ms",   "latency_grid"),
    (["p50"],                    "p50_ms",    "latency_grid"),
    (["p95"],                    "p95_ms",    "latency_grid"),
    (["p99"],                    "p99_ms",    "latency_grid"),
    (["perplexity", "ppl"],      "perplexity","perplexity_sweep"),
    (["entropy"],                "mean_entropy","entropy_sweep"),
    (["bpb"],                    "bpb",       "perplexity_sweep"),
    (["top1"],                   "mean_top1_prob","entropy_sweep"),
]


def _resolve_metric(metric: str) -> tuple[str, str]:
    """Return (grid_key, source) for a user-supplied metric string."""
    m = metric.lower()
    for triggers, grid_key, source in _METRIC_ALIASES:
        if any(t in m for t in triggers):
            return grid_key, source
    # Fall-through: assume the metric string IS the grid key, source unknown
    return metric, "any"


def _extract_grid(profile: Any, source: str) -> list[dict[str, Any]]:
    """Pull the sweep / grid list from a RunProfile or result object."""
    # RunProfile case
    if hasattr(profile, "throughput") and source == "throughput_grid":
        r = profile.throughput
        return r.grid if r is not None else []
    if hasattr(profile, "latency") and source == "latency_grid":
        r = profile.latency
        return r.grid if r is not None else []
    if hasattr(profile, "perplexity") and source == "perplexity_sweep":
        r = profile.perplexity
        return r.sweep if r is not None else []
    if hasattr(profile, "entropy") and source == "entropy_sweep":
        r = profile.entropy
        return r.sweep if r is not None else []
    if source == "any":
        for attr in ("throughput", "latency", "perplexity", "entropy"):
            r = getattr(profile, attr, None)
            if r is None:
                continue
            g = getattr(r, "grid", None) or getattr(r, "sweep", None)
            if g:
                return g
    return []


# ── Colour palette ─────────────────────────────────────────────────────────────

_COLORSCALES = [
    "Blues", "Greens", "Reds", "Purples", "Oranges",
    "YlOrRd", "YlGnBu", "Hot", "Electric", "Viridis",
]

_LINE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE-RUN SURFACE
# ══════════════════════════════════════════════════════════════════════════════


def plot_3d_surface(
    profile:    Any,          # RunProfile or any object with .throughput/.latency/…
    metric:     str = "tps",
    title:      str | None = None,
    colorscale: str = "Viridis",
    show:       bool = True,
    output:     str | None = None,
):
    """3D surface for a **single** run: X=batch_size, Y=seq_len, Z=metric.

    Args:
        profile    : A :class:`RunProfile` (or any object that exposes
                     ``.throughput.grid``, ``.latency.grid``, etc.).
        metric     : Metric to visualise (see module docstring for aliases).
        title      : Plot title — defaults to ``run_name + metric``.
        colorscale : Plotly colorscale name.
        show       : If True, open the figure in a browser tab.
        output     : If given, write the figure to this HTML file.

    Returns:
        ``plotly.graph_objects.Figure``
    """
    import plotly.graph_objects as go
    import numpy as np

    grid_key, source = _resolve_metric(metric)
    grid = _extract_grid(profile, source)

    if not grid:
        raise ValueError(
            f"No sweep / grid data found for metric={metric!r} "
            f"(source={source!r}) in profile {getattr(profile, 'run_name', '?')!r}. "
            "Did you run the profiler with lists for seq_lens and batch_sizes?"
        )

    xs = sorted(set(r["batch_size"] for r in grid))
    ys = sorted(set(r["seq_len"]    for r in grid))

    # Build Z matrix (batch_size × seq_len grid)
    lookup: dict[tuple, float] = {
        (r["batch_size"], r["seq_len"]): r.get(grid_key, float("nan"))
        for r in grid
    }
    z = np.array(
        [[lookup.get((bs, sl), float("nan")) for sl in ys] for bs in xs],
        dtype=float,
    )

    run_name = getattr(profile, "run_name", "run")
    fig_title = title or f"{run_name} — {metric}"

    z_label = _z_label(grid_key)
    surf = go.Surface(
        x=np.array(xs),
        y=np.array(ys),
        z=z,
        colorscale=colorscale,
        colorbar=dict(title=z_label),
        name=run_name,
        hovertemplate=(
            f"batch_size=%{{x}}<br>seq_len=%{{y}}<br>"
            f"{z_label}=%{{z:.3f}}<extra>{run_name}</extra>"
        ),
    )

    fig = go.Figure(data=[surf])
    fig.update_layout(
        title=fig_title,
        scene=dict(
            xaxis=dict(title="batch_size"),
            yaxis=dict(title="seq_len"),
            zaxis=dict(title=z_label),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=1.2)),
        ),
        margin=dict(l=0, r=0, b=0, t=50),
    )

    _save_or_show(fig, show=show, output=output)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# MULTI-RUN COMPARISON
# ══════════════════════════════════════════════════════════════════════════════


def plot_3d_compare(
    report:  Any,             # MultiRunReport
    metric:  str = "tps",
    mode:    str = "surface", # "surface" | "scatter"
    title:   str | None = None,
    show:    bool = True,
    output:  str | None = None,
):
    """Overlay one surface (or scatter) per run for multi-run comparison.

    Args:
        report  : A :class:`MultiRunReport`.
        metric  : Metric to visualise.
        mode    : ``"surface"`` renders opaque surfaces (good for 2-4 runs),
                  ``"scatter"`` renders coloured 3D scatter points (many runs).
        title   : Figure title.
        show    : Open in browser.
        output  : Write HTML to this path.

    Returns:
        ``plotly.graph_objects.Figure``
    """
    import plotly.graph_objects as go
    import numpy as np

    grid_key, source = _resolve_metric(metric)
    profiles = getattr(report, "profiles", [report])

    traces = []
    for i, profile in enumerate(profiles):
        grid = _extract_grid(profile, source)
        if not grid:
            continue

        run_name = getattr(profile, "run_name", f"run_{i}")
        color_i  = i % len(_COLORSCALES)

        xs = sorted(set(r["batch_size"] for r in grid))
        ys = sorted(set(r["seq_len"]    for r in grid))
        lookup = {
            (r["batch_size"], r["seq_len"]): r.get(grid_key, float("nan"))
            for r in grid
        }
        z_label = _z_label(grid_key)

        if mode == "surface" and len(xs) > 1 and len(ys) > 1:
            z = np.array(
                [[lookup.get((bs, sl), float("nan")) for sl in ys] for bs in xs],
                dtype=float,
            )
            traces.append(go.Surface(
                x=np.array(xs),
                y=np.array(ys),
                z=z,
                colorscale=_COLORSCALES[color_i],
                showscale=False,
                opacity=0.85,
                name=run_name,
                hovertemplate=(
                    f"batch_size=%{{x}}<br>seq_len=%{{y}}<br>"
                    f"{z_label}=%{{z:.3f}}<extra>{run_name}</extra>"
                ),
            ))
        else:
            # Fallback to scatter (also used when mode="scatter")
            bss = [r["batch_size"] for r in grid]
            sls = [r["seq_len"]    for r in grid]
            zs  = [r.get(grid_key, float("nan")) for r in grid]
            clr = _LINE_COLORS[i % len(_LINE_COLORS)]
            traces.append(go.Scatter3d(
                x=bss, y=sls, z=zs,
                mode="markers",
                marker=dict(size=6, color=clr, opacity=0.9),
                name=run_name,
                hovertemplate=(
                    f"batch_size=%{{x}}<br>seq_len=%{{y}}<br>"
                    f"{z_label}=%{{z:.3f}}<extra>{run_name}</extra>"
                ),
            ))

    if not traces:
        raise ValueError(
            f"No sweep data found for metric={metric!r} in any run. "
            "Ensure you used seq_lens and batch_sizes lists in RunsProfiler."
        )

    fig = go.Figure(data=traces)
    z_label = _z_label(grid_key)
    fig.update_layout(
        title=title or f"Multi-run comparison — {metric}",
        scene=dict(
            xaxis=dict(title="batch_size"),
            yaxis=dict(title="seq_len"),
            zaxis=dict(title=z_label),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=1.2)),
        ),
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=0, r=0, b=0, t=50),
    )

    _save_or_show(fig, show=show, output=output)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 1-D BAR / LINE COMPARISON (for scalar metrics without grid)
# ══════════════════════════════════════════════════════════════════════════════


def plot_bar_compare(
    report:  Any,
    metric:  str,
    title:   str | None = None,
    show:    bool = True,
    output:  str | None = None,
):
    """Simple bar chart comparing a scalar metric across runs.

    Useful when the profiler was run without sweep lists (single bs/sl).

    Args:
        report : A :class:`MultiRunReport`.
        metric : A key from :meth:`RunProfile.to_dict` (e.g. ``"lat_mean_ms"``).
    """
    import plotly.graph_objects as go

    profiles = getattr(report, "profiles", [report])
    names, vals = [], []
    for p in profiles:
        d = p.to_dict()
        if metric in d and not _is_nan(d[metric]):
            names.append(p.run_name)
            vals.append(float(d[metric]))

    if not names:
        raise ValueError(f"Metric {metric!r} not found in any run.")

    fig = go.Figure(
        go.Bar(x=names, y=vals, marker_color=_LINE_COLORS[0])
    )
    fig.update_layout(
        title=title or f"{metric} — per run",
        xaxis_title="Run",
        yaxis_title=metric,
        margin=dict(l=40, r=20, b=80, t=50),
    )
    _save_or_show(fig, show=show, output=output)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def _z_label(key: str) -> str:
    _MAP = {
        "tps":             "tokens/s",
        "mean_ms":         "latency (ms)",
        "p50_ms":          "p50 (ms)",
        "p95_ms":          "p95 (ms)",
        "p99_ms":          "p99 (ms)",
        "perplexity":      "perplexity",
        "mean_entropy":    "entropy (nats)",
        "bpb":             "bpb",
        "mean_top1_prob":  "top-1 prob",
    }
    return _MAP.get(key, key)


def _is_nan(v: Any) -> bool:
    try:
        return math.isnan(v)
    except (TypeError, ValueError):
        return False


def _save_or_show(fig, show: bool, output: str | None) -> None:
    if output:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        fig.write_html(output, include_plotlyjs="cdn")
    if show:
        fig.show()


# ══════════════════════════════════════════════════════════════════════════════
# SHARED 2-D MATPLOTLIB STYLE
# ══════════════════════════════════════════════════════════════════════════════

# Canonical paradigm order + style — override via the style dicts if needed.
_PARADIGMS = ["AR", "Discrete", "Continuous"]

_COLORS: dict[str, str] = {
    "AR":         "#2166ac",
    "Discrete":   "#d62728",
    "Continuous": "#2ca02c",
}
_LIGHT: dict[str, str] = {
    "AR":         "#9ecae1",
    "Discrete":   "#fc9272",
    "Continuous": "#a1d99b",
}
_MARKERS: dict[str, str] = {
    "AR":         "o",
    "Discrete":   "s",
    "Continuous": "^",
}
_NICE: dict[str, str] = {
    "AR":         "Autoregressive",
    "Discrete":   "Masked Diffusion (LLaDA)",
    "Continuous": "Continuous Diffusion (ELF)",
}


def _mpl_style() -> None:
    """Apply consistent rcParams for all 2-D figures."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plt.rcParams.update({
        "font.size":            9,
        "axes.titlesize":       10,
        "axes.labelsize":       9,
        "legend.fontsize":      8,
        "xtick.labelsize":      8,
        "ytick.labelsize":      8,
        "figure.dpi":           130,
        "savefig.bbox":         "tight",
        "axes.grid":            True,
        "grid.alpha":           0.22,
        "grid.linestyle":       ":",
        "axes.spines.top":      False,
        "axes.spines.right":    False,
    })


def _mpl_save_or_show(fig, show: bool, output: str | None) -> None:
    import matplotlib.pyplot as plt
    if output:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")
    if show:
        plt.show()


def _paradigm_iter(df: Any):
    """Yield (paradigm, sub-DataFrame) in canonical order."""
    import pandas as pd
    seen = set(df["paradigm"].unique())
    for par in _PARADIGMS:
        if par in seen:
            yield par, df[df["paradigm"] == par]
    for par in sorted(seen - set(_PARADIGMS)):
        yield par, df[df["paradigm"] == par]


# ══════════════════════════════════════════════════════════════════════════════
# 3-D CONVENIENCE — plot directly from a DataFrame
# ══════════════════════════════════════════════════════════════════════════════


def plot_3d_from_df(
    df:            Any,            # pandas DataFrame
    z:             str  = "tps",   # column to use as Z-axis
    paradigm_col:  str  = "paradigm",
    x_col:         str  = "batch_size",
    y_col:         str  = "seq_len",
    mode:          str  = "surface",  # "surface" | "scatter"
    title:         str | None = None,
    show:          bool = True,
    output:        str | None = None,
):
    """3-D Plotly surface / scatter directly from a sweep DataFrame.

    No ``RunProfile`` or ``MultiRunReport`` objects needed — just pass the
    DataFrame produced by your sweep loop.

    Args:
        df           : DataFrame with at least ``paradigm_col``, ``x_col``,
                       ``y_col``, and ``z`` columns.
        z            : Column to map to the Z-axis (e.g. ``"tps"``, ``"e2e_ms"``).
        paradigm_col : Column that identifies each run / paradigm.
        x_col        : Column for the X-axis (default ``"batch_size"``).
        y_col        : Column for the Y-axis (default ``"seq_len"``).
        mode         : ``"surface"`` (smooth mesh) or ``"scatter"`` (points).
        title        : Figure title.
        show         : Open in browser / notebook.
        output       : Save as HTML to this path.

    Returns:
        ``plotly.graph_objects.Figure``

    Example::

        fig = plot_3d_from_df(batch_df, z="tps")
        fig = plot_3d_from_df(batch_df, z="e2e_ms", mode="scatter")
        fig = plot_3d_from_df(steps_df, z="tps", y_col="steps")
    """
    import plotly.graph_objects as go
    import numpy as np

    traces = []
    paradigms = [p for p in _PARADIGMS if p in df[paradigm_col].unique()]
    paradigms += [p for p in sorted(df[paradigm_col].unique()) if p not in paradigms]

    for i, par in enumerate(paradigms):
        sub = df[df[paradigm_col] == par]
        xs  = sorted(sub[x_col].unique())
        ys  = sorted(sub[y_col].unique())
        color_i = i % len(_COLORSCALES)

        if mode == "surface" and len(xs) > 1 and len(ys) > 1:
            lookup = {
                (row[x_col], row[y_col]): row[z]
                for _, row in sub.iterrows()
            }
            Z = np.array(
                [[lookup.get((x, y), float("nan")) for y in ys] for x in xs],
                dtype=float,
            )
            traces.append(go.Surface(
                x=np.array(xs), y=np.array(ys), z=Z,
                colorscale=_COLORSCALES[color_i],
                showscale=False, opacity=0.85,
                name=par,
                hovertemplate=(
                    f"{x_col}=%{{x}}<br>{y_col}=%{{y}}<br>"
                    f"{z}=%{{z:.2f}}<extra>{par}</extra>"
                ),
            ))
        else:
            clr = _LINE_COLORS[i % len(_LINE_COLORS)]
            traces.append(go.Scatter3d(
                x=sub[x_col].tolist(),
                y=sub[y_col].tolist(),
                z=sub[z].tolist(),
                mode="markers",
                marker=dict(size=6, color=clr, opacity=0.9),
                name=par,
                hovertemplate=(
                    f"{x_col}=%{{x}}<br>{y_col}=%{{y}}<br>"
                    f"{z}=%{{z:.2f}}<extra>{par}</extra>"
                ),
            ))

    if not traces:
        raise ValueError(f"No data found in DataFrame for paradigms {paradigms!r}.")

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title or f"{z}  ·  {x_col} × {y_col}",
        scene=dict(
            xaxis=dict(title=x_col),
            yaxis=dict(title=y_col),
            zaxis=dict(title=z),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=1.2)),
        ),
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=0, r=0, b=0, t=50),
    )
    _save_or_show(fig, show=show, output=output)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 2-D MATPLOTLIB — SCALE SWEEP (Fig 1)
# ══════════════════════════════════════════════════════════════════════════════


def plot_scale(
    df:     Any,             # scale_df: paradigm, params_m, tok_s
    title:  str | None = None,
    show:   bool = True,
    output: str | None = None,
):
    """Fig 1 — throughput vs model size + speedup ratio vs AR.

    Args:
        df     : DataFrame with columns ``paradigm``, ``params_m``, ``tok_s``.
        title  : Figure suptitle.
        show   : Call ``plt.show()``.
        output : Save to this path (PNG/PDF).

    Returns:
        ``(fig, axes)``  — matplotlib figure and 2-element axes array.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    _mpl_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

    # (a) absolute throughput
    ax = axes[0]
    for par, sub in _paradigm_iter(df):
        sub = sub.sort_values("params_m")
        ax.plot(sub["params_m"], sub["tok_s"] / 1e3,
                color=_COLORS.get(par, "#555"),
                marker=_MARKERS.get(par, "o"),
                lw=2.0, ms=6.5, label=_NICE.get(par, par))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Model parameters (M)")
    ax.set_ylabel("Throughput (k tok/s)")
    ax.set_title("(a) Scale: throughput vs model size", fontweight="bold")
    ax.legend(fontsize=7.5)

    # (b) speedup ratio vs AR
    ax2 = axes[1]
    ar_df = df[df["paradigm"] == "AR"]
    if not ar_df.empty:
        ar_idx = ar_df.set_index("params_m")["tok_s"]
        for par, sub in _paradigm_iter(df):
            if par == "AR":
                continue
            sub = sub.sort_values("params_m")
            xs, ys = [], []
            for _, row in sub.iterrows():
                near = ar_idx.index[np.argmin(np.abs(ar_idx.index - row["params_m"]))]
                if ar_idx[near] > 0:
                    xs.append(row["params_m"])
                    ys.append(row["tok_s"] / ar_idx[near])
            ax2.plot(xs, ys, color=_COLORS.get(par, "#555"),
                     marker=_MARKERS.get(par, "o"), lw=2.0, ms=6.5,
                     label=_NICE.get(par, par))
        ax2.axhline(1.0, color=_COLORS["AR"], lw=1.5, ls="--",
                    label="AR baseline (1×)", alpha=0.7)
    ax2.set_xscale("log")
    ax2.set_xlabel("Model parameters (M)")
    ax2.set_ylabel("Speedup ratio vs AR")
    ax2.set_title("(b) Speedup ratio over AR", fontweight="bold")
    ax2.legend(fontsize=7.5, loc="lower right")

    fig.suptitle(
        title or "Scale sweep: diffusion is consistently faster than AR",
        fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    _mpl_save_or_show(fig, show=show, output=output)
    return fig, axes


# ══════════════════════════════════════════════════════════════════════════════
# 2-D MATPLOTLIB — BATCH-SIZE SWEEP (Fig 2)
# ══════════════════════════════════════════════════════════════════════════════


def plot_batch(
    df:       Any,
    seq_lens: list | None = None,  # which seq_lens to show; default: [first, last]
    title:    str | None = None,
    show:     bool = True,
    output:   str | None = None,
):
    """Fig 2 — throughput vs batch_size at selected seq_len values.

    Args:
        df        : DataFrame with columns ``paradigm``, ``batch_size``,
                    ``seq_len``, ``tps``.
        seq_lens  : List of seq_len values to plot (two panels).  Defaults to
                    the smallest and largest available.
        title     : Figure suptitle.
        show      : Call ``plt.show()``.
        output    : Save path.

    Returns:
        ``(fig, axes)``
    """
    import matplotlib.pyplot as plt

    _mpl_style()
    all_sls = sorted(df["seq_len"].unique())
    if seq_lens is None:
        seq_lens = [all_sls[0], all_sls[-1]] if len(all_sls) >= 2 else all_sls[:1]
    n = len(seq_lens)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.0))
    if n == 1:
        axes = [axes]

    for ax_i, sl in enumerate(seq_lens):
        ax = axes[ax_i]
        sub = df[df["seq_len"] == sl]
        for par, d in _paradigm_iter(sub):
            d = d.sort_values("batch_size")
            ax.plot(d["batch_size"], d["tps"] / 1e3,
                    color=_COLORS.get(par, "#555"),
                    marker=_MARKERS.get(par, "o"),
                    lw=2.0, ms=6.0, label=_NICE.get(par, par))
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Throughput (k tok/s)")
        ax.set_title(f"({'ab'[ax_i]}) seq_len={sl}", fontweight="bold")
        if ax_i == 0:
            ax.legend(fontsize=7.5)

    fig.suptitle(
        title or "Batch-size sweep: throughput vs concurrent sequences",
        fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    _mpl_save_or_show(fig, show=show, output=output)
    return fig, axes


# ══════════════════════════════════════════════════════════════════════════════
# 2-D MATPLOTLIB — STEPS SWEEP (Fig 3)
# ══════════════════════════════════════════════════════════════════════════════


def plot_steps(
    df:     Any,
    slo_ms: float = 200.0,
    title:  str | None = None,
    show:   bool = True,
    output: str | None = None,
):
    """Fig 3 — throughput + latency vs number of diffusion steps.

    Args:
        df     : DataFrame with columns ``paradigm``, ``steps``, ``tps``,
                 ``e2e_ms``.
        slo_ms : Latency SLO guide line (default 200 ms).
        title  : Figure suptitle.
        show   : Call ``plt.show()``.
        output : Save path.

    Returns:
        ``(fig, axes)``
    """
    import matplotlib.pyplot as plt

    _mpl_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

    for ax_i, (ycol, ylabel) in enumerate([
        ("tps",    "Throughput (k tok/s)"),
        ("e2e_ms", "End-to-end latency (ms)"),
    ]):
        ax = axes[ax_i]
        for par, d in _paradigm_iter(df):
            d = d.sort_values("steps")
            ls = "--" if par == "AR" else "-"
            y  = d[ycol] / 1e3 if ycol == "tps" else d[ycol]
            ax.plot(d["steps"], y,
                    color=_COLORS.get(par, "#555"),
                    marker=_MARKERS.get(par, "o"),
                    ls=ls, lw=2.0, ms=6.0, label=_NICE.get(par, par))
        if ycol == "e2e_ms":
            ax.axhline(slo_ms, color="#888888", lw=1.0, ls=":", label=f"{slo_ms:.0f} ms SLO")
        ax.set_xlabel("Diffusion steps")
        ax.set_ylabel(ylabel)
        ax.set_title(f"({'ab'[ax_i]}) {'Throughput' if ax_i == 0 else 'Latency'} vs steps",
                     fontweight="bold")
        ax.legend(fontsize=7.5)

    fig.suptitle(
        title or "Steps sweep: more steps = higher quality, lower throughput",
        fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    _mpl_save_or_show(fig, show=show, output=output)
    return fig, axes


# ══════════════════════════════════════════════════════════════════════════════
# 2-D MATPLOTLIB — DTYPE COMPARISON (Fig 4)
# ══════════════════════════════════════════════════════════════════════════════


def plot_dtype(
    df:     Any,
    dtypes: list[str] | None = None,  # default: ["fp32", "bf16"]
    title:  str | None = None,
    show:   bool = True,
    output: str | None = None,
):
    """Fig 4 — fp32 vs bf16 throughput bar chart.

    Args:
        df     : DataFrame with columns ``paradigm``, ``dtype``, ``tps``.
        dtypes : Which dtype values to show (default ``["fp32", "bf16"]``).
        title  : Figure title.
        show   : Call ``plt.show()``.
        output : Save path.

    Returns:
        ``(fig, ax)``
    """
    import matplotlib.pyplot as plt
    import numpy as np

    _mpl_style()
    if dtypes is None:
        dtypes = sorted(df["dtype"].unique(), key=lambda d: d != "fp32")

    paradigms = [p for p in _PARADIGMS if p in df["paradigm"].unique()]
    paradigms += [p for p in sorted(df["paradigm"].unique()) if p not in paradigms]
    x = np.arange(len(paradigms))
    W = 0.8 / max(len(dtypes), 1)

    BAR_COLORS = ["#4393c3", "#f4a582", "#74c476", "#fdbb84"]

    fig, ax = plt.subplots(figsize=(7, 4.0))
    for i, dtype in enumerate(dtypes):
        vals = []
        for par in paradigms:
            row = df[(df["paradigm"] == par) & (df["dtype"] == dtype)]
            vals.append(float(row["tps"].iloc[0]) / 1e3 if not row.empty else 0.0)
        offset = (i - (len(dtypes) - 1) / 2) * W
        bars = ax.bar(x + offset, vals, W,
                      color=BAR_COLORS[i % len(BAR_COLORS)],
                      label=dtype, zorder=3)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.03,
                        f"{v:.1f}k", ha="center", va="bottom", fontsize=7.5)

    # speedup annotation between first two dtypes
    if len(dtypes) >= 2:
        base_dt, fast_dt = dtypes[0], dtypes[1]
        for i, par in enumerate(paradigms):
            base = df[(df["paradigm"] == par) & (df["dtype"] == base_dt)]["tps"]
            fast = df[(df["paradigm"] == par) & (df["dtype"] == fast_dt)]["tps"]
            if not base.empty and not fast.empty and float(base.iloc[0]) > 0:
                ratio = float(fast.iloc[0]) / float(base.iloc[0])
                peak = max(float(base.iloc[0]), float(fast.iloc[0])) / 1e3
                ax.text(i, peak + 0.35, f"{ratio:.1f}×",
                        ha="center", fontsize=8, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels([_NICE.get(p, p) for p in paradigms], rotation=12)
    ax.set_ylabel("Throughput (k tok/s)")
    ax.legend(fontsize=8)
    ax.set_title(title or "dtype comparison: fp32 vs bf16 throughput", fontweight="bold")
    plt.tight_layout()
    _mpl_save_or_show(fig, show=show, output=output)
    return fig, ax


# ══════════════════════════════════════════════════════════════════════════════
# 2-D MATPLOTLIB — COMPOSITE 2×2 PANEL (Fig 5)
# ══════════════════════════════════════════════════════════════════════════════


def plot_composite(
    scale_df:  Any,
    batch_df:  Any,
    steps_df:  Any,
    slo_ms:    float = 200.0,
    title:     str | None = None,
    show:      bool = True,
    output:    str | None = None,
):
    """Fig 5 — 2×2 composite panel.

    Panels:
        (a) MFU % vs model size  (requires ``mfu_pct`` in scale_df)
        (b) Step latency vs model size  (requires ``step_ms`` in scale_df)
        (c) Serving Pareto: latency vs throughput (from batch_df)
        (d) Quality-speed knob: throughput vs diffusion steps (from steps_df)

    Args:
        scale_df : As produced by the scale sweep (needs ``mfu_pct``, ``step_ms``).
        batch_df : As produced by the batch sweep.
        steps_df : As produced by the steps sweep.
        slo_ms   : Latency SLO guide line for the Pareto panel.
        title    : Figure suptitle.
        show     : Call ``plt.show()``.
        output   : Save path.

    Returns:
        ``(fig, axes)``  — 2×2 axes array.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    _mpl_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # (a) MFU %
    ax = axes[0, 0]
    if "mfu_pct" in scale_df.columns:
        for par, sub in _paradigm_iter(scale_df):
            sub = sub.sort_values("params_m")
            ax.plot(sub["params_m"], sub["mfu_pct"],
                    color=_COLORS.get(par, "#555"), marker=_MARKERS.get(par, "o"),
                    lw=2.0, ms=5.5, label=_NICE.get(par, par))
        ax.set_xscale("log")
        ax.set_xlabel("Parameters (M)"); ax.set_ylabel("MFU %")
        ax.legend(fontsize=7)
    ax.set_title("(a) Hardware utilisation (MFU %)", fontweight="bold")

    # (b) Step latency
    ax2 = axes[0, 1]
    if "step_ms" in scale_df.columns:
        for par, sub in _paradigm_iter(scale_df):
            sub = sub.sort_values("params_m")
            ax2.plot(sub["params_m"], sub["step_ms"],
                     color=_COLORS.get(par, "#555"), marker=_MARKERS.get(par, "o"),
                     lw=2.0, ms=5.5, label=_NICE.get(par, par))
        ax2.set_xscale("log")
        ax2.set_xlabel("Parameters (M)"); ax2.set_ylabel("Single-step latency (ms)")
        ax2.legend(fontsize=7)
    ax2.set_title("(b) Step latency vs model size", fontweight="bold")

    # (c) Pareto frontier (latency × throughput from batch sweep)
    ax3 = axes[1, 0]
    if not batch_df.empty and "e2e_ms" in batch_df.columns:
        all_sls = sorted(batch_df["seq_len"].unique())
        mid_sl  = all_sls[len(all_sls) // 2]
        all_bs  = sorted(batch_df["batch_size"].unique())
        sub_all = batch_df[batch_df["seq_len"] == mid_sl]
        for par, d in _paradigm_iter(sub_all):
            d = d.sort_values("batch_size")
            ax3.plot(d["e2e_ms"], d["tps"] / 1e3,
                     color=_COLORS.get(par, "#555"), marker=_MARKERS.get(par, "o"),
                     lw=2.0, ms=5.5, label=_NICE.get(par, par))
            for _, row in d.iterrows():
                if int(row["batch_size"]) in (all_bs[0], all_bs[-1]):
                    ax3.annotate(f"B={int(row['batch_size'])}",
                                 (row["e2e_ms"], row["tps"] / 1e3),
                                 textcoords="offset points", xytext=(5, 4), fontsize=6.5,
                                 color=_COLORS.get(par, "#555"))
        ax3.axvline(slo_ms, color="#888888", lw=1.0, ls=":", label=f"{slo_ms:.0f} ms SLO")
        ax3.set_xscale("log"); ax3.set_yscale("log")
        ax3.set_xlabel("Per-request latency (ms)"); ax3.set_ylabel("Throughput (k tok/s)")
        ax3.legend(fontsize=7)
    ax3.set_title("(c) Serving Pareto frontier", fontweight="bold")

    # (d) Steps sweep scatter
    ax4 = axes[1, 1]
    if not steps_df.empty:
        ar_tps = steps_df[steps_df["paradigm"] == "AR"]["tps"].mean()
        for par, d in _paradigm_iter(steps_df):
            if par == "AR":
                continue
            d = d.sort_values("steps")
            ax4.scatter(d["steps"], d["tps"] / 1e3, s=80,
                        c=_COLORS.get(par, "#555"), marker=_MARKERS.get(par, "o"),
                        label=_NICE.get(par, par), zorder=3)
            ax4.plot(d["steps"], d["tps"] / 1e3, color=_COLORS.get(par, "#555"),
                     lw=1.5, alpha=0.5)
        if not math.isnan(ar_tps):
            ax4.axhline(ar_tps / 1e3, color=_COLORS["AR"], lw=1.5, ls="--", label="AR baseline")
        ax4.set_xlabel("Diffusion steps"); ax4.set_ylabel("Throughput (k tok/s)")
        ax4.legend(fontsize=7)
    ax4.set_title("(d) Quality-speed knob: diffusion steps", fontweight="bold")

    fig.suptitle(
        title or "DantinoX Paradigm Benchmark — Composite panel",
        fontsize=11, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    _mpl_save_or_show(fig, show=show, output=output)
    return fig, axes
