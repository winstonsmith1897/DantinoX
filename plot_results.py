"""
Comparative plots from benchmark_results.csv.

Produces four figures saved to plots/:
  1. core_metrics.png      — grouped bar charts (throughput, cache, prefill, val-loss)
  2. seq_scaling.png       — throughput vs sequence length per family
  3. batch_scaling.png     — throughput vs batch size + efficiency (tps / batch_size)
  4. tradeoffs.png         — quality-efficiency scatter plots
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #
IN_CSV   = "benchmark_results.csv"
OUT_DIR  = "plots"
DPI      = 150

TYPE_COLORS  = {"MLA": "#4C9BE8", "GQA": "#E87B4C", "MHA": "#4CE87B"}
TYPE_ORDER   = ["MLA", "GQA", "MHA"]
MOE_MARKERS  = {False: "o", True: "^"}   # Dense vs MoE in scatter plots
MOE_LSTYLE   = {False: "-",  True: "--"} # Dense vs MoE in line plots

SEQ_LENS  = [64, 128, 256, 512]
BATCH_SIZES = [1, 4, 16, 64, 128, 256]

# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _family(row) -> str:
    moe = "MoE" if row["moe"] else "Dense"
    return f"D{int(row['dim'])}_L{int(row['num_blocks'])}_{moe}"


def _add_family(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["family"] = df.apply(_family, axis=1)
    return df


def _types_in(df: pd.DataFrame):
    return [t for t in TYPE_ORDER if t in df["type"].unique()]


def _grouped_bar(ax, df, col, ylabel, title, *, fmt=".1f", log=False):
    """Grouped bar chart: one cluster per family, one bar per attention type."""
    families = sorted(df["family"].unique())
    types    = _types_in(df)
    n_f, n_t = len(families), len(types)
    width    = 0.75 / n_t
    x        = np.arange(n_f)

    for ti, t in enumerate(types):
        sub   = df[df["type"] == t].groupby("family")[col].agg(["mean", "std"])
        means = [sub.loc[f, "mean"] if f in sub.index else np.nan for f in families]
        stds  = [sub.loc[f, "std"]  if f in sub.index else 0.0   for f in families]
        stds  = [0.0 if np.isnan(s) else s for s in stds]
        off   = (ti - n_t / 2 + 0.5) * width
        bars  = ax.bar(x + off, means, width, label=t,
                       color=TYPE_COLORS[t], edgecolor="black",
                       linewidth=0.5, zorder=3)
        ax.errorbar(x + off, means, yerr=stds,
                    fmt="none", color="black", capsize=3, linewidth=1, zorder=4)
        for bar, m in zip(bars, means):
            if not np.isnan(m):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.02,
                        f"{m:{fmt}}", ha="center", va="bottom",
                        fontsize=6.5, rotation=40)

    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontweight="bold", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    if log:
        ax.set_yscale("log")


# --------------------------------------------------------------------------- #
# Figure 1 — Core metrics (grouped bar charts)                                 #
# --------------------------------------------------------------------------- #

def fig_core_metrics(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Core Metrics — MLA vs GQA vs MHA (per architecture family)",
                 fontsize=13, fontweight="bold")
    fig.subplots_adjust(hspace=0.55, wspace=0.35)

    _grouped_bar(axes[0, 0], df, "tps_512",
                 "Tokens / sec", "Decode Throughput  (seq=512)")
    _grouped_bar(axes[0, 1], df, "theoretical_cache_mb",
                 "MB", "KV Cache Size  (theoretical)")
    _grouped_bar(axes[1, 0], df, "prefill_ms",
                 "ms", "Prefill Latency  (seq=512)", fmt=".0f")
    _grouped_bar(axes[1, 1], df, "val_loss",
                 "NLL", "Final Validation Loss", fmt=".3f")

    _save(fig, "core_metrics.png")


# --------------------------------------------------------------------------- #
# Figure 2 — Throughput vs sequence length                                     #
# --------------------------------------------------------------------------- #

def fig_seq_scaling(df: pd.DataFrame):
    families = sorted(df["family"].unique())
    n        = len(families)
    ncols    = 3
    nrows    = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(6 * ncols, 4.5 * nrows),
                              squeeze=False)
    fig.suptitle("Throughput vs Sequence Length  (single-sample decode)",
                 fontsize=13, fontweight="bold")
    fig.subplots_adjust(hspace=0.5, wspace=0.35)

    for idx, fam in enumerate(families):
        ax  = axes[idx // ncols][idx % ncols]
        sub = df[df["family"] == fam]
        for t in _types_in(sub):
            rows = sub[sub["type"] == t]
            ys   = [rows[f"tps_{s}"].dropna().mean() for s in SEQ_LENS]
            stds = [rows[f"tps_{s}"].dropna().std()  for s in SEQ_LENS]
            stds = [0 if np.isnan(s) else s for s in stds]
            ax.plot(SEQ_LENS, ys, marker="o", linewidth=2,
                    label=t, color=TYPE_COLORS[t])
            ax.fill_between(SEQ_LENS,
                            [y - s for y, s in zip(ys, stds)],
                            [y + s for y, s in zip(ys, stds)],
                            alpha=0.15, color=TYPE_COLORS[t])
        ax.set_title(fam, fontweight="bold", fontsize=9)
        ax.set_xlabel("Sequence length", fontsize=8)
        ax.set_ylabel("Tokens / sec", fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_locator(ticker.FixedLocator(SEQ_LENS))

    # hide unused subplots
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    _save(fig, "seq_scaling.png")


# --------------------------------------------------------------------------- #
# Figure 3 — Throughput vs batch size                                          #
# --------------------------------------------------------------------------- #

def fig_batch_scaling(df: pd.DataFrame):
    families = sorted(df["family"].unique())
    n        = len(families)
    ncols    = 3
    nrows    = (n + ncols - 1) // ncols

    # Two rows of subplots per family: raw tps and per-sample efficiency
    fig = plt.figure(figsize=(6 * ncols, 8 * nrows))
    fig.suptitle("Throughput vs Batch Size", fontsize=13, fontweight="bold")
    outer = gridspec.GridSpec(nrows, ncols, figure=fig,
                              hspace=0.60, wspace=0.35)

    for idx, fam in enumerate(families):
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[idx // ncols, idx % ncols],
            hspace=0.40)
        ax_tps = fig.add_subplot(inner[0])
        ax_eff = fig.add_subplot(inner[1])

        sub = df[df["family"] == fam]
        for t in _types_in(sub):
            rows = sub[sub["type"] == t]
            ys   = [rows[f"tps_bs{b}"].dropna().mean() for b in BATCH_SIZES]
            effs = [y / b if not np.isnan(y) else np.nan
                    for y, b in zip(ys, BATCH_SIZES)]

            ax_tps.plot(BATCH_SIZES, ys, marker="o", linewidth=2,
                        label=t, color=TYPE_COLORS[t])
            ax_eff.plot(BATCH_SIZES, effs, marker="o", linewidth=1.5,
                        linestyle="--", label=t, color=TYPE_COLORS[t])

        ax_tps.set_title(fam, fontweight="bold", fontsize=9)
        ax_tps.set_ylabel("Total tok/s", fontsize=8)
        ax_tps.set_xscale("log")
        ax_tps.legend(fontsize=7)
        ax_tps.grid(alpha=0.3)

        ax_eff.set_xlabel("Batch size", fontsize=8)
        ax_eff.set_ylabel("tok/s per sample", fontsize=8)
        ax_eff.set_xscale("log")
        ax_eff.grid(alpha=0.3)

    _save(fig, "batch_scaling.png")


# --------------------------------------------------------------------------- #
# Figure 4 — Quality-efficiency tradeoffs                                      #
# --------------------------------------------------------------------------- #

def fig_tradeoffs(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Quality-Efficiency Tradeoffs", fontsize=13, fontweight="bold")
    fig.subplots_adjust(wspace=0.35)

    families = sorted(df["family"].unique())
    fam_markers = ["o", "s", "^", "D", "v", "P", "X", "h", "*",
                   "<", ">", "8"][:len(families)]
    fam_marker  = {f: fam_markers[i] for i, f in enumerate(families)}

    def _scatter(ax, xcol, ycol, xlabel, ylabel, title, logx=False):
        for t in _types_in(df):
            for fam in families:
                pts = df[(df["type"] == t) & (df["family"] == fam)]
                ax.scatter(pts[xcol], pts[ycol],
                           color=TYPE_COLORS[t],
                           marker=fam_marker[fam],
                           edgecolors="black", linewidth=0.5,
                           s=70, zorder=3,
                           label=f"{t} / {fam}")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.grid(alpha=0.3)
        if logx:
            ax.set_xscale("log")

    _scatter(axes[0], "tps_512", "val_loss",
             "Decode throughput @ seq=512  (tok/s)",
             "Validation loss (NLL)",
             "Quality vs Decode Speed")

    _scatter(axes[1], "theoretical_cache_mb", "val_loss",
             "KV Cache size  (MB, theoretical)",
             "Validation loss (NLL)",
             "Quality vs Memory Footprint")

    _scatter(axes[2], "tps_bs1", "tps_bs256",
             "Throughput @ bs=1  (tok/s)",
             "Throughput @ bs=256  (tok/s)",
             "Single-sample vs High-batch Throughput",
             logx=True)

    # Legend: type (color) + family (marker)
    type_handles = [
        Line2D([0], [0], marker="o", color=TYPE_COLORS[t],
               label=t, markeredgecolor="black", linewidth=0)
        for t in _types_in(df)
    ]
    fam_handles = [
        Line2D([0], [0], marker=fam_marker[f], color="grey",
               label=f, markeredgecolor="black", linewidth=0)
        for f in families
    ]
    axes[2].legend(handles=type_handles + fam_handles,
                   fontsize=7, ncol=2, loc="upper left")

    _save(fig, "tradeoffs.png")


# --------------------------------------------------------------------------- #
# Figure 5 — KV cache analysis                                                 #
# --------------------------------------------------------------------------- #

def fig_cache(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("KV Cache Analysis", fontsize=13, fontweight="bold")
    fig.subplots_adjust(wspace=0.35)

    # Panel 1: theoretical vs measured
    ax = axes[0]
    for t in _types_in(df):
        sub = df[df["type"] == t]
        ax.scatter(sub["theoretical_cache_mb"], sub["measured_cache_mb"],
                   color=TYPE_COLORS[t], edgecolors="black",
                   linewidth=0.5, s=70, label=t, zorder=3)
    lim = max(df["theoretical_cache_mb"].max(), df["measured_cache_mb"].max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, label="y=x")
    ax.set_xlabel("Theoretical cache (MB)", fontsize=9)
    ax.set_ylabel("Measured cache (MB)", fontsize=9)
    ax.set_title("Theoretical vs Measured KV Cache", fontweight="bold", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 2: cache per parameter (memory efficiency)
    ax = axes[1]
    df2 = df.copy()
    df2["cache_per_param"] = df2["theoretical_cache_mb"] / df2["params_m"].replace(0, np.nan)
    _grouped_bar(ax, df2, "cache_per_param",
                 "MB / M params", "Cache / Parameters Ratio")

    # Panel 3: max batch size that fits
    ax = axes[2]
    if "max_batch_survived" in df.columns and df["max_batch_survived"].notna().any():
        _grouped_bar(ax, df, "max_batch_survived",
                     "Max batch size", "Max Batch Size (before OOM)", fmt=".0f")
    else:
        # Compute from tps_bs* columns: largest bs with valid tps
        def _max_bs(row):
            for bs in reversed(BATCH_SIZES):
                v = row.get(f"tps_bs{bs}", np.nan)
                if not (isinstance(v, float) and np.isnan(v)):
                    return bs
            return np.nan
        df2["max_bs"] = df.apply(_max_bs, axis=1)
        _grouped_bar(ax, df2, "max_bs",
                     "Max batch size", "Max Batch Size (before OOM)", fmt=".0f")

    _save(fig, "cache_analysis.png")


# --------------------------------------------------------------------------- #
# Figure 6 — Summary heatmap                                                   #
# --------------------------------------------------------------------------- #

def fig_heatmap(df: pd.DataFrame):
    """
    Normalised performance heatmap: rows = (type, family),
    columns = key metrics.  Each column is normalised within
    the column (0 = worst, 1 = best) so patterns across metrics
    are easy to see.
    """
    metrics = {
        "tps_512":              ("Throughput\n@seq512",  True),   # higher better
        "tps_bs64":             ("Throughput\n@bs64",    True),
        "theoretical_cache_mb": ("Cache MB\n(theor.)",   False),  # lower better
        "prefill_ms":           ("Prefill\nlatency",     False),
        "val_loss":             ("Val Loss\n(NLL)",       False),
    }

    agg = df.groupby(["type", "family"])[list(metrics)].mean()
    norm = agg.copy()
    for col, (_, higher_better) in metrics.items():
        mn, mx = agg[col].min(), agg[col].max()
        if mx > mn:
            norm[col] = (agg[col] - mn) / (mx - mn)
            if not higher_better:
                norm[col] = 1 - norm[col]
        else:
            norm[col] = 0.5

    labels = [f"{t}\n{f}" for t, f in norm.index]
    col_labels = [v[0] for v in metrics.values()]
    data = norm.values

    fig, ax = plt.subplots(figsize=(len(metrics) * 1.8, max(6, len(labels) * 0.45)))
    fig.suptitle("Normalised Performance Heatmap  (green = best)",
                 fontsize=12, fontweight="bold")

    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)

    for i in range(len(labels)):
        for j in range(len(col_labels)):
            raw = agg.values[i, j]
            txt = f"{raw:.1f}" if not np.isnan(raw) else "—"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=7, color="black")

    # Color rows by type
    type_colors = {"MLA": "#4C9BE8", "GQA": "#E87B4C", "MHA": "#4CE87B"}
    for i, (t, _) in enumerate(agg.index):
        ax.add_patch(plt.Rectangle((-0.5 + len(col_labels), i - 0.5),
                                   0.25, 1.0,
                                   color=type_colors.get(t, "grey"),
                                   clip_on=False, transform=ax.transData))

    plt.colorbar(im, ax=ax, fraction=0.03, label="Normalised score")
    fig.tight_layout()
    _save(fig, "heatmap.png")


# --------------------------------------------------------------------------- #
# Figure 7 — Parameters vs Generation Speed                                    #
# --------------------------------------------------------------------------- #

def _trend_line(ax, x, y, color, alpha=0.35):
    """Fit and draw a log-log power-law trend per type."""
    mask = (~np.isnan(x)) & (~np.isnan(y)) & (x > 0) & (y > 0)
    if mask.sum() < 3:
        return
    lx, ly = np.log10(x[mask]), np.log10(y[mask])
    m, b = np.polyfit(lx, ly, 1)
    xs = np.linspace(x[mask].min(), x[mask].max(), 100)
    ys = 10 ** (m * np.log10(xs) + b)
    ax.plot(xs, ys, "--", color=color, linewidth=1.5, alpha=alpha)


def fig_params_speed(df: pd.DataFrame):
    """Six panels exploring model size vs generation speed tradeoffs."""
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("Model Size vs Generation Speed — MLA / GQA / MHA",
                 fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax_s1  = fig.add_subplot(gs[0, 0])   # params vs tps (bs=1)
    ax_s64 = fig.add_subplot(gs[0, 1])   # params vs tps (bs=64)
    ax_eff = fig.add_subplot(gs[0, 2])   # tok/s per M-param (efficiency)
    ax_ql  = fig.add_subplot(gs[1, 0])   # params vs val_loss
    ax_pa  = fig.add_subplot(gs[1, 1])   # pareto: speed vs quality, sized by params
    ax_pr  = fig.add_subplot(gs[1, 2])   # prefill latency vs params

    # ── helpers ────────────────────────────────────────────────────────── #
    def _sc(ax, xcol, ycol, xlabel, ylabel, title, logx=True, logy=False):
        for t in _types_in(df):
            sub = df[df["type"] == t].dropna(subset=[xcol, ycol])
            for moe_val, grp in sub.groupby("moe"):
                mk = MOE_MARKERS[moe_val]
                lbl = f"{t} ({'MoE' if moe_val else 'Dense'})"
                ax.scatter(grp[xcol], grp[ycol],
                           color=TYPE_COLORS[t], marker=mk,
                           edgecolors="black", linewidth=0.5,
                           s=70, zorder=3, label=lbl)
            _trend_line(ax,
                        sub[xcol].values.astype(float),
                        sub[ycol].values.astype(float),
                        TYPE_COLORS[t])
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.grid(alpha=0.3)
        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")

    # ── Panel 1: params vs throughput, bs=1 ────────────────────────────── #
    _sc(ax_s1, "params_m", "tps_512",
        "Parameters (M)", "tok/s  (seq=512, bs=1)",
        "Model Size vs Single-Sample Throughput")
    ax_s1.legend(fontsize=7, ncol=2)

    # ── Panel 2: params vs throughput, bs=64 ───────────────────────────── #
    _sc(ax_s64, "params_m", "tps_bs64",
        "Parameters (M)", "tok/s  (batch=64)",
        "Model Size vs Batch Throughput  (bs=64)")

    # ── Panel 3: throughput efficiency (tok/s per M-param) ─────────────── #
    df3 = df.copy()
    df3["eff_s1"]  = df3["tps_512"]   / df3["params_m"].replace(0, np.nan)
    df3["eff_b64"] = df3["tps_bs64"]  / df3["params_m"].replace(0, np.nan)
    ax_eff.set_title("Throughput Efficiency  (tok/s per M-param)",
                     fontweight="bold", fontsize=10)
    width = 0.35
    types = _types_in(df)
    x = np.arange(len(types))
    for i, (col, label, offset) in enumerate([
            ("eff_s1",  "bs=1",  -width/2),
            ("eff_b64", "bs=64", +width/2)]):
        means = [df3[df3["type"] == t][col].mean() for t in types]
        stds  = [df3[df3["type"] == t][col].std()  for t in types]
        bars = ax_eff.bar(x + offset, means, width,
                          label=label,
                          color=[TYPE_COLORS[t] for t in types],
                          edgecolor="black", linewidth=0.5,
                          alpha=0.6 + 0.4 * i, zorder=3)
        ax_eff.errorbar(x + offset, means, yerr=[s if not np.isnan(s) else 0 for s in stds],
                        fmt="none", color="black", capsize=3, linewidth=1, zorder=4)
        for bar, m in zip(bars, means):
            if not np.isnan(m):
                ax_eff.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() * 1.03,
                            f"{m:.2f}", ha="center", va="bottom", fontsize=8)
    ax_eff.set_xticks(x)
    ax_eff.set_xticklabels(types)
    ax_eff.set_ylabel("tok/s / M-param", fontsize=9)
    ax_eff.legend(fontsize=8)
    ax_eff.grid(axis="y", alpha=0.3, zorder=0)

    # ── Panel 4: params vs val_loss ─────────────────────────────────────── #
    _sc(ax_ql, "params_m", "val_loss",
        "Parameters (M)", "Validation Loss (NLL)",
        "Model Size vs Quality")
    # annotate: lower-right = big+good; upper-left = small+bad
    ax_ql.legend(fontsize=7, ncol=2)

    # ── Panel 5: Pareto — speed vs quality, bubble size = params ────────── #
    ax_pa.set_title("Quality–Speed Pareto  (bubble ∝ params)",
                    fontweight="bold", fontsize=10)
    for t in _types_in(df):
        sub = df[df["type"] == t].dropna(subset=["tps_512", "val_loss", "params_m"])
        for moe_val, grp in sub.groupby("moe"):
            sz  = (grp["params_m"] / df["params_m"].max() * 400).clip(lower=20)
            mk  = MOE_MARKERS[moe_val]
            ax_pa.scatter(grp["tps_512"], grp["val_loss"],
                          s=sz, color=TYPE_COLORS[t], marker=mk,
                          edgecolors="black", linewidth=0.5,
                          alpha=0.80, zorder=3,
                          label=f"{t} ({'MoE' if moe_val else 'Dense'})")
    ax_pa.set_xlabel("Decode throughput  (tok/s, bs=1, seq=512)", fontsize=9)
    ax_pa.set_ylabel("Validation Loss (NLL)  ← better", fontsize=9)
    ax_pa.grid(alpha=0.3)
    ax_pa.legend(fontsize=7, ncol=2)
    # Pareto frontier annotation
    _draw_pareto(ax_pa, df)

    # ── Panel 6: params vs prefill latency ──────────────────────────────── #
    _sc(ax_pr, "params_m", "prefill_ms",
        "Parameters (M)", "Prefill latency  (ms, seq=512)",
        "Model Size vs Prefill Latency")

    # ── shared legend for Dense/MoE markers ────────────────────────────── #
    from matplotlib.lines import Line2D
    extra = [
        Line2D([0],[0], marker="o", color="grey", label="Dense",
               markeredgecolor="black", linewidth=0, markersize=8),
        Line2D([0],[0], marker="^", color="grey", label="MoE",
               markeredgecolor="black", linewidth=0, markersize=8),
        Line2D([0],[0], linestyle="--", color="grey", label="trend (log-log)",
               linewidth=1.5, alpha=0.5),
    ]
    fig.legend(handles=extra, loc="lower center", ncol=3,
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.01))

    _save(fig, "params_vs_speed.png")


def _draw_pareto(ax, df: pd.DataFrame):
    """Highlight the Pareto-optimal front (fastest + best quality)."""
    pts = df[["tps_512", "val_loss"]].dropna()
    if pts.empty:
        return
    # dominance: higher tps AND lower val_loss
    xs, ys = pts["tps_512"].values, pts["val_loss"].values
    pareto = []
    for i in range(len(xs)):
        dominated = any(
            xs[j] >= xs[i] and ys[j] <= ys[i] and (xs[j] > xs[i] or ys[j] < ys[i])
            for j in range(len(xs)) if j != i
        )
        if not dominated:
            pareto.append((xs[i], ys[i]))
    if pareto:
        pareto.sort()
        px, py = zip(*pareto)
        ax.step(px, py, where="post", color="red", linewidth=1.5,
                linestyle=":", label="Pareto front", zorder=5)
        ax.legend(fontsize=7, ncol=2)


# --------------------------------------------------------------------------- #
# I/O                                                                          #
# --------------------------------------------------------------------------- #

def _save(fig, name: str):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    df = pd.read_csv(IN_CSV)
    df = _add_family(df)

    # Coerce numeric columns
    num_cols = (["tps_512", "tps_bs1", "tps_bs4", "tps_bs16",
                 "tps_bs64", "tps_bs128", "tps_bs256",
                 "theoretical_cache_mb", "measured_cache_mb",
                 "prefill_ms", "val_loss", "params_m",
                 "max_batch_survived"]
                + [f"tps_{s}" for s in SEQ_LENS])
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    print(f"Loaded {len(df)} runs  ({df['type'].value_counts().to_dict()})")
    print(f"Families: {sorted(df['family'].unique())}")
    print()

    print("Generating figures...")
    fig_core_metrics(df)
    fig_seq_scaling(df)
    fig_batch_scaling(df)
    fig_tradeoffs(df)
    fig_cache(df)
    fig_heatmap(df)
    fig_params_speed(df)
    print("Done.")
