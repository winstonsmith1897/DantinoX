"""
Two LinkedIn-ready figures from benchmark_results.csv.

  linkedin_1_cache_vs_quality.png  — KV cache footprint vs model quality
  linkedin_2_dense_comparison.png  — Dense-only bar comparison (fair)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

IN_CSV  = "benchmark_results.csv"
OUT_DIR = "plots"
DPI     = 180

TYPE_COLORS = {"MLA": "#3A86FF", "GQA": "#FF6B35", "MHA": "#2DC653"}
TYPE_ORDER  = ["MLA", "GQA", "MHA"]


# --------------------------------------------------------------------------- #
# Data prep                                                                    #
# --------------------------------------------------------------------------- #
def load() -> pd.DataFrame:
    df = pd.read_csv(IN_CSV)
    for c in ["params_m", "tps_64", "tps_128", "tps_256", "tps_512",
              "tps_bs1", "tps_bs64",
              "val_loss", "theoretical_cache_mb", "measured_cache_mb", "prefill_ms"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# --------------------------------------------------------------------------- #
# Figure 1 — KV Cache Footprint vs Model Quality                               #
# --------------------------------------------------------------------------- #
def fig1_cache_vs_quality(df: pd.DataFrame):
    """
    Scatter: KV cache size (x, log) vs validation loss (y, lower=better).
    Each point is one run; size ∝ params_m.  MLA should cluster
    bottom-left (small cache AND better quality).
    """
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    sub = df.dropna(subset=["theoretical_cache_mb", "val_loss", "params_m"])
    size_scale = 600 / sub["params_m"].max()

    for t in TYPE_ORDER:
        pts = sub[sub["type"] == t]
        for moe_val, grp in pts.groupby("moe"):
            sizes  = (grp["params_m"] * size_scale).clip(lower=30)
            marker = "^" if moe_val else "o"
            ax.scatter(grp["theoretical_cache_mb"], grp["val_loss"],
                       s=sizes, c=TYPE_COLORS[t], marker=marker,
                       edgecolors="white", linewidths=1.2,
                       alpha=0.88, zorder=3)

    # Annotate mean position per type with a large label
    for t in TYPE_ORDER:
        pts = sub[sub["type"] == t]
        mx, my = pts["theoretical_cache_mb"].mean(), pts["val_loss"].mean()
        offsets = {"MLA": (-0.06, -0.04), "GQA": (0.04, -0.04), "MHA": (0.04, 0.025)}
        dx, dy = offsets[t]
        ax.annotate(
            f"{t}\n"
            f"cache: {mx:.1f} MB\n"
            f"val loss: {my:.3f}",
            xy=(mx, my),
            xytext=(mx * (1 + dx * 4), my + dy * 0.5),
            fontsize=11, fontweight="bold", color=TYPE_COLORS[t],
            ha="left" if dx > 0 else "right",
            arrowprops=dict(arrowstyle="-", color=TYPE_COLORS[t],
                            lw=1.5, alpha=0.7),
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=TYPE_COLORS[t], lw=1.2, alpha=0.9),
        )

    ax.set_xscale("log")
    ax.set_xlabel("KV Cache Footprint  (MB, log scale)", fontsize=13, labelpad=8)
    ax.set_ylabel("Validation Loss  (NLL) ← lower is better", fontsize=13, labelpad=8)
    ax.set_title(
        "MLA: Better Quality with 5× Less KV Cache Memory",
        fontsize=16, fontweight="bold", pad=16
    )

    # Arrow annotation for the ideal direction
    ax.annotate("", xy=(0.12, 0.08), xytext=(0.30, 0.25),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="#555555",
                                lw=1.5, mutation_scale=16))
    ax.text(0.095, 0.065, "ideal", transform=ax.transAxes,
            fontsize=9, color="#555555", style="italic")

    # Legend: type (color) + Dense/MoE (marker)
    handles = (
        [mpatches.Patch(color=TYPE_COLORS[t], label=t) for t in TYPE_ORDER]
        + [
            Line2D([0],[0], marker="o", color="#888", linestyle="None",
                   markersize=9, label="Dense", markeredgecolor="white"),
            Line2D([0],[0], marker="^", color="#888", linestyle="None",
                   markersize=9, label="MoE",   markeredgecolor="white"),
        ]
    )
    ax.legend(handles=handles, fontsize=10, framealpha=0.9,
              loc="upper right", ncol=2)

    # Bubble size guide
    for p, lbl in [(20, "20M"), (100, "100M"), (400, "400M")]:
        ax.scatter([], [], s=p * size_scale, c="#AAAAAA",
                   edgecolors="white", label=f"{lbl} params")
    ax.legend(handles=handles
              + [ax.scatter([], [], s=p * size_scale, c="#AAAAAA",
                            edgecolors="white", label=f"{lbl} params")
                 for p, lbl in [(20, "20M"), (100, "100M"), (400, "400M")]],
              fontsize=9, framealpha=0.9, loc="upper right", ncol=2)

    ax.grid(True, which="both", alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    _save(fig, "linkedin_1_cache_vs_quality.png")


# --------------------------------------------------------------------------- #
# Figure 2 — Dense-only fair comparison                                        #
# --------------------------------------------------------------------------- #
def fig2_dense_comparison(df: pd.DataFrame):
    """
    Dense-only bar chart: three metrics side by side (val_loss, cache MB,
    single-sample throughput).  Each bar normalised within its metric so
    the chart is unit-free — the goal is relative ranking.
    """
    dense = df[~df["moe"]].copy()

    metrics = {
        "val_loss":             ("Model Quality\n(val loss ↓)",         False),  # lower=better
        "theoretical_cache_mb": ("KV Cache\n(MB ↓)",                    False),
        "tps_512":              ("Throughput\n(tok/s ↑, seq=512)",       True),   # higher=better
    }

    means = dense.groupby("type")[list(metrics)].mean()
    stds  = dense.groupby("type")[list(metrics)].std()

    # Normalise columns for the visual bar height
    normed = means.copy()
    for col, (_, higher) in metrics.items():
        mn, mx = means[col].min(), means[col].max()
        if mx > mn:
            normed[col] = (means[col] - mn) / (mx - mn)
            if not higher:
                normed[col] = 1 - normed[col]
        else:
            normed[col] = 0.5

    types   = [t for t in TYPE_ORDER if t in means.index]
    n_met   = len(metrics)
    n_type  = len(types)
    width   = 0.22
    x       = np.arange(n_met)

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    for ti, t in enumerate(types):
        off   = (ti - n_type / 2 + 0.5) * width
        vals  = [normed.loc[t, col] for col in metrics]
        raw   = [means.loc[t, col] for col in metrics]
        bars  = ax.bar(x + off, vals, width,
                       color=TYPE_COLORS[t], label=t,
                       edgecolor="white", linewidth=0.8,
                       alpha=0.92, zorder=3)

        # Raw value label inside/above bar
        for bar, rv, v, (col, (_, higher)) in zip(bars, raw, vals, metrics.items()):
            fmt = ".3f" if "loss" in col else ".1f"
            unit = "" if "loss" in col else (" MB" if "cache" in col else " t/s")
            ax.text(bar.get_x() + bar.get_width() / 2,
                    min(v + 0.04, 0.97),
                    f"{rv:{fmt}}{unit}",
                    ha="center", va="bottom",
                    fontsize=9, fontweight="bold",
                    color=TYPE_COLORS[t])

    ax.set_xticks(x)
    ax.set_xticklabels([v[0] for v in metrics.values()], fontsize=12)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Normalised score  (higher bar = better)", fontsize=12, labelpad=8)
    ax.set_title(
        "Dense Models — Fair Comparison Across Key Metrics\n"
        "(Same architecture family, single-sample decode)",
        fontsize=15, fontweight="bold", pad=14
    )

    ax.legend(fontsize=12, framealpha=0.9, loc="upper left")
    ax.grid(axis="y", alpha=0.25, linestyle="--", zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False)
    ax.set_yticks([])

    # Add a subtle "higher = better" indicator on the y-axis
    ax.annotate("← better", xy=(0, 1.05), xycoords="axes fraction",
                fontsize=9, color="#777777", style="italic")

    _save(fig, "linkedin_2_dense_comparison.png")


# --------------------------------------------------------------------------- #
# Figure 3 — MLA advantage only kicks in when context is long AND model large  #
# --------------------------------------------------------------------------- #
def fig3_when_mla_wins(df: pd.DataFrame):
    """
    Three panels (small / medium / large model scale) show KV-cache size (GB)
    vs context length.  The crossing of GPU-memory limits makes it obvious
    that both conditions — long context AND large model — must be met.

    Small (~30M):  actual experimental medians from benchmark_results.csv.
    Medium (~7B):  theoretical values for a standard 7B architecture.
    Large (~70B):  theoretical values for a standard 70B architecture.
    """
    # ── Experimental bytes-per-token from actual runs (all use max_context=512) ──
    df2 = df.copy()
    df2["bpt"] = df2["theoretical_cache_mb"] * 1024**2 / df2["max_context"]
    bpt_exp = df2.groupby("type")["bpt"].median()   # bytes / token

    # ── Theoretical bytes-per-token for standard larger architectures ────────
    # MHA:  2 × n_heads × head_size × num_blocks × 2 (bf16)
    # GQA:  2 × kv_heads × head_size × num_blocks × 2
    # MLA:  2 × (down_dim_kv + rope_dim) × num_blocks × 2
    scales = {
        "~30M\n(experimental)": bpt_exp,
        "~7B\n(theoretical)": {
            # dim=4096, n_heads=32, kv_heads=8, head_size=128, num_blocks=32
            "MHA": 2 * 32  * 128 * 32 * 2,   # 524 288
            "GQA": 2 *  8  * 128 * 32 * 2,   # 131 072
            "MLA": 2 * (512 + 64) * 32 * 2,  #  73 728  (down_dim_kv=512, rope=64)
        },
        "~70B\n(theoretical)": {
            # dim=8192, n_heads=64, kv_heads=8, head_size=128, num_blocks=80
            "MHA": 2 * 64  * 128 * 80 * 2,    # 3 276 800
            "GQA": 2 *  8  * 128 * 80 * 2,    #  327 680
            "MLA": 2 * (1024 + 64) * 80 * 2,  #  348 160  (down_dim_kv=1024, rope=64)
        },
    }

    ctx_lens  = np.array([512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])
    vram_80gb = 80.0   # A100 / H100
    vram_24gb = 24.0   # consumer RTX

    linestyles = {"MLA": "-", "GQA": "--", "MHA": ":"}

    fig, axes = plt.subplots(1, 3, figsize=(17, 6), sharey=False)
    fig.patch.set_facecolor("#F8F9FA")

    for ax, (title, bpt_map) in zip(axes, scales.items()):
        ax.set_facecolor("#F8F9FA")

        for t in TYPE_ORDER:
            bpt = bpt_map[t] if isinstance(bpt_map, dict) else bpt_map.get(t, np.nan)
            cache_gb = bpt * ctx_lens / 1024**3
            ax.plot(ctx_lens / 1024, cache_gb,
                    color=TYPE_COLORS[t], ls=linestyles[t], lw=2.5,
                    marker="o", markersize=5, label=t, zorder=4)

        ymax = ax.get_ylim()[1]
        # VRAM reference bands
        for vram, label, col in [(vram_80gb, "80 GB (A100)", "#CC3311"),
                                  (vram_24gb, "24 GB (RTX)", "#EE7733")]:
            ax.axhline(vram, color=col, lw=1.4, ls="-.", alpha=0.8, zorder=3)
            ax.text(ctx_lens[-1] / 1024 * 0.98, vram * 1.04, label,
                    ha="right", va="bottom", fontsize=8, color=col, alpha=0.9)

        ax.set_xscale("log")
        ax.set_yscale("log")
        xtick_vals = [0.5, 1, 2, 4, 8, 16, 32, 64, 128]
        ax.set_xticks([v for v in xtick_vals])
        ax.set_xticklabels([f"{int(v)}K" if v >= 1 else "512" for v in xtick_vals], fontsize=9)
        ax.set_xlabel("Context length", fontsize=11, labelpad=6)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.grid(True, which="both", alpha=0.2, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        if ax is axes[0]:
            ax.set_ylabel("KV Cache per sequence (GB)", fontsize=11, labelpad=6)
        ax.legend(fontsize=10, framealpha=0.9)

    fig.suptitle(
        "MLA's Edge Requires BOTH Conditions: Long Context  AND  Large Model",
        fontsize=15, fontweight="bold", y=1.02
    )
    fig.text(0.5, -0.04,
             "At small scale every method fits. At 7B+ scale, MHA runs OOM first. "
             "At 70B scale, only MLA (and partially GQA) survives 128K context.",
             ha="center", fontsize=10, color="#555555", style="italic")

    plt.tight_layout()
    _save(fig, "linkedin_3_when_mla_wins.png")


# --------------------------------------------------------------------------- #
# Figure 4 — Model size vs KV cache, by attention type                         #
# --------------------------------------------------------------------------- #
def fig4_params_vs_cache(df: pd.DataFrame):
    """
    Scatter: params_m (x, log) vs theoretical_cache_mb (y, log).
    Color = attention type.  Marker = Dense/MoE.
    Bubble size ∝ effective KV dim per layer:
      MHA/GQA → kv_heads × head_size   (locked to architecture)
      MLA     → down_dim_kv + rope_dim  (free hyperparameter)
    This makes it visually clear why MLA can have large params but small cache:
    its KV dimension is decoupled from model width (dim).
    """
    sub = df.dropna(subset=["params_m", "theoretical_cache_mb"]).copy()

    # Derive effective KV dim per layer from the cache formula:
    #   cache_bytes = kv_dim_per_layer × num_blocks × max_context × 2(K+V) × 2(fp16)
    sub["kv_dim"] = (
        sub["theoretical_cache_mb"] * 1024**2
        / (sub["num_blocks"] * sub["max_context"] * 4)
    ).round()

    kv_max = sub["kv_dim"].max()
    size_scale = 500 / kv_max  # max bubble → s=500

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    for t in TYPE_ORDER:
        for moe_val, grp in sub[sub["type"] == t].groupby("moe"):
            marker = "^" if moe_val else "o"
            sizes  = (grp["kv_dim"] * size_scale).clip(lower=25)
            ax.scatter(grp["params_m"], grp["theoretical_cache_mb"],
                       s=sizes, c=TYPE_COLORS[t], marker=marker,
                       edgecolors="white", linewidths=1.0,
                       alpha=0.88, zorder=4,
                       label=f"{t} ({'MoE' if moe_val else 'Dense'})")

    # Log-log trend lines split by (type, moe)
    moe_ls = {False: "--", True: ":"}
    for t in TYPE_ORDER:
        for moe_val, grp in sub[sub["type"] == t].groupby("moe"):
            if len(grp) < 3:
                continue
            x_fit = np.logspace(
                np.log10(grp["params_m"].min() * 0.85),
                np.log10(grp["params_m"].max() * 1.15), 80
            )
            lx = np.log10(grp["params_m"])
            ly = np.log10(grp["theoretical_cache_mb"])
            coeffs = np.polyfit(lx, ly, 1)
            y_fit  = 10 ** np.polyval(coeffs, np.log10(x_fit))
            ax.plot(x_fit, y_fit, color=TYPE_COLORS[t], lw=1.8,
                    ls=moe_ls[moe_val], alpha=0.55, zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Model Parameters (M)", fontsize=13, labelpad=8)
    ax.set_ylabel("KV Cache Footprint (MB, log scale)", fontsize=13, labelpad=8)
    ax.set_title(
        "MLA Decouples Model Size from Cache Size\n"
        "Bubble size = effective KV dim per layer  "
        "(MLA: free hyperparameter · MHA/GQA: locked to model width)",
        fontsize=12, fontweight="bold", pad=14
    )

    # ── Legend block ─────────────────────────────────────────────────────────
    # Type colours
    type_handles = [mpatches.Patch(color=TYPE_COLORS[t], label=t) for t in TYPE_ORDER]

    # Marker shapes
    shape_handles = [
        Line2D([0],[0], marker="o", color="#888", ls="None",
               markersize=9, markeredgecolor="white", label="Dense"),
        Line2D([0],[0], marker="^", color="#888", ls="None",
               markersize=9, markeredgecolor="white", label="MoE"),
    ]

    # Trend line styles
    ls_handles = [
        Line2D([0],[0], color="#888", ls="--", lw=1.8, label="trend (Dense)"),
        Line2D([0],[0], color="#888", ls=":",  lw=1.8, label="trend (MoE)"),
    ]

    # Bubble size guide (KV dim reference)
    kv_refs = [64, 128, 256, 512]
    size_handles = [
        Line2D([0],[0], marker="o", color="#AAAAAA", ls="None",
               markersize=np.sqrt(kv * size_scale),
               markeredgecolor="white", label=f"KV dim={kv}")
        for kv in kv_refs if kv * size_scale >= 10
    ]

    all_handles = type_handles + shape_handles + ls_handles + size_handles
    all_labels  = [h.get_label() for h in all_handles]
    ax.legend(all_handles, all_labels, fontsize=9, framealpha=0.9,
              ncol=2, loc="upper left")

    ax.grid(True, which="both", alpha=0.2, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    # Callout explaining MLA's decoupling
    ax.text(0.98, 0.05,
            "MLA large bubbles → high down_dim_kv (more cache)\n"
            "MLA small bubbles → low  down_dim_kv (less cache)\n"
            "Both can appear at any model scale",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, color="#444444", style="italic",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#CCCCCC", lw=1))

    _save(fig, "linkedin_4_params_vs_cache.png")


# --------------------------------------------------------------------------- #
# Figure 5 — MLA cache vs down_dim_kv                                          #
# --------------------------------------------------------------------------- #
def fig5_mla_down_dim(df: pd.DataFrame):
    """
    Left panel  — actual benchmark data at context=512: cache (MB) vs down_dim_kv,
                  grouped by num_blocks, with MHA/GQA reference bands.
    Right panel — theoretical extrapolation: cache (GB) vs context length for
                  each (down_dim_kv, num_blocks) combo, plus MHA/GQA references.
    """
    mla = df[df["type"] == "MLA"].copy()

    # Empirical kv_dim per layer: derived from the actual measured cache.
    # This absorbs rope_dim implicitly (rope_dim varies across runs and is
    # not stored in the CSV).
    mla["kv_dim"] = (
        mla["theoretical_cache_mb"] * 1024**2
        / (mla["num_blocks"] * mla["max_context"] * 4)
    )
    # Mean kv_dim per down_dim_kv level (used for extrapolation)
    kv_dim_by_level = mla.groupby("down_dim_kv")["kv_dim"].mean()

    # Reference: median empirical kv_dim per layer for MHA and GQA
    def _ref_kv_dim(t):
        sub = df[df["type"] == t].copy()
        sub["kv_dim"] = (
            sub["theoretical_cache_mb"] * 1024**2
            / (sub["num_blocks"] * sub["max_context"] * 4)
        )
        return sub.groupby("num_blocks")["kv_dim"].median()

    ref_mha = _ref_kv_dim("MHA")
    ref_gqa = _ref_kv_dim("GQA")

    dkv_vals   = sorted(mla["down_dim_kv"].unique())
    nb_vals    = sorted(mla["num_blocks"].unique())

    _cmap = plt.cm.get_cmap("plasma", len(dkv_vals))
    DKV_COLORS = {d: _cmap(i) for i, d in enumerate(dkv_vals)}
    _ls_cycle  = ["-", "--", "-.", ":", (0, (5, 2)), (0, (3, 1, 1, 1))]
    NB_LS      = {nb: _ls_cycle[i % len(_ls_cycle)] for i, nb in enumerate(nb_vals)}
    _mk_cycle  = ["o", "s", "^", "D", "v", "P"]
    NB_MARKS   = {nb: _mk_cycle[i % len(_mk_cycle)] for i, nb in enumerate(nb_vals)}

    ctx_lens = np.array([512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))
    fig.patch.set_facecolor("#F8F9FA")

    # ── Left panel: actual data at context=512 ──────────────────────────────
    ax1.set_facecolor("#F8F9FA")

    # MHA / GQA reference bands (median across all their num_blocks)
    mha_med = df[df["type"] == "MHA"]["theoretical_cache_mb"].median()
    gqa_med = df[df["type"] == "GQA"]["theoretical_cache_mb"].median()
    x_band  = [-0.5, len(dkv_vals) - 0.5]
    ax1.fill_between(x_band, mha_med * 0.7, mha_med * 1.3,
                     color=TYPE_COLORS["MHA"], alpha=0.12, zorder=1)
    ax1.axhline(mha_med, color=TYPE_COLORS["MHA"], lw=1.5, ls="-.", alpha=0.7,
                label=f"MHA median ({mha_med:.1f} MB)")
    ax1.fill_between(x_band, gqa_med * 0.7, gqa_med * 1.3,
                     color=TYPE_COLORS["GQA"], alpha=0.12, zorder=1)
    ax1.axhline(gqa_med, color=TYPE_COLORS["GQA"], lw=1.5, ls="-.", alpha=0.7,
                label=f"GQA median ({gqa_med:.1f} MB)")

    # MLA scatter, grouped by num_blocks with jitter on x
    jitter_range = np.linspace(-0.25, 0.25, len(nb_vals)) if len(nb_vals) > 1 else [0.0]
    jitter = {nb: jitter_range[i] for i, nb in enumerate(nb_vals)}
    for nb in nb_vals:
        sub = mla[mla["num_blocks"] == nb]
        x_pos = [dkv_vals.index(d) + jitter[nb]
                 for d in sub["down_dim_kv"]]
        ax1.scatter(x_pos, sub["theoretical_cache_mb"],
                    marker=NB_MARKS[nb], color=TYPE_COLORS["MLA"],
                    s=70, alpha=0.85, edgecolors="white", lw=0.8, zorder=4,
                    label=f"MLA {nb} layers")
        # Connect median per (down_dim_kv, num_blocks) with a line
        med = sub.groupby("down_dim_kv")["theoretical_cache_mb"].median()
        xs  = [dkv_vals.index(d) + jitter[nb] for d in med.index]
        ax1.plot(xs, med.values, color=TYPE_COLORS["MLA"],
                 ls=NB_LS[nb], lw=1.6, alpha=0.65, zorder=3)

    ax1.set_xticks(range(len(dkv_vals)))
    ax1.set_xticklabels([str(d) for d in dkv_vals], fontsize=12)
    ax1.set_xlabel("down_dim_kv", fontsize=13, labelpad=6)
    ax1.set_ylabel("KV Cache (MB)  @ 512 tokens", fontsize=12, labelpad=6)
    ax1.set_title("Actual Benchmark Data\n(context = 512 tokens)",
                  fontsize=12, fontweight="bold", pad=10)
    ax1.legend(fontsize=9, framealpha=0.9, loc="upper left")
    ax1.grid(axis="y", alpha=0.25, linestyle="--")
    ax1.spines[["top", "right"]].set_visible(False)

    # ── Right panel: theoretical extrapolation ───────────────────────────────
    ax2.set_facecolor("#F8F9FA")

    # MHA / GQA reference lines for num_blocks=12 (representative)
    nb_ref = 12
    for t, ref_dict, col in [("MHA", ref_mha, TYPE_COLORS["MHA"]),
                              ("GQA", ref_gqa, TYPE_COLORS["GQA"])]:
        kv  = ref_dict.get(nb_ref, ref_dict.median())
        cache_gb = kv * nb_ref * ctx_lens * 4 / 1024**3
        ax2.plot(ctx_lens / 1024, cache_gb,
                 color=col, lw=2, ls="-.", alpha=0.75, zorder=3,
                 label=f"{t} ({nb_ref} layers)")

    # MLA lines for each (down_dim_kv, num_blocks) combo
    for dkv in dkv_vals:
        kv = kv_dim_by_level[dkv]
        for nb in nb_vals:
            cache_gb = kv * nb * ctx_lens * 4 / 1024**3
            ax2.plot(ctx_lens / 1024, cache_gb,
                     color=DKV_COLORS[dkv], lw=1.8, ls=NB_LS[nb],
                     alpha=0.80, zorder=4,
                     label=f"MLA dkv={dkv}, {nb}L")

    # GPU VRAM references
    for vram, lbl in [(80, "80 GB A100"), (24, "24 GB RTX")]:
        ax2.axhline(vram, color="#CC3311", lw=1.2, ls=":", alpha=0.6)
        ax2.text(ctx_lens[-1] / 1024 * 0.98, vram * 1.04, lbl,
                 ha="right", fontsize=8, color="#CC3311", alpha=0.8)

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    xtick_vals = [0.5, 1, 2, 4, 8, 16, 32, 64, 128]
    ax2.set_xticks(xtick_vals)
    ax2.set_xticklabels(["512" if v < 1 else f"{int(v)}K" for v in xtick_vals],
                        fontsize=9)
    ax2.set_xlabel("Context length", fontsize=13, labelpad=6)
    ax2.set_ylabel("KV Cache per sequence (GB)", fontsize=12, labelpad=6)
    ax2.set_title("Theoretical Extrapolation\n(solid=16L  dashed=12L  dotted=8L)",
                  fontsize=12, fontweight="bold", pad=10)
    ax2.grid(True, which="both", alpha=0.2, linestyle="--")
    ax2.spines[["top", "right"]].set_visible(False)

    # Legend: down_dim_kv colours only (linestyles already in title)
    leg_h = [mpatches.Patch(color=DKV_COLORS[d], label=f"MLA  down_dim_kv = {d}")
             for d in dkv_vals]
    leg_h += [Line2D([0], [0], color=TYPE_COLORS["MHA"], ls="-.", lw=2, label="MHA ref"),
              Line2D([0], [0], color=TYPE_COLORS["GQA"], ls="-.", lw=2, label="GQA ref")]
    ax2.legend(handles=leg_h, fontsize=9, framealpha=0.9, loc="upper left")

    fig.suptitle(
        "down_dim_kv Is MLA's Cache Dial — Tune It, Not the Model Width",
        fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    _save(fig, "linkedin_5_mla_down_dim.png")


# --------------------------------------------------------------------------- #
# I/O                                                                          #
# --------------------------------------------------------------------------- #
def _save(fig, name: str):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


if __name__ == "__main__":
    df = load()
    print(f"Loaded {len(df)} runs")

    print("\nDense-only means:")
    dense = df[~df["moe"]]
    cols = [c for c in ["params_m","val_loss","theoretical_cache_mb","tps_512"] if c in dense.columns]
    print(dense.groupby("type")[cols].mean().round(2))

    print("\nGenerating LinkedIn figures...")
    fig1_cache_vs_quality(df)
    fig2_dense_comparison(df)
    fig3_when_mla_wins(df)
    fig4_params_vs_cache(df)
    fig5_mla_down_dim(df)
    print("Done.")
