"""
plot_insights.py — three high-signal plots from benchmark_results.csv.

  insight_1_pareto.png        — quality-cache Pareto front (MLA dominates)
  insight_2_serving.png       — aggregate tok/s at fixed VRAM budget
  insight_3_mla_dial.png      — MLA's down_dim_kv quality-cache tradeoff knob

Run: python3 plot_insights.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull

IN_CSV       = "benchmark_results.csv"
OUT_DIR      = "plots"
DPI          = 180
TYPE_COLORS  = {"MLA": "#3A86FF", "GQA": "#FF6B35", "MHA": "#2DC653"}
TYPE_ORDER   = ["MLA", "GQA", "MHA"]


def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, name)
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p}")


def load():
    df = pd.read_csv(IN_CSV)
    for c in ["params_m", "val_loss", "theoretical_cache_mb",
              "tps_64", "tps_128", "tps_256", "tps_512"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _pareto_front(xs, ys):
    """Return indices of points on the lower-left Pareto front (minimize both)."""
    pts = sorted(enumerate(zip(xs, ys)), key=lambda p: p[1][0])
    front, best_y = [], float("inf")
    for idx, (x, y) in pts:
        if y <= best_y:
            front.append(idx)
            best_y = y
    return front


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Quality-Cache Pareto front
# ─────────────────────────────────────────────────────────────────────────────
def fig1_pareto(df):
    """
    x: KV cache (MB)  — lower is better
    y: val_loss       — lower is better
    Lower-left corner is ideal.  MLA should populate the Pareto front.

    Pareto-dominated points are shown semi-transparent; the Pareto frontier
    is highlighted with a step-function line.
    """
    sub = df.dropna(subset=["theoretical_cache_mb", "val_loss", "params_m"]).copy()
    dense = sub[~sub["moe"]]

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    fig.patch.set_facecolor("#F8F9FA")

    for ax_i, (data, title_sfx) in enumerate([(dense, "Dense"), (sub, "Dense + MoE")]):
        ax = axes[ax_i]
        ax.set_facecolor("#F8F9FA")

        xs_all = data["theoretical_cache_mb"].values
        ys_all = data["val_loss"].values
        pareto_idx = set(_pareto_front(xs_all, ys_all))

        for t in TYPE_ORDER:
            pts = data[data["type"] == t].reset_index(drop=True)
            if pts.empty:
                continue
            local_idx = data[data["type"] == t].index
            sz = (pts["params_m"] / sub["params_m"].max() * 400).clip(lower=25)
            mk = "^" if data.get("moe", pd.Series(False)).any() else "o"

            for moe_v, grp in pts.groupby("moe"):
                mk = "^" if moe_v else "o"
                g_sz = (grp["params_m"] / sub["params_m"].max() * 400).clip(lower=25)
                orig_idx = grp.index
                alphas = [0.90 if i in pareto_idx else 0.22 for i in orig_idx]
                for (_, row), a, s in zip(grp.iterrows(), alphas, g_sz):
                    ax.scatter(row["theoretical_cache_mb"], row["val_loss"],
                               s=s, c=TYPE_COLORS[t], marker=mk,
                               edgecolors="white", lw=0.7, alpha=a, zorder=4)

        # Draw Pareto step-function
        pareto_pts = sorted(
            [(xs_all[i], ys_all[i]) for i in pareto_idx],
            key=lambda p: p[0]
        )
        if pareto_pts:
            px = [p[0] for p in pareto_pts]
            py = [p[1] for p in pareto_pts]
            # step line: move right then down
            step_x, step_y = [px[0]], [py[0]]
            for i in range(1, len(px)):
                step_x += [px[i], px[i]]
                step_y += [py[i - 1], py[i]]
            ax.plot(step_x, step_y, color="#222222", lw=2, ls="-",
                    alpha=0.65, zorder=5, label="Pareto front")
            # Identify which types are on the front
            front_types = set(
                data.iloc[i]["type"] for i in pareto_idx
                if i < len(data)
            )
            ax.text(step_x[0] * 1.05, step_y[0] * 0.997,
                    f"Pareto front\n({', '.join(sorted(front_types))})",
                    fontsize=9, color="#222", style="italic",
                    va="top")

        # Ideal direction arrow
        ax.annotate("", xy=(0.06, 0.06), xytext=(0.22, 0.22),
                    xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color="#888",
                                    lw=1.5, mutation_scale=14))
        ax.text(0.04, 0.055, "ideal", transform=ax.transAxes,
                fontsize=9, color="#888", style="italic")

        ax.set_xlabel("KV Cache (MB) @ 512 tokens  ← lower is better", fontsize=12)
        ax.set_ylabel("Validation Loss (NLL)  ← lower is better", fontsize=12)
        ax.set_title(f"{title_sfx} — Quality vs Cache Pareto\n"
                     "(semi-transparent = Pareto-dominated)",
                     fontsize=12, fontweight="bold")
        ax.grid(alpha=0.2, ls="--")
        ax.spines[["top", "right"]].set_visible(False)

        handles = [mpatches.Patch(color=TYPE_COLORS[t], label=t) for t in TYPE_ORDER
                   if not data[data["type"] == t].empty]
        handles += [
            Line2D([0],[0], marker="o", color="#888", ls="None",
                   markersize=8, markeredgecolor="white", label="Dense"),
            Line2D([0],[0], marker="^", color="#888", ls="None",
                   markersize=8, markeredgecolor="white", label="MoE"),
            Line2D([0],[0], color="#222", lw=2, label="Pareto front"),
        ]
        ax.legend(handles=handles, fontsize=9, framealpha=0.9,
                  loc="upper right", ncol=2)

    fig.suptitle(
        "MLA Pareto-Dominates MHA: Better Quality AND Smaller KV Cache",
        fontsize=14, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    _save(fig, "insight_1_pareto.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Aggregate serving throughput at fixed VRAM budget
# ─────────────────────────────────────────────────────────────────────────────
def fig2_serving(df):
    """
    Key insight: a faster-per-sequence model is NOT always best for serving.
    When VRAM is fixed, a model with a smaller KV cache fits more concurrent
    sequences, which can outweigh its per-sequence latency.

    Metric: total_tps(budget) = (budget_MB / cache_per_seq_MB) × tps_per_seq

    We compute the MEDIAN tps and cache across all runs of each type (Dense).
    Then we sweep VRAM budgets from 500 MB to 80 GB.
    """
    dense = df[~df["moe"]].dropna(subset=["tps_512", "theoretical_cache_mb"])

    # Median per-sequence stats per type
    stats = dense.groupby("type").agg(
        tps=("tps_512", "median"),
        cache_mb=("theoretical_cache_mb", "median"),
        tps_lo=("tps_512", lambda x: x.quantile(0.25)),
        tps_hi=("tps_512", lambda x: x.quantile(0.75)),
        cache_lo=("theoretical_cache_mb", lambda x: x.quantile(0.25)),
        cache_hi=("theoretical_cache_mb", lambda x: x.quantile(0.75)),
    )

    budgets_mb = np.logspace(np.log10(500), np.log10(80_000), 300)

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    fig.patch.set_facecolor("#F8F9FA")

    # ── Left: total tok/s vs budget ─────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#F8F9FA")

    for t in TYPE_ORDER:
        if t not in stats.index:
            continue
        tps   = stats.loc[t, "tps"]
        cache = stats.loc[t, "cache_mb"]
        total = (budgets_mb / cache) * tps
        ax.plot(budgets_mb / 1024, total / 1000,
                color=TYPE_COLORS[t], lw=2.8, label=t, zorder=4)

        # Uncertainty band (IQR of tps and cache)
        t_lo = stats.loc[t, "tps_lo"]
        t_hi = stats.loc[t, "tps_hi"]
        c_lo = stats.loc[t, "cache_lo"]
        c_hi = stats.loc[t, "cache_hi"]
        tot_lo = (budgets_mb / c_hi) * t_lo
        tot_hi = (budgets_mb / c_lo) * t_hi
        ax.fill_between(budgets_mb / 1024, tot_lo / 1000, tot_hi / 1000,
                        color=TYPE_COLORS[t], alpha=0.12, zorder=3)

    # GPU memory reference lines
    for gb, lbl in [(24, "24 GB\n(RTX)"), (40, "40 GB\n(A100)"), (80, "80 GB\n(H100)")]:
        ax.axvline(gb, color="#888", lw=1, ls="--", alpha=0.6)
        ax.text(gb * 1.02, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1,
                lbl, fontsize=8, color="#888", va="top")

    ax.set_xscale("log")
    ax.set_xlabel("KV Cache VRAM budget (GB)", fontsize=12)
    ax.set_ylabel("Aggregate throughput (k tok/s)", fontsize=12)
    ax.set_title("Total Serving Throughput at Fixed VRAM Budget\n"
                 "(median per-sequence tps × max concurrent sequences)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, which="both", alpha=0.2, ls="--")
    ax.spines[["top", "right"]].set_visible(False)

    # ── Right: ratio vs MHA baseline ────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#F8F9FA")
    mha_total = (budgets_mb / stats.loc["MHA", "cache_mb"]) * stats.loc["MHA", "tps"]

    for t in TYPE_ORDER:
        if t not in stats.index:
            continue
        tps   = stats.loc[t, "tps"]
        cache = stats.loc[t, "cache_mb"]
        ratio = ((budgets_mb / cache) * tps) / mha_total
        ax2.plot(budgets_mb / 1024, ratio,
                 color=TYPE_COLORS[t], lw=2.8, label=t, zorder=4)

    ax2.axhline(1.0, color=TYPE_COLORS["MHA"], lw=1.2, ls="--", alpha=0.6)
    ax2.axhline(2.0, color="#CCCCCC", lw=0.8, ls=":", alpha=0.8)
    ax2.axhline(2.5, color="#CCCCCC", lw=0.8, ls=":", alpha=0.8)

    # Annotate at 80 GB
    for t in TYPE_ORDER:
        if t not in stats.index:
            continue
        tps   = stats.loc[t, "tps"]
        cache = stats.loc[t, "cache_mb"]
        val_at_80 = ((80_000 / cache) * tps) / ((80_000 / stats.loc["MHA","cache_mb"]) * stats.loc["MHA","tps"])
        ax2.annotate(f"{t}: {val_at_80:.1f}×",
                     xy=(80, val_at_80),
                     xytext=(-50, 8), textcoords="offset points",
                     fontsize=11, fontweight="bold", color=TYPE_COLORS[t],
                     arrowprops=dict(arrowstyle="-", color=TYPE_COLORS[t], lw=1))

    for gb, lbl in [(24, "24 GB"), (40, "40 GB"), (80, "80 GB")]:
        ax2.axvline(gb, color="#888", lw=1, ls="--", alpha=0.6)
        ax2.text(gb * 1.02, 0.05, lbl, fontsize=8, color="#888",
                 transform=ax2.get_xaxis_transform())

    ax2.set_xscale("log")
    ax2.set_xlabel("KV Cache VRAM budget (GB)", fontsize=12)
    ax2.set_ylabel("Throughput ratio  (MHA = 1.0×)", fontsize=12)
    ax2.set_title("Relative Serving Advantage vs MHA\n"
                  "(above 1.0 = more total tokens/s for the same VRAM)",
                  fontsize=12, fontweight="bold")
    ax2.legend(fontsize=11, framealpha=0.9)
    ax2.grid(True, which="both", alpha=0.2, ls="--")
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "MLA's Smaller Cache Enables 2–3× More Total Throughput at Fixed VRAM\n"
        "(even though MLA is slower per-sequence, it fits more concurrent users)",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    _save(fig, "insight_2_serving.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — MLA's down_dim_kv dial: quality vs cache
# ─────────────────────────────────────────────────────────────────────────────
def fig3_mla_dial(df):
    """
    MLA has a unique hyperparameter: down_dim_kv.
    Increasing it improves quality but grows the KV cache linearly.
    MHA and GQA have no equivalent knob — their cache is fixed by dim and kv_heads.

    Left panel  — down_dim_kv vs val_loss (quality cost of compression)
    Right panel — down_dim_kv vs cache (linear relationship)
    Both panels: MHA/GQA reference bands for context.
    """
    mla   = df[(df["type"] == "MLA") & (~df["moe"])].copy()
    other = df[(df["type"] != "MLA") & (~df["moe"])].copy()

    if mla["down_dim_kv"].isna().all():
        print("[fig3] No down_dim_kv data — skipping.")
        return

    # Aggregate MLA per down_dim_kv
    agg = mla.groupby("down_dim_kv").agg(
        val_loss_mean=("val_loss", "mean"),
        val_loss_min =("val_loss", "min"),
        val_loss_max =("val_loss", "max"),
        cache_mean   =("theoretical_cache_mb", "mean"),
        cache_min    =("theoretical_cache_mb", "min"),
        cache_max    =("theoretical_cache_mb", "max"),
        n            =("val_loss", "count"),
    ).reset_index()

    # MHA / GQA reference: overall median val_loss and cache across Dense runs
    ref_mha_loss  = other[other["type"] == "MHA"]["val_loss"].median()
    ref_gqa_loss  = other[other["type"] == "GQA"]["val_loss"].median()
    ref_mha_cache = other[other["type"] == "MHA"]["theoretical_cache_mb"].median()
    ref_gqa_cache = other[other["type"] == "GQA"]["theoretical_cache_mb"].median()

    dkv = agg["down_dim_kv"].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))
    fig.patch.set_facecolor("#F8F9FA")

    # ── Left: quality ────────────────────────────────────────────────────────
    ax1.set_facecolor("#F8F9FA")

    # MHA / GQA reference bands
    for val, col, lbl in [
        (ref_mha_loss, TYPE_COLORS["MHA"], "MHA median"),
        (ref_gqa_loss, TYPE_COLORS["GQA"], "GQA median"),
    ]:
        ax1.axhline(val, color=col, lw=2, ls="--", alpha=0.75, label=lbl)
        ax1.fill_between(
            [dkv.min() * 0.8, dkv.max() * 1.2],
            val * 0.97, val * 1.03,
            color=col, alpha=0.08
        )

    # MLA line + IQR band
    ax1.plot(dkv, agg["val_loss_mean"], color=TYPE_COLORS["MLA"],
             lw=2.8, marker="o", markersize=8, label="MLA (mean)", zorder=5)
    ax1.fill_between(dkv, agg["val_loss_min"], agg["val_loss_max"],
                     color=TYPE_COLORS["MLA"], alpha=0.18, zorder=3,
                     label="MLA (min–max range)")

    # Annotate best MLA point
    best = agg.loc[agg["val_loss_mean"].idxmin()]
    ax1.scatter(best["down_dim_kv"], best["val_loss_mean"],
                s=180, c=TYPE_COLORS["MLA"], edgecolors="black", lw=2, zorder=6)
    ax1.annotate(f"best:\ndkv={int(best['down_dim_kv'])}\nloss={best['val_loss_mean']:.3f}",
                 (best["down_dim_kv"], best["val_loss_mean"]),
                 xytext=(14, -28), textcoords="offset points",
                 fontsize=9, color=TYPE_COLORS["MLA"],
                 arrowprops=dict(arrowstyle="-", color=TYPE_COLORS["MLA"], lw=1))

    ax1.set_xlabel("down_dim_kv  (MLA latent KV dimension)", fontsize=12)
    ax1.set_ylabel("Validation Loss (NLL)  ← lower is better", fontsize=12)
    ax1.set_title("Quality vs down_dim_kv\n"
                  "MHA / GQA have no equivalent tuning knob",
                  fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10, framealpha=0.9, loc="upper right")
    ax1.grid(alpha=0.2, ls="--")
    ax1.spines[["top", "right"]].set_visible(False)

    # ── Right: cache ─────────────────────────────────────────────────────────
    ax2.set_facecolor("#F8F9FA")

    for val, col, lbl in [
        (ref_mha_cache, TYPE_COLORS["MHA"], "MHA median"),
        (ref_gqa_cache, TYPE_COLORS["GQA"], "GQA median"),
    ]:
        ax2.axhline(val, color=col, lw=2, ls="--", alpha=0.75, label=lbl)
        ax2.fill_between(
            [dkv.min() * 0.8, dkv.max() * 1.2],
            val * 0.85, val * 1.15,
            color=col, alpha=0.08
        )

    ax2.plot(dkv, agg["cache_mean"], color=TYPE_COLORS["MLA"],
             lw=2.8, marker="o", markersize=8, label="MLA (mean)", zorder=5)
    ax2.fill_between(dkv, agg["cache_min"], agg["cache_max"],
                     color=TYPE_COLORS["MLA"], alpha=0.18, zorder=3,
                     label="MLA (min–max range)")

    # Annotate the crossover points
    for ref_val, ref_col, ref_lbl in [
        (ref_mha_cache, TYPE_COLORS["MHA"], "MHA"),
        (ref_gqa_cache, TYPE_COLORS["GQA"], "GQA"),
    ]:
        # Find where MLA cache crosses the reference
        crosses = agg[agg["cache_mean"] <= ref_val]
        if not crosses.empty:
            cross_dkv = crosses["down_dim_kv"].max()
            ax2.axvline(cross_dkv, color=ref_col, lw=1.2, ls=":", alpha=0.8)
            ax2.text(cross_dkv, ref_val * 0.6,
                     f"MLA = {ref_lbl}\ncache at\ndkv={int(cross_dkv)}",
                     fontsize=8, color=ref_col, ha="center",
                     bbox=dict(boxstyle="round,pad=0.3", fc="white",
                               ec=ref_col, lw=0.8, alpha=0.85))

    ax2.set_xlabel("down_dim_kv  (MLA latent KV dimension)", fontsize=12)
    ax2.set_ylabel("KV Cache (MB) @ 512 tokens", fontsize=12)
    ax2.set_title("Cache Size vs down_dim_kv\n"
                  "Linear relationship — easy to predict at any scale",
                  fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10, framealpha=0.9, loc="upper left")
    ax2.grid(alpha=0.2, ls="--")
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "MLA's Unique Tuning Dial: down_dim_kv Controls the Quality-Cache Tradeoff\n"
        "MHA / GQA are fixed — MLA lets you choose where on the curve to land",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    _save(fig, "insight_3_mla_dial.png")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load()
    print(f"Loaded {len(df)} runs")
    print("\nGenerating insight figures...")
    fig1_pareto(df)
    fig2_serving(df)
    fig3_mla_dial(df)
    print("Done.")
