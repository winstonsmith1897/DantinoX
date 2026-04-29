"""
plot_sweeps.py — controlled-sweep plots for sweeps B and C.

  sweep_B_down_dim_kv.png  — down_dim_kv vs val_loss + cache (MLA quality-cache Pareto)
  sweep_C_width_ladder.png — dim vs cache for MHA / GQA / MLA (flat MLA line)

Run after:
  1. wandb agents finish (sweeps B + C)
  2. python3 benchmark_analysis.py   (populates new runs into benchmark_results.csv)
  3. python3 plot_sweeps.py

The script works at any stage: if new-sweep data is not yet in the CSV it
plots only the existing anchor points and prints a warning.
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

# down_dim_kv values that exist before sweep B (from the original random sweep)
EXISTING_DKV = {64, 96, 128}
# down_dim_kv values added by sweep B
NEW_DKV      = {32, 48, 80, 112, 160, 192, 256}

# dims that exist before sweep C (for Dense num_blocks=12 runs)
EXISTING_DIMS_C = {256, 512}
# dims added by sweep C
NEW_DIMS_C      = {128, 192, 384, 768}


# ─────────────────────────────────────────────────────────────────────────────
def load() -> pd.DataFrame:
    df = pd.read_csv(IN_CSV)
    for c in ["params_m", "val_loss", "theoretical_cache_mb", "down_dim_kv",
              "dim", "num_blocks", "n_heads", "kv_heads"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _save(fig, name: str):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Sweep B — down_dim_kv ladder
# ─────────────────────────────────────────────────────────────────────────────
def fig_sweep_b(df: pd.DataFrame):
    """
    Twin-axis chart: down_dim_kv on x.
      Left  y (blue)   — val_loss (quality, lower = better)
      Right y (orange) — theoretical_cache_mb (memory, lower = better)

    Points from sweep B (new, controlled training):  filled markers
    Points from existing random-sweep runs:           hollow markers (backdrop)

    Architecture anchor: MLA Dense, dim=256, num_blocks=12.
    """
    anchor = df[
        (df["type"] == "MLA") &
        (~df["moe"]) &
        (df["dim"] == 256) &
        (df["num_blocks"] == 12)
    ].copy()

    if anchor.empty:
        print("[sweep B] No MLA Dense dim=256 num_blocks=12 runs found — skipping.")
        return

    # Tag as new (sweep B) vs existing
    anchor["is_new"] = anchor["down_dim_kv"].isin(NEW_DKV)

    new_runs = anchor[anchor["is_new"]]
    old_runs = anchor[~anchor["is_new"]]

    if new_runs.empty:
        print("[sweep B] Sweep B runs not yet in CSV — plotting existing anchor only.")

    # Aggregate existing runs: best (min) val_loss per dkv, cache is architectural
    old_agg = old_runs.groupby("down_dim_kv").agg(
        val_loss_mean=("val_loss", "mean"),
        val_loss_min =("val_loss", "min"),
        val_loss_max =("val_loss", "max"),
        cache        =("theoretical_cache_mb", "min"),   # same arch → same cache
    ).reset_index()

    # Sweep B runs: one controlled run per dkv
    new_agg = new_runs.groupby("down_dim_kv").agg(
        val_loss_mean=("val_loss", "mean"),
        cache        =("theoretical_cache_mb", "min"),
    ).reset_index()

    # Combined for the control line (new only if available, else existing best)
    ctrl = pd.concat([
        old_agg[["down_dim_kv","val_loss_min","cache"]].rename(
            columns={"val_loss_min": "val_loss"}),
        new_agg[["down_dim_kv","val_loss_mean","cache"]].rename(
            columns={"val_loss_mean": "val_loss"}),
    ]).sort_values("down_dim_kv")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(11, 6.5))
    fig.patch.set_facecolor("#F8F9FA")
    ax1.set_facecolor("#F8F9FA")
    ax2 = ax1.twinx()

    # Existing runs — scatter (light, hollow)
    if not old_agg.empty:
        for _, row in old_agg.iterrows():
            ax1.scatter(row["down_dim_kv"], row["val_loss_mean"],
                        color=TYPE_COLORS["MLA"], s=60,
                        facecolors="none", edgecolors=TYPE_COLORS["MLA"],
                        lw=1.5, zorder=3, alpha=0.7)
            # Error range
            ax1.vlines(row["down_dim_kv"],
                       row["val_loss_min"], row["val_loss_max"],
                       color=TYPE_COLORS["MLA"], lw=1, alpha=0.4)

    # Sweep B runs — filled markers
    if not new_agg.empty:
        ax1.scatter(new_agg["down_dim_kv"], new_agg["val_loss_mean"],
                    color=TYPE_COLORS["MLA"], s=90,
                    zorder=5, edgecolors="white", lw=1.2, label="Sweep B (controlled)")

    # Control line val_loss
    ax1.plot(ctrl["down_dim_kv"], ctrl["val_loss"],
             color=TYPE_COLORS["MLA"], lw=2, ls="--", alpha=0.7, zorder=4)

    # Cache on right axis (pure architecture, perfectly linear with dkv)
    ax2.plot(ctrl["down_dim_kv"], ctrl["cache"],
             color="#E63946", lw=2.5, ls="-", zorder=4, label="KV cache")
    ax2.scatter(ctrl["down_dim_kv"], ctrl["cache"],
                color="#E63946", s=70, zorder=5, edgecolors="white", lw=1)

    # ── Cosmetics ─────────────────────────────────────────────────────────────
    ax1.set_xlabel("down_dim_kv  (MLA latent KV dimension)", fontsize=13, labelpad=8)
    ax1.set_ylabel("Validation Loss  (NLL) ← lower is better",
                   fontsize=12, color=TYPE_COLORS["MLA"], labelpad=8)
    ax2.set_ylabel("KV Cache (MB) ← lower is better",
                   fontsize=12, color="#E63946", labelpad=8)
    ax1.tick_params(axis="y", colors=TYPE_COLORS["MLA"])
    ax2.tick_params(axis="y", colors="#E63946")

    ax1.set_title(
        "MLA: Choosing down_dim_kv Trades Quality for Cache\n"
        "Architecture: dim=256, num_blocks=12, Dense",
        fontsize=14, fontweight="bold", pad=12
    )

    handles = [
        Line2D([0],[0], color=TYPE_COLORS["MLA"], lw=2, ls="--",
               label="val_loss (best / controlled)"),
        mpatches.Patch(facecolor="none", edgecolor=TYPE_COLORS["MLA"],
                       linewidth=1.5, label="existing runs (range)"),
        Line2D([0],[0], color="#E63946", lw=2.5, label="KV cache (MB)"),
    ]
    ax1.legend(handles=handles, fontsize=10, framealpha=0.9, loc="upper left")

    ax1.grid(axis="both", alpha=0.2, linestyle="--")
    ax1.spines[["top"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)

    _save(fig, "sweep_B_down_dim_kv.png")


# ─────────────────────────────────────────────────────────────────────────────
# Sweep C — model width ladder
# ─────────────────────────────────────────────────────────────────────────────
def fig_sweep_c(df: pd.DataFrame):
    """
    Two-panel figure.

    Left  — dim vs theoretical_cache_mb (architectural, no training noise):
             MLA is a horizontal line; GQA and MHA rise linearly with dim.
    Right — dim vs val_loss (quality comparison across scales):
             Shows whether MLA maintains its quality advantage at larger dim.

    Controlled runs (sweep C, fixed training HPs): filled markers + solid line.
    Existing anchor runs (dim=256 and dim=512): hollow markers.

    MLA filter: down_dim_kv=64 only (keeps the flat-cache story clean).
    """
    # MLA: only dkv=64 runs, Dense, num_blocks=12
    mla = df[
        (df["type"] == "MLA") &
        (~df["moe"]) &
        (df["num_blocks"] == 12) &
        (df["down_dim_kv"] == 64)
    ].copy()
    mla["is_new"] = mla["dim"].isin(NEW_DIMS_C)

    # MHA / GQA: all Dense num_blocks=12 runs
    other = df[
        (df["type"] != "MLA") &
        (~df["moe"]) &
        (df["num_blocks"] == 12)
    ].copy()
    other["is_new"] = other["dim"].isin(NEW_DIMS_C)

    combined = pd.concat([mla, other])

    if combined.empty:
        print("[sweep C] No Dense num_blocks=12 runs found — skipping.")
        return

    new_count = combined["is_new"].sum()
    if new_count == 0:
        print("[sweep C] Sweep C runs not yet in CSV — plotting existing anchors only.")

    # Best val_loss and de-duplicated cache per (type, dim)
    agg = (
        combined
        .groupby(["type", "dim", "is_new"])
        .agg(
            val_loss_best =("val_loss", "min"),
            val_loss_mean =("val_loss", "mean"),
            val_loss_std  =("val_loss", "std"),
            cache         =("theoretical_cache_mb", "median"),
            params_m      =("params_m", "median"),
        )
        .reset_index()
    )

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, (ax_cache, ax_loss) = plt.subplots(1, 2, figsize=(16, 6.5))
    fig.patch.set_facecolor("#F8F9FA")

    for ax in (ax_cache, ax_loss):
        ax.set_facecolor("#F8F9FA")
        ax.grid(axis="both", alpha=0.2, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Model width  (dim)", fontsize=13, labelpad=8)

    # ── Left: cache ────────────────────────────────────────────────────────
    for t in TYPE_ORDER:
        sub = agg[agg["type"] == t].sort_values("dim")
        if sub.empty:
            continue

        new_pts = sub[sub["is_new"]]
        old_pts = sub[~sub["is_new"]]

        # Hollow markers for existing anchor runs
        if not old_pts.empty:
            ax_cache.scatter(old_pts["dim"], old_pts["cache"],
                             s=80, color=TYPE_COLORS[t],
                             facecolors="none", edgecolors=TYPE_COLORS[t],
                             lw=1.8, zorder=4, alpha=0.8)

        # Filled markers for new sweep C runs
        if not new_pts.empty:
            ax_cache.scatter(new_pts["dim"], new_pts["cache"],
                             s=80, color=TYPE_COLORS[t],
                             edgecolors="white", lw=1.2, zorder=5)

        # Connect all points with a line
        ax_cache.plot(sub["dim"], sub["cache"],
                      color=TYPE_COLORS[t], lw=2.2, ls="-",
                      alpha=0.8, zorder=3, label=t)

        # Label the last point
        last = sub.iloc[-1]
        ax_cache.annotate(
            f"{t}  {last['cache']:.1f} MB",
            xy=(last["dim"], last["cache"]),
            xytext=(8, 0), textcoords="offset points",
            fontsize=9, color=TYPE_COLORS[t], fontweight="bold", va="center"
        )

    ax_cache.set_ylabel("KV Cache (MB)  @ 512 tokens  ← lower is better",
                        fontsize=12, labelpad=8)
    ax_cache.set_title(
        "Cache Grows with dim for MHA/GQA\n— MLA Stays Flat (down_dim_kv=64 fixed)",
        fontsize=12, fontweight="bold", pad=12
    )
    ax_cache.legend(fontsize=10, framealpha=0.9, loc="upper left")

    # Marker style guide
    ax_cache.scatter([], [], s=70, color="#888", facecolors="none",
                     edgecolors="#888", lw=1.8, label="existing run")
    ax_cache.scatter([], [], s=70, color="#888",
                     edgecolors="white", lw=1.2, label="sweep C (controlled)")
    ax_cache.legend(fontsize=9, framealpha=0.9, loc="upper left", ncol=1)

    # ── Right: val_loss ────────────────────────────────────────────────────
    for t in TYPE_ORDER:
        sub = agg[agg["type"] == t].sort_values("dim")
        if sub.empty:
            continue

        new_pts = sub[sub["is_new"]]
        old_pts = sub[~sub["is_new"]]

        if not old_pts.empty:
            ax_loss.scatter(old_pts["dim"], old_pts["val_loss_best"],
                            s=80, color=TYPE_COLORS[t],
                            facecolors="none", edgecolors=TYPE_COLORS[t],
                            lw=1.8, zorder=4, alpha=0.8)

        if not new_pts.empty:
            # Error bars for std (if multiple runs at same dim)
            ax_loss.errorbar(
                new_pts["dim"], new_pts["val_loss_mean"],
                yerr=new_pts["val_loss_std"].fillna(0),
                fmt="o", color=TYPE_COLORS[t],
                ecolor=TYPE_COLORS[t], elinewidth=1.2,
                capsize=4, markersize=7,
                markeredgecolor="white", markeredgewidth=1.2,
                zorder=5
            )

        ax_loss.plot(sub["dim"], sub["val_loss_best"],
                     color=TYPE_COLORS[t], lw=2.2, ls="-",
                     alpha=0.8, zorder=3, label=t)

    ax_loss.set_ylabel("Validation Loss  (NLL)  ← lower is better",
                       fontsize=12, labelpad=8)
    ax_loss.set_title(
        "Quality vs Model Width\n(all three types, same training config)",
        fontsize=12, fontweight="bold", pad=12
    )
    ax_loss.legend(fontsize=10, framealpha=0.9, loc="upper right")

    fig.suptitle(
        "Sweep C — Width Ladder: Dense, num_blocks=12, MLA with down_dim_kv=64",
        fontsize=14, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    _save(fig, "sweep_C_width_ladder.png")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load()
    print(f"Loaded {len(df)} runs from {IN_CSV}")

    mla_anchor = df[
        (df["type"] == "MLA") & (~df["moe"]) &
        (df["dim"] == 256) & (df["num_blocks"] == 12)
    ]
    new_b = mla_anchor[mla_anchor["down_dim_kv"].isin(NEW_DKV)]
    new_c = df[df["dim"].isin(NEW_DIMS_C) & (~df["moe"]) & (df["num_blocks"] == 12)]

    print(f"\nSweep B anchor runs : {len(mla_anchor)}  "
          f"(new controlled: {len(new_b)}/{len(NEW_DKV)} expected)")
    print(f"Sweep C new-dim runs: {len(new_c)}  "
          f"(expected {len(NEW_DIMS_C) * 3} after sweeps complete)")

    print("\nGenerating sweep plots...")
    fig_sweep_b(df)
    fig_sweep_c(df)
    print("Done.")
