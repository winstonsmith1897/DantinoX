"""
plot_perf.py — performance comparison: memory, KV cache, speed, FLOPs.

Figures produced
  perf_1_cache_breakdown.png   — KV cache vs model size (theoretical, all types)
  perf_2_seqlen_throughput.png — tok/s vs sequence length (existing data, bs=1)
  perf_3_flops_vs_cache.png    — analytical decode FLOPs vs KV cache (Pareto)
  perf_4_batch_throughput.png  — tok/s vs batch size  (needs batch_sweep_results.csv)
  perf_5_prefill.png           — prefill latency vs model size

Run with existing data:
  python3 plot_perf.py

Run after batch sweep:
  CUDA_VISIBLE_DEVICES=0 python3 benchmark_batch_sweep.py
  python3 plot_perf.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

IN_CSV       = "benchmark_results.csv"
BATCH_CSV    = "batch_sweep_results.csv"
OUT_DIR      = "plots"
DPI          = 180

TYPE_COLORS  = {"MLA": "#3A86FF", "GQA": "#FF6B35", "MHA": "#2DC653"}
TYPE_ORDER   = ["MLA", "GQA", "MHA"]
SEQ_LENS     = [64, 128, 256, 512]


# ─────────────────────────────────────────────────────────────────────────────
def _save(fig, name: str):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def load() -> pd.DataFrame:
    df = pd.read_csv(IN_CSV)
    num_cols = ["params_m", "val_loss", "theoretical_cache_mb",
                "measured_cache_mb", "prefill_ms"] + [f"tps_{s}" for s in SEQ_LENS]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # head_size derived from architecture
    df["head_size"] = (df["dim"] / df["n_heads"]).round().astype(int)
    return df


def load_batch() -> pd.DataFrame | None:
    if not os.path.exists(BATCH_CSV):
        return None
    df = pd.read_csv(BATCH_CSV)
    for c in ["tps", "cache_mb_total", "batch_size", "params_m",
              "theoretical_cache_mb"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Analytical FLOPs for one decode step (T=1, context length S, batch B=1)
# ─────────────────────────────────────────────────────────────────────────────
def _decode_flops(row: pd.Series, S: int = 256) -> float:
    """Approximate decode FLOPs (×2 counted, i.e. MACs×2) for one token."""
    dim      = int(row["dim"])
    n_heads  = int(row["n_heads"])
    kv_heads = int(row["kv_heads"])
    h        = int(row["head_size"])   # = dim // n_heads
    nb       = int(row["num_blocks"])
    exp      = 4   # expansion factor; SwiGLU ≈ same total

    if row["type"] == "MLA":
        down_dim_q  = row.get("down_dim_q",  dim // 2) or dim // 2
        down_dim_kv = row["down_dim_kv"] if not pd.isna(row["down_dim_kv"]) else dim // 4
        rope_dim    = max(16, int(down_dim_kv) // 4)
        down_dim_q  = int(down_dim_q)
        down_dim_kv = int(down_dim_kv)
        # Per decode step, per batch=1:
        attn = (
            2 * dim * down_dim_q                              # down_q
            + 2 * dim * rope_dim * 2                          # q_pe + k_pe
            + 2 * n_heads * down_dim_q * down_dim_kv * h      # wt-wt attn_proj (per layer)
            + 2 * n_heads * down_dim_q * down_dim_kv          # q × attn_proj
            + 2 * n_heads * S * down_dim_kv                   # vs compressed cache
            + 2 * n_heads * S * rope_dim                      # rope attention
            + 2 * n_heads * h * down_dim_kv * dim             # W_vo wt-wt (per layer)
            + 2 * n_heads * down_dim_kv * dim                 # output einsum
        )
    else:
        attn = (
            2 * dim * dim                                     # Q proj (n_heads×h = dim)
            + 2 * dim * kv_heads * h * 2                      # K + V proj
            + 2 * n_heads * S * h                             # QK^T
            + 2 * n_heads * S * h                             # Attn × V
            + 2 * dim * dim                                   # O proj
        )

    mlp = 2 * dim * exp * dim * 3   # SwiGLU: two up projections + down

    return (attn + mlp) * nb


def _prefill_flops(row: pd.Series) -> float:
    """Prefill FLOPs scale quadratically with sequence length."""
    T = int(row.get("max_context", 512))
    return _decode_flops(row, S=T // 2) * T   # rough: O(T²) for attention


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — KV cache vs model parameters, split by type + Dense/MoE
# ─────────────────────────────────────────────────────────────────────────────
def fig1_cache_breakdown(df: pd.DataFrame):
    sub = df.dropna(subset=["params_m", "theoretical_cache_mb"]).copy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    fig.patch.set_facecolor("#F8F9FA")

    # Left: scatter params vs cache
    ax = axes[0]
    ax.set_facecolor("#F8F9FA")
    for t in TYPE_ORDER:
        for moe, grp in sub[sub["type"] == t].groupby("moe"):
            mk = "^" if moe else "o"
            ax.scatter(grp["params_m"], grp["theoretical_cache_mb"],
                       s=60, c=TYPE_COLORS[t], marker=mk,
                       edgecolors="white", lw=0.8, alpha=0.85, zorder=4)
        # mean annotation
        pts = sub[sub["type"] == t]
        ax.scatter([], [], c=TYPE_COLORS[t], s=80, label=t)

    ax.set_xlabel("Parameters (M)", fontsize=12)
    ax.set_ylabel("KV Cache (MB) @ 512 tokens", fontsize=12)
    ax.set_title("Params vs KV Cache — MLA decoupled",
                 fontsize=12, fontweight="bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.2, ls="--")
    ax.spines[["top","right"]].set_visible(False)

    # Right: bar chart — cache per type (Dense only), split by num_blocks
    ax2 = axes[1]
    ax2.set_facecolor("#F8F9FA")
    dense = sub[~sub["moe"]]
    agg   = dense.groupby(["type","num_blocks"])["theoretical_cache_mb"].mean().reset_index()
    nb_vals = sorted(agg["num_blocks"].unique())
    w       = 0.8 / len(nb_vals)
    x       = np.arange(len(TYPE_ORDER))
    cmap    = plt.cm.Blues(np.linspace(0.4, 0.9, len(nb_vals)))

    for i, nb in enumerate(nb_vals):
        sub2 = agg[agg["num_blocks"] == nb]
        vals = [sub2[sub2["type"] == t]["theoretical_cache_mb"].mean()
                if not sub2[sub2["type"] == t].empty else 0 for t in TYPE_ORDER]
        off  = (i - len(nb_vals) / 2 + 0.5) * w
        bars = ax2.bar(x + off, vals, w, color=[TYPE_COLORS[t] for t in TYPE_ORDER],
                       alpha=0.55 + 0.45 * i / max(len(nb_vals)-1, 1),
                       edgecolor="white", lw=0.8, label=f"{nb} layers", zorder=3)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, v * 1.04,
                         f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    ax2.set_xticks(x)
    ax2.set_xticklabels(TYPE_ORDER, fontsize=12)
    ax2.set_ylabel("KV Cache (MB) @ 512 tokens", fontsize=12)
    ax2.set_title("Dense models: cache scales with depth\n(MLA remains smallest at every depth)",
                  fontsize=11, fontweight="bold")
    ax2.legend(title="num_blocks", fontsize=9, framealpha=0.9)
    ax2.grid(axis="y", alpha=0.2, ls="--", zorder=0)
    ax2.spines[["top","right","left"]].set_visible(False)
    ax2.tick_params(left=False)

    fig.suptitle("KV Cache Footprint by Attention Type",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, "perf_1_cache_breakdown.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Decode throughput vs sequence length (bs=1, existing data)
# ─────────────────────────────────────────────────────────────────────────────
def fig2_seqlen_throughput(df: pd.DataFrame):
    dense = df[~df["moe"]].copy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("#F8F9FA")

    # Left: median tps per type vs seq_len
    ax = axes[0]
    ax.set_facecolor("#F8F9FA")
    for t in TYPE_ORDER:
        sub = dense[dense["type"] == t]
        if sub.empty:
            continue
        ys_med = [sub[f"tps_{s}"].median() for s in SEQ_LENS]
        ys_p25 = [sub[f"tps_{s}"].quantile(0.25) for s in SEQ_LENS]
        ys_p75 = [sub[f"tps_{s}"].quantile(0.75) for s in SEQ_LENS]
        ax.plot(SEQ_LENS, ys_med, marker="o", color=TYPE_COLORS[t],
                lw=2.5, label=t, zorder=4)
        ax.fill_between(SEQ_LENS, ys_p25, ys_p75,
                        color=TYPE_COLORS[t], alpha=0.15, zorder=3)

    ax.set_xlabel("Context / sequence length (tokens)", fontsize=12)
    ax.set_ylabel("Tokens / sec  (batch=1)", fontsize=12)
    ax.set_title("Decode Throughput vs Sequence Length\n(batch=1, all Dense models)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(alpha=0.2, ls="--")
    ax.spines[["top","right"]].set_visible(False)

    # Annotation: why MLA is slower at bs=1
    ax.text(0.97, 0.97,
            "At bs=1, GPU is underutilized.\n"
            "MLA has more ops per step\n"
            "(up-projections + fused einsum).\n"
            "Cache savings don't matter yet.",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color="#555", style="italic",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ccc", lw=1))

    # Right: normalized (MLA = 1.0 at each seq_len)
    ax2 = axes[1]
    ax2.set_facecolor("#F8F9FA")
    mla_med = {s: dense[dense["type"] == "MLA"][f"tps_{s}"].median() for s in SEQ_LENS}
    for t in TYPE_ORDER:
        sub = dense[dense["type"] == t]
        if sub.empty:
            continue
        ys = [sub[f"tps_{s}"].median() / mla_med[s] for s in SEQ_LENS]
        ax2.plot(SEQ_LENS, ys, marker="o", color=TYPE_COLORS[t],
                 lw=2.5, label=t, zorder=4)

    ax2.axhline(1.0, color=TYPE_COLORS["MLA"], lw=1.2, ls="--", alpha=0.6)
    ax2.set_xlabel("Context / sequence length (tokens)", fontsize=12)
    ax2.set_ylabel("Relative throughput  (MLA = 1.0)", fontsize=12)
    ax2.set_title("Relative Throughput — MLA baseline\n"
                  "(below 1 = slower than MLA, above 1 = faster)",
                  fontsize=12, fontweight="bold")
    ax2.legend(fontsize=11, framealpha=0.9)
    ax2.grid(alpha=0.2, ls="--")
    ax2.spines[["top","right"]].set_visible(False)

    fig.suptitle("Decode Speed: Sequence Length Scaling (Batch=1)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, "perf_2_seqlen_throughput.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Analytical FLOPs vs KV cache (Pareto front)
# ─────────────────────────────────────────────────────────────────────────────
def fig3_flops_vs_cache(df: pd.DataFrame):
    sub = df.dropna(subset=["theoretical_cache_mb"]).copy()
    sub["decode_flops_m"] = sub.apply(lambda r: _decode_flops(r, S=256), axis=1) / 1e6

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    fig.patch.set_facecolor("#F8F9FA")

    for ax_idx, (moe_val, title_sfx) in enumerate([(False, "Dense"), (True, "MoE")]):
        ax  = axes[ax_idx]
        ax.set_facecolor("#F8F9FA")
        grp = sub[sub["moe"] == moe_val]
        if grp.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(title_sfx)
            continue

        # Pareto front (lower-left is better)
        for t in TYPE_ORDER:
            pts = grp[grp["type"] == t]
            if pts.empty:
                continue
            sz = (pts["params_m"] / pts["params_m"].max() * 250).clip(lower=25)
            ax.scatter(pts["theoretical_cache_mb"], pts["decode_flops_m"],
                       s=sz, c=TYPE_COLORS[t], edgecolors="white", lw=0.8,
                       alpha=0.85, zorder=4, label=t)

        # Annotate centroid per type
        for t in TYPE_ORDER:
            pts = grp[grp["type"] == t]
            if pts.empty:
                continue
            cx = pts["theoretical_cache_mb"].median()
            cy = pts["decode_flops_m"].median()
            ax.annotate(t, (cx, cy), fontsize=11, fontweight="bold",
                        color=TYPE_COLORS[t],
                        xytext=(6, 4), textcoords="offset points")

        ax.set_xlabel("KV Cache (MB) @ 512 tokens  ← lower is better", fontsize=12)
        ax.set_ylabel("Decode FLOPs (M, analytical, S=256)  ← lower is better", fontsize=12)
        ax.set_title(f"{title_sfx} models — FLOPs vs Cache tradeoff\n"
                     "(bubble size ∝ params)",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=10, framealpha=0.9)
        ax.grid(alpha=0.2, ls="--")
        ax.spines[["top","right"]].set_visible(False)

        # Ideal direction arrow
        ax.annotate("", xy=(0.08, 0.08), xytext=(0.28, 0.28),
                    xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.4,
                                    mutation_scale=14))
        ax.text(0.06, 0.06, "ideal", transform=ax.transAxes,
                fontsize=8, color="#888", style="italic")

    fig.suptitle(
        "FLOPs vs Cache Pareto: MLA Pays More Compute for Less Memory\n"
        "(analytical decode FLOPs per token, context S=256)",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    _save(fig, "perf_3_flops_vs_cache.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Throughput vs batch size (requires batch_sweep_results.csv)
# ─────────────────────────────────────────────────────────────────────────────
def fig4_batch_throughput(bdf: pd.DataFrame):
    bdf = bdf[~bdf["oom"]].copy()

    dims = sorted(bdf["dim"].unique())
    n_dims = len(dims)

    fig, axes = plt.subplots(1, n_dims, figsize=(8 * n_dims, 6.5), squeeze=False)
    fig.patch.set_facecolor("#F8F9FA")

    for col, dim in enumerate(dims):
        ax  = axes[0, col]
        ax.set_facecolor("#F8F9FA")
        sub = bdf[bdf["dim"] == dim]

        for t in TYPE_ORDER:
            pts = sub[sub["type"] == t].sort_values("batch_size")
            if pts.empty:
                continue
            ax.plot(pts["batch_size"], pts["tps"],
                    marker="o", color=TYPE_COLORS[t], lw=2.5, label=t, zorder=4)
            # Mark max achieved tps
            best = pts.loc[pts["tps"].idxmax()]
            ax.scatter(best["batch_size"], best["tps"],
                       s=120, c=TYPE_COLORS[t], edgecolors="black",
                       lw=1.2, zorder=5)

        ax.set_xlabel("Batch size", fontsize=12)
        ax.set_ylabel("Tokens / sec  (aggregate)", fontsize=12)
        nb = int(sub["num_blocks"].iloc[0]) if not sub.empty else "?"
        ax.set_title(f"dim={dim}, {nb} layers\n"
                     f"seq_len={int(sub['seq_len'].iloc[0]) if not sub.empty else '?'} tokens",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=11, framealpha=0.9)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(bdf["batch_size"].unique()))
        ax.set_xticklabels([str(b) for b in sorted(bdf["batch_size"].unique())])
        ax.grid(alpha=0.2, ls="--")
        ax.spines[["top","right"]].set_visible(False)

    fig.suptitle(
        "Decode Throughput vs Batch Size\n"
        "MLA's smaller KV cache allows more sequences per GPU "
        "→ crossover at larger batches",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    _save(fig, "perf_4_batch_throughput.png")


def fig4_missing():
    """Placeholder if batch sweep hasn't been run yet."""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")
    ax.text(0.5, 0.6,
            "Run the batch sweep first:",
            ha="center", va="center", fontsize=14, fontweight="bold",
            transform=ax.transAxes)
    ax.text(0.5, 0.42,
            "CUDA_VISIBLE_DEVICES=0 python3 benchmark_batch_sweep.py",
            ha="center", va="center", fontsize=12, family="monospace",
            color="#3A86FF", transform=ax.transAxes)
    ax.text(0.5, 0.25,
            "Then re-run plot_perf.py",
            ha="center", va="center", fontsize=11, color="#555",
            transform=ax.transAxes)
    ax.axis("off")
    ax.set_title("Figure 4: Throughput vs Batch Size — data pending",
                 fontsize=13, fontweight="bold")
    _save(fig, "perf_4_batch_throughput.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Prefill latency + theoretical KV cache per context length
# ─────────────────────────────────────────────────────────────────────────────
def fig5_prefill(df: pd.DataFrame):
    dense = df[~df["moe"]].dropna(subset=["prefill_ms", "params_m"]).copy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("#F8F9FA")

    # Left: prefill ms vs params_m, coloured by type
    ax = axes[0]
    ax.set_facecolor("#F8F9FA")
    for t in TYPE_ORDER:
        pts = dense[dense["type"] == t]
        if pts.empty:
            continue
        ax.scatter(pts["params_m"], pts["prefill_ms"],
                   s=70, c=TYPE_COLORS[t], edgecolors="white", lw=0.8,
                   alpha=0.85, zorder=4, label=t)
        # Trend line
        if len(pts) >= 3:
            lx = np.log10(pts["params_m"])
            ly = np.log10(pts["prefill_ms"])
            c  = np.polyfit(lx, ly, 1)
            xf = np.logspace(lx.min() - 0.1, lx.max() + 0.1, 50)
            ax.plot(xf, 10**np.polyval(c, np.log10(xf)),
                    color=TYPE_COLORS[t], lw=2, ls="--", alpha=0.6, zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Model parameters (M)", fontsize=12)
    ax.set_ylabel("Prefill latency (ms)  @ max_context tokens", fontsize=12)
    ax.set_title("Prefill Latency vs Model Size\n(Dense, batch=1)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, which="both", alpha=0.2, ls="--")
    ax.spines[["top","right"]].set_visible(False)
    ax.text(0.97, 0.05,
            "MLA prefill is slower: it runs\nup-projections for every token\n"
            "rather than storing compressed KV.",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color="#555", style="italic",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ccc", lw=1))

    # Right: theoretical cache vs context length for median representative models
    ax2 = axes[1]
    ax2.set_facecolor("#F8F9FA")
    ctx = np.array([512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])
    # Use median bytes-per-token per type
    for t in TYPE_ORDER:
        sub = dense[dense["type"] == t]
        if sub.empty:
            continue
        bpt_median = (sub["theoretical_cache_mb"] * 1e6 / sub["max_context"]).median()
        cache_gb   = bpt_median * ctx / 1e9
        ax2.plot(ctx / 1024, cache_gb, color=TYPE_COLORS[t], lw=2.5, label=t,
                 marker="o", markersize=5)

    for vram, lbl, col in [(80, "80 GB (A100/H100)", "#CC3311"),
                            (24, "24 GB (RTX)", "#EE7733")]:
        ax2.axhline(vram, color=col, lw=1.3, ls="-.", alpha=0.8)
        ax2.text(ctx[-1] / 1024 * 0.97, vram * 1.06, lbl,
                 ha="right", fontsize=9, color=col)

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Context length (K tokens)", fontsize=12)
    ax2.set_ylabel("KV Cache per sequence (GB)", fontsize=12)
    ax2.set_title("KV Cache Growth with Context\n"
                  "(theoretical, median bytes-per-token from experiments)",
                  fontsize=12, fontweight="bold")
    ax2.legend(fontsize=11, framealpha=0.9)
    ax2.grid(True, which="both", alpha=0.2, ls="--")
    ax2.spines[["top","right"]].set_visible(False)

    fig.suptitle("Prefill Cost & KV Cache Scaling",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, "perf_5_prefill.png")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df  = load()
    bdf = load_batch()
    print(f"Loaded {len(df)} benchmark runs")
    if bdf is not None:
        print(f"Loaded {len(bdf)} batch-sweep rows")
    else:
        print("No batch_sweep_results.csv found — fig4 will be a placeholder.")

    print("\nGenerating performance figures...")
    fig1_cache_breakdown(df)
    fig2_seqlen_throughput(df)
    fig3_flops_vs_cache(df)

    if bdf is not None and not bdf.empty:
        fig4_batch_throughput(bdf)
    else:
        fig4_missing()

    fig5_prefill(df)
    print("Done.")

    # Quick summary
    print("\n── Dense model summary ──")
    dense = df[~df["moe"]]
    dense = dense.copy()
    dense["decode_flops_m"] = dense.apply(lambda r: _decode_flops(r, S=256), axis=1) / 1e6
    cols = ["params_m", "theoretical_cache_mb", "prefill_ms",
            "decode_flops_m"] + [f"tps_{s}" for s in SEQ_LENS]
    cols = [c for c in cols if c in dense.columns]
    print(dense.groupby("type")[cols].mean().round(2).to_string())
