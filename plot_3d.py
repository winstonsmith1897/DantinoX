#!/home/marco.simoni/miniconda3/bin/python3
"""
plot_3d.py — 3D visualisations of the MLA vs GQA vs MHA tradeoff.

Run with: /home/marco.simoni/miniconda3/bin/python3 plot_3d.py

Figures:
  3d_1_cache_surface.png   — KV-cache surface per type (params × seq_len → cache GB)
  3d_2_quality_cube.png    — 3D scatter: quality × cache × params
  3d_3_efficiency_cube.png — 3D scatter: speed × cache_efficiency × quality
  3d_4_serving_surface.png — aggregate serving throughput: VRAM budget × seq_len
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # registers the 3d projection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

IN_CSV      = "benchmark_results.csv"
OUT_DIR     = "plots"
DPI         = 210

TYPE_COLORS = {"MLA": "#3A86FF", "GQA": "#FF6B35", "MHA": "#2DC653"}
TYPE_ORDER  = ["MLA", "GQA", "MHA"]
MOE_ALPHA   = {False: 0.88, True: 0.40}
MOE_MARKER  = {False: "o",  True: "^"}

SEQ_GRID    = np.array([512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])
PARAMS_GRID = np.logspace(np.log10(3), np.log10(200), 60)
VRAM_GB     = [(80, "#CC3311", "80 GB  H100"),
               (24, "#EE7733", "24 GB  RTX")]


# ─── helpers ─────────────────────────────────────────────────────────────────
def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, name)
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p}")


def _ax3(fig, pos, elev=22, azim=-50, fc="#F8F9FA"):
    ax = fig.add_subplot(*pos, projection="3d")
    ax.set_facecolor(fc)
    ax.view_init(elev=elev, azim=azim)
    return ax


def load():
    df = pd.read_csv(IN_CSV)
    for c in ["params_m", "theoretical_cache_mb", "val_loss",
              "tps_512", "tps_64", "max_context", "num_blocks",
              "n_heads", "kv_heads", "dim", "down_dim_kv"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["bpt"]      = df["theoretical_cache_mb"] * 1e6 / df["max_context"]
    df["head_size"] = (df["dim"] / df["n_heads"]).round().astype("Int64")
    return df


def _bpt_fn(df, t):
    """
    Returns bpt(params_array) → array  (bytes per token, as float).
    MHA/GQA: log-log fit  bpt ~ a * params^b.
    MLA:     median bpt (decoupled from params).
    """
    sub = df[(df["type"] == t) & (~df["moe"])].dropna(subset=["params_m", "bpt"])
    if sub.empty:
        return lambda p: np.zeros_like(p, dtype=float)
    if t == "MLA":
        m = float(sub["bpt"].median())
        return lambda p: np.full(np.asarray(p).shape, m)
    lx, ly = np.log10(sub["params_m"].values), np.log10(sub["bpt"].values)
    b, la  = np.polyfit(lx, ly, 1)
    a = 10 ** la
    return lambda p: a * np.asarray(p, dtype=float) ** b


def _decode_flops_m(row, S=256):
    """Approximate analytical decode FLOPs (M) for one token."""
    dim  = int(row["dim"]);  n_heads = int(row["n_heads"])
    kv   = int(row["kv_heads"]); h = int(row["head_size"]); nb = int(row["num_blocks"])
    exp  = 4
    mlp  = 2 * dim * exp * dim * 3
    if row["type"] == "MLA":
        dq   = dim // 2
        dkv  = int(row["down_dim_kv"]) if not pd.isna(row["down_dim_kv"]) else dim // 4
        rope = max(16, dkv // 4)
        attn = (2*dim*dq + 4*dim*rope
                + 2*n_heads*dq*dkv*h + 2*n_heads*dq*dkv
                + 2*n_heads*S*dkv   + 2*n_heads*S*rope
                + 2*n_heads*h*dkv*dim + 2*n_heads*dkv*dim)
    else:
        attn = (2*dim*dim + 4*dim*kv*h
                + 4*n_heads*S*h + 2*dim*dim)
    return (attn + mlp) * nb / 1e6


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 — KV-cache surface: params × seq_len → cache (GB)
#          Three subplots + one combined, same z-scale.
# ─────────────────────────────────────────────────────────────────────────────
def fig1_cache_surface(df):
    dense = df[~df["moe"]]
    P2D, S2D = np.meshgrid(PARAMS_GRID, SEQ_GRID)

    surfaces = {}
    z_global = 0.0
    for t in TYPE_ORDER:
        fn   = _bpt_fn(dense, t)
        Z    = fn(P2D.ravel()).reshape(P2D.shape) * S2D / 1e9
        surfaces[t] = Z
        z_global    = max(z_global, float(np.nanmax(Z)))
    z_cap = min(z_global, 300)

    fig = plt.figure(figsize=(22, 10))
    fig.patch.set_facecolor("#F8F9FA")

    # ── Three separate subplots (top row) ───────────────────────────────────
    for col, t in enumerate(TYPE_ORDER):
        ax = _ax3(fig, (2, 4, col + 1), elev=24, azim=-52)
        Z  = np.minimum(surfaces[t], z_cap)
        c  = TYPE_COLORS[t]

        # Surface
        ax.plot_surface(np.log10(P2D), np.log10(S2D / 1024), Z,
                        color=c, alpha=0.55, linewidth=0, antialiased=True)

        # Projected contour on the floor (z=0)
        ax.contourf(np.log10(P2D), np.log10(S2D / 1024), Z,
                    zdir="z", offset=0,
                    levels=np.linspace(0, z_cap, 12),
                    cmap="Blues" if t == "MLA" else
                         "Oranges" if t == "GQA" else "Greens",
                    alpha=0.35)

        # VRAM planes
        lp = np.log10(PARAMS_GRID[[0, -1]])
        ls = np.log10(SEQ_GRID[[0, -1]] / 1024)
        for vram, vc, vlbl in VRAM_GB:
            if vram > z_cap: continue
            verts = [[(lp[0], ls[0], vram), (lp[1], ls[0], vram),
                      (lp[1], ls[1], vram), (lp[0], ls[1], vram)]]
            poly = Poly3DCollection(verts, alpha=0.20)
            poly.set_facecolor(vc); poly.set_edgecolor(vc)
            ax.add_collection3d(poly)
            ax.text(lp[1], ls[0], vram + 2, vlbl,
                    color=vc, fontsize=7, fontweight="bold")

        # Axis ticks & labels
        _set_cache_axes(ax, z_cap)
        ax.set_title(t, fontsize=14, fontweight="bold", color=c, pad=6)

    # ── Combined plot (bottom row, spans all columns) ────────────────────────
    ax_c = _ax3(fig, (2, 1, 2), elev=26, azim=-46)

    for t in TYPE_ORDER:
        Z = np.minimum(surfaces[t], z_cap)
        c = TYPE_COLORS[t]
        ax_c.plot_surface(np.log10(P2D), np.log10(S2D / 1024), Z,
                          color=c, alpha=0.32, linewidth=0, antialiased=True)
        ax_c.plot_wireframe(np.log10(P2D), np.log10(S2D / 1024), Z,
                            color=c, alpha=0.18, linewidth=0.5,
                            rstride=3, cstride=3)

    lp = np.log10(PARAMS_GRID[[0, -1]])
    ls = np.log10(SEQ_GRID[[0, -1]] / 1024)
    for vram, vc, vlbl in VRAM_GB:
        if vram > z_cap: continue
        verts = [[(lp[0], ls[0], vram), (lp[1], ls[0], vram),
                  (lp[1], ls[1], vram), (lp[0], ls[1], vram)]]
        poly = Poly3DCollection(verts, alpha=0.22)
        poly.set_facecolor(vc); poly.set_edgecolor(vc)
        ax_c.add_collection3d(poly)
        ax_c.text(lp[0], ls[1], vram + 3, f"  {vlbl}",
                  color=vc, fontsize=10, fontweight="bold", va="bottom")

    _set_cache_axes(ax_c, z_cap)
    ax_c.set_title("All types — same scale", fontsize=12, fontweight="bold", pad=6)

    # Proxy legend for combined plot
    handles = [mpatches.Patch(color=TYPE_COLORS[t], alpha=0.7, label=t)
               for t in TYPE_ORDER]
    for vram, vc, vlbl in VRAM_GB:
        handles.append(mpatches.Patch(color=vc, alpha=0.45, label=vlbl))
    ax_c.legend(handles=handles, fontsize=10, loc="upper left",
                framealpha=0.9, bbox_to_anchor=(-0.02, 1.0))

    fig.suptitle(
        "KV Cache (GB) = f(Model Parameters, Context Length)\n"
        "MLA surface is flat along the params axis — "
        "growing the model doesn't inflate its cache",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    _save(fig, "3d_1_cache_surface.png")


def _set_cache_axes(ax, z_cap):
    p_ticks = [3, 10, 30, 100]
    s_ticks = [0.512, 1, 4, 16, 64, 128]
    ax.set_xticks([np.log10(v) for v in p_ticks])
    ax.set_xticklabels([f"{v}M" for v in p_ticks], fontsize=7)
    ax.set_yticks([np.log10(v) for v in s_ticks])
    ax.set_yticklabels(["512" if v < 1 else f"{int(v)}K" for v in s_ticks], fontsize=7)
    ax.set_xlabel("Model parameters", fontsize=9, labelpad=6)
    ax.set_ylabel("Context length", fontsize=9, labelpad=6)
    ax.set_zlabel("KV Cache per seq (GB)", fontsize=9, labelpad=8)
    ax.set_zlim(0, z_cap)
    ax.tick_params(axis="z", labelsize=7)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 — 3D scatter: params × KV-cache × val_loss
#          The "quality cube": where do types land in the 3-way tradeoff?
# ─────────────────────────────────────────────────────────────────────────────
def fig2_quality_cube(df):
    sub = df.dropna(subset=["params_m", "theoretical_cache_mb", "val_loss"]).copy()

    fig = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor("#F8F9FA")

    for pi, (moe_v, title) in enumerate([(False, "Dense"), (True, "MoE")]):
        ax = _ax3(fig, (1, 2, pi + 1), elev=20, azim=-55)
        grp = sub[sub["moe"] == moe_v]

        for t in TYPE_ORDER:
            pts = grp[grp["type"] == t]
            if pts.empty: continue
            ax.scatter(
                pts["params_m"],
                pts["theoretical_cache_mb"],
                pts["val_loss"],
                c=TYPE_COLORS[t], s=55,
                marker=MOE_MARKER[moe_v],
                edgecolors="white", linewidths=0.6,
                alpha=0.85, zorder=4, label=t
            )

        # Ideal corner arrow annotation
        ax.text(sub["params_m"].min(), sub["theoretical_cache_mb"].min(),
                sub["val_loss"].min() - 0.04,
                "← ideal", fontsize=9, color="#555", style="italic")

        ax.set_xlabel("Parameters (M)", fontsize=10, labelpad=6)
        ax.set_ylabel("KV Cache (MB)", fontsize=10, labelpad=6)
        ax.set_zlabel("Validation Loss ↓", fontsize=10, labelpad=8)
        ax.set_title(f"{title} — Quality · Cache · Size cube",
                     fontsize=12, fontweight="bold", pad=8)
        ax.tick_params(labelsize=8)

        handles = [mpatches.Patch(color=TYPE_COLORS[t], label=t) for t in TYPE_ORDER
                   if not grp[grp["type"] == t].empty]
        ax.legend(handles=handles, fontsize=10, framealpha=0.9,
                  loc="upper right")

    fig.suptitle(
        "The Quality–Cache–Size Cube\n"
        "Lower-left-front corner = ideal: fewer params, smaller cache, better quality",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    _save(fig, "3d_2_quality_cube.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 — 3D scatter: throughput × cache-efficiency × quality
#          All three axes: higher = better.
#          cache-efficiency = tps / cache_mb  (tok/s per MB of KV cache)
#          Upper-right-front corner = Pareto-optimal.
# ─────────────────────────────────────────────────────────────────────────────
def fig3_efficiency_cube(df):
    sub = df.dropna(subset=["tps_512", "theoretical_cache_mb", "val_loss"]).copy()
    sub["cache_eff"] = sub["tps_512"] / sub["theoretical_cache_mb"]
    sub["quality"]   = 1.0 / sub["val_loss"]   # higher = better

    fig = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor("#F8F9FA")

    for pi, (moe_v, title) in enumerate([(False, "Dense"), (True, "MoE")]):
        ax = _ax3(fig, (1, 2, pi + 1), elev=22, azim=-48)
        grp = sub[sub["moe"] == moe_v]

        for t in TYPE_ORDER:
            pts = grp[grp["type"] == t]
            if pts.empty: continue
            sz = (pts["params_m"] / sub["params_m"].max() * 300).clip(lower=20)
            ax.scatter(
                pts["tps_512"],
                pts["cache_eff"],
                pts["quality"],
                c=TYPE_COLORS[t], s=sz,
                marker=MOE_MARKER[moe_v],
                edgecolors="white", linewidths=0.6,
                alpha=0.85, zorder=4, label=t
            )

        # Ideal corner
        ax.text(grp["tps_512"].max() * 0.95,
                grp["cache_eff"].max() * 0.95,
                grp["quality"].max() * 1.01,
                "ideal ↗", fontsize=9, color="#555", style="italic")

        ax.set_xlabel("Decode Throughput (tok/s) ↑", fontsize=9, labelpad=6)
        ax.set_ylabel("Cache Efficiency\n(tok/s per MB cache) ↑", fontsize=9, labelpad=8)
        ax.set_zlabel("Quality  (1/val_loss) ↑", fontsize=9, labelpad=8)
        ax.set_title(f"{title} — Efficiency cube\n(bubble size ∝ params)",
                     fontsize=11, fontweight="bold", pad=8)
        ax.tick_params(labelsize=8)

        handles = [mpatches.Patch(color=TYPE_COLORS[t], label=t) for t in TYPE_ORDER
                   if not grp[grp["type"] == t].empty]
        ax.legend(handles=handles, fontsize=10, framealpha=0.9, loc="upper left")

    fig.suptitle(
        "Efficiency Cube: Speed × Cache Efficiency × Quality\n"
        "All three axes ↑ = better. Upper-right-front = Pareto-optimal.",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    _save(fig, "3d_3_efficiency_cube.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 — Serving surface: VRAM budget × seq_len → aggregate tok/s
#          Shows at what (budget, context) each type breaks down.
# ─────────────────────────────────────────────────────────────────────────────
def fig4_serving_surface(df):
    dense = df[~df["moe"]]

    budgets_gb = np.logspace(np.log10(1), np.log10(80), 60)   # 1–80 GB
    seq_lens   = SEQ_GRID

    B2D, S2D = np.meshgrid(budgets_gb, seq_lens)

    fig = plt.figure(figsize=(22, 8))
    fig.patch.set_facecolor("#F8F9FA")

    surfaces = {}
    z_cap = 0.0

    for t in TYPE_ORDER:
        fn_bpt = _bpt_fn(dense, t)
        # median tps at 512 — we'll scale it by 512/S to approximate seqlen effect
        tps_512_med = float(dense[dense["type"] == t]["tps_512"].median())

        # bpt at MEDIAN params (the representative model)
        bpt_med = float(dense[dense["type"] == t]["bpt"].median())

        # Cache per seq (GB) at each seq_len
        cache_per_seq_gb = bpt_med * S2D / 1e9

        # Max concurrent sequences in budget
        n_seq = B2D / cache_per_seq_gb

        # Throughput approximation: tps scales slightly with seq_len (from our data)
        # Use measured ratios: tps doesn't change much with seqlen in our data
        # so we use tps_512_med as a flat estimate
        total_tps = n_seq * tps_512_med / 1e3  # k tok/s

        # Clip where cache_per_seq > budget (can't fit even 1 sequence)
        total_tps = np.where(cache_per_seq_gb > B2D, 0.0, total_tps)
        surfaces[t] = total_tps
        z_cap = max(z_cap, float(np.nanmax(total_tps)))

    z_cap = min(z_cap, 5000)   # k tok/s cap

    for col, t in enumerate(TYPE_ORDER):
        ax = _ax3(fig, (1, 3, col + 1), elev=26, azim=-52)
        Z  = np.minimum(surfaces[t], z_cap)
        c  = TYPE_COLORS[t]

        ax.plot_surface(np.log10(B2D), np.log10(S2D / 1024), Z,
                        color=c, alpha=0.58, linewidth=0, antialiased=True)

        # Floor projection
        ax.contourf(np.log10(B2D), np.log10(S2D / 1024), Z,
                    zdir="z", offset=0,
                    levels=np.linspace(0, z_cap, 10),
                    cmap="Blues" if t == "MLA" else
                         "Oranges" if t == "GQA" else "Greens",
                    alpha=0.40)

        # Budget reference lines (vertical planes at 24 GB and 80 GB)
        ls = np.log10(seq_lens[[0, -1]] / 1024)
        for gb, vc, vlbl in VRAM_GB:
            x_v = np.log10(gb)
            z_line = np.minimum(surfaces[t][:, np.argmin(np.abs(budgets_gb - gb))], z_cap)
            ax.plot([x_v] * len(seq_lens), np.log10(seq_lens / 1024), z_line,
                    color=vc, lw=2.2, alpha=0.9, zorder=5)
            ax.text(x_v, ls[0], z_line[0] + z_cap * 0.02,
                    vlbl, color=vc, fontsize=7, fontweight="bold")

        # Axes
        b_ticks = [1, 5, 10, 40, 80]
        s_ticks = [0.512, 1, 4, 16, 64, 128]
        ax.set_xticks([np.log10(v) for v in b_ticks])
        ax.set_xticklabels([f"{v}GB" for v in b_ticks], fontsize=7)
        ax.set_yticks([np.log10(v) for v in s_ticks])
        ax.set_yticklabels(["512" if v < 1 else f"{int(v)}K" for v in s_ticks], fontsize=7)
        ax.set_zlim(0, z_cap)
        ax.tick_params(axis="z", labelsize=7)
        ax.set_xlabel("KV-cache VRAM budget", fontsize=9, labelpad=6)
        ax.set_ylabel("Context length", fontsize=9, labelpad=6)
        ax.set_zlabel("Aggregate throughput (k tok/s)", fontsize=9, labelpad=8)
        ax.set_title(t, fontsize=14, fontweight="bold", color=c, pad=6)

    fig.suptitle(
        "Aggregate Serving Throughput = f(VRAM Budget, Context Length)\n"
        "The coloured lines mark throughput at 24 GB / 80 GB GPU. "
        "MLA's surface stays higher across every (budget, context) point.",
        fontsize=12, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    _save(fig, "3d_4_serving_surface.png")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    df = load()
    print(f"Loaded {len(df)} runs")
    print("\nGenerating 3D figures...")
    fig1_cache_surface(df)
    fig2_quality_cube(df)
    fig3_efficiency_cube(df)
    fig4_serving_surface(df)
    print("Done.")
