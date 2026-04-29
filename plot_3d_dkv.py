#!/home/marco.simoni/miniconda3/bin/python3
"""
plot_3d_dkv.py — 3D surfaces focused on down_dim_kv (MLA's latent KV dimension).

Run with: /home/marco.simoni/miniconda3/bin/python3 plot_3d_dkv.py

Figures:
  3d_5_dkv_cache_seqlen.png   — MLA: down_dim_kv × seq_len → KV cache
                                  (with GQA/MHA fixed reference planes)
  3d_6_kv_decoupling.png      — kv_dim_eff × params_m → cache
                                  (MLA cluster is flat; MHA/GQA cluster rises)
  3d_7_mla_quality.png        — MLA: down_dim_kv × params_m → val_loss quality landscape
  3d_8_dkv_num_blocks.png     — MLA: down_dim_kv × num_blocks → cache at multiple seq_lens
"""

import os
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.cm as cm

IN_CSV      = "benchmark_results.csv"
OUT_DIR     = "plots"
DPI         = 210

TYPE_COLORS = {"MLA": "#3A86FF", "GQA": "#FF6B35", "MHA": "#2DC653"}
TYPE_ORDER  = ["MLA", "GQA", "MHA"]
SEQ_GRID    = np.array([512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])
NB_REF      = 12          # representative num_blocks for surfaces
VRAM_GB     = [(80, "#CC3311", "80 GB  H100"),
               (24, "#EE7733", "24 GB  RTX")]


def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, name)
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p}")


def _ax3(fig, pos, elev=24, azim=-52, fc="#F8F9FA"):
    ax = fig.add_subplot(*pos, projection="3d")
    ax.set_facecolor(fc)
    ax.view_init(elev=elev, azim=azim)
    return ax


def load():
    df = pd.read_csv(IN_CSV)
    num_cols = ["params_m", "theoretical_cache_mb", "val_loss",
                "tps_512", "max_context", "num_blocks",
                "n_heads", "kv_heads", "dim", "down_dim_kv"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Effective KV dimension: the quantity that directly sets the cache.
    # cache_bytes = kv_dim_eff × num_blocks × seq_len × 4  (fp32)
    df["kv_dim_eff"] = (
        df["theoretical_cache_mb"] * 1e6
        / (df["num_blocks"] * df["max_context"] * 4)
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5 — MLA surface: down_dim_kv × seq_len → KV cache (GB)
#          GQA and MHA shown as horizontal reference planes.
# ─────────────────────────────────────────────────────────────────────────────
def fig5_dkv_cache_seqlen(df):
    """
    For MLA, cache scales with (down_dim_kv + rope_dim) × num_blocks × seq_len.
    kv_dim_eff(down_dim_kv) is fitted from actual run data (it absorbs rope_dim).
    GQA/MHA reference: their median kv_dim_eff at num_blocks=NB_REF.
    """
    mla   = df[(df["type"] == "MLA") & (~df["moe"])].dropna(subset=["down_dim_kv", "kv_dim_eff"])
    dense = df[~df["moe"]]

    # Fit kv_dim_eff vs down_dim_kv for MLA (linear — nearly perfect)
    agg_mla = mla.groupby("down_dim_kv")["kv_dim_eff"].mean()
    dkv_pts = agg_mla.index.values.astype(float)
    kve_pts = agg_mla.values
    slope, intercept = np.polyfit(dkv_pts, kve_pts, 1)   # kv_dim_eff ≈ slope*dkv + intercept

    # Surface grid
    dkv_range = np.linspace(dkv_pts.min(), 256, 60)
    S_range   = SEQ_GRID
    DKV2D, S2D = np.meshgrid(dkv_range, S_range)

    kve_2d = slope * DKV2D + intercept
    Z_mla  = kve_2d * NB_REF * S2D * 4 / 1e9   # cache in GB

    # GQA/MHA reference kv_dim_eff at num_blocks=NB_REF
    ref_cache = {}
    for t in ["GQA", "MHA"]:
        sub = dense[(dense["type"] == t) & (dense["num_blocks"] == NB_REF)]
        if sub.empty:
            sub = dense[dense["type"] == t]
        ref_cache[t] = float(sub["kv_dim_eff"].median()) * NB_REF * S_range * 4 / 1e9

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 8))
    fig.patch.set_facecolor("#F8F9FA")
    ax  = _ax3(fig, (1, 1, 1), elev=26, azim=-55)

    # MLA surface
    surf = ax.plot_surface(DKV2D, np.log10(S2D / 1024), Z_mla,
                           color=TYPE_COLORS["MLA"], alpha=0.55,
                           linewidth=0, antialiased=True, zorder=3)

    # Actual MLA scatter points at their real seq_len=max_context
    for _, row in mla.iterrows():
        z_pt = row["kv_dim_eff"] * row["num_blocks"] * row["max_context"] * 4 / 1e9
        ax.scatter(row["down_dim_kv"], np.log10(row["max_context"] / 1024), z_pt,
                   c=TYPE_COLORS["MLA"], s=30, edgecolors="white", lw=0.5,
                   alpha=0.80, zorder=6)

    # GQA / MHA reference planes (constant in down_dim_kv direction)
    dkv_lims = [dkv_range[0], dkv_range[-1]]
    log_s    = np.log10(S_range[[0, -1]] / 1024)
    for t, vcol in [("GQA", TYPE_COLORS["GQA"]), ("MHA", TYPE_COLORS["MHA"])]:
        cache_lo = ref_cache[t][0]
        cache_hi = ref_cache[t][-1]
        for si, sl in enumerate(S_range):
            z_val = ref_cache[t][si]
            if z_val > ax.get_zlim()[1] if ax.get_zlim()[1] > 0 else True:
                pass
            ax.plot(dkv_lims, [np.log10(sl / 1024)] * 2, [z_val, z_val],
                    color=vcol, lw=1.2, alpha=0.55, zorder=4)

        # One bold reference line at 512 tokens
        ax.plot(dkv_lims, [np.log10(0.512)] * 2,
                [ref_cache[t][0]] * 2,
                color=vcol, lw=2.5, alpha=0.85, zorder=5,
                label=f"{t} cache @ any dkv (num_blocks={NB_REF})")

    # Scatter actual data points for fitted line (validation)
    ax.plot(dkv_pts, [np.log10(0.512)] * len(dkv_pts),
            kve_pts * NB_REF * 512 * 4 / 1e9,
            "o", color=TYPE_COLORS["MLA"], markersize=6,
            markeredgecolor="white", lw=0, zorder=7,
            label="MLA measured (512 tokens)")

    # VRAM planes
    z_now = float(np.nanmax(Z_mla))
    for vram, vc, vlbl in VRAM_GB:
        if vram > z_now: continue
        verts = [[(dkv_lims[0], log_s[0], vram), (dkv_lims[1], log_s[0], vram),
                  (dkv_lims[1], log_s[1], vram), (dkv_lims[0], log_s[1], vram)]]
        poly = Poly3DCollection(verts, alpha=0.18)
        poly.set_facecolor(vc); poly.set_edgecolor(vc)
        ax.add_collection3d(poly)
        ax.text(dkv_lims[1], log_s[1], vram + 0.3, f"  {vlbl}",
                color=vc, fontsize=9, fontweight="bold")

    # Axes
    s_ticks = [0.512, 1, 4, 16, 64, 128]
    ax.set_yticks([np.log10(v) for v in s_ticks])
    ax.set_yticklabels(["512" if v < 1 else f"{int(v)}K" for v in s_ticks], fontsize=8)
    ax.set_xlabel("down_dim_kv  (MLA latent KV dim)", fontsize=10, labelpad=8)
    ax.set_ylabel("Context length", fontsize=10, labelpad=8)
    ax.set_zlabel("KV Cache per seq (GB)", fontsize=10, labelpad=10)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="z", labelsize=8)

    # Annotation on the surface
    ax.text2D(0.02, 0.92,
              f"MLA surface: cache = f(down_dim_kv, seq_len)\n"
              f"kv_dim_eff ≈ {slope:.2f} × down_dim_kv + {intercept:.1f}  (num_blocks={NB_REF})\n"
              "GQA/MHA lines: their cache is fixed regardless of down_dim_kv",
              transform=ax.transAxes, fontsize=8.5, color="#333",
              bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ccc", lw=0.8, alpha=0.9))

    handles = [
        mpatches.Patch(color=TYPE_COLORS["MLA"], alpha=0.6, label="MLA surface"),
        Line2D([0],[0], color=TYPE_COLORS["GQA"], lw=2.5, label=f"GQA reference (num_blocks={NB_REF})"),
        Line2D([0],[0], color=TYPE_COLORS["MHA"], lw=2.5, label=f"MHA reference (num_blocks={NB_REF})"),
    ] + [mpatches.Patch(color=vc, alpha=0.4, label=vlbl) for _, vc, vlbl in VRAM_GB]
    ax.legend(handles=handles, fontsize=9, framealpha=0.9,
              loc="upper left", bbox_to_anchor=(-0.01, 1.0))

    ax.set_title(
        "MLA: down_dim_kv × Context Length → KV Cache\n"
        "Slide down_dim_kv to land below GQA or MHA cache levels at any context",
        fontsize=12, fontweight="bold", pad=12
    )
    plt.tight_layout()
    _save(fig, "3d_5_dkv_cache_seqlen.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6 — kv_dim_eff × params_m → KV cache: the decoupling plot
#          MLA: kv_dim_eff is flat vs params (free parameter)
#          GQA/MHA: kv_dim_eff rises with params (locked to architecture)
# ─────────────────────────────────────────────────────────────────────────────
def fig6_kv_decoupling(df):
    dense = df[~df["moe"]].dropna(subset=["params_m", "kv_dim_eff", "theoretical_cache_mb"])

    fig = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor("#F8F9FA")

    # ── Left: 3D scatter kv_dim_eff × params_m × cache ──────────────────────
    ax = _ax3(fig, (1, 2, 1), elev=20, azim=-50)

    for t in TYPE_ORDER:
        pts = dense[dense["type"] == t]
        if pts.empty: continue
        ax.scatter(pts["params_m"], pts["kv_dim_eff"],
                   pts["theoretical_cache_mb"],
                   c=TYPE_COLORS[t], s=60,
                   edgecolors="white", lw=0.6,
                   alpha=0.85, zorder=4, label=t)

    # Trend surfaces (fit per type)
    params_fit = np.linspace(dense["params_m"].min(), dense["params_m"].max(), 40)
    for t in TYPE_ORDER:
        pts = dense[dense["type"] == t]
        if len(pts) < 4: continue
        lx = np.log10(pts["params_m"])
        ly = pts["kv_dim_eff"]
        b, a = np.polyfit(lx, ly, 1)
        kve_fit = b * np.log10(params_fit) + a

        # Surface at fixed seq_len=512: cache = kv_dim_eff × num_blocks × 512 × 4 / 1e6
        # Use median num_blocks per type
        nb_med = float(pts["num_blocks"].median())
        cache_fit = kve_fit * nb_med * 512 * 4 / 1e6

        ax.plot(params_fit, kve_fit, cache_fit,
                color=TYPE_COLORS[t], lw=2.2, ls="--", alpha=0.65, zorder=3)

    ax.set_xlabel("Model parameters (M)", fontsize=9, labelpad=6)
    ax.set_ylabel("kv_dim_eff\n(bytes / token / layer / 4)", fontsize=9, labelpad=8)
    ax.set_zlabel("KV Cache (MB) @ 512 tok", fontsize=9, labelpad=8)
    ax.tick_params(labelsize=7)
    ax.set_title("kv_dim_eff × Params × Cache\n"
                 "MLA: flat vs params — GQA/MHA: rises",
                 fontsize=11, fontweight="bold", pad=8)
    handles = [mpatches.Patch(color=TYPE_COLORS[t], label=t) for t in TYPE_ORDER]
    ax.legend(handles=handles, fontsize=10, framealpha=0.9)

    # ── Right: 2D projection kv_dim_eff vs params (the core message) ─────────
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_facecolor("#F8F9FA")

    for t in TYPE_ORDER:
        pts = dense[dense["type"] == t]
        if pts.empty: continue
        ax2.scatter(pts["params_m"], pts["kv_dim_eff"],
                    c=TYPE_COLORS[t], s=65, edgecolors="white", lw=0.7,
                    alpha=0.85, zorder=4, label=t)
        if len(pts) >= 4:
            lx = np.log10(pts["params_m"])
            b, a = np.polyfit(lx, pts["kv_dim_eff"], 1)
            xf = np.logspace(lx.min() - 0.05, lx.max() + 0.05, 80)
            ax2.plot(xf, b * np.log10(xf) + a,
                     color=TYPE_COLORS[t], lw=2.2, ls="--", alpha=0.65)

    # Annotate slopes
    ax2.text(0.98, 0.96,
             "MHA:  kv_dim_eff ∝ dim  (locked to width)\n"
             "GQA:  kv_dim_eff ∝ dim × (kv_heads/n_heads)\n"
             "MLA:  kv_dim_eff = down_dim_kv  (free param)\n\n"
             "→ MLA's cache doesn't grow with the model",
             transform=ax2.transAxes, ha="right", va="top",
             fontsize=9.5, color="#333", style="italic",
             bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#ccc", lw=1, alpha=0.92))

    ax2.set_xscale("log")
    ax2.set_xlabel("Model parameters (M)", fontsize=11)
    ax2.set_ylabel("kv_dim_eff  (effective KV dim per layer)", fontsize=11)
    ax2.set_title("The Decoupling: kv_dim_eff vs Model Size\n"
                  "MHA/GQA locked to architecture — MLA freely chosen",
                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=10, framealpha=0.9)
    ax2.grid(alpha=0.2, ls="--")
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "kv_dim_eff: The Root Cause of MLA's Cache Advantage\n"
        "For MHA/GQA it is an architectural constant. For MLA it is a free hyperparameter.",
        fontsize=12, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    _save(fig, "3d_6_kv_decoupling.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 7 — MLA: down_dim_kv × params_m → val_loss  (quality landscape)
#          Both levers affect quality; this surface shows the full tradeoff.
# ─────────────────────────────────────────────────────────────────────────────
def fig7_mla_quality(df):
    mla = df[(df["type"] == "MLA") & (~df["moe"])].dropna(
        subset=["down_dim_kv", "params_m", "val_loss"]
    ).copy()

    # Reference: Dense GQA/MHA median val_loss (what MLA competes against)
    ref = df[~df["moe"]].dropna(subset=["val_loss"])
    ref_mha = float(ref[ref["type"] == "MHA"]["val_loss"].median())
    ref_gqa = float(ref[ref["type"] == "GQA"]["val_loss"].median())

    xs = mla["params_m"].values
    ys = mla["down_dim_kv"].values
    zs = mla["val_loss"].values

    # Dense interpolated surface
    xi = np.linspace(xs.min(), xs.max(), 50)
    yi = np.linspace(ys.min(), ys.max(), 50)
    XI, YI = np.meshgrid(xi, yi)
    ZI = griddata((xs, ys), zs, (XI, YI), method="linear")

    fig = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor("#F8F9FA")

    # ── Left: 3D surface ─────────────────────────────────────────────────────
    ax = _ax3(fig, (1, 2, 1), elev=22, azim=-48)

    # Interpolated surface coloured by quality
    valid = ~np.isnan(ZI)
    vmin, vmax = float(np.nanmin(ZI)), float(np.nanmax(ZI))
    surf = ax.plot_surface(XI, YI, ZI,
                           facecolors=cm.RdYlGn_r((ZI - vmin) / (vmax - vmin + 1e-9)),
                           alpha=0.55, linewidth=0, antialiased=True, zorder=3)

    # Actual data scatter
    ax.scatter(xs, ys, zs, c=zs, cmap="RdYlGn_r",
               vmin=vmin, vmax=vmax,
               s=55, edgecolors="white", lw=0.6, alpha=0.90, zorder=6)

    # MHA/GQA reference planes
    xl = [xs.min(), xs.max()]
    yl = [ys.min(), ys.max()]
    for ref_val, rcol, rlbl in [(ref_mha, TYPE_COLORS["MHA"], "MHA median"),
                                 (ref_gqa, TYPE_COLORS["GQA"], "GQA median")]:
        verts = [[(xl[0], yl[0], ref_val), (xl[1], yl[0], ref_val),
                  (xl[1], yl[1], ref_val), (xl[0], yl[1], ref_val)]]
        poly = Poly3DCollection(verts, alpha=0.20)
        poly.set_facecolor(rcol); poly.set_edgecolor(rcol)
        ax.add_collection3d(poly)
        ax.text(xl[1], yl[1], ref_val + 0.01, f"  {rlbl}",
                color=rcol, fontsize=8, fontweight="bold")

    ax.set_xlabel("Parameters (M)", fontsize=9, labelpad=6)
    ax.set_ylabel("down_dim_kv", fontsize=9, labelpad=6)
    ax.set_zlabel("Validation Loss ↓", fontsize=9, labelpad=8)
    ax.tick_params(labelsize=7)
    ax.set_title("MLA Quality Landscape\n(surface colour: red=worse, green=better)",
                 fontsize=11, fontweight="bold", pad=8)

    # Colorbar proxy
    sm = cm.ScalarMappable(cmap="RdYlGn_r",
                           norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.1, label="val_loss")

    # ── Right: top-down heatmap (cleaner reading) ─────────────────────────────
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_facecolor("#F8F9FA")

    # Scatter (actual runs)
    sc = ax2.scatter(xs, ys, c=zs, cmap="RdYlGn_r",
                     vmin=vmin, vmax=vmax,
                     s=120, edgecolors="white", lw=1, alpha=0.92, zorder=4)

    # Contour lines on interpolated grid
    CS = ax2.contour(XI, YI, ZI, levels=8, cmap="RdYlGn_r",
                     vmin=vmin, vmax=vmax, alpha=0.55, linewidths=1.2)
    ax2.clabel(CS, fmt="%.3f", fontsize=8, inline=True)

    # MHA/GQA reference horizontal bands
    for i, (ref_val, rcol, rlbl) in enumerate([(ref_mha, TYPE_COLORS["MHA"], "MHA median"),
                                               (ref_gqa, TYPE_COLORS["GQA"], "GQA median")]):
        ax2.text(xs.max() * 1.01, ys.max() * (0.98 - i * 0.08),
                 f"{rlbl}: {ref_val:.3f}",
                 ha="left", fontsize=8, color=rcol, fontweight="bold")

    ax2.set_xlabel("Parameters (M)", fontsize=11)
    ax2.set_ylabel("down_dim_kv", fontsize=11)
    ax2.set_title("Top-down View: Quality Heatmap\n"
                  "(green = lower loss = better quality)",
                  fontsize=11, fontweight="bold")
    ax2.grid(alpha=0.15, ls="--")
    ax2.spines[["top", "right"]].set_visible(False)
    plt.colorbar(sc, ax=ax2, label="Validation Loss ↓")

    fig.suptitle(
        "MLA Quality Landscape: down_dim_kv × Model Size → Validation Loss\n"
        "Both levers matter — but params_m drives quality more than down_dim_kv",
        fontsize=12, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    _save(fig, "3d_7_mla_quality.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 8 — MLA: down_dim_kv × num_blocks → KV cache at multiple seq_lens
#          Two independent knobs that both drive cache independently.
# ─────────────────────────────────────────────────────────────────────────────
def fig8_dkv_numblocks(df):
    mla = df[(df["type"] == "MLA") & (~df["moe"])].dropna(
        subset=["down_dim_kv", "num_blocks", "kv_dim_eff"]
    )

    # Fit: kv_dim_eff ≈ slope * down_dim_kv + intercept
    agg  = mla.groupby("down_dim_kv")["kv_dim_eff"].mean()
    sl, ic = np.polyfit(agg.index.values.astype(float), agg.values, 1)

    dkv_range = np.linspace(32, 256, 50)
    nb_range  = np.array([4, 8, 12, 16, 24, 32])     # include realistic LLM depths

    DKV2D, NB2D = np.meshgrid(dkv_range, nb_range)
    kve_2d = sl * DKV2D + ic

    seq_levels = [512, 4096, 32768, 131072]
    seq_labels = ["512", "4K", "32K", "128K"]
    colors_seq = ["#3A86FF", "#8338EC", "#FF006E", "#FB5607"]

    fig = plt.figure(figsize=(18, 9))
    fig.patch.set_facecolor("#F8F9FA")

    # ── One subplot per seq_len ───────────────────────────────────────────────
    for pi, (sl_v, sl_lbl, sl_col) in enumerate(zip(seq_levels, seq_labels, colors_seq)):
        ax = _ax3(fig, (1, 4, pi + 1), elev=28, azim=-55)
        Z  = kve_2d * NB2D * sl_v * 4 / 1e9   # GB

        surf = ax.plot_surface(DKV2D, NB2D, Z,
                               color=sl_col, alpha=0.50,
                               linewidth=0, antialiased=True, zorder=3)

        # Floor heatmap
        ax.contourf(DKV2D, NB2D, Z, zdir="z", offset=0,
                    levels=np.linspace(0, float(np.nanmax(Z)), 10),
                    cmap="Blues", alpha=0.40)

        # VRAM planes
        for vram, vc, vlbl in VRAM_GB:
            z_cap = float(np.nanmax(Z))
            if vram > z_cap: continue
            dkv_lims = [dkv_range[0], dkv_range[-1]]
            nb_lims  = [nb_range[0], nb_range[-1]]
            verts = [[(dkv_lims[0], nb_lims[0], vram),
                      (dkv_lims[1], nb_lims[0], vram),
                      (dkv_lims[1], nb_lims[1], vram),
                      (dkv_lims[0], nb_lims[1], vram)]]
            poly = Poly3DCollection(verts, alpha=0.22)
            poly.set_facecolor(vc); poly.set_edgecolor(vc)
            ax.add_collection3d(poly)
            ax.text(dkv_lims[1], nb_lims[0], vram + float(np.nanmax(Z)) * 0.02,
                    f" {vlbl}", color=vc, fontsize=7, fontweight="bold")

        # Actual data points (projected to the nearest seq_len)
        for _, row in mla.iterrows():
            nb_v = int(row["num_blocks"])
            if nb_v not in nb_range: continue
            z_pt = row["kv_dim_eff"] * nb_v * sl_v * 4 / 1e9
            ax.scatter(row["down_dim_kv"], nb_v, z_pt,
                       c="white", s=22, edgecolors=sl_col, lw=1.2,
                       alpha=0.85, zorder=7)

        ax.set_xlabel("down_dim_kv", fontsize=8, labelpad=5)
        ax.set_ylabel("num_blocks", fontsize=8, labelpad=5)
        ax.set_zlabel("KV Cache (GB)", fontsize=8, labelpad=7)
        ax.tick_params(labelsize=7)
        ax.set_title(f"seq = {sl_lbl}", fontsize=12,
                     fontweight="bold", color=sl_col, pad=6)

    fig.suptitle(
        "MLA KV Cache = f(down_dim_kv, num_blocks)  — shown at four context lengths\n"
        "Both knobs are tunable: reduce down_dim_kv OR use fewer layers to save cache",
        fontsize=12, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    _save(fig, "3d_8_dkv_num_blocks.png")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    df = load()
    print(f"Loaded {len(df)} runs")
    print("\nGenerating down_dim_kv 3D figures...")
    fig5_dkv_cache_seqlen(df)
    fig6_kv_decoupling(df)
    fig7_mla_quality(df)
    fig8_dkv_numblocks(df)
    print("Done.")
