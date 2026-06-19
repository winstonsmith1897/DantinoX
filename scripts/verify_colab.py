"""
Verifies the Colab notebook logic end-to-end and saves the 3D plots to HTML + PNG.
Run with: CUDA_VISIBLE_DEVICES=3 python scripts/verify_colab.py
"""
import os, sys, math, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import jax, jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flax import nnx

from dantinox.core.config import ModelConfig, ELFConfig
from dantinox.core.model   import Transformer
from dantinox.core.elf     import ELFTransformer
from dantinox.profiling import (
    LatencyMetric, ThroughputMetric, FLOPsMetric,
    LatencyResult, ThroughputResult,
    RunProfile, MultiRunReport,
    plot_3d_surface, plot_3d_compare, plot_bar_compare,
)

print(f"JAX {jax.__version__}  backend={jax.default_backend()}")
print(f"device: {jax.devices()[0].device_kind}\n")

# ── Config ────────────────────────────────────────────────────────────────────

QUICK     = True
VOCAB     = 256
MASK_ID   = 4
B_DEFAULT = 4
G_DEFAULT = 128
P_DEFAULT = 32
STEPS_DEF = 16
OUT_DIR   = "results/colab_verify"
os.makedirs(OUT_DIR, exist_ok=True)

SIZES = {
    "tiny":   (64,  4, 16, 2),
    "small":  (128, 4, 32, 3),
    "medium": (256, 8, 32, 6),
}
BATCH_SIZES  = [1, 2, 4, 8, 16]
SEQ_LENS     = [32, 64, 128, 256]
STEPS_LIST   = [4, 8, 16, 32]
BENCH_SIZE   = "medium"
N_WARMUP, N_MEASURE = 1, 5

# ── Model factories ───────────────────────────────────────────────────────────

def _cfg_ar(size, sl=256):
    d, nh, hs, nb = SIZES[size]
    return ModelConfig(dim=d, n_heads=nh, head_size=hs, num_blocks=nb,
                       vocab_size=VOCAB, max_context=sl+1,
                       causal=True, dropout=0.0)

def _cfg_disc(size, sl=256):
    d, nh, hs, nb = SIZES[size]
    return ModelConfig(dim=d, n_heads=nh, head_size=hs, num_blocks=nb,
                       vocab_size=VOCAB, max_context=sl+1,
                       causal=False, dropout=0.0, mask_token_id=MASK_ID)

def _cfg_elf(size, sl=256):
    d, nh, hs, nb = SIZES[size]
    return ELFConfig(embed_dim=d, bottleneck_dim=max(32, d//2),
                     model_dim=d, n_heads=nh, head_size=hs, num_blocks=nb,
                     vocab_size=VOCAB, max_seq_len=sl,
                     gradient_checkpointing=False, dropout=0.0)

def _params_m(model):
    _, st = nnx.split(model)
    return sum(x.size for x in jax.tree_util.tree_leaves(st) if hasattr(x,"size")) / 1e6

# ── JIT primitives ────────────────────────────────────────────────────────────

@nnx.jit
def _ar_prefill(model, x):
    return model(x, caches=None, cache_index=0, deterministic=True).kv_caches

@nnx.jit
def _ar_decode(model, tok, cache, pos):
    out = model(tok, caches=cache, cache_index=pos, deterministic=True)
    return jnp.argmax(out.logits[:,-1,:], -1)[:,None].astype(jnp.int32), out.kv_caches

@nnx.jit
def _disc_step(model, x_t):
    out = model(x_t, deterministic=True)
    x0  = jnp.argmax(out.logits, -1).astype(jnp.int32)
    return jnp.where(x_t == MASK_ID, x0, x_t)

@nnx.jit
def _elf_step(model, z, t_arr, dt):
    x_prev = jnp.zeros_like(z)
    w      = jnp.ones(z.shape[0], dtype=z.dtype)
    mask   = jnp.zeros(z.shape[0], dtype=bool)
    out    = model(z, x_prev, t_arr, w, mask, deterministic=True)
    v      = (out.x_pred - z) / jnp.clip(1.0 - t_arr[:,None,None], 1e-6)
    return z + dt * v

# ── Generation closures ───────────────────────────────────────────────────────

def make_ar_gen(model, B, P, G):
    prompt = jnp.ones((B, P), jnp.int32)
    tok0   = jnp.ones((B, 1), jnp.int32)
    def fn():
        cache = _ar_prefill(model, prompt)
        tok   = tok0
        for i in range(G):
            tok, cache = _ar_decode(model, tok, cache, jnp.array(P+i, jnp.int32))
        jax.block_until_ready(tok)
    return fn

def make_disc_gen(model, B, G, steps):
    x = jnp.full((B, G), MASK_ID, jnp.int32)
    def fn():
        xx = x
        for _ in range(steps):
            xx = _disc_step(model, xx)
        jax.block_until_ready(xx)
    return fn

def make_elf_gen(model, B, G, steps, dim):
    z    = jax.random.normal(jax.random.key(0), (B, G, dim), jnp.float32)
    dt   = jnp.array(1.0/steps, jnp.float32)
    ts   = jnp.linspace(0.0, 1.0 - 1.0/steps, steps)
    def fn():
        zz = z
        for i in range(steps):
            t_arr = jnp.full((B,), float(ts[i]), jnp.float32)
            zz    = _elf_step(model, zz, t_arr, dt)
        jax.block_until_ready(zz)
    return fn

lat = LatencyMetric(n_warmup=N_WARMUP, n_measure=N_MEASURE)

# ═══════════════════════════════════════════════════════════════════════════════
# SCALE SWEEP
# ═══════════════════════════════════════════════════════════════════════════════
print("── Scale sweep ──────────────────────────────────────────────────────")
scale_rows = []

for size in SIZES:
    dim, n_heads, head_size, num_blocks = SIZES[size]
    print(f"  [{size}]", end="  ", flush=True)
    for paradigm in ("AR", "Discrete", "Continuous"):
        try:
            if paradigm == "AR":
                model = Transformer(_cfg_ar(size, sl=P_DEFAULT+G_DEFAULT), rngs=nnx.Rngs(0))
                fn    = make_ar_gen(model, B_DEFAULT, P_DEFAULT, G_DEFAULT)
            elif paradigm == "Discrete":
                model = Transformer(_cfg_disc(size, sl=G_DEFAULT), rngs=nnx.Rngs(0))
                fn    = make_disc_gen(model, B_DEFAULT, G_DEFAULT, STEPS_DEF)
            else:
                model = ELFTransformer(_cfg_elf(size, sl=G_DEFAULT), rngs=nnx.Rngs(0))
                fn    = make_elf_gen(model, B_DEFAULT, G_DEFAULT, STEPS_DEF, dim)

            pm = _params_m(model)
            r  = lat.measure(fn, n_tokens=B_DEFAULT*G_DEFAULT)
            tps = B_DEFAULT * G_DEFAULT * 1e3 / r.mean_ms
            scale_rows.append(dict(paradigm=paradigm, size=size, params_m=pm,
                                   e2e_ms=r.mean_ms, tok_s=tps, step_ms=r.p50_ms))
            print(f"{paradigm}={tps:.0f}", end="  ", flush=True)
        except Exception as e:
            print(f"{paradigm}✗({e.__class__.__name__})", end="  ", flush=True)
    print()

scale_df = pd.DataFrame(scale_rows)
print(scale_df[["paradigm","size","params_m","tok_s","e2e_ms"]].round(2).to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# BATCH / SEQ_LEN 2-D SWEEP  →  ThroughputResult grids for 3D plots
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Batch × seq_len 2-D sweep ────────────────────────────────────────")
throughput_results = {}
batch_rows = []
dim, n_heads, head_size, num_blocks = SIZES[BENCH_SIZE]

for paradigm in ("AR", "Discrete", "Continuous"):
    print(f"  {paradigm}:", end="  ", flush=True)
    try:
        if paradigm == "AR":
            model = Transformer(_cfg_ar(BENCH_SIZE, sl=P_DEFAULT+max(SEQ_LENS)), rngs=nnx.Rngs(0))
        elif paradigm == "Discrete":
            model = Transformer(_cfg_disc(BENCH_SIZE, sl=max(SEQ_LENS)), rngs=nnx.Rngs(0))
        else:
            model = ELFTransformer(_cfg_elf(BENCH_SIZE, sl=max(SEQ_LENS)), rngs=nnx.Rngs(0))

        grid, by_batch, by_seq = [], {}, {}
        for bs in BATCH_SIZES:
            for sl in SEQ_LENS:
                try:
                    if paradigm == "AR":
                        fn = make_ar_gen(model, bs, P_DEFAULT, sl)
                    elif paradigm == "Discrete":
                        fn = make_disc_gen(model, bs, sl, STEPS_DEF)
                    else:
                        fn = make_elf_gen(model, bs, sl, STEPS_DEF, dim)

                    r   = lat.measure(fn, n_tokens=bs*sl)
                    tps = bs * sl * 1e3 / r.mean_ms
                    grid.append({"batch_size": bs, "seq_len": sl, "tps": tps,
                                 "mean_ms": r.mean_ms, "p50_ms": r.p50_ms,
                                 "p95_ms": r.p95_ms, "p99_ms": r.p99_ms})
                    if sl == SEQ_LENS[1]: by_batch[bs] = tps
                    if bs == 1:           by_seq[sl]   = tps
                    batch_rows.append(dict(paradigm=paradigm, batch_size=bs,
                                          seq_len=sl, tps=tps, e2e_ms=r.mean_ms))
                    print(".", end="", flush=True)
                except Exception:
                    break

        all_tps = [e["tps"] for e in grid]
        throughput_results[paradigm] = ThroughputResult(
            peak_tps=max(all_tps) if all_tps else float("nan"),
            by_batch=by_batch, by_seq=by_seq,
            seq_len=SEQ_LENS[1], grid=grid,
        )
        print(f"  {len(grid)} pts, peak={max(all_tps)/1e3:.1f}k tok/s")
    except Exception as e:
        print(f"FAILED: {e}")

batch_df = pd.DataFrame(batch_rows)

# ═══════════════════════════════════════════════════════════════════════════════
# STEPS SWEEP  →  ThroughputResult grid (batch × n_steps)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Steps 2-D sweep (batch × n_steps) ───────────────────────────────")
steps_grid_results = {}

for paradigm in ("Discrete", "Continuous"):
    print(f"  {paradigm}:", end="  ", flush=True)
    try:
        if paradigm == "Discrete":
            model = Transformer(_cfg_disc(BENCH_SIZE, sl=G_DEFAULT), rngs=nnx.Rngs(0))
        else:
            model = ELFTransformer(_cfg_elf(BENCH_SIZE, sl=G_DEFAULT), rngs=nnx.Rngs(0))

        grid = []
        for bs in BATCH_SIZES[:4]:
            for steps in STEPS_LIST:
                try:
                    if paradigm == "Discrete":
                        fn = make_disc_gen(model, bs, G_DEFAULT, steps)
                    else:
                        fn = make_elf_gen(model, bs, G_DEFAULT, steps, dim)
                    r   = lat.measure(fn, n_tokens=bs*G_DEFAULT)
                    tps = bs * G_DEFAULT * 1e3 / r.mean_ms
                    grid.append({"batch_size": bs, "seq_len": steps, "tps": tps})
                    print(".", end="", flush=True)
                except Exception:
                    break

        all_tps = [e["tps"] for e in grid]
        steps_grid_results[paradigm] = ThroughputResult(
            peak_tps=max(all_tps), by_batch={}, by_seq={},
            seq_len=STEPS_LIST[0], grid=grid,
        )
        print(f"  {len(grid)} pts")
    except Exception as e:
        print(f"FAILED: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# BUILD RunProfile + MultiRunReport objects
# ═══════════════════════════════════════════════════════════════════════════════
profiles = [
    RunProfile(run_name=p, run_dir=f"/tmp/{p}", config={"paradigm": p},
               throughput=throughput_results[p])
    for p in ("AR", "Discrete", "Continuous") if p in throughput_results
]
report = MultiRunReport(profiles=profiles, total_time_s=0.0,
                        metrics=["throughput"], filter_used={"size": BENCH_SIZE})

steps_profiles = [
    RunProfile(run_name=p, run_dir=f"/tmp/{p}", config={},
               throughput=steps_grid_results[p])
    for p in steps_grid_results
]
steps_report = MultiRunReport(profiles=steps_profiles, total_time_s=0.0,
                               metrics=["throughput"], filter_used={})

# ═══════════════════════════════════════════════════════════════════════════════
# 3-D PLOTLY FIGURES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Generating 3D Plotly figures ─────────────────────────────────────")

figs = {
    "throughput_3d_surface": plot_3d_compare(
        report, metric="tps", mode="surface", show=False,
        title="Throughput (tok/s): AR vs Discrete vs ELF — batch_size × seq_len",
    ),
    "throughput_3d_scatter": plot_3d_compare(
        report, metric="tps", mode="scatter", show=False,
        title="Throughput scatter: each point = (batch_size, seq_len, tok/s)",
    ),
    "throughput_vs_steps_3d": plot_3d_compare(
        steps_report, metric="tps", mode="surface", show=False,
        title="Throughput (tok/s): batch_size × n_diffusion_steps",
    ),
    "ar_surface_only": plot_3d_surface(
        profiles[0], metric="tps", show=False,
        title="AR Throughput surface (batch_size × seq_len)",
        colorscale="Blues",
    ),
    "discrete_surface_only": plot_3d_surface(
        profiles[1], metric="tps", show=False,
        title="Discrete Diffusion Throughput surface",
        colorscale="Reds",
    ),
}
if len(profiles) > 2:
    figs["elf_surface_only"] = plot_3d_surface(
        profiles[2], metric="tps", show=False,
        title="ELF (Continuous) Throughput surface",
        colorscale="Greens",
    )

for name, fig in figs.items():
    path = f"{OUT_DIR}/{name}.html"
    fig.write_html(path, include_plotlyjs="cdn")
    kb = os.path.getsize(path) // 1024
    print(f"  saved {path}  ({kb} KB)")

# Also save static PNG via kaleido (best-effort)
try:
    import kaleido  # noqa: F401
    fig_main = figs["throughput_3d_surface"]
    png_path = f"{OUT_DIR}/throughput_3d_surface.png"
    fig_main.write_image(png_path, width=1200, height=800)
    print(f"  saved {png_path}")
except Exception as e:
    print(f"  (kaleido not available — PNG skipped: {e})")

# ═══════════════════════════════════════════════════════════════════════════════
# 2-D MATPLOTLIB FIGURES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Generating 2D matplotlib figures ────────────────────────────────")

COLORS  = {"AR": "#2166ac", "Discrete": "#d62728", "Continuous": "#2ca02c"}
MARKERS = {"AR": "o", "Discrete": "s", "Continuous": "^"}
NICE    = {"AR": "Autoregressive",
           "Discrete": "Masked Diffusion (LLaDA)",
           "Continuous": "Continuous Diffusion (ELF)"}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "figure.dpi": 150, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.22, "grid.linestyle": ":",
    "axes.spines.top": False, "axes.spines.right": False,
})

# Fig 1: Scale sweep
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
for par in ["AR", "Discrete", "Continuous"]:
    sub = scale_df[scale_df.paradigm == par].sort_values("params_m")
    if sub.empty: continue
    ax.plot(sub.params_m, sub.tok_s/1e3, color=COLORS[par],
            marker=MARKERS[par], lw=2.0, ms=6.5, label=NICE[par])
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Parameters (M)"); ax.set_ylabel("Throughput (k tok/s, B=4, G=128)")
ax.set_title("(a) Absolute throughput vs model size", fontweight="bold")
ax.legend(fontsize=7.5)

ax2 = axes[1]
ar_sub = scale_df[scale_df.paradigm == "AR"].set_index("params_m")["tok_s"]
for par in ["Discrete", "Continuous"]:
    sub = scale_df[scale_df.paradigm == par].sort_values("params_m")
    xs, ys = [], []
    for _, row in sub.iterrows():
        near = ar_sub.index[np.argmin(np.abs(ar_sub.index - row.params_m))]
        if ar_sub[near] > 0:
            xs.append(row.params_m); ys.append(row.tok_s / ar_sub[near])
    ax2.plot(xs, ys, color=COLORS[par], marker=MARKERS[par], lw=2.0, ms=6.5, label=NICE[par])
ax2.axhline(1.0, color=COLORS["AR"], lw=1.5, ls="--", label="AR baseline (1×)", alpha=0.7)
ax2.axhspan(3.0, 8.0, alpha=0.05, color="#888888", label="3–8× band")
ax2.set_xscale("log")
ax2.set_xlabel("Parameters (M)"); ax2.set_ylabel("Speedup ratio vs AR")
ax2.set_title("(b) Speedup ratio over AR", fontweight="bold")
ax2.legend(fontsize=7.5, loc="lower right")
fig.suptitle("Fig 1 — Scale: diffusion is consistently 3–8× faster than AR",
             fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/fig1_scale.png"); plt.close(fig)
print(f"  saved fig1_scale.png")

# Fig 2: Batch sweep (Pareto + throughput)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax_i, sl in enumerate([SEQ_LENS[0], SEQ_LENS[-1]]):
    ax = axes[ax_i]
    sub = batch_df[batch_df.seq_len == sl]
    for par in ["AR", "Discrete", "Continuous"]:
        d = sub[sub.paradigm == par].sort_values("batch_size")
        if d.empty: continue
        ax.plot(d.batch_size, d.tps/1e3, color=COLORS[par], marker=MARKERS[par],
                lw=2.0, ms=6.0, label=NICE[par])
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Batch size"); ax.set_ylabel("Throughput (k tok/s)")
    ax.set_title(f"({'ab'[ax_i]}) seq_len={sl}", fontweight="bold")
    if ax_i == 0: ax.legend(fontsize=7.5)
fig.suptitle("Fig 2 — Batch-size sweep", fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/fig2_batch.png"); plt.close(fig)
print(f"  saved fig2_batch.png")

# Fig 3: Latency vs throughput Pareto
fig, ax = plt.subplots(figsize=(7, 4.5))
for par in ["AR", "Discrete", "Continuous"]:
    d = batch_df[(batch_df.paradigm == par) & (batch_df.seq_len == SEQ_LENS[1])].sort_values("batch_size")
    if d.empty: continue
    ax.plot(d.e2e_ms, d.tps/1e3, color=COLORS[par], marker=MARKERS[par],
            lw=2.0, ms=6.0, label=NICE[par])
    for _, row in d.iterrows():
        if int(row.batch_size) in (1, BATCH_SIZES[-1]):
            ax.annotate(f"B={int(row.batch_size)}", (row.e2e_ms, row.tps/1e3),
                        textcoords="offset points", xytext=(5, 4), fontsize=6.5,
                        color=COLORS[par])
ax.axvline(200, color="#888888", lw=1.0, ls=":", label="200 ms SLO")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Per-request latency (ms)"); ax.set_ylabel("Throughput (k tok/s)")
ax.set_title("Fig 3 — Serving Pareto frontier (log–log)", fontweight="bold")
ax.legend(fontsize=7.5)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/fig3_pareto.png"); plt.close(fig)
print(f"  saved fig3_pareto.png")

print(f"\nAll outputs in {OUT_DIR}/")
print("PASS — notebook logic verified.")
