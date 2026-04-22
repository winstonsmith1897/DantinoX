import os
import time
import yaml
import dataclasses
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import jax
import jax.numpy as jnp
from flax import nnx
import msgpack
from flax.serialization import _msgpack_ext_unpack

# Ensure these match your project structure
from core import Config, Transformer

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE      = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
RUNS_DIR    = "runs"
OUT_CSV     = "benchmark_results.csv"
OUT_PLOT    = "plots/benchmark_mla_vs_gqa_mha.png"
SEQ_LENS    = [64, 128, 256, 512]   # sequence lengths tested for throughput scaling
N_WARMUP    = 3
N_MEASURE   = 20                    # decode steps measured per seq-len trial

# --------------------------------------------------------------------------- #
# Module-level JIT so the cache key is stable across runs                     #
# --------------------------------------------------------------------------- #
@nnx.jit
def _decode_step(model, tok, cache, idx):
    # Depending on your model's exact signature, it may return just logits, 
    # or (logits, updated_cache). We execute the forward pass here.
    return model(tok, use_cache=True, kv_caches=cache, cache_index=idx)

@nnx.jit
def _prefill_step(model, prompt):
    return model(prompt, use_cache=False, kv_caches=None, cache_index=0)

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _load_config(run_path: str) -> Config:
    with open(os.path.join(run_path, "config.yaml"), "r") as f:
        raw = yaml.safe_load(f)
    if any(isinstance(v, dict) for v in raw.values()):
        flat = {}
        for section in raw.values():
            if isinstance(section, dict):
                flat.update(section)
    else:
        flat = raw
    valid = {f.name for f in dataclasses.fields(Config)}
    return Config(**{k: v for k, v in flat.items() if k in valid})

def _detect_actual_vocab(state_dict, dim: int) -> int | None:
    def _get(d, key: str):
        if not isinstance(d, dict):
            return None
        v = d.get(key)
        if v is None:
            v = d.get(key.encode())  # bytes key fallback
        return v

    def _unwrap(obj):
        if isinstance(obj, dict):
            for k in ("value", "raw_value", b"value", b"raw_value"):
                if k in obj:
                    return obj[k]
        return obj

    wte_state = _get(state_dict, "wte")
    if wte_state is None:
        return None
    emb = _unwrap(_get(wte_state, "embedding"))
    if emb is None or not hasattr(emb, "shape") or emb.ndim != 2:
        return None
    
    if emb.shape[1] == dim:
        return int(emb.shape[0])
    if emb.shape[0] == dim:
        return int(emb.shape[1])
    return None

def _load_model(run_path: str, config: Config) -> Transformer:
    weights_path = os.path.join(run_path, "model_weights.msgpack")
    with open(weights_path, "rb") as f:
        state_dict = msgpack.unpackb(f.read(), ext_hook=_msgpack_ext_unpack,
                                     strict_map_key=False)

    actual_vocab = _detect_actual_vocab(state_dict, config.dim)
    if actual_vocab is not None and actual_vocab != config.vocab_size:
        config = dataclasses.replace(config, vocab_size=actual_vocab)

    rngs  = nnx.Rngs(42)
    model = Transformer(config, rngs=rngs)
    nnx.update(model, state_dict)
    return model

def _attn_type(config: Config) -> str:
    if getattr(config, "mla", False):
        return "MLA"
    if getattr(config, "kv_heads", config.n_heads) < config.n_heads:
        return "GQA"
    return "MHA"

def _theoretical_kv_cache_mb(config: Config) -> float:
    S = config.max_context
    if getattr(config, "mla", False):
        per_layer = S * (getattr(config, "down_dim_kv", 0) + getattr(config, "rope_dim", 0)) * 4
    else:
        per_layer = 2 * S * getattr(config, "kv_heads", config.n_heads) * getattr(config, "head_size", config.dim // config.n_heads) * 4
    return per_layer * config.num_blocks / 1e6

def _val_loss(run_path: str) -> float | None:
    log = os.path.join(run_path, "training_log.csv")
    if not os.path.exists(log):
        return None
    try:
        df = pd.read_csv(log)
        return float(df["val_loss"].dropna().iloc[-1])
    except Exception:
        return None

def _xla_costs(fn, *args):
    """Return (flops, bytes_accessed) from XLA, or (nan, nan) on failure."""
    try:
        costs = fn.lower(*args).cost_analysis()
        if isinstance(costs, list):
            flops = sum(c.get("flops", 0) for c in costs)
            mem   = sum(c.get("bytes accessed", 0) for c in costs)
        else:
            flops = costs.get("flops", float("nan"))
            mem   = costs.get("bytes accessed", float("nan"))
        return float(flops), float(mem)
    except Exception as e:
        return float("nan"), float("nan")

def _count_params(model) -> float:
    # Get parameters via nnx graph traversal
    _, state = nnx.split(model)
    total_params = sum(x.size for x in jax.tree_util.tree_leaves(state) if hasattr(x, "size"))
    return total_params / 1e6

# --------------------------------------------------------------------------- #
# Core Benchmarking Logic                                                     #
# --------------------------------------------------------------------------- #
def benchmark_run(run_name: str) -> dict:
    run_path = os.path.join(RUNS_DIR, run_name)
    print(f"\nEvaluating: {run_name}")
    
    config = _load_config(run_path)
    model = _load_model(run_path, config)
    attn_type = _attn_type(config)
    params_m = _count_params(model)
    
    prompt_len = min(config.max_context, 256)
    prompt = jnp.ones((1, prompt_len), dtype=jnp.int32)
    tok = jnp.ones((1, 1), dtype=jnp.int32)
    
    # 1. Prefill latency measurement
    _ = _prefill_step(model, prompt) # Warmup
    jax.block_until_ready(_)
    
    t0 = time.perf_counter()
    out = _prefill_step(model, prompt)
    jax.block_until_ready(out)
    prefill_ms = (time.perf_counter() - t0) * 1000

    # 2. Decode Throughput
    tps_by_seqlen = {}
    for seq_len in SEQ_LENS:
        if seq_len > config.max_context:
            tps_by_seqlen[seq_len] = float("nan")
            print(f"  ⏭️  Skipping seq_len {seq_len} (exceeds max_context {config.max_context})")
            continue
            
        cache = None 
        
        try:
            # --- Warmup ---
            for _ in range(N_WARMUP):
                out = _decode_step(model, tok, cache, seq_len)
                
            jax.block_until_ready(out)
            
            # --- Measure ---
            t0 = time.perf_counter()
            for i in range(N_MEASURE):
                out = _decode_step(model, tok, cache, seq_len)
                
                # Optional: print a tiny dot every 5 steps just to see it's alive
                # if (i + 1) % 5 == 0:
                #     print(".", end="", flush=True)
                    
            jax.block_until_ready(out)
            
            duration = time.perf_counter() - t0
            tps_by_seqlen[seq_len] = N_MEASURE / duration
            
            # Print Success!
            print(f"  ✅ [Success] seq_len {seq_len:>3}: {tps_by_seqlen[seq_len]:.2f} tokens/sec")

        except Exception as e:
            # Print Failure (e.g., OOM error) and record NaN so the benchmark continues
            print(f"  ❌ [Failed]  seq_len {seq_len:>3}: {str(e).splitlines()[0]}")
            tps_by_seqlen[seq_len] = float("nan")

    # 3. FLOPs & memory traffic via XLA
    cache_mid = None
    mid_idx = min(config.max_context // 2, config.max_context - 1)
    
    decode_flops, decode_bytes = _xla_costs(_decode_step, model, tok, cache_mid, mid_idx)
    prefill_flops, prefill_bytes = _xla_costs(_prefill_step, model, prompt)

    decode_gflops      = round(decode_flops  / 1e9,  4) if not np.isnan(decode_flops)  else float("nan")
    prefill_gflops     = round(prefill_flops / 1e9,  4) if not np.isnan(prefill_flops) else float("nan")
    decode_arith_int   = round(decode_flops  / max(decode_bytes,  1), 4) if not np.isnan(decode_bytes)  else float("nan")
    prefill_arith_int  = round(prefill_flops / max(prefill_bytes, 1), 4) if not np.isnan(prefill_bytes) else float("nan")

    # Achieved TFLOP/s at the largest measured seq-len
    best_tps = tps_by_seqlen.get(max(s for s in SEQ_LENS if s <= config.max_context), float("nan"))
    decode_tflops_s = round(decode_gflops * best_tps / 1e3, 4) if not np.isnan(decode_gflops) else float("nan")

    vram_cache_mb = round(_theoretical_kv_cache_mb(config), 2) # Fallback if direct profiling isn't implemented

    result = {
        "run":                    run_name,
        "type":                   attn_type,
        "params_m":               params_m,
        "moe":                    getattr(config, "use_moe", False),
        "num_blocks":             config.num_blocks,
        "dim":                    config.dim,
        "n_heads":                config.n_heads,
        "kv_heads":               getattr(config, "kv_heads", config.n_heads),
        "max_context":            config.max_context,
        "down_dim_kv":            getattr(config, "down_dim_kv", None),
        "theoretical_cache_mb":   round(_theoretical_kv_cache_mb(config), 2),
        "measured_cache_mb":      vram_cache_mb,
        "prefill_ms":             prefill_ms,
        "val_loss":               _val_loss(run_path),
        "decode_gflops":          decode_gflops,
        "prefill_gflops":         prefill_gflops,
        "decode_arith_int":       decode_arith_int,
        "prefill_arith_int":      prefill_arith_int,
        "decode_tflops_s":        decode_tflops_s,
        **{f"tps_{s}": tps_by_seqlen.get(s, float("nan")) for s in SEQ_LENS},
    }

    del model
    return result

# --------------------------------------------------------------------------- #
# Grouping helpers                                                            #
# --------------------------------------------------------------------------- #
TYPE_COLORS   = {"MLA": "#4C9BE8", "GQA": "#E87B4C", "MHA": "#4CE87B"}
TYPE_ORDER    = ["MLA", "GQA", "MHA"]
MOE_LABELS    = {True: "MoE", False: "Dense"}

def _family_key(row) -> str:
    moe = "MoE" if row["moe"] else "Dense"
    return f"L{row['num_blocks']}_D{row['dim']}_H{row['n_heads']}_C{row['max_context']}_{moe}"

def _add_family(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["family"] = df.apply(_family_key, axis=1)
    return df

# --------------------------------------------------------------------------- #
# Plotting Functions                                                          #
# --------------------------------------------------------------------------- #
def _grouped_bar(ax, df, col, ylabel, title, log=False):
    families = sorted(df["family"].unique())
    types    = [t for t in TYPE_ORDER if t in df["type"].unique()]

    n_fam  = len(families)
    n_type = len(types)
    width  = 0.8 / max(n_type, 1)
    x      = np.arange(n_fam)

    for ti, t in enumerate(types):
        sub    = df[df["type"] == t].groupby("family")[col].agg(["mean", "std"])
        means  = [sub.loc[f, "mean"] if f in sub.index else float("nan") for f in families]
        stds   = [sub.loc[f, "std"]  if f in sub.index else 0.0          for f in families]
        stds   = [s if not np.isnan(s) else 0.0 for s in stds]
        offset = (ti - n_type / 2 + 0.5) * width
        bars   = ax.bar(x + offset, means, width, label=t,
                        color=TYPE_COLORS[t], edgecolor="black", linewidth=0.5, zorder=3)
        ax.errorbar(x + offset, means, yerr=stds,
                    fmt="none", color="black", capsize=3, linewidth=1, zorder=4)
        for bar, m in zip(bars, means):
            if not np.isnan(m):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                        f"{m:.1f}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    if log:
        ax.set_yscale("log")

def _line_scaling(ax, df):
    families = sorted(df["family"].unique())
    types    = [t for t in TYPE_ORDER if t in df["type"].unique()]
    linestyles = ["-", "--", ":", "-."]

    for fi, fam in enumerate(families):
        sub_fam = df[df["family"] == fam]
        for t in types:
            sub = sub_fam[sub_fam["type"] == t]
            if sub.empty:
                continue
            ys = [sub[f"tps_{s}"].dropna().mean() for s in SEQ_LENS]
            ls = linestyles[fi % len(linestyles)]
            ax.plot(SEQ_LENS, ys, marker="o", linestyle=ls,
                    label=f"{fam} / {t}",
                    color=TYPE_COLORS.get(t, "grey"), linewidth=1.8)

    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Tokens / sec")
    ax.set_title("Throughput vs Sequence Length (per family)", fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

def _scatter_params_tps(ax, df):
    families = sorted(df["family"].unique())
    markers  = ["o", "s", "^", "D", "v", "P", "X"]
    for fi, fam in enumerate(families):
        sub = df[df["family"] == fam]
        for t in TYPE_ORDER:
            pts = sub[sub["type"] == t]
            if pts.empty:
                continue
            ax.scatter(pts["params_m"], pts[f"tps_{SEQ_LENS[-1]}"],
                       label=f"{fam}/{t}",
                       color=TYPE_COLORS.get(t, "grey"),
                       marker=markers[fi % len(markers)],
                       edgecolors="black", linewidth=0.5, s=80, zorder=3)

    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel(f"Tokens/sec  (seq={SEQ_LENS[-1]})")
    ax.set_title("Parameters vs Throughput", fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

def _cache_comparison(ax, df):
    families = sorted(df["family"].unique())
    markers  = ["o", "s", "^", "D", "v", "P", "X"]
    for fi, fam in enumerate(families):
        sub = df[df["family"] == fam]
        for t in TYPE_ORDER:
            pts = sub[sub["type"] == t]
            if pts.empty:
                continue
            ax.scatter(pts["theoretical_cache_mb"], pts["measured_cache_mb"],
                       label=f"{fam}/{t}",
                       color=TYPE_COLORS.get(t, "grey"),
                       marker=markers[fi % len(markers)],
                       edgecolors="black", linewidth=0.5, s=80, zorder=3)

    lim = max(df["theoretical_cache_mb"].max(), df["measured_cache_mb"].max(), 1) * 1.1
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, label="y=x")
    ax.set_xlabel("Theoretical cache (MB)")
    ax.set_ylabel("Measured cache (MB)")
    ax.set_title("KV Cache: Theoretical vs Measured", fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

def make_plots(df: pd.DataFrame):
    os.makedirs(os.path.dirname(OUT_PLOT), exist_ok=True)
    df = _add_family(df)

    fig = plt.figure(figsize=(22, 20))
    fig.suptitle("MLA vs GQA vs MHA — Fair Comparison by Architecture Family",
                 fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.60, wspace=0.38)

    ax_tps      = fig.add_subplot(gs[0, 0])
    ax_cache    = fig.add_subplot(gs[0, 1])
    ax_prefill  = fig.add_subplot(gs[0, 2])
    ax_loss     = fig.add_subplot(gs[1, 0])
    ax_flops    = fig.add_subplot(gs[1, 1])
    ax_tflops_s = fig.add_subplot(gs[1, 2])
    ax_scale    = fig.add_subplot(gs[2, :])
    ax_scatter  = fig.add_subplot(gs[3, 0:2])
    ax_cmp      = fig.add_subplot(gs[3, 2])

    _grouped_bar(ax_tps,     df, f"tps_{SEQ_LENS[-1]}",  "Tokens/sec", f"Decode throughput (seq={SEQ_LENS[-1]})")
    _grouped_bar(ax_cache,   df, "theoretical_cache_mb", "MB",          "KV Cache size (theoretical)")
    _grouped_bar(ax_prefill, df, "prefill_ms",           "ms",          "Prefill latency")

    if df["val_loss"].notna().any():
        _grouped_bar(ax_loss, df, "val_loss", "Val loss (NLL)", "Final Validation Loss")
    else:
        ax_loss.text(0.5, 0.5, "No val-loss data", ha="center", va="center", transform=ax_loss.transAxes, fontsize=11)
        ax_loss.set_title("Final Validation Loss", fontweight="bold")

    if df["decode_gflops"].notna().any():
        _grouped_bar(ax_flops,    df, "decode_gflops",   "GFLOPs",   "Decode FLOPs (T=1, XLA)")
        _grouped_bar(ax_tflops_s, df, "decode_tflops_s", "TFLOP/s",  "Achieved TFLOP/s (decode)")
    else:
        for ax, lbl in ((ax_flops, "Decode FLOPs"), (ax_tflops_s, "Achieved TFLOP/s")):
            ax.text(0.5, 0.5, "FLOPs unavailable", ha="center", va="center", transform=ax.transAxes, fontsize=11)
            ax.set_title(lbl, fontweight="bold")

    _line_scaling(ax_scale, df)
    _scatter_params_tps(ax_scatter, df)
    _cache_comparison(ax_cmp, df)

    plt.savefig(OUT_PLOT, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {OUT_PLOT}")

# --------------------------------------------------------------------------- #
# Console summary helpers                                                     #
# --------------------------------------------------------------------------- #
def _print_fair_summary(df: pd.DataFrame):
    df = _add_family(df)
    agg_cols = (["theoretical_cache_mb", "measured_cache_mb", "prefill_ms", "val_loss",
                 "decode_gflops", "prefill_gflops", "decode_arith_int", "prefill_arith_int",
                 "decode_tflops_s"]
                + [f"tps_{s}" for s in SEQ_LENS])
    
    agg_cols = [c for c in agg_cols if c in df.columns]

    print("\n=== Per-family comparison (apples-to-apples) ===")
    for fam, grp in df.groupby("family"):
        print(f"\n  [{fam}]  runs: {len(grp)}")
        pivot = grp.groupby("type")[agg_cols].mean()
        print(pivot.to_string())

    if "moe" in df.columns and df["moe"].any():
        print("\n=== Dense vs MoE — throughput ===")
        print(df.groupby(["moe", "type"])[[f"tps_{s}" for s in SEQ_LENS]].mean().to_string())


def main():
    if not os.path.exists(RUNS_DIR):
        print(f"Error: {RUNS_DIR} directory not found.")
        return

    run_dirs = [d for d in os.listdir(RUNS_DIR) if os.path.isdir(os.path.join(RUNS_DIR, d))]
    
    if not run_dirs:
        print(f"No run directories found in {RUNS_DIR}/")
        return

    results = []
    for run in run_dirs:
        try:
            res = benchmark_run(run)
            results.append(res)
        except Exception as e:
            print(f"Failed to benchmark {run}: {e}")

    if not results:
        print("No valid benchmark results collected.")
        return

    df = pd.DataFrame(results)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved raw metrics to {OUT_CSV}")

    _print_fair_summary(df)
    make_plots(df)

if __name__ == "__main__":
    main()