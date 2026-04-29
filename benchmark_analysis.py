import os
import time
import yaml
import json
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
from core.config import Config
from core.model import Transformer
import os

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
    return model(tok, use_cache=True, kv_caches=cache, cache_index=idx)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
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
    """Navigate to wte.embedding in the saved state dict to find the real vocab_size.

    config.yaml stores the initial vocab_size before tokenizer training; the
    actual vocab (unique chars) is reflected in the saved embedding shape.
    Handles both str and bytes dict keys produced by different msgpack settings.
    """
    def _get(d, key: str):
        if not isinstance(d, dict):
            return None
        v = d.get(key)
        if v is None:
            v = d.get(key.encode())   # bytes key fallback
        return v

    def _unwrap(obj):
        """Unwrap NNX Variable container if present."""
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
    # embedding is (vocab_size, dim)
    if emb.shape[1] == dim:
        return int(emb.shape[0])
    # weight-tying may have stored it transposed
    if emb.shape[0] == dim:
        return int(emb.shape[1])
    return None


def _load_model(run_path: str, config: Config) -> Transformer:
    weights_path = os.path.join(run_path, "model_weights.msgpack")
    with open(weights_path, "rb") as f:
        state_dict = msgpack.unpackb(f.read(), ext_hook=_msgpack_ext_unpack,
                                     strict_map_key=False)

    # The config.yaml stores the *initial* vocab_size, but the char tokenizer
    # shrinks it to the actual number of unique characters found in the text.
    # Detect the real vocab_size from the saved embedding matrix.
    actual_vocab = _detect_actual_vocab(state_dict, config.dim)
    if actual_vocab is not None and actual_vocab != config.vocab_size:
        config = dataclasses.replace(config, vocab_size=actual_vocab)

    rngs  = nnx.Rngs(42)
    model = Transformer(config, rngs=rngs)
    nnx.update(model, state_dict)
    return model


def _attn_type(config: Config) -> str:
    if config.mla:
        return "MLA"
    if config.kv_heads < config.n_heads:
        return "GQA"
    return "MHA"


def _theoretical_kv_cache_mb(config: Config) -> float:
    """KV cache bytes per layer × num_layers, at max_context, in MB."""
    S = config.max_context
    if config.mla:
        # compressed cache: (S, down_dim_kv) + rope cache (S, rope_dim)
        per_layer = S * (config.down_dim_kv + config.rope_dim) * 4
    else:
        # standard: 2 × (S, kv_heads, head_size)
        per_layer = 2 * S * config.kv_heads * config.head_size * 4
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


# --------------------------------------------------------------------------- #
# Per-run benchmark                                                            #
# --------------------------------------------------------------------------- #
def run_benchmark(run_name: str) -> dict:
    run_path = os.path.join(RUNS_DIR, run_name)
    config   = _load_config(run_path)

    if config.mla:
        config.inference = True  # use compressed-cache path for generation

    model = _load_model(run_path, config)

    attn_type = _attn_type(config)

    params_m = 0
    summary  = os.path.join(run_path, "model_summary.json")
    if os.path.exists(summary):
        try:
            params_m = json.load(open(summary)).get("total_params_M", 0)
        except Exception:
            pass

    tok = jnp.zeros((1, 1), dtype=jnp.int32)

    # ------------------------------------------------------------------ #
    # Throughput at each sequence length                                  #
    # ------------------------------------------------------------------ #
    tps_by_seqlen = {}
    for seq_len in SEQ_LENS:
        if seq_len > config.max_context:
            tps_by_seqlen[seq_len] = float("nan")
            continue

        # fresh cache for this trial
        cache = None
        _, cache, _ = _decode_step(model, tok, cache, 0)  # init cache

        # warmup
        for i in range(1, N_WARMUP + 1):
            _, cache, _ = _decode_step(model, tok, cache, i)
        jax.block_until_ready(cache)

        t0 = time.time()
        for i in range(N_WARMUP + 1, N_WARMUP + 1 + N_MEASURE):
            _, cache, _ = _decode_step(model, tok, cache, min(i, seq_len - 1))
        jax.block_until_ready(cache)
        t1 = time.time()

        tps_by_seqlen[seq_len] = round(N_MEASURE / (t1 - t0), 2)

    # ------------------------------------------------------------------ #
    # Prefill latency at max_context                                      #
    # ------------------------------------------------------------------ #
    prompt = jnp.zeros((1, config.max_context), dtype=jnp.int32)

    @nnx.jit
    def _prefill(m, x):
        return m(x, use_cache=False, kv_caches=None, cache_index=0)

    _prefill(model, prompt)  # compile
    jax.block_until_ready(_prefill(model, prompt))

    t0 = time.time()
    out = _prefill(model, prompt)
    jax.block_until_ready(out)
    t1 = time.time()
    prefill_ms = round((t1 - t0) * 1000, 2)

    # ------------------------------------------------------------------ #
    # VRAM used by KV cache (measured)                                    #
    # ------------------------------------------------------------------ #
    jax.clear_caches()
    dev  = jax.devices()[0]
    vram_before = dev.memory_stats().get("bytes_in_use", 0)

    cache = None
    _, cache, _ = _decode_step(model, tok, cache, 0)
    jax.block_until_ready(cache)
    vram_after = dev.memory_stats().get("bytes_in_use", 0)
    vram_cache_mb = round(max(0, vram_after - vram_before) / 1e6, 2)

    # ------------------------------------------------------------------ #
    # FLOPs & memory traffic (via XLA cost analysis)                      #
    # ------------------------------------------------------------------ #
    def _xla_costs(fn, *args):
        """Return (flops, bytes_accessed) from XLA, or (nan, nan) on failure."""
        try:
            costs = fn.lower(*args).cost_analysis()
            # cost_analysis() returns a list of dicts (one per partition)
            if isinstance(costs, list):
                flops = sum(c.get("flops", 0) for c in costs)
                mem   = sum(c.get("bytes accessed", 0) for c in costs)
            else:
                flops = costs.get("flops", float("nan"))
                mem   = costs.get("bytes accessed", float("nan"))
            return float(flops), float(mem)
        except Exception:
            return float("nan"), float("nan")

    # Decode step (T=1, with filled cache at mid-context position)
    cache_mid = None
    _, cache_mid, _ = _decode_step(model, tok, cache_mid, 0)
    _decode_jit = nnx.jit(lambda m, t, c, i: m(t, use_cache=True,
                                                 kv_caches=c, cache_index=i))
    mid_idx = min(config.max_context // 2, config.max_context - 1)
    decode_flops, decode_bytes = _xla_costs(_decode_jit, model, tok, cache_mid, mid_idx)

    # Prefill (full context, no cache)
    _prefill_jit = nnx.jit(lambda m, x: m(x, use_cache=False,
                                            kv_caches=None, cache_index=0))
    prefill_flops, prefill_bytes = _xla_costs(_prefill_jit, model, prompt)

    decode_gflops        = round(decode_flops  / 1e9,  4) if not np.isnan(decode_flops)  else float("nan")
    prefill_gflops       = round(prefill_flops / 1e9,  4) if not np.isnan(prefill_flops) else float("nan")
    decode_arith_int     = round(decode_flops  / max(decode_bytes,  1), 4) if not np.isnan(decode_bytes)  else float("nan")
    prefill_arith_int    = round(prefill_flops / max(prefill_bytes, 1), 4) if not np.isnan(prefill_bytes) else float("nan")

    # Achieved TFLOP/s at the largest measured seq-len
    best_tps = tps_by_seqlen.get(max(s for s in SEQ_LENS if s <= config.max_context), float("nan"))
    decode_tflops_s = round(decode_gflops * best_tps / 1e3, 4) if not np.isnan(decode_gflops) else float("nan")

    result = {
        "run":                    run_name,
        "type":                   attn_type,
        "params_m":               params_m,
        "moe":                    config.use_moe,
        "num_blocks":             config.num_blocks,
        "dim":                    config.dim,
        "n_heads":                config.n_heads,
        "kv_heads":               config.kv_heads,
        "max_context":            config.max_context,
        "down_dim_kv":            getattr(config, "down_dim_kv", None),
        "theoretical_cache_mb":   round(_theoretical_kv_cache_mb(config), 2),
        "measured_cache_mb":      vram_cache_mb,
        "prefill_ms":             prefill_ms,
        "val_loss":               _val_loss(run_path),
        # FLOPs
        "decode_gflops":          decode_gflops,
        "prefill_gflops":         prefill_gflops,
        "decode_arith_int":       decode_arith_int,
        "prefill_arith_int":      prefill_arith_int,
        "decode_tflops_s":        decode_tflops_s,
        **{f"tps_{s}": tps_by_seqlen[s] for s in SEQ_LENS},
    }

    del model, cache, cache_mid
    return result


# --------------------------------------------------------------------------- #
# Grouping helpers                                                             #
# --------------------------------------------------------------------------- #
TYPE_COLORS   = {"MLA": "#4C9BE8", "GQA": "#E87B4C", "MHA": "#4CE87B"}
TYPE_ORDER    = ["MLA", "GQA", "MHA"]
MOE_LABELS    = {True: "MoE", False: "Dense"}

def _family_key(row) -> str:
    """Stable label for a set of runs that share the same architecture."""
    moe = "MoE" if row["moe"] else "Dense"
    return f"L{row['num_blocks']}_D{row['dim']}_H{row['n_heads']}_C{row['max_context']}_{moe}"


def _add_family(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["family"] = df.apply(_family_key, axis=1)
    return df


# --------------------------------------------------------------------------- #
# Grouped-bar plot (one cluster per family, one bar per attention type)        #
# --------------------------------------------------------------------------- #
def _grouped_bar(ax, df, col, ylabel, title, log=False):
    families = sorted(df["family"].unique())
    types     = [t for t in TYPE_ORDER if t in df["type"].unique()]

    n_fam  = len(families)
    n_type = len(types)
    width  = 0.8 / n_type
    x      = np.arange(n_fam)

    for ti, t in enumerate(types):
        sub    = df[df["type"] == t].groupby("family")[col].agg(["mean", "std"])
        means  = [sub.loc[f, "mean"] if f in sub.index else float("nan") for f in families]
        stds   = [sub.loc[f, "std"]  if f in sub.index else 0.0           for f in families]
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


# --------------------------------------------------------------------------- #
# Throughput scaling — one line per (family, type) pair                        #
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Params vs throughput scatter                                                  #
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# KV cache: theoretical vs measured                                            #
# --------------------------------------------------------------------------- #
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

    lim = max(df["theoretical_cache_mb"].max(), df["measured_cache_mb"].max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, label="y=x")
    ax.set_xlabel("Theoretical cache (MB)")
    ax.set_ylabel("Measured cache (MB)")
    ax.set_title("KV Cache: Theoretical vs Measured", fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)


# --------------------------------------------------------------------------- #
# Main plot                                                                    #
# --------------------------------------------------------------------------- #
def make_plots(df: pd.DataFrame):
    os.makedirs(os.path.dirname(OUT_PLOT), exist_ok=True)
    if "family" not in df.columns:
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
    _grouped_bar(ax_cache,   df, "theoretical_cache_mb",  "MB",          "KV Cache size (theoretical)")
    _grouped_bar(ax_prefill, df, "prefill_ms",             "ms",          "Prefill latency")

    if df["val_loss"].notna().any():
        _grouped_bar(ax_loss, df, "val_loss", "Val loss (NLL)", "Final Validation Loss")
    else:
        ax_loss.text(0.5, 0.5, "No val-loss data", ha="center", va="center",
                     transform=ax_loss.transAxes, fontsize=11)
        ax_loss.set_title("Final Validation Loss", fontweight="bold")

    if df["decode_gflops"].notna().any():
        _grouped_bar(ax_flops,    df, "decode_gflops",   "GFLOPs",   "Decode FLOPs (T=1, XLA)")
        _grouped_bar(ax_tflops_s, df, "decode_tflops_s", "TFLOP/s",  "Achieved TFLOP/s (decode)")
    else:
        for ax, lbl in ((ax_flops, "Decode FLOPs"), (ax_tflops_s, "Achieved TFLOP/s")):
            ax.text(0.5, 0.5, "FLOPs unavailable", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11)
            ax.set_title(lbl, fontweight="bold")

    _line_scaling(ax_scale, df)
    _scatter_params_tps(ax_scatter, df)
    _cache_comparison(ax_cmp, df)

    plt.savefig(OUT_PLOT, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {OUT_PLOT}")


# --------------------------------------------------------------------------- #
# Console summary helpers                                                       #
# --------------------------------------------------------------------------- #
def _print_fair_summary(df: pd.DataFrame):
    agg_cols = (["theoretical_cache_mb", "measured_cache_mb", "prefill_ms", "val_loss",
                 "decode_gflops", "prefill_gflops", "decode_arith_int", "prefill_arith_int",
                 "decode_tflops_s"]
                + [f"tps_{s}" for s in SEQ_LENS])
    # keep only columns that actually exist (FLOPs may be nan if cost analysis failed)
    agg_cols = [c for c in agg_cols if c in df.columns]

    print("\n=== Per-family comparison (apples-to-apples) ===")
    for fam, grp in df.groupby("family"):
        print(f"\n  [{fam}]  runs: {len(grp)}")
        pivot = grp.groupby("type")[agg_cols].mean()
        print(pivot.to_string())

    if df["moe"].any():
        print("\n=== Dense vs MoE — throughput ===")
        print(df.groupby(["moe", "type"])[[f"tps_{s}" for s in SEQ_LENS]].mean().to_string())


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Load existing results so we can skip already-benchmarked runs
    if os.path.exists(OUT_CSV):
        existing_df   = pd.read_csv(OUT_CSV)
        already_done  = set(existing_df["run"].astype(str))
        existing_rows = existing_df.to_dict("records")
        print(f"Loaded {len(already_done)} existing results from {OUT_CSV}")
    else:
        already_done  = set()
        existing_rows = []

    all_runs = sorted([
        r for r in os.listdir(RUNS_DIR)
        if os.path.isdir(os.path.join(RUNS_DIR, r))
        and os.path.exists(os.path.join(RUNS_DIR, r, "model_weights.msgpack"))
    ])
    pending = [r for r in all_runs if r not in already_done]
    print(f"Found {len(all_runs)} runs total — {len(already_done)} already done, "
          f"{len(pending)} to benchmark.")

    new_results = []
    for i, run in enumerate(pending, 1):
        print(f"  [{i}/{len(pending)}] {run} ...", end=" ", flush=True)
        try:
            new_results.append(run_benchmark(run))
            r = new_results[-1]
            print(f"OK  type={r['type']}  family={_family_key(r)}  tps@{SEQ_LENS[-1]}={r[f'tps_{SEQ_LENS[-1]}']}")
        except Exception as e:
            print(f"SKIP ({e})")

    all_results = existing_rows + new_results
    if not all_results:
        print("No valid runs found.")
        raise SystemExit(1)

    df = pd.DataFrame(all_results)
    df.to_csv(OUT_CSV, index=False)
    if new_results:
        print(f"\nAdded {len(new_results)} new results → {OUT_CSV}")
    else:
        print("\nNo new runs — CSV unchanged.")

    df = _add_family(df)
    _print_fair_summary(df)
    make_plots(df)
