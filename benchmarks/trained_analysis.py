#!/usr/bin/env python3
"""
benchmarks/trained_analysis.py
================================

Per-run latency, throughput, memory, and validation-loss analysis for
trained DantinoX checkpoints stored under a ``runs/`` directory.

Each run directory must contain:
  config.yaml              — model configuration
  model_weights.msgpack    — serialised Flax NNX state
  training_log.csv         — step, train_loss, val_loss  (optional)

Metrics collected
-----------------
  run                    run directory name
  type                   MHA | GQA | MLA
  model_type             autoregressive | diffusion
  dim, n_heads, kv_heads, num_blocks, max_context
  params_m               trainable parameters (millions)
  theoretical_cache_mb   KV-cache size at max_context (MB, fp32)
  val_ppl                exp(final val_loss)  — NaN if no training log
  prefill_ms             prefill latency (prompt=256 tokens, median over N trials)
  decode_ms              single decode step latency (median)
  tok_s                  decode throughput at BS=1 (tokens/s)
  diff_step_ms           one full denoising step (Diffusion only)
  down_dim_kv            MLA latent KV dimension (None for MHA/GQA)

Output CSV
----------
  results/benchmark_results.csv   (default)

Usage
-----
  python benchmarks/trained_analysis.py
  python benchmarks/trained_analysis.py --runs-dir runs --out-csv results/trained.csv
  python benchmarks/trained_analysis.py --device 1 --n-trials 20
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

import jax
import jax.numpy as jnp
import msgpack
import numpy as np
from flax import nnx
from flax.serialization import _msgpack_ext_unpack

from core.config import Config
from core.model import DiffusionTransformer, Transformer

log = logging.getLogger(__name__)

_N_WARMUP  = 3
_N_MEASURE = 20

# ── XLA compilation cache ──────────────────────────────────────────────────────
_XLA_CACHE = Path.home() / ".cache" / "jax_xla" / "dantinox_trained"
_XLA_CACHE.mkdir(parents=True, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(_XLA_CACHE))


# ── Helpers (also imported by trained_batch_sweep) ────────────────────────────

def _load_config(run_path: str) -> Config:
    import yaml
    cfg_path = os.path.join(run_path, "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"config.yaml not found in {run_path}")
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    flat: dict = {}
    for section in raw.values():
        if isinstance(section, dict):
            flat.update(section)
    if not flat:
        flat = raw
    return Config.from_dict(flat)


def _load_model(
    run_path: str,
    config: Config,
) -> Transformer | DiffusionTransformer:
    """Load model weights into a freshly initialised Transformer or DiffusionTransformer."""
    weights_path = os.path.join(run_path, "model_weights.msgpack")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"model_weights.msgpack not found in {run_path}")
    with open(weights_path, "rb") as f:
        state_dict = msgpack.unpackb(
            f.read(), ext_hook=_msgpack_ext_unpack, strict_map_key=False
        )

    # Detect actual vocab size from the embedding weight shape
    def _get(d: object, key: str) -> object:
        if not isinstance(d, dict):
            return None
        v = d.get(key) or d.get(key.encode() if isinstance(key, str) else key)
        return v

    def _unwrap(obj: object) -> object:
        if isinstance(obj, dict):
            for k in ("value", "raw_value", b"value", b"raw_value"):
                if k in obj:
                    return obj[k]
        return obj

    wte = _get(state_dict, "wte")
    if wte is not None:
        emb = _unwrap(_get(wte, "embedding"))
        if emb is not None and hasattr(emb, "shape") and emb.ndim == 2:
            vocab = int(emb.shape[0]) if emb.shape[1] == config.dim else int(emb.shape[1])
            if vocab != config.vocab_size:
                config = dataclasses.replace(config, vocab_size=vocab)

    rngs = nnx.Rngs(42)
    if config.model_type == "diffusion":
        model: Transformer | DiffusionTransformer = DiffusionTransformer(config, rngs=rngs)
    else:
        model = Transformer(config, rngs=rngs)
    nnx.update(model, state_dict)
    return model


def _attn_type(config: Config) -> str:
    if getattr(config, "mla", False):
        return "MLA"
    if getattr(config, "kv_heads", config.n_heads) < config.n_heads:
        return "GQA"
    return "MHA"


def _theoretical_kv_cache_mb(
    config: Config, batch_size: int = 1, bf16: bool = False
) -> float:
    bpp = 2 if bf16 else 4
    S   = config.max_context
    if getattr(config, "mla", False):
        per_layer = S * (getattr(config, "down_dim_kv", 0) + getattr(config, "rope_dim", 0)) * bpp * batch_size
    else:
        hs        = config.dim // config.n_heads
        kv_heads  = getattr(config, "kv_heads", config.n_heads)
        per_layer = 2 * S * kv_heads * hs * bpp * batch_size
    return round(config.num_blocks * per_layer / 1e6, 3)


def _val_ppl(run_path: str) -> float:
    """Return exp(final_val_loss) from training_log.csv, or NaN."""
    log_csv = os.path.join(run_path, "training_log.csv")
    if not os.path.exists(log_csv):
        return float("nan")
    try:
        import pandas as pd
        df = pd.read_csv(log_csv)
        val_loss = float(df["val_loss"].dropna().iloc[-1])
        return float(np.exp(val_loss))
    except Exception:
        return float("nan")


def _count_params_m(model: nnx.Module) -> float:
    _, state = nnx.split(model)
    return sum(x.size for x in jax.tree_util.tree_leaves(state) if hasattr(x, "size")) / 1e6


# ── JIT kernels ───────────────────────────────────────────────────────────────

@nnx.jit
def _ar_prefill(model: nnx.Module, x: jnp.ndarray) -> jnp.ndarray:
    logits, _, _ = model(x, deterministic=True)
    return logits


@nnx.jit
def _ar_prefill_cached(model: nnx.Module, x: jnp.ndarray, cache: tuple) -> tuple:
    logits, new_cache, _ = model(x, caches=cache, cache_index=0, deterministic=True)
    return logits, new_cache


@nnx.jit
def _ar_decode(model: nnx.Module, tok: jnp.ndarray, cache: tuple, pos: jax.Array) -> tuple:
    logits, new_cache, _ = model(tok, caches=cache, cache_index=pos, deterministic=True)
    return logits, new_cache


@nnx.jit
def _diff_step(model: nnx.Module, x_t: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    out = model(x_t, dual_cache=None, deterministic=True)  # type: ignore[call-arg]
    return out.logits


def _time_fn(fn, *args, n_warmup: int, n_trials: int) -> np.ndarray:
    t0 = time.perf_counter()
    jax.block_until_ready(fn(*args))
    compile_s = time.perf_counter() - t0
    if compile_s > 2.0:
        log.info("    compile %.1fs", compile_s)
    for _ in range(max(0, n_warmup - 1)):
        jax.block_until_ready(fn(*args))
    ts: list[float] = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append((time.perf_counter() - t0) * 1_000.0)
    return np.array(ts)


# ── Per-run benchmark ─────────────────────────────────────────────────────────

def benchmark_run(run_path: str, n_warmup: int = _N_WARMUP, n_trials: int = _N_MEASURE) -> dict:
    """Benchmark one trained run, returning a metrics dict."""
    nan = float("nan")
    config = _load_config(run_path)
    model  = _load_model(run_path, config)
    params_m = _count_params_m(model)

    prompt_len = min(256, config.max_context - 1)
    x     = jnp.ones((1, prompt_len), dtype=jnp.int32)
    tok   = jnp.ones((1, 1),          dtype=jnp.int32)
    pos   = jnp.array(prompt_len,     dtype=jnp.int32)

    row: dict = {
        "run":                  Path(run_path).name,
        "model_type":           config.model_type,
        "type":                 _attn_type(config),
        "dim":                  config.dim,
        "n_heads":              config.n_heads,
        "kv_heads":             getattr(config, "kv_heads", config.n_heads),
        "num_blocks":           config.num_blocks,
        "max_context":          config.max_context,
        "params_m":             round(params_m, 3),
        "theoretical_cache_mb": _theoretical_kv_cache_mb(config),
        "down_dim_kv":          getattr(config, "down_dim_kv", None) if config.mla else None,
        "val_ppl":              _val_ppl(run_path),
        "prefill_ms": nan, "decode_ms": nan, "tok_s": nan, "diff_step_ms": nan,
    }

    if config.model_type == "autoregressive":
        try:
            init_cache = tuple((None, None) for _ in range(config.num_blocks))
            pre_ms = _time_fn(_ar_prefill, model, x, n_warmup=n_warmup, n_trials=n_trials)
            _, kv = _ar_prefill_cached(model, x, init_cache)
            jax.block_until_ready(kv)
            dec_ms = _time_fn(_ar_decode, model, tok, kv, pos, n_warmup=n_warmup, n_trials=n_trials)
            row["prefill_ms"] = round(float(np.median(pre_ms)), 3)
            row["decode_ms"]  = round(float(np.median(dec_ms)), 3)
            row["tok_s"]      = round(1_000.0 / float(np.median(dec_ms)), 2)
        except Exception as exc:
            log.warning("AR benchmark failed for %s: %s", run_path, exc)

    else:  # diffusion
        try:
            t_mid = jnp.full((1,), config.diffusion_steps // 2, dtype=jnp.int32)
            step_ms = _time_fn(_diff_step, model, x, t_mid, n_warmup=n_warmup, n_trials=n_trials)
            row["diff_step_ms"] = round(float(np.median(step_ms)), 3)
            # Estimate tok/s: B×T tokens per step, N steps for full generation
            n_steps = getattr(config, "num_sampling_steps", 50)
            row["tok_s"] = round(1.0 * prompt_len * 1_000.0 / (n_steps * float(np.median(step_ms))), 2)
        except Exception as exc:
            log.warning("Diffusion benchmark failed for %s: %s", run_path, exc)

    return row


# ── Entry point ───────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Trained-model per-run analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--runs-dir",  default="runs",
                        help="Directory containing run subdirs (default: runs)")
    parser.add_argument("--runs",       nargs="*",
                        help="Specific run names (default: all in --runs-dir)")
    parser.add_argument("--run-prefix", nargs="+", default=["ar_", "diff_"],
                        metavar="PREFIX",
                        help="Only analyse runs whose name starts with one of these prefixes (default: ar_ diff_). Pass empty string to include all.")
    parser.add_argument("--out-csv",   default="results/benchmark_results.csv",
                        help="Output CSV path (default: results/benchmark_results.csv)")
    parser.add_argument("--out-plot",  default="results/plots/trained_analysis.png",
                        help="Output figure path (default: results/plots/trained_analysis.png)")
    parser.add_argument("--n-warmup",  type=int, default=_N_WARMUP)
    parser.add_argument("--n-trials",  type=int, default=_N_MEASURE)
    parser.add_argument("--device",    default=None,
                        help="CUDA_VISIBLE_DEVICES override (e.g. '0', '1')")
    args = parser.parse_args(argv)

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_dir():
        print(f"Runs directory not found: {runs_dir}", file=sys.stderr)
        sys.exit(1)

    prefixes = tuple(p for p in (args.run_prefix or []) if p)
    run_names = args.runs or [
        d for d in os.listdir(runs_dir)
        if (runs_dir / d).is_dir()
        and (not prefixes or any(d.startswith(p) for p in prefixes))
    ]
    if not run_names:
        print(f"No run directories found in {runs_dir}", file=sys.stderr)
        sys.exit(0)

    print(f"Trained-model analysis — {len(run_names)} runs")
    print(f"  device  : {jax.default_backend()}")
    print(f"  runs    : {runs_dir}")
    print(f"  output  : {args.out_csv}")
    print()

    rows = []
    for name in run_names:
        path = str(runs_dir / name)
        log.info("Benchmarking %s ...", name)
        try:
            row = benchmark_run(path, n_warmup=args.n_warmup, n_trials=args.n_trials)
            rows.append(row)
            print(f"  {name:<40}  type={row['type']:<4}  "
                  f"params={row['params_m']:.1f}M  "
                  f"val_ppl={row['val_ppl']:.1f}  "
                  f"tok/s={row['tok_s']:.1f}")
        except Exception as exc:
            log.warning("Skipped %s: %s", name, exc)

    if not rows:
        print("No results — check --runs-dir path.")
        sys.exit(0)

    import pandas as pd
    df = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {len(df)} rows → {out_csv}")

    # ── Quick summary plot ────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        palette = {"MHA": "#1565C0", "GQA": "#43A047", "MLA": "#8E24AA"}

        for ax, (col, ylabel) in zip(axes, [
            ("val_ppl",              "Validation PPL ↓"),
            ("tok_s",                "Throughput (tok/s) ↑"),
            ("theoretical_cache_mb", "KV-cache (MB) ↓"),
        ]):
            sub = df.dropna(subset=[col])
            for attn in ["MHA", "GQA", "MLA"]:
                pts = sub[sub["type"] == attn]
                if pts.empty:
                    continue
                ax.scatter(pts["params_m"], pts[col],
                           label=attn, color=palette.get(attn, "grey"),
                           s=60, zorder=3)
            ax.set_xlabel("Parameters (M)")
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        fig.suptitle("DantinoX — Trained-model summary", fontweight="bold")
        plt.tight_layout()
        out_plot = Path(args.out_plot)
        out_plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_plot, dpi=150, bbox_inches="tight")
        print(f"Plot saved → {out_plot}")
    except Exception as exc:
        log.warning("Plotting failed: %s", exc)


if __name__ == "__main__":
    main()
