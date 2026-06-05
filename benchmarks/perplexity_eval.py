#!/usr/bin/env python3
"""
benchmarks/perplexity_eval.py
==============================

Evaluate perplexity on EMNLP-standard NLP benchmarks for trained DantinoX models.

Primary metric: bits-per-byte (bpb)
  bpb is tokenizer-agnostic and enables fair comparison between models with
  different vocabularies (char-level, BPE, byte-level).
  bpb = NLL_nats / ln(2) / n_bytes

Secondary metric: PPL = exp(NLL_nats)

Evaluation method
-----------------
AR models
  Sliding-window cross-entropy over token sequences.
  stride = max_context // 2  (overlapping windows for unbiased estimates).
  PPL = exp(mean_NLL),  bpb = mean_NLL / ln(2) / avg_bytes_per_token

Diffusion models (ELBO-bpb)
  For each window, corrupt at T//5, 2T//5, 3T//5, 4T//5, T
  and evaluate masked cross-entropy.  Average across timesteps.
  ELBO-bpb is the primary quality metric for Diffusion.

Tokenisation
  The model's saved tokenizer.json is used.
  Unknown chars → byte-level fallback: each UTF-8 byte is encoded as token
  min(byte_value, vocab_size-1) so that any text is encodeable regardless of
  the training vocabulary.  This is required for evaluating char-level models
  trained on Italian text on English benchmarks.

Benchmark datasets (HuggingFace)
---------------------------------
Primary (EMNLP-standard):
  wikitext-103   wikitext / wikitext-103-raw-v1  (test)   — standard LM benchmark
  ptb            ptb_text_only                    (test)   — classic LM benchmark
  lambada        EleutherAI/lambada_openai         (test)   — long-range coherence
  c4             allenai/c4 / en                  (val, 2k docs) — generalist web

Secondary (in-domain):
  dante          Daniele/dante-corpus              (last 10% as test)  — training dist.
  wikipedia-it   Wikimedia/wikipedia / it          (val, 500 docs)     — Italian gen.

Output CSV columns
------------------
  run, model_type, attn_variant, dataset, bpb, ppl,
  n_tokens, n_bytes, n_windows, params_m, max_context, dim, train_val_ppl

Usage
-----
  python benchmarks/perplexity_eval.py --runs-dir runs
  python benchmarks/perplexity_eval.py --runs-dir runs --datasets wikitext-103 ptb lambada
  python benchmarks/perplexity_eval.py --runs-dir runs --max-windows 200 --out results/ppl.csv
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from benchmarks.trained_analysis import (
    _attn_type,
    _count_params_m,
    _load_config,
    _load_model,
    _val_ppl,
)
from core.config import Config
from core.diffusion import NoiseSchedule, corrupt, make_noise_schedule, masked_cross_entropy

log = logging.getLogger(__name__)

_XLA_CACHE = Path.home() / ".cache" / "jax_xla" / "dantinox_ppl"
_XLA_CACHE.mkdir(parents=True, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(_XLA_CACHE))

# ── Dataset registry ──────────────────────────────────────────────────────────

# (hf_repo, hf_config, split, text_field, max_docs)
_DATASET_SPECS: dict[str, tuple[str, str | None, str, str, int]] = {
    # EMNLP-standard benchmarks
    "wikitext-103": ("wikitext",                  "wikitext-103-raw-v1", "test",       "text",     0),
    "ptb":          ("ptb_text_only",             None,                  "test",       "sentence", 0),
    "lambada":      ("EleutherAI/lambada_openai", None,                  "test",       "text",     0),
    "c4":           ("allenai/c4",                "en",                  "validation", "text",     2_000),
    # In-domain / secondary
    "dante":        ("Daniele/dante-corpus",       None,                  "train",      "text",     0),
    "wikipedia-it": ("Wikimedia/wikipedia",        "20231101.it",         "train",      "text",     500),
}

_EMNLP_PRIMARY = ["wikitext-103", "ptb", "lambada", "c4"]
_ALL_DATASETS  = list(_DATASET_SPECS.keys())


def _load_hf_text(dataset_name: str, max_chars: int = 5_000_000) -> str | None:
    """Download and return text from a HuggingFace dataset split."""
    spec = _DATASET_SPECS.get(dataset_name)
    if spec is None:
        log.warning("Unknown dataset: %s (available: %s)", dataset_name, list(_DATASET_SPECS))
        return None
    hf_name, config_name, split, text_field, max_docs = spec

    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError:
        log.warning("datasets not installed: pip install datasets")
        return None

    try:
        kwargs: dict[str, Any] = {"trust_remote_code": True}
        if config_name:
            kwargs["name"] = config_name
        # Use streaming for large datasets to avoid full download
        use_streaming = max_docs > 0 or dataset_name in ("c4", "wikipedia-it", "dante")
        if use_streaming:
            kwargs["streaming"] = True
            kwargs["split"] = split
        else:
            kwargs["split"] = split

        ds = load_dataset(hf_name, **kwargs)

        chunks: list[str] = []
        total_chars = 0
        limit = max_docs if max_docs > 0 else 10_000_000  # effectively unlimited
        for i, ex in enumerate(ds):
            if i >= limit:
                break
            t = str(ex.get(text_field, "") or "")
            if not t.strip():
                continue
            chunks.append(t)
            total_chars += len(t)
            if total_chars >= max_chars:
                break

        if not chunks:
            log.warning("No text found in %s (field=%s)", dataset_name, text_field)
            return None

        # For dante: use last 10% as held-out test (training used first 90%)
        combined = "\n".join(chunks)
        if dataset_name == "dante":
            split_idx = int(0.9 * len(combined))
            combined = combined[split_idx:]

        log.info("Loaded %s: %d chars", dataset_name, len(combined))
        return combined

    except Exception as exc:
        log.warning("Failed to load %s: %s", dataset_name, exc)
        return None


# ── Tokenisation ──────────────────────────────────────────────────────────────

def _load_tokenizer(run_path: str) -> Any | None:
    tok_path = os.path.join(run_path, "tokenizer.json")
    if not os.path.exists(tok_path):
        return None
    try:
        from utils.tokenizer import load_tokenizer_from_file
        return load_tokenizer_from_file(tok_path)
    except Exception as exc:
        log.warning("Could not load tokenizer from %s: %s", tok_path, exc)
        return None


def _encode_text(text: str, tokenizer: Any) -> tuple[list[int], int]:
    """Encode text, returning (token_ids, n_bytes).

    Unknown chars use byte-level fallback: each UTF-8 byte is mapped to
    min(byte_value, vocab_size-1).  This ensures any text is fully encodeable
    while preserving byte count for accurate bpb computation.
    """
    vocab_size = getattr(tokenizer, "vocab_size", 256)
    n_bytes    = len(text.encode("utf-8"))
    ids: list[int] = []

    try:
        # Try encoding the full text first (works for BPE and clean char text)
        raw = tokenizer.encode(text)
        ids = [i for i in raw if i is not None]
        return ids, n_bytes
    except Exception:
        pass

    # Char tokenizer: char-by-char with byte-level fallback for unknowns
    for ch in text:
        try:
            ids.extend(tokenizer.encode(ch))
        except (KeyError, Exception):
            # Byte-level fallback
            for b in ch.encode("utf-8"):
                ids.append(min(b, vocab_size - 1))

    return ids, n_bytes


# ── JIT kernels ───────────────────────────────────────────────────────────────

@nnx.jit
def _ar_forward(model: nnx.Module, x: jnp.ndarray) -> jnp.ndarray:
    logits, _, _ = model(x, use_cache=False, kv_caches=None, cache_index=0, deterministic=True)
    return logits


@nnx.jit
def _diff_forward(model: nnx.Module, x_t: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    out = model(x_t, t, dual_cache=None, deterministic=True)  # type: ignore[call-arg]
    return out.logits


# ── Window iterator ───────────────────────────────────────────────────────────

def _windows(
    ids: list[int], window: int, stride: int
) -> list[tuple[list[int], list[int]]]:
    """Yield (input, target) windows for sliding-window LM evaluation."""
    windows = []
    for start in range(0, max(1, len(ids) - window), stride):
        end = start + window + 1
        chunk = ids[start:end]
        if len(chunk) < 2:
            continue
        windows.append((chunk[:-1], chunk[1:]))
    return windows


# ── AR perplexity ─────────────────────────────────────────────────────────────

def _nll_to_bpb(nll_nats: float, n_bytes: int, n_tokens: int) -> float:
    """Convert mean token-NLL (nats) to bits-per-byte."""
    if n_bytes == 0 or n_tokens == 0:
        return float("nan")
    # nll_nats is per-token average; scale by n_tokens to get total nats
    return (nll_nats * n_tokens) / (n_bytes * math.log(2))


def _eval_ar_ppl(
    model: nnx.Module,
    ids: list[int],
    n_bytes: int,
    config: Config,
    max_windows: int,
) -> tuple[float, float, int, int]:
    """Returns (ppl, bpb, n_tokens, n_windows)."""
    window = min(config.max_context, 512)
    stride = window // 2
    wins   = _windows(ids, window, stride)[:max_windows]
    if not wins:
        return float("nan"), float("nan"), 0, 0

    total_nll  = 0.0
    total_toks = 0
    n_wins     = 0

    for inp, tgt in wins:
        x = jnp.array([inp], dtype=jnp.int32)
        T = x.shape[1]
        try:
            logits = _ar_forward(model, x)
            log_p  = jax.nn.log_softmax(logits[0])
            tgt_j  = jnp.array(tgt, dtype=jnp.int32)
            nll    = -float(jnp.mean(log_p[jnp.arange(T), tgt_j]))
            total_nll  += nll * T
            total_toks += T
            n_wins     += 1
        except Exception as exc:
            log.debug("AR window error: %s", exc)

    if total_toks == 0:
        return float("nan"), float("nan"), 0, 0

    mean_nll = total_nll / total_toks
    ppl = float(np.exp(mean_nll))
    bpb = _nll_to_bpb(mean_nll, n_bytes, total_toks)
    return ppl, bpb, total_toks, n_wins


# ── Diffusion ELBO-bpb ────────────────────────────────────────────────────────

def _eval_diff_elbo_ppl(
    model: nnx.Module,
    ids: list[int],
    n_bytes: int,
    config: Config,
    schedule: NoiseSchedule,
    max_windows: int,
) -> tuple[float, float, int, int]:
    """Returns (elbo_ppl, elbo_bpb, n_tokens, n_windows)."""
    T_diff  = config.diffusion_steps
    window  = min(config.max_context, 256)
    stride  = window // 2
    wins    = _windows(ids, window, stride)[:max_windows]
    if not wins:
        return float("nan"), float("nan"), 0, 0

    # Uniform timestep grid: 5 points spanning [T/5, T]
    t_vals  = [T_diff // 5, 2 * T_diff // 5, 3 * T_diff // 5, 4 * T_diff // 5, T_diff]
    rng     = jax.random.key(0)

    total_elbo = 0.0
    total_toks = 0
    n_wins     = 0

    for inp, _ in wins:
        x0 = jnp.array([inp], dtype=jnp.int32)
        T  = x0.shape[1]
        win_elbo = 0.0
        for t_val in t_vals:
            t   = jnp.array([t_val], dtype=jnp.int32)
            rng, sub = jax.random.split(rng)
            x_t = corrupt(x0, t, sub, schedule, config.mask_token_id)
            try:
                logits = _diff_forward(model, x_t, t)
                loss   = float(masked_cross_entropy(logits, x0, x_t, config.mask_token_id))
                win_elbo += loss
            except Exception as exc:
                log.debug("Diff window error: %s", exc)
        total_elbo += (win_elbo / len(t_vals)) * T
        total_toks += T
        n_wins     += 1

    if total_toks == 0:
        return float("nan"), float("nan"), 0, 0

    mean_elbo = total_elbo / total_toks
    elbo_ppl  = float(np.exp(mean_elbo))
    elbo_bpb  = _nll_to_bpb(mean_elbo, n_bytes, total_toks)
    return elbo_ppl, elbo_bpb, total_toks, n_wins


# ── Per-run evaluation ────────────────────────────────────────────────────────

def eval_run(
    run_path: str,
    dataset_texts: dict[str, str],
    max_windows: int = 200,
) -> list[dict]:
    """Evaluate one run on all available datasets. Returns list of row dicts."""
    nan = float("nan")
    rows: list[dict] = []

    try:
        config  = _load_config(run_path)
        model   = _load_model(run_path, config)
    except Exception as exc:
        log.warning("Could not load %s: %s", run_path, exc)
        return []

    tokenizer  = _load_tokenizer(run_path)
    params_m   = _count_params_m(model)
    attn       = _attn_type(config)
    train_ppl  = _val_ppl(run_path)
    run_name   = Path(run_path).name

    schedule: NoiseSchedule | None = None
    if config.model_type == "diffusion":
        schedule = make_noise_schedule(config)

    # val-PPL from training log (always available if training log exists)
    rows.append({
        "run": run_name, "model_type": config.model_type, "attn_variant": attn,
        "dataset": "train_val", "ppl": train_ppl, "bpb": nan,
        "n_tokens": 0, "n_windows": 0,
        "params_m": round(params_m, 3),
        "max_context": config.max_context, "dim": config.dim,
        "train_val_ppl": train_ppl,
    })

    if tokenizer is None:
        log.warning("No tokenizer.json in %s — skipping external datasets", run_path)
        return rows

    for ds_name, text in dataset_texts.items():
        if not text:
            continue
        ids, n_bytes = _encode_text(text, tokenizer)
        if len(ids) < 10:
            log.warning("%s: too few tokens after encoding (%d) for %s", run_name, len(ids), ds_name)
            continue

        try:
            if config.model_type == "autoregressive":
                ppl, bpb, n_tok, n_win = _eval_ar_ppl(model, ids, n_bytes, config, max_windows)
            else:
                ppl, bpb, n_tok, n_win = _eval_diff_elbo_ppl(  # type: ignore[arg-type]
                    model, ids, n_bytes, config, schedule, max_windows
                )
        except Exception as exc:
            log.warning("Eval failed for %s / %s: %s", run_name, ds_name, exc)
            ppl, bpb, n_tok, n_win = nan, nan, 0, 0

        rows.append({
            "run": run_name, "model_type": config.model_type, "attn_variant": attn,
            "dataset": ds_name,
            "bpb": round(bpb, 4), "ppl": round(ppl, 3),
            "n_tokens": n_tok, "n_bytes": n_bytes, "n_windows": n_win,
            "params_m": round(params_m, 3),
            "max_context": config.max_context, "dim": config.dim,
            "train_val_ppl": round(train_ppl, 3),
        })
        log.info("  %s / %s: bpb=%.4f  PPL=%.2f  (%d windows)", run_name, ds_name, bpb, ppl, n_win)

    return rows


# ── Entry point ───────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Perplexity evaluation on standard NLP benchmarks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--runs-dir",   default="runs")
    parser.add_argument("--runs",       nargs="*")
    parser.add_argument("--run-prefix", nargs="+", default=["ar_", "diff_"],
                        metavar="PREFIX",
                        help="Only evaluate runs whose name starts with a prefix (default: ar_ diff_). Pass empty string for all.")
    parser.add_argument("--datasets",   nargs="+",
                        default=_EMNLP_PRIMARY,
                        choices=_ALL_DATASETS,
                        help=f"Datasets to evaluate (default: EMNLP primary: {_EMNLP_PRIMARY})")
    parser.add_argument("--local-text", default=None,
                        help="Local text file used as fallback when HF datasets fail")
    parser.add_argument("--max-windows", type=int, default=200,
                        help="Max evaluation windows per dataset (default: 200)")
    parser.add_argument("--out",        default="results/perplexity.csv")
    parser.add_argument("--device",     default=None)
    args = parser.parse_args(argv)

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # Load dataset texts once (shared across all runs)
    print(f"Loading datasets: {', '.join(args.datasets)}")
    dataset_texts: dict[str, str] = {}
    local_text: str | None = None
    if args.local_text and Path(args.local_text).exists():
        local_text = Path(args.local_text).read_text(encoding="utf-8", errors="ignore")

    for ds in args.datasets:
        text = _load_hf_text(ds)
        if text is None and local_text is not None:
            log.info("Using local text as fallback for %s", ds)
            text = local_text
        if text:
            dataset_texts[ds] = text
            print(f"  {ds}: {len(text):,} chars")
        else:
            print(f"  {ds}: SKIPPED (unavailable)")

    runs_dir = Path(args.runs_dir)
    prefixes  = tuple(p for p in (args.run_prefix or []) if p)
    run_names = args.runs or [
        d for d in os.listdir(runs_dir)
        if (runs_dir / d).is_dir()
        and (not prefixes or any(d.startswith(p) for p in prefixes))
    ]

    print(f"\nEvaluating {len(run_names)} runs on {len(dataset_texts)} datasets ...")
    all_rows: list[dict] = []
    for name in run_names:
        path = str(runs_dir / name)
        rows = eval_run(path, dataset_texts, max_windows=args.max_windows)
        all_rows.extend(rows)

    if not all_rows:
        print("No results.")
        sys.exit(0)

    import pandas as pd
    df = pd.DataFrame(all_rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} rows → {out}")

    # Summary table — bpb as primary EMNLP metric
    ext = df[df["dataset"] != "train_val"]
    if not ext.empty:
        pivot = ext.pivot_table(
            index=["attn_variant", "model_type"], columns="dataset",
            values="bpb", aggfunc="mean"
        )
        if not pivot.empty:
            print("\nMean bpb (↓ lower is better) by attention × paradigm:")
            print(pivot.round(4).to_string())
        pivot_ppl = ext.pivot_table(
            index=["attn_variant", "model_type"], columns="dataset",
            values="ppl", aggfunc="mean"
        )
        if not pivot_ppl.empty:
            print("\nMean PPL (↓) by attention × paradigm:")
            print(pivot_ppl.round(2).to_string())


if __name__ == "__main__":
    main()
