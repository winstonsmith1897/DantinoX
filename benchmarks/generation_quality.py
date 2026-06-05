#!/usr/bin/env python3
"""
benchmarks/generation_quality.py
==================================

Open-ended generation quality metrics for trained DantinoX models.

Metrics
-------
distinct_1   Fraction of unique unigrams across all generated samples.
             Higher = more diverse vocabulary usage.
distinct_2   Fraction of unique bigrams.  Higher = more phrase-level diversity.
rep_4        Fraction of 4-gram tokens that are repetitions of an earlier
             4-gram in the same sample (Megatron / CTRL measure).
             Lower = less repetitive.
self_bleu_4  Mean BLEU-4 of each sample against the rest.
             Lower = more diverse generations.
mauve        MAUVE score vs. reference corpus (requires ``mauve-text``).
             Higher = closer to human-text distribution (0–1).

AR models   : decoded with nucleus sampling (top_p=0.9) via ``core.generation.generate``.
Diffusion   : decoded with ``diffusion_generate`` (confidence threshold=0.9).
MLA inference path is activated automatically for MLA runs.

Output CSV columns
------------------
  run, model_type, attn_variant, n_samples, gen_len,
  distinct_1, distinct_2, rep_4, self_bleu_4, mauve, params_m

Usage
-----
  python benchmarks/generation_quality.py --runs-dir runs
  python benchmarks/generation_quality.py --runs-dir runs --n-samples 50 --gen-len 128
  python benchmarks/generation_quality.py --runs-dir runs --out results/gen_quality.csv
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.trained_analysis import (
    _attn_type,
    _count_params_m,
    _load_config,
    _load_model,
)
from core.config import Config
from core.diffusion import make_noise_schedule
from core.generation import diffusion_generate, generate

log = logging.getLogger(__name__)

_XLA_CACHE = Path.home() / ".cache" / "jax_xla" / "dantinox_genq"
_XLA_CACHE.mkdir(parents=True, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(_XLA_CACHE))

_DEFAULT_N_SAMPLES = 100
_DEFAULT_GEN_LEN   = 128


# ── Text quality metrics ──────────────────────────────────────────────────────

def _ngrams(tokens: list[int], n: int) -> list[tuple[int, ...]]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(samples: list[list[int]], n: int) -> float:
    """Fraction of unique n-grams across all samples."""
    all_ng  = [ng for s in samples for ng in _ngrams(s, n)]
    if not all_ng:
        return float("nan")
    return len(set(all_ng)) / len(all_ng)


def rep_4(samples: list[list[int]]) -> float:
    """Mean fraction of 4-gram token positions that repeat an earlier 4-gram."""
    scores = []
    for s in samples:
        ngs = _ngrams(s, 4)
        if not ngs:
            continue
        seen: set[tuple[int, ...]] = set()
        reps = 0
        for ng in ngs:
            if ng in seen:
                reps += 1
            seen.add(ng)
        scores.append(reps / len(ngs))
    return float(np.mean(scores)) if scores else float("nan")


def self_bleu_4(samples: list[list[int]], max_refs: int = 50) -> float:
    """Mean BLEU-4 of each sample against a random subset of the others."""
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    except ImportError:
        log.warning("nltk not installed — skipping Self-BLEU (pip install nltk)")
        return float("nan")

    smooth = SmoothingFunction().method1
    scores = []
    n = len(samples)
    for i in range(n):
        hypothesis = [str(t) for t in samples[i]]
        refs_idx   = [j for j in range(n) if j != i]
        if len(refs_idx) > max_refs:
            refs_idx = list(np.random.choice(refs_idx, max_refs, replace=False))
        references = [[str(t) for t in samples[j]] for j in refs_idx]
        try:
            score = corpus_bleu(
                [references],
                [hypothesis],
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smooth,
            )
            scores.append(score)
        except Exception:
            pass
    return float(np.mean(scores)) if scores else float("nan")


def mauve_score(generated: list[list[int]], reference: list[list[int]]) -> float:
    """MAUVE score: generated vs. reference distribution."""
    try:
        import mauve  # type: ignore[import]
        gen_flat = [" ".join(str(t) for t in s) for s in generated]
        ref_flat = [" ".join(str(t) for t in s) for s in reference]
        out = mauve.compute_mauve(p_text=gen_flat, q_text=ref_flat, device_id=0, verbose=False)
        return float(out.mauve)
    except ImportError:
        return float("nan")
    except Exception as exc:
        log.debug("MAUVE failed: %s", exc)
        return float("nan")


# ── Generation ────────────────────────────────────────────────────────────────

def _generate_ar(
    model, config: Config, n_samples: int, gen_len: int, seed: int = 42
) -> list[list[int]]:
    """Generate n_samples sequences of length gen_len using AR decoding."""
    samples: list[list[int]] = []
    prompt = jnp.zeros((1, 1), dtype=jnp.int32)  # single BOS-like token
    for i in range(n_samples):
        try:
            out = generate(
                model, prompt, max_generations=gen_len,
                greedy=False, top_p=0.9, seed=seed + i,
                use_cache=True, temperature=1.0,
            )
            tokens = out[0, 1:].tolist()
            samples.append(tokens)
        except Exception as exc:
            log.debug("AR generation failed sample %d: %s", i, exc)
    return samples


def _generate_diff(
    model, config: Config, n_samples: int, gen_len: int, seed: int = 42
) -> list[list[int]]:
    """Generate n_samples sequences using diffusion decoding."""
    schedule = make_noise_schedule(config)
    samples: list[list[int]] = []
    prefix = jnp.zeros((1, 0), dtype=jnp.int32)  # empty prefix
    for i in range(n_samples):
        try:
            out = diffusion_generate(
                model, prefix, gen_len, schedule,
                mask_token_id=config.mask_token_id,
                seed=seed + i,
                num_sampling_steps=min(50, config.num_sampling_steps),
                temperature=1.0,
            )
            tokens = out[0].tolist()
            samples.append(tokens)
        except Exception as exc:
            log.debug("Diffusion generation failed sample %d: %s", i, exc)
    return samples


# ── Per-run evaluation ────────────────────────────────────────────────────────

def eval_run(
    run_path: str,
    n_samples: int = _DEFAULT_N_SAMPLES,
    gen_len: int = _DEFAULT_GEN_LEN,
    seed: int = 42,
) -> dict:
    nan = float("nan")
    run_name = Path(run_path).name

    try:
        config = _load_config(run_path)
        model  = _load_model(run_path, config)
    except Exception as exc:
        log.warning("Could not load %s: %s", run_path, exc)
        return {}

    params_m = _count_params_m(model)
    attn     = _attn_type(config)

    print(f"  {run_name}  [{config.model_type}/{attn}]  generating {n_samples}×{gen_len} ...")

    if config.model_type == "autoregressive":
        samples = _generate_ar(model, config, n_samples, gen_len, seed)
    else:
        samples = _generate_diff(model, config, n_samples, gen_len, seed)

    if not samples:
        log.warning("No samples generated for %s", run_name)
        return {}

    d1   = distinct_n(samples, 1)
    d2   = distinct_n(samples, 2)
    r4   = rep_4(samples)
    sb4  = self_bleu_4(samples)

    # MAUVE needs a reference; use the training corpus if available, else skip
    mv = float("nan")
    ref_text_path = os.path.join(run_path, "training_corpus.txt")
    if os.path.exists(ref_text_path):
        from benchmarks.perplexity_eval import _encode_text, _load_tokenizer
        tok = _load_tokenizer(run_path)
        if tok:
            ref_text  = Path(ref_text_path).read_text(encoding="utf-8", errors="ignore")[:200_000]
            ref_ids   = _encode_text(ref_text, tok)
            ref_samps = [ref_ids[i * gen_len:(i + 1) * gen_len]
                         for i in range(min(n_samples, len(ref_ids) // gen_len))]
            mv = mauve_score(samples, ref_samps)

    return {
        "run":          run_name,
        "model_type":   config.model_type,
        "attn_variant": attn,
        "n_samples":    len(samples),
        "gen_len":      gen_len,
        "distinct_1":   round(d1, 4),
        "distinct_2":   round(d2, 4),
        "rep_4":        round(r4, 4),
        "self_bleu_4":  round(sb4, 4),
        "mauve":        round(mv, 4) if not (mv != mv) else nan,
        "params_m":     round(params_m, 3),
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Open-ended generation quality metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--runs-dir",  default="runs")
    parser.add_argument("--runs",      nargs="*")
    parser.add_argument("--run-prefix", nargs="+", default=["ar_", "diff_"],
                        metavar="PREFIX",
                        help="Only evaluate runs whose name starts with a prefix (default: ar_ diff_).")
    parser.add_argument("--n-samples", type=int, default=_DEFAULT_N_SAMPLES,
                        help=f"Samples to generate per run (default: {_DEFAULT_N_SAMPLES})")
    parser.add_argument("--gen-len",   type=int, default=_DEFAULT_GEN_LEN,
                        help=f"Tokens per sample (default: {_DEFAULT_GEN_LEN})")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--out",       default="results/generation_quality.csv")
    parser.add_argument("--device",    default=None)
    args = parser.parse_args(argv)

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    runs_dir  = Path(args.runs_dir)
    prefixes  = tuple(p for p in (args.run_prefix or []) if p)
    run_names = args.runs or [
        d for d in os.listdir(runs_dir)
        if (runs_dir / d).is_dir()
        and (not prefixes or any(d.startswith(p) for p in prefixes))
    ]

    print(f"Generation quality — {len(run_names)} runs, "
          f"{args.n_samples} samples × {args.gen_len} tokens")
    print(f"  device: {jax.default_backend()}\n")

    rows = []
    for name in run_names:
        row = eval_run(
            str(runs_dir / name),
            n_samples=args.n_samples,
            gen_len=args.gen_len,
            seed=args.seed,
        )
        if row:
            rows.append(row)
            print(f"    distinct1={row['distinct_1']:.3f}  distinct2={row['distinct_2']:.3f}  "
                  f"rep4={row['rep_4']:.3f}  selfBLEU4={row['self_bleu_4']:.3f}  "
                  f"mauve={row['mauve']:.3f}")

    if not rows:
        print("No results.")
        sys.exit(0)

    import pandas as pd
    df  = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} rows → {out}")

    # Summary
    print("\nMean metrics by attention × paradigm:")
    summary = df.groupby(["model_type", "attn_variant"])[
        ["distinct_1", "distinct_2", "rep_4", "self_bleu_4"]
    ].mean().round(4)
    print(summary.to_string())


if __name__ == "__main__":
    main()
