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
self_bleu_4  Mean BLEU-4 of each sample against the rest (Self-BLEU).
             Lower = more diverse generations.
mauve        MAUVE score vs. WikiText-103 test set (requires ``mauve-text``).
             Higher = closer to human-text distribution (0–1).
ppl_gpt2     Mean perplexity of decoded samples under GPT-2.
             Lower = more fluent text (requires ``transformers``).
bleu4_cond   BLEU-4 on conditional continuations (64-token prefix → 64 tokens).
             Measured against real WikiText-103 continuations.
             ELF models return NaN (unconditional only).

AR models   : decoded with nucleus sampling (top_p=0.9) via ``core.generation.generate``.
Diffusion   : decoded with ``diffusion_generate`` (confidence threshold=0.9).
ELF         : decoded with ``flow_generate`` (flow-matching, unconditional).

Output CSV columns
------------------
  run, model_type, attn_variant, n_samples, gen_len,
  distinct_1, distinct_2, rep_4, self_bleu_4,
  mauve, ppl_gpt2, bleu4_cond,
  params_m

Usage
-----
  python benchmarks/generation_quality.py --runs-dir runs
  python benchmarks/generation_quality.py --runs-dir runs --n-samples 100 --gen-len 128
  python benchmarks/generation_quality.py --runs-dir runs --no-mauve --no-ppl-gpt2
  python benchmarks/generation_quality.py --runs-dir runs --gpt2-model gpt2-medium
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

from benchmarks.perplexity_eval import _encode_text, _load_tokenizer
from benchmarks.trained_analysis import (
    _attn_type,
    _count_params_m,
    _load_config,
    _load_model,
)
from dantinox.core.config import Config
from dantinox.core.diffusion import make_noise_schedule
from dantinox.core.generation import diffusion_generate, flow_generate, generate

log = logging.getLogger(__name__)

_XLA_CACHE = Path.home() / ".cache" / "jax_xla" / "dantinox_genq"
_XLA_CACHE.mkdir(parents=True, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(_XLA_CACHE))

_DEFAULT_N_SAMPLES  = 100
_DEFAULT_GEN_LEN    = 128
_DEFAULT_PROMPT_LEN = 64   # for conditional BLEU-4
_DEFAULT_CONT_LEN   = 64   # for conditional BLEU-4


# ── Diversity / repetition metrics (token-ID space) ──────────────────────────

def _ngrams(tokens: list[int], n: int) -> list[tuple[int, ...]]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(samples: list[list[int]], n: int) -> float:
    all_ng = [ng for s in samples for ng in _ngrams(s, n)]
    if not all_ng:
        return float("nan")
    return len(set(all_ng)) / len(all_ng)


def rep_4(samples: list[list[int]]) -> float:
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
    try:
        from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
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
                [references], [hypothesis],
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smooth,
            )
            scores.append(score)
        except Exception:
            pass
    return float(np.mean(scores)) if scores else float("nan")


# ── Text-space metrics ────────────────────────────────────────────────────────

def _decode_samples(samples: list[list[int]], tokenizer) -> list[str]:
    texts = []
    for s in samples:
        try:
            texts.append(tokenizer.decode(s))
        except Exception:
            texts.append("")
    return [t for t in texts if t.strip()]


def mauve_score_text(gen_texts: list[str], ref_texts: list[str]) -> float:
    """MAUVE score between generated and human-text distributions (text input)."""
    try:
        import mauve as mauve_lib
        out = mauve_lib.compute_mauve(
            p_text=gen_texts,
            q_text=ref_texts,
            device_id=0,
            verbose=False,
            featurize_model_name="gpt2",
            max_text_length=256,
        )
        return float(out.mauve)
    except ImportError:
        log.warning("mauve-text not installed — pip install mauve-text")
        return float("nan")
    except Exception as exc:
        log.debug("MAUVE failed: %s", exc)
        return float("nan")


def ppl_under_gpt2(
    text_samples: list[str],
    gpt2_tok,
    gpt2_mod,
    max_length: int = 512,
) -> float:
    """Mean perplexity of generated text under a pre-loaded GPT-2 model."""
    try:
        import torch
    except ImportError:
        return float("nan")

    device = next(gpt2_mod.parameters()).device
    ppls = []
    for text in text_samples:
        if not text.strip():
            continue
        enc = gpt2_tok(
            text, return_tensors="pt", truncation=True, max_length=max_length
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        if enc["input_ids"].shape[1] < 2:
            continue
        with torch.no_grad():
            loss = gpt2_mod(**enc, labels=enc["input_ids"]).loss
        ppls.append(float(torch.exp(loss)))

    return float(np.mean(ppls)) if ppls else float("nan")


def load_gpt2(model_name: str = "gpt2"):
    """Load GPT-2 tokenizer and model once; return (tok, model) or (None, None)."""
    try:
        import torch
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tok = GPT2TokenizerFast.from_pretrained(model_name)
        mod = GPT2LMHeadModel.from_pretrained(model_name).eval().to(device)
        log.info("Loaded %s on %s", model_name, device)
        return tok, mod
    except ImportError:
        log.warning("transformers/torch not installed — pip install transformers torch")
        return None, None
    except Exception as exc:
        log.warning("Could not load GPT-2: %s", exc)
        return None, None


# ── WikiText-103 reference data ───────────────────────────────────────────────

def load_wikitext103_chunks(
    tokenizer,
    n_chunks: int = 200,
    chunk_len: int = 128,
    split: str = "test",
) -> list[list[int]]:
    """Load WikiText-103 test set; return token-ID chunks of chunk_len each."""
    try:
        from datasets import load_dataset
        ds   = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
        text = "\n\n".join(t for t in ds["text"] if t.strip())
        ids, _ = _encode_text(text, tokenizer)
        return [
            ids[i * chunk_len:(i + 1) * chunk_len]
            for i in range(min(n_chunks, len(ids) // chunk_len))
        ]
    except Exception as exc:
        log.warning("Could not load WikiText-103: %s", exc)
        return []


# ── Conditional BLEU-4 ────────────────────────────────────────────────────────

def conditional_bleu4(
    model,
    config: Config,
    ref_chunks: list[list[int]],
    prompt_len: int = _DEFAULT_PROMPT_LEN,
    cont_len: int = _DEFAULT_CONT_LEN,
    seed: int = 42,
) -> float:
    """BLEU-4 on prompt-conditioned continuations vs. real WikiText-103 text.

    ELF is fundamentally unconditional (generates from noise), so returns NaN.
    AR and Diffusion models use the first ``prompt_len`` tokens as prefix and
    generate ``cont_len`` continuation tokens.
    """
    if config.model_type == "elf":
        return float("nan")

    try:
        from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
    except ImportError:
        log.warning("nltk not installed — pip install nltk")
        return float("nan")

    schedule = (
        make_noise_schedule(config)
        if config.model_type != "autoregressive"
        else None
    )
    smooth      = SmoothingFunction().method1
    hypotheses  = []
    references  = []

    for i, chunk in enumerate(ref_chunks):
        if len(chunk) < prompt_len + cont_len:
            continue
        prefix_ids = chunk[:prompt_len]
        cont_ids   = chunk[prompt_len:prompt_len + cont_len]
        prefix_arr = jnp.array([prefix_ids], dtype=jnp.int32)

        try:
            if config.model_type == "autoregressive":
                out    = generate(
                    model, prefix_arr, max_generations=cont_len,
                    greedy=False, top_p=0.9, seed=seed + i,
                    use_cache=True, temperature=1.0,
                )
                gen_ids = out[0, len(prefix_ids):].tolist()
            else:
                out    = diffusion_generate(
                    model, prefix_arr, cont_len, schedule,
                    mask_token_id=config.mask_token_id,
                    seed=seed + i,
                    num_sampling_steps=64,
                    temperature=1.0,
                )
                gen_ids = out[0].tolist()
        except Exception as exc:
            log.debug("Conditional gen failed chunk %d: %s", i, exc)
            continue

        hypotheses.append([str(t) for t in gen_ids])
        references.append([[str(t) for t in cont_ids]])

    if not hypotheses:
        return float("nan")

    return float(
        corpus_bleu(references, hypotheses,
                    weights=(0.25, 0.25, 0.25, 0.25),
                    smoothing_function=smooth)
    )


# ── Unconditional generation ──────────────────────────────────────────────────

def _generate_ar(
    model, config: Config, n_samples: int, gen_len: int, seed: int = 42
) -> list[list[int]]:
    samples: list[list[int]] = []
    prompt = jnp.zeros((1, 1), dtype=jnp.int32)
    for i in range(n_samples):
        try:
            out    = generate(
                model, prompt, max_generations=gen_len,
                greedy=False, top_p=0.9, seed=seed + i,
                use_cache=True, temperature=1.0,
            )
            samples.append(out[0, 1:].tolist())
        except Exception as exc:
            log.debug("AR generation failed sample %d: %s", i, exc)
    return samples


def _generate_diff(
    model, config: Config, n_samples: int, gen_len: int, seed: int = 42
) -> list[list[int]]:
    schedule = make_noise_schedule(config)
    samples: list[list[int]] = []
    prefix = jnp.zeros((1, 0), dtype=jnp.int32)
    for i in range(n_samples):
        try:
            out    = diffusion_generate(
                model, prefix, gen_len, schedule,
                mask_token_id=config.mask_token_id,
                seed=seed + i,
                num_sampling_steps=64,
                temperature=1.0,
            )
            samples.append(out[0].tolist())
        except Exception as exc:
            log.debug("Diffusion generation failed sample %d: %s", i, exc)
    return samples


def _generate_elf(
    model, config: Config, n_samples: int, gen_len: int, seed: int = 42,
) -> list[list[int]]:
    n_steps   = getattr(config, "flow_n_steps",   64)
    cfg_scale = getattr(config, "flow_cfg_scale", 1.0)
    samples: list[list[int]] = []
    for i in range(n_samples):
        try:
            out    = flow_generate(model, gen_len=gen_len, batch_size=1,
                                  n_steps=n_steps, cfg_scale=cfg_scale, seed=seed + i)
            samples.append(out[0].tolist())
        except Exception as exc:
            log.debug("ELF generation failed sample %d: %s", i, exc)
    return samples


# ── Per-run evaluation ────────────────────────────────────────────────────────

def eval_run(
    run_path: str,
    n_samples: int = _DEFAULT_N_SAMPLES,
    gen_len: int = _DEFAULT_GEN_LEN,
    seed: int = 42,
    ref_chunks: list[list[int]] | None = None,
    gpt2_tok=None,
    gpt2_mod=None,
    skip_mauve: bool = False,
    skip_ppl_gpt2: bool = False,
    skip_cond_bleu: bool = False,
) -> dict:
    nan = float("nan")
    run_name = Path(run_path).name

    try:
        config = _load_config(run_path)
        model  = _load_model(run_path, config)
    except Exception as exc:
        log.warning("Could not load %s: %s", run_path, exc)
        return {}

    params_m  = _count_params_m(model)
    attn      = _attn_type(config)
    tokenizer = _load_tokenizer(run_path)

    print(f"  {run_name}  [{config.model_type}/{attn}]  generating {n_samples}×{gen_len} ...")

    if config.model_type == "autoregressive":
        samples = _generate_ar(model, config, n_samples, gen_len, seed)
    elif config.model_type == "elf":
        samples = _generate_elf(model, config, n_samples, gen_len, seed)
    else:
        samples = _generate_diff(model, config, n_samples, gen_len, seed)

    if not samples:
        log.warning("No samples generated for %s", run_name)
        return {}

    # Token-space diversity metrics
    d1  = distinct_n(samples, 1)
    d2  = distinct_n(samples, 2)
    r4  = rep_4(samples)
    sb4 = self_bleu_4(samples)

    # Decode to text for text-space metrics
    gen_texts = _decode_samples(samples, tokenizer) if tokenizer else []

    # MAUVE vs. WikiText-103
    mv = nan
    if not skip_mauve and gen_texts and ref_chunks:
        ref_texts = _decode_samples(ref_chunks[:len(gen_texts)], tokenizer)
        if ref_texts:
            mv = mauve_score_text(gen_texts, ref_texts)

    # PPL under GPT-2
    pg = nan
    if not skip_ppl_gpt2 and gen_texts and gpt2_tok is not None and gpt2_mod is not None:
        pg = ppl_under_gpt2(gen_texts, gpt2_tok, gpt2_mod)

    # Conditional BLEU-4 (prompt → continuation vs. real WikiText-103)
    cb = nan
    if not skip_cond_bleu and ref_chunks:
        cb = conditional_bleu4(
            model, config, ref_chunks,
            prompt_len=_DEFAULT_PROMPT_LEN,
            cont_len=_DEFAULT_CONT_LEN,
            seed=seed,
        )

    return {
        "run":          run_name,
        "model_type":   config.model_type,
        "attn_variant": attn,
        "n_samples":    len(samples),
        "gen_len":      gen_len,
        "distinct_1":   round(d1,  4),
        "distinct_2":   round(d2,  4),
        "rep_4":        round(r4,  4),
        "self_bleu_4":  round(sb4, 4),
        "mauve":        round(mv,  4) if mv == mv else nan,
        "ppl_gpt2":     round(pg,  2) if pg == pg else nan,
        "bleu4_cond":   round(cb,  4) if cb == cb else nan,
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
    parser.add_argument("--runs-dir",    default="runs")
    parser.add_argument("--runs",        nargs="*")
    parser.add_argument("--run-prefix",  nargs="+", default=["ar_", "diff_", "elf_"],
                        metavar="PREFIX")
    parser.add_argument("--n-samples",   type=int, default=_DEFAULT_N_SAMPLES)
    parser.add_argument("--gen-len",     type=int, default=_DEFAULT_GEN_LEN)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--out",         default="results/generation_quality.csv")
    parser.add_argument("--device",      default=None)
    parser.add_argument("--gpt2-model",  default="gpt2",
                        help="HuggingFace model name for PPL-GPT2 (default: gpt2)")
    parser.add_argument("--wt103-chunks", type=int, default=200,
                        help="Number of WikiText-103 test chunks for MAUVE/BLEU (default: 200)")
    parser.add_argument("--no-mauve",    action="store_true")
    parser.add_argument("--no-ppl-gpt2", action="store_true")
    parser.add_argument("--no-cond-bleu", action="store_true")
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

    # ── Shared resources (loaded once) ────────────────────────────────────────
    ref_chunks = []
    gpt2_tok   = None
    gpt2_mod   = None

    if not (args.no_mauve and args.no_cond_bleu):
        # Load tokenizer from first valid run to tokenize WikiText-103
        for name in run_names:
            tok = _load_tokenizer(str(runs_dir / name))
            if tok:
                print(f"Loading WikiText-103 reference ({args.wt103_chunks} chunks)...")
                ref_chunks = load_wikitext103_chunks(
                    tok, n_chunks=args.wt103_chunks, chunk_len=_DEFAULT_PROMPT_LEN + _DEFAULT_CONT_LEN
                )
                print(f"  Loaded {len(ref_chunks)} chunks of "
                      f"{_DEFAULT_PROMPT_LEN}+{_DEFAULT_CONT_LEN} tokens.\n")
                break

    if not args.no_ppl_gpt2:
        print(f"Loading {args.gpt2_model} for PPL evaluation...")
        gpt2_tok, gpt2_mod = load_gpt2(args.gpt2_model)
        print()

    # ── Per-run evaluation ────────────────────────────────────────────────────
    rows = []
    for name in run_names:
        row = eval_run(
            str(runs_dir / name),
            n_samples=args.n_samples,
            gen_len=args.gen_len,
            seed=args.seed,
            ref_chunks=ref_chunks or None,
            gpt2_tok=gpt2_tok,
            gpt2_mod=gpt2_mod,
            skip_mauve=args.no_mauve,
            skip_ppl_gpt2=args.no_ppl_gpt2,
            skip_cond_bleu=args.no_cond_bleu,
        )
        if row:
            rows.append(row)
            print(
                f"    d1={row['distinct_1']:.3f}  d2={row['distinct_2']:.3f}  "
                f"rep4={row['rep_4']:.3f}  sb4={row['self_bleu_4']:.3f}  "
                f"mauve={row['mauve'] if row['mauve']==row['mauve'] else 'nan'!s:.4}  "
                f"ppl_gpt2={row['ppl_gpt2'] if row['ppl_gpt2']==row['ppl_gpt2'] else 'nan'!s:.5}  "
                f"bleu4_cond={row['bleu4_cond'] if row['bleu4_cond']==row['bleu4_cond'] else 'nan'!s:.4}"
            )

    if not rows:
        print("No results.")
        sys.exit(0)

    import pandas as pd
    df  = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} rows → {out}")

    print("\nMean metrics by paradigm × attention:")
    cols = ["distinct_1", "distinct_2", "rep_4", "self_bleu_4", "mauve", "ppl_gpt2", "bleu4_cond"]
    print(df.groupby(["model_type", "attn_variant"])[cols].mean().round(4).to_string())


if __name__ == "__main__":
    main()
