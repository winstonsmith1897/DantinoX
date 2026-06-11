#!/usr/bin/env python3
"""
scripts/test_generation_quality.py
====================================
Qualitative + quantitative evaluation of the trained diffusion and ELF models.

Tests:
  diff_mha_512d_12b_Dense  — 28 000 training steps, best diffusion candidate
  elf_mha_512d_12b_Dense   — 9 500 training steps

Outputs
-------
  - Decoded text samples for human inspection
  - distinct_1, distinct_2, rep_4 metrics
  - Verdict on whether quality is sufficient for EMNLP 2026

Usage
-----
  cd DantinoX
  python scripts/test_generation_quality.py
  python scripts/test_generation_quality.py --n-samples 20 --gen-len 128
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import jax
import jax.numpy as jnp
import msgpack
import numpy as np
from flax import nnx
from flax.serialization import _msgpack_ext_unpack

from core.config import Config
from core.diffusion import make_noise_schedule
from core.elf import ELFTransformer
from core.generation import diffusion_generate, elf_generate
from core.model import DiffusionTransformer

RUNS_DIR = _ROOT / "runs"

_EVAL_RUNS = [
    "diff_mha_512d_12b_Dense",
    "elf_mha_512d_12b_Dense",
]


# ── Tokenizer ─────────────────────────────────────────────────────────────────

def _load_tokenizer(run_path: Path):
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("t5-base")
    except Exception as exc:
        print(f"  [warn] Could not load tokenizer: {exc}")
        return None


def _decode_tokens(token_ids: list[int], tokenizer) -> str:
    if tokenizer is None:
        return " ".join(str(t) for t in token_ids[:20]) + " ..."
    try:
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        return text.strip() or "<empty>"
    except Exception:
        return " ".join(str(t) for t in token_ids[:20]) + " ..."


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_config(run_path: Path) -> Config:
    import yaml
    with open(run_path / "config.yaml") as f:
        raw = yaml.safe_load(f)
    flat: dict = {}
    for v in raw.values():
        if isinstance(v, dict):
            flat.update(v)
    return Config.from_dict(flat if flat else raw)


def _load_model(run_path: Path, config: Config):
    for fname in ("best_model_weights.msgpack", "model_weights.msgpack"):
        weights_path = run_path / fname
        if weights_path.exists():
            break
    else:
        raise FileNotFoundError(f"No weights in {run_path}")

    with open(weights_path, "rb") as f:
        state_dict = msgpack.unpackb(
            f.read(), ext_hook=_msgpack_ext_unpack, strict_map_key=False
        )

    rngs = nnx.Rngs(42)
    if config.model_type == "elf":
        model = ELFTransformer(config.to_elf_config(), rngs=rngs)
    else:
        model = DiffusionTransformer(config, rngs=rngs)
    nnx.update(model, state_dict)
    return model


# ── Metrics ───────────────────────────────────────────────────────────────────

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


def vocab_coverage(samples: list[list[int]], vocab_size: int) -> float:
    unique_toks = set(t for s in samples for t in s)
    return len(unique_toks) / vocab_size


# ── Generation ────────────────────────────────────────────────────────────────

def generate_diffusion(
    model, config: Config, n_samples: int, gen_len: int, seed: int = 42
) -> list[list[int]]:
    schedule = make_noise_schedule(config)
    prefix   = jnp.zeros((1, 0), dtype=jnp.int32)
    samples  = []
    for i in range(n_samples):
        out    = diffusion_generate(
            model, prefix, gen_len, schedule,
            mask_token_id=config.mask_token_id,
            seed=seed + i,
            num_sampling_steps=min(50, config.num_sampling_steps),
        )
        samples.append(out[0].tolist())
    return samples


def generate_elf(
    model, config: Config, n_samples: int, gen_len: int, seed: int = 42
) -> list[list[int]]:
    n_steps   = getattr(config, "elf_n_steps",   64)
    cfg_scale = getattr(config, "elf_cfg_scale", 1.5)
    samples   = []
    for i in range(n_samples):
        out    = elf_generate(model, gen_len=gen_len, batch_size=1,
                              n_steps=n_steps, cfg_scale=cfg_scale, seed=seed + i)
        samples.append(out[0].tolist())
    return samples


# ── Per-run evaluation ────────────────────────────────────────────────────────

def eval_run(
    name: str,
    n_samples: int = 10,
    gen_len: int = 64,
    seed: int = 42,
    show_text: int = 5,
) -> dict:
    run_path = RUNS_DIR / name
    print(f"\n{'═'*70}")
    print(f"  {name}")
    print(f"{'═'*70}")

    config = _load_config(run_path)
    print(f"  model_type : {config.model_type}")
    print(f"  dim        : {config.dim}  n_heads={config.n_heads}  num_blocks={config.num_blocks}")
    if config.model_type == "elf":
        print(f"  embed_dim  : {config.embed_dim}  elf_n_steps={config.elf_n_steps}  cfg_scale={config.elf_cfg_scale}")
    else:
        print(f"  mask_id    : {config.mask_token_id}  noise={config.noise_schedule}  diff_steps={config.diffusion_steps}")

    print("\n  Loading model ...", end=" ", flush=True)
    t0    = time.perf_counter()
    model = _load_model(run_path, config)
    params_m = sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))) / 1e6
    print(f"done  ({time.perf_counter()-t0:.1f}s)  {params_m:.1f}M params")

    tokenizer = _load_tokenizer(run_path)

    print(f"\n  Generating {n_samples} × {gen_len} tokens ...", end=" ", flush=True)
    t0 = time.perf_counter()
    if config.model_type == "elf":
        samples = generate_elf(model, config, n_samples, gen_len, seed)
    else:
        samples = generate_diffusion(model, config, n_samples, gen_len, seed)
    elapsed = time.perf_counter() - t0
    tok_per_s = n_samples * gen_len / elapsed
    print(f"done  ({elapsed:.1f}s  {tok_per_s:.0f} tok/s)")

    # ── Metrics ───────────────────────────────────────────────────────────────
    d1   = distinct_n(samples, 1)
    d2   = distinct_n(samples, 2)
    r4   = rep_4(samples)
    vcov = vocab_coverage(samples, config.vocab_size)

    print(f"\n  ── Metrics ──────────────────────────────────────────────────────")
    print(f"  distinct_1    : {d1:.4f}  (higher = more diverse unigrams; random ≈ 0.8+)")
    print(f"  distinct_2    : {d2:.4f}  (higher = more diverse bigrams;  random ≈ 0.9+)")
    print(f"  rep_4         : {r4:.4f}  (lower  = less repetitive 4-grams; good < 0.1)")
    print(f"  vocab_cov     : {vcov:.5f} ({len(set(t for s in samples for t in s))} unique tokens)")

    # ── Token frequency analysis ──────────────────────────────────────────────
    all_tokens = [t for s in samples for t in s]
    tok_counts = {}
    for t in all_tokens:
        tok_counts[t] = tok_counts.get(t, 0) + 1
    top10 = sorted(tok_counts.items(), key=lambda x: -x[1])[:10]
    print(f"\n  ── Top-10 most frequent token IDs ───────────────────────────────")
    for tid, cnt in top10:
        pct  = 100 * cnt / len(all_tokens)
        text = ""
        if tokenizer:
            try:
                text = repr(tokenizer.decode([tid], skip_special_tokens=False))[:30]
            except Exception:
                pass
        print(f"    id={tid:6d}  count={cnt:4d}  ({pct:5.1f}%)  {text}")

    # ── Qualitative samples ───────────────────────────────────────────────────
    n_show = min(show_text, len(samples))
    print(f"\n  ── First {n_show} decoded samples ───────────────────────────────────")
    for i, sample in enumerate(samples[:n_show]):
        text = _decode_tokens(sample, tokenizer)
        print(f"\n  [{i+1}] {text[:200]}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print(f"\n  ── Quality verdict ──────────────────────────────────────────────")
    issues = []
    if d1 < 0.05:
        issues.append(f"distinct_1={d1:.3f} is very low — severe repetition / mode collapse")
    if d2 < 0.10:
        issues.append(f"distinct_2={d2:.3f} is very low — phrases are repetitive")
    if r4 > 0.50:
        issues.append(f"rep_4={r4:.3f} is high — strong 4-gram repetition")
    if vcov < 0.001:
        issues.append(f"vocab_cov={vcov:.5f} — model is using very few tokens (< 0.1% of vocab)")

    if issues:
        print(f"  [FAIL] Quality INSUFFICIENT for EMNLP 2026:")
        for iss in issues:
            print(f"    • {iss}")
    else:
        print(f"  [PASS] Diversity metrics look reasonable — inspect text samples above.")

    return {
        "run":        name,
        "model_type": config.model_type,
        "params_m":   round(params_m, 1),
        "distinct_1": round(d1, 4),
        "distinct_2": round(d2, 4),
        "rep_4":      round(r4, 4),
        "vocab_cov":  round(vcov, 6),
        "tok_per_s":  round(tok_per_s, 1),
        "issues":     issues,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qualitative + quantitative generation test for trained models."
    )
    parser.add_argument("--runs",      nargs="*", default=_EVAL_RUNS,
                        help="Run names to evaluate (default: diff_mha + elf_mha).")
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--gen-len",   type=int, default=64)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--show-text", type=int, default=5,
                        help="Number of text samples to print per model.")
    args = parser.parse_args()

    print(f"DantinoX generation quality test")
    print(f"Device: {jax.default_backend()}")
    print(f"Config: {args.n_samples} samples × {args.gen_len} tokens")

    results = []
    for run_name in args.runs:
        try:
            result = eval_run(
                run_name,
                n_samples=args.n_samples,
                gen_len=args.gen_len,
                seed=args.seed,
                show_text=args.show_text,
            )
            results.append(result)
        except Exception as exc:
            print(f"\n[ERROR] {run_name}: {exc}")
            import traceback
            traceback.print_exc()

    # ── Summary table ─────────────────────────────────────────────────────────
    if len(results) > 1:
        print(f"\n{'═'*70}")
        print("  SUMMARY")
        print(f"{'═'*70}")
        hdr = f"  {'run':<35} {'type':<12} {'d1':>6} {'d2':>6} {'rep4':>6} {'vcov':>8}  status"
        print(hdr)
        print("  " + "-" * 66)
        for r in results:
            status = "PASS" if not r["issues"] else "FAIL"
            print(f"  {r['run']:<35} {r['model_type']:<12} "
                  f"{r['distinct_1']:>6.3f} {r['distinct_2']:>6.3f} "
                  f"{r['rep_4']:>6.3f} {r['vocab_cov']:>8.5f}  {status}")

    print("\nDone.")


if __name__ == "__main__":
    main()
