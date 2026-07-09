#!/usr/bin/env python3
"""
benchmarks/reproduce_table2.py
===============================

End-to-end reproduction of Table 2 ("DANTINOX open-ended generation quality
at Small scale, 512-d/12-layer") using ONLY DantinoX's public Level-2 API
(``dx.ModelConfig``, ``dx.TrainingConfig``, ``dx.Paradigm``, ``dx.Trainer``,
``dx.Generator``) — the same API surface documented in the paper's Figure 3.

For all 9 paradigm x attention combinations (AR / Discrete Diffusion /
Continuous Flow-Matching, each with MHA / GQA / MLA), across 3 seeds
(42, 43, 44), this script:

  1. Trains a 512-d, 12-layer model on WikiText-103-raw-v1 (T5 SentencePiece
     tokenizer, Muon optimizer, effective batch of 256 sequences of 512
     tokens) via ``dx.Trainer(dx.Paradigm(cfg), tcfg).fit(...)``.
  2. Generates 100 unconditional samples of 128 tokens per run via
     ``dx.Generator.generate(...)``, at a matched inference budget of 64
     steps (AR: token-by-token; Discrete: 64 denoising steps; Continuous:
     64 ODE steps).
  3. Scores each run on MAUVE, PPL under GPT-2, Distinct-2, Rep-4, and
     conditional BLEU-4 (AR/Discrete only), reusing the metric
     implementations from ``benchmarks/generation_quality.py`` (those are
     generic text-metric utilities, independent of the model/paradigm).
  4. Aggregates mean +/- std over the 3 seeds and emits both a CSV and a
     LaTeX table matching ``docs/tab_genquality_results.tex``.

This is a substantial compute job: 9 architectures x 3 seeds = 27 full
training runs plus 27 x 100-sample generation/scoring passes. It is
resumable — completed (paradigm, attention, seed) cells are recorded in
a JSON manifest and skipped on a re-run — but a first full run is a
multi-hour-to-multi-day job depending on hardware. Use --dry-run first,
and --archs / --seeds to scope a smaller run (e.g. one cell) when
smoke-testing.

Usage
-----
  # Sanity-check configs without training anything:
  python benchmarks/reproduce_table2.py --dry-run

  # Smoke-test a single cell:
  python benchmarks/reproduce_table2.py --archs ar:mha --seeds 42

  # Full reproduction (default: all 9 archs x seeds 42,43,44):
  python benchmarks/reproduce_table2.py

  # Re-run after an interruption — completed cells are skipped:
  python benchmarks/reproduce_table2.py --manifest results/table2_manifest.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

import dantinox as dx
from benchmarks.generation_quality import (
    _DEFAULT_CONT_LEN,
    _DEFAULT_PROMPT_LEN,
    _decode_samples,
    conditional_bleu4,
    distinct_n,
    load_gpt2,
    load_wikitext103_chunks,
    mauve_score_text,
    ppl_under_gpt2,
    rep_4,
)
from benchmarks.trained_analysis import _count_params_m

log = logging.getLogger(__name__)

ARCH_GRID = [
    ("ar", "mha"), ("ar", "gqa"), ("ar", "mla"),
    ("discrete", "mha"), ("discrete", "gqa"), ("discrete", "mla"),
    ("continuous", "mha"), ("continuous", "gqa"), ("continuous", "mla"),
]
PARADIGM_NICE = {
    "ar": "Autoregressive", "discrete": "Discrete Diffusion",
    "continuous": "Continuous Flow-Matching",
}
ATTN_NICE = {"mha": "MHA", "gqa": "GQA", "mla": "MLA"}

_METRIC_DIRECTION = {  # True = higher is better
    "mauve": True, "ppl_gpt2": False, "distinct_2": True,
    "rep_4": False, "bleu4_cond": True,
}


# ── Config builders (public API only) ────────────────────────────────────────

def build_model_config(paradigm: str, attention: str) -> dx.ModelConfig:
    """Small-scale (512-d, 12-layer) architecture, one config field per
    axis of variation — this is exactly the ablation DantinoX is built for."""
    base = dict(
        paradigm=paradigm, attention=attention,
        dim=512, n_heads=8, num_blocks=12, max_context=512,
        expansion=4, use_swiglu=True, weight_tying=True,
    )
    if attention == "gqa":
        base["kv_heads"] = 2  # n_heads/4, i.e. "GQA-1/4"
    elif attention == "mla":
        # Small-scale MLA compression dims used throughout this paper.
        base["down_dim_q"] = 128
        base["down_dim_kv"] = 96
        base["rope_dim"] = 16

    if paradigm == "discrete":
        base["noise_schedule"] = "linear"
    elif paradigm == "continuous":
        base["embed_dim"] = 768          # frozen T5-base encoder width
        base["bottleneck_dim"] = 128
        base["flow_n_steps"] = 64        # matched inference budget (below)
        base["flow_cfg_scale"] = 1.0

    return dx.ModelConfig(**base)


def build_training_config(seed: int, max_train_tokens: int, epochs: float) -> dx.TrainingConfig:
    """WikiText-103, Muon, T5 tokenizer, effective batch of 256x512 tokens —
    the shared recipe described in Section 4.2. vocab_size and
    mask_token_id are intentionally left unset on ModelConfig: Trainer.fit()
    auto-syncs both from the tokenizer."""
    return dx.TrainingConfig(
        lr=3e-4, optimizer="muon", lr_schedule="cosine", warmup_steps=400,
        batch_size=256, grad_accum=1, epochs=epochs,
        max_train_tokens=max_train_tokens,
        dataset_source="huggingface", dataset_name="wikitext",
        dataset_config="wikitext-103-raw-v1",
        tokenizer_type="t5", seed=seed, use_bf16=True,
    )


# ── Manifest (resumability) ───────────────────────────────────────────────────

def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))


def _cell_key(paradigm: str, attention: str, seed: int) -> str:
    return f"{paradigm}:{attention}:seed{seed}"


# ── Training ───────────────────────────────────────────────────────────────

def train_cell(
    paradigm: str, attention: str, seed: int,
    max_train_tokens: int, epochs: float, runs_dir: Path,
) -> str:
    cfg = build_model_config(paradigm, attention)
    tcfg = build_training_config(seed, max_train_tokens, epochs)
    run_dir = str(runs_dir / f"{paradigm}_{attention}_512d_12b_seed{seed}")

    log.info("Training %s/%s seed=%d -> %s", paradigm, attention, seed, run_dir)
    trainer = dx.Trainer(dx.Paradigm(cfg), tcfg)
    return trainer.fit(dataset_source="huggingface", dataset_name="wikitext",
                       run_dir=run_dir)


# ── Generation + metrics (dx.Generator only) ─────────────────────────────────

def _generate_samples(
    gen: dx.Generator, paradigm: str, n_samples: int, gen_len: int,
    n_steps: int, seed: int,
) -> list[str]:
    """Unconditional generation, matched inference budget: AR decodes
    token-by-token (n_steps is not applicable); Discrete/Continuous use
    n_steps denoising/ODE steps. Varying gen.seed per call (rather than
    rebuilding Generator each time) reuses the loaded checkpoint while
    still giving n_samples independently-seeded generations."""
    texts = []
    for i in range(n_samples):
        gen.seed = seed + i
        try:
            if paradigm == "ar":
                text = gen.generate(" ", max_new_tokens=gen_len, top_p=0.9,
                                    temperature=1.0, use_cache=True)
            elif paradigm == "discrete":
                text = gen.generate("", max_new_tokens=gen_len, n_steps=n_steps,
                                    decoding_strategy="sample", temperature=1.0)
            else:  # continuous — prompt is ignored, always unconditional
                text = gen.generate("", max_new_tokens=gen_len, n_steps=n_steps)
            texts.append(text)
        except Exception as exc:
            log.debug("Generation failed sample %d: %s", i, exc)
    return texts


def eval_cell(
    run_dir: str, paradigm: str, seed: int, n_samples: int, gen_len: int,
    n_steps: int, ref_chunks: list[list[int]], gpt2_tok, gpt2_mod,
) -> dict:
    nan = float("nan")
    gen = dx.Generator(run_dir, seed=seed)
    params_m = _count_params_m(gen.model)

    texts = _generate_samples(gen, paradigm, n_samples, gen_len, n_steps, seed)
    if not texts:
        log.warning("No samples generated for %s", run_dir)
        return {}

    token_samples = [gen.tokenizer.encode(t) for t in texts]
    d2 = distinct_n(token_samples, 2)
    r4 = rep_4(token_samples)

    mv = nan
    if ref_chunks:
        ref_texts = _decode_samples(ref_chunks[:len(texts)], gen.tokenizer)
        if ref_texts:
            mv = mauve_score_text(texts, ref_texts)

    pg = nan
    if gpt2_tok is not None and gpt2_mod is not None:
        pg = ppl_under_gpt2(texts, gpt2_tok, gpt2_mod)

    cb = nan
    if paradigm != "continuous" and ref_chunks:
        cb = conditional_bleu4(
            gen.model, gen.config, ref_chunks,
            prompt_len=_DEFAULT_PROMPT_LEN, cont_len=_DEFAULT_CONT_LEN, seed=seed,
        )

    return {
        "paradigm": paradigm, "seed": seed, "params_m": round(params_m, 3),
        "mauve": mv, "ppl_gpt2": pg, "distinct_2": round(d2, 4),
        "rep_4": round(r4, 4), "bleu4_cond": cb,
    }


# ── Aggregation + LaTeX ────────────────────────────────────────────────────────

def aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    agg = rows.groupby(["paradigm", "attention"]).agg(
        params_m=("params_m", "mean"),
        mauve_mean=("mauve", "mean"), mauve_std=("mauve", "std"),
        ppl_mean=("ppl_gpt2", "mean"), ppl_std=("ppl_gpt2", "std"),
        d2_mean=("distinct_2", "mean"), d2_std=("distinct_2", "std"),
        r4_mean=("rep_4", "mean"), r4_std=("rep_4", "std"),
        b4_mean=("bleu4_cond", "mean"), b4_std=("bleu4_cond", "std"),
    ).reset_index()
    return agg


def _fmt(mean: float, std: float, decimals: int, bold: bool) -> str:
    if mean != mean:  # NaN
        return "---"
    s = f"{mean:.{decimals}f}{{\\tiny$\\pm${std:.{decimals}f}}}"
    return f"\\textbf{{{mean:.{decimals}f}}}{{\\tiny$\\pm${std:.{decimals}f}}}" if bold else s


def _best_attn(sub: pd.DataFrame, mean_col: str, std_col: str, higher_better: bool) -> str | None:
    """Attention variant to bold: best mean, only if the gap to the runner-up
    exceeds the sum of their stds (margin-exceeds-seed-variance heuristic)."""
    d = sub.dropna(subset=[mean_col]).sort_values(mean_col, ascending=not higher_better)
    if len(d) < 2:
        return d["attention"].iloc[0] if len(d) == 1 else None
    best, second = d.iloc[0], d.iloc[1]
    gap = abs(best[mean_col] - second[mean_col])
    noise = (best[std_col] if best[std_col] == best[std_col] else 0.0) + \
            (second[std_col] if second[std_col] == second[std_col] else 0.0)
    return best["attention"] if gap > noise else None


def to_latex(agg: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "% Generation quality table -- Small scale (512-d, 12-layer), all three paradigms",
        "% Auto-generated by benchmarks/reproduce_table2.py -- do not hand-edit.",
        "% Needs: booktabs, xcolor (colortbl)",
        "\\begin{table}[t]",
        "\\centering",
        "\\footnotesize",
        "\\renewcommand{\\arraystretch}{1.2}",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{} l rrrrr @{}}",
        "\\toprule",
        "\\textbf{Arch (Params)}",
        "  & MV~$\\uparrow$ & PPL~$\\downarrow$",
        "  & D-2~$\\uparrow$ & R-4~$\\downarrow$ & B-4~$\\uparrow$ \\\\",
        "\\midrule",
    ]
    paradigms = ["ar", "discrete", "continuous"]
    for pi, paradigm in enumerate(paradigms):
        sub = agg[agg["paradigm"] == paradigm]
        if sub.empty:
            continue
        best_ppl = _best_attn(sub, "ppl_mean", "ppl_std", higher_better=False)
        best_d2  = _best_attn(sub, "d2_mean",  "d2_std",  higher_better=True)
        best_r4  = _best_attn(sub, "r4_mean",  "r4_std",  higher_better=False)
        best_b4  = _best_attn(sub, "b4_mean",  "b4_std",  higher_better=True)

        lines.append("%")
        lines.append(f"\\rowcolor{{gray!10}}")
        lines.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\textbf{{{PARADIGM_NICE[paradigm]}}}}} \\\\")
        for attn in ["mha", "gqa", "mla"]:
            r = sub[sub["attention"] == attn]
            if r.empty:
                continue
            r = r.iloc[0]
            row = (
                f"\\quad {ATTN_NICE[attn]} ({r['params_m']:.0f}M)  & "
                f"{_fmt(r['mauve_mean'], r['mauve_std'], 2, False)} & "
                f"{_fmt(r['ppl_mean'], r['ppl_std'], 0 if r['ppl_mean'] > 10 else 1, attn == best_ppl)} & "
                f"{_fmt(r['d2_mean'], r['d2_std'], 3, attn == best_d2)} & "
                f"{_fmt(r['r4_mean'], r['r4_std'], 3, attn == best_r4)} & "
                f"{_fmt(r['b4_mean'], r['b4_std'], 3, attn == best_b4)} \\\\"
            )
            lines.append(row)
        if pi < len(paradigms) - 1:
            lines.append("%")
            lines.append("\\midrule")

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{%",
        "  \\textsc{DantinoX} open-ended generation quality at Small scale",
        "  (512-d, 12-layer).",
        "  MV = MAUVE~$\\uparrow$; PPL = PPL\\textsubscript{GPT-2}~$\\downarrow$;",
        "  D-2 = Distinct-2~$\\uparrow$; R-4 = Rep-4~$\\downarrow$;",
        "  B-4 = BLEU-4\\textsubscript{cond}~$\\uparrow$",
        "  (``---'' = paradigm does not support prefix conditioning).",
        "  All metrics: mean~$\\pm$~std over 3 generation seeds, 100 unconditional",
        "  samples of 128 tokens each; matched inference budget of 64 steps",
        "  (AR decodes token-by-token; Diffusion: 64 denoising steps; ",
        "  Flow-Matching: 64 ODE steps).",
        "  \\textbf{Bold} = best per paradigm where the margin exceeds seed",
        "  variance; within-paradigm MAUVE differences are left unmarked.%",
        "}",
        "\\label{tab:genquality}",
        "\\end{table}",
    ]
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Saved LaTeX table -> {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archs", nargs="*", default=None,
                        help="Subset as 'paradigm:attention', e.g. ar:mha discrete:gqa. "
                             "Default: all 9 combinations.")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--gen-len", type=int, default=128)
    parser.add_argument("--n-steps", type=int, default=64,
                        help="Denoising/ODE steps for Discrete/Continuous (matched budget)")
    parser.add_argument("--max-train-tokens", type=int, default=50_000_000)
    parser.add_argument("--epochs", type=float, default=6,
                        help="Upper bound on passes over the capped corpus")
    parser.add_argument("--runs-dir", default="runs/table2")
    parser.add_argument("--manifest", default="results/table2_manifest.json")
    parser.add_argument("--out-csv", default="results/table2_dantinox_api.csv")
    parser.add_argument("--out-tex", default="docs/tab_genquality_results.tex")
    parser.add_argument("--gpt2-model", default="gpt2")
    parser.add_argument("--wt103-chunks", type=int, default=200)
    parser.add_argument("--no-mauve", action="store_true")
    parser.add_argument("--no-ppl-gpt2", action="store_true")
    parser.add_argument("--no-cond-bleu", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build and print configs for every cell; train/generate nothing.")
    args = parser.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",")]
    grid = ARCH_GRID
    if args.archs:
        wanted = {tuple(a.split(":")) for a in args.archs}
        grid = [c for c in grid if c in wanted]

    cells = [(p, a, s) for (p, a) in grid for s in seeds]
    print(f"Table 2 reproduction: {len(grid)} architectures x {len(seeds)} seeds "
          f"= {len(cells)} runs")

    if args.dry_run:
        for paradigm, attention, seed in cells:
            cfg = build_model_config(paradigm, attention)
            print(f"  [{_cell_key(paradigm, attention, seed)}] {cfg}")
        print("\n--dry-run: no training or generation was performed.")
        return

    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest)
    manifest = _load_manifest(manifest_path)

    # ── Train (resumable via manifest) ───────────────────────────────────────
    for paradigm, attention, seed in cells:
        key = _cell_key(paradigm, attention, seed)
        if key in manifest:
            log.info("Skipping %s (already in manifest -> %s)", key, manifest[key])
            continue
        run_dir = train_cell(paradigm, attention, seed,
                             args.max_train_tokens, args.epochs, runs_dir)
        manifest[key] = run_dir
        _save_manifest(manifest_path, manifest)

    # ── Shared eval resources (loaded once) ──────────────────────────────────
    ref_chunks = []
    if not (args.no_mauve and args.no_cond_bleu):
        print(f"Loading WikiText-103 reference ({args.wt103_chunks} chunks)...")
        first_run = manifest[_cell_key(*cells[0])]
        tokenizer = dx.Generator(first_run, seed=42).tokenizer
        ref_chunks = load_wikitext103_chunks(
            tokenizer, n_chunks=args.wt103_chunks,
            chunk_len=_DEFAULT_PROMPT_LEN + _DEFAULT_CONT_LEN,
        )
        print(f"  Loaded {len(ref_chunks)} chunks.\n")

    gpt2_tok = gpt2_mod = None
    if not args.no_ppl_gpt2:
        print(f"Loading {args.gpt2_model} for PPL evaluation...")
        gpt2_tok, gpt2_mod = load_gpt2(args.gpt2_model)

    # ── Evaluate every cell ──────────────────────────────────────────────────
    rows = []
    for paradigm, attention, seed in cells:
        key = _cell_key(paradigm, attention, seed)
        run_dir = manifest[key]
        print(f"Evaluating {key} -> {run_dir}")
        row = eval_cell(
            run_dir, paradigm, seed, args.n_samples, args.gen_len, args.n_steps,
            ref_chunks, gpt2_tok, gpt2_mod,
        )
        if row:
            row["attention"] = attention
            rows.append(row)

    if not rows:
        print("No results.")
        sys.exit(0)

    df = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {len(df)} per-seed rows -> {out_csv}")

    agg = aggregate(df)
    to_latex(agg, Path(args.out_tex))


if __name__ == "__main__":
    main()
