---
title: "EMNLP 2026 System Demonstration"
hide:
  - toc
---

# DantinoX: A Unified Framework for Systematic Comparison of Autoregressive and Masked Diffusion Language Models

<div class="hero-badges" markdown>
[![EMNLP 2026](https://img.shields.io/badge/EMNLP-2026%20System%20Demo-blue?style=flat-square)](paper.md)
[![JAX](https://img.shields.io/badge/JAX-000000?style=flat-square&logo=JAX&logoColor=white)](https://github.com/google/jax)
[![Flax NNX](https://img.shields.io/badge/Flax_NNX-5E17EB?style=flat-square&logoColor=white)](https://github.com/google/flax)
[![License MIT](https://img.shields.io/badge/License-MIT-16a34a?style=flat-square)](https://opensource.org/licenses/MIT)
</div>

> **EMNLP 2026 System Demonstrations** — official companion page for the paper
> *"DantinoX: A Unified JAX/Flax Framework for Autoregressive and Masked Diffusion Language Models across MHA, GQA, and MLA Attention Variants"*

---

## Abstract

DantinoX is a unified, configurable framework for systematically comparing autoregressive (AR) and masked diffusion (MD) language models under strictly identical training conditions. The library is implemented natively in JAX and Flax NNX, exposing every architectural choice — attention mechanism, feed-forward network, normalisation, positional encoding, and noise schedule — as a single field in a typed `Config` dataclass, without subclassing or source modifications.

The system demonstration presents an end-to-end experimental pipeline that spans **architecture ablations across three attention families** (Multi-Head Attention, Grouped-Query Attention, and Multi-head Latent Attention), multiple model sizes, and feed-forward network variants (Dense and Mixture-of-Experts). All 180 training runs are conducted on WikiText-103, and the resulting checkpoints are evaluated across three complementary axes: **language modelling quality** (perplexity and bits-per-byte on WikiText-103, Penn Treebank, LAMBADA, and C4), **inference throughput** (tokens per second at varied batch sizes, sequence lengths, and precisions), and **open-ended generation quality** (Distinct-1/2, Self-BLEU, Rep-4, and MAUVE). The framework is designed so that every figure in the paper can be reproduced from raw training to publication-ready PDFs with a single shell command.

---

## Research Questions

The paper addresses four primary research questions:

- **RQ1 — Quality–efficiency tradeoff (AR vs. Diffusion):** Under identical model architectures and training budgets, does masked diffusion achieve competitive perplexity relative to autoregressive language modelling, and at what throughput cost? Does the bidirectional context of diffusion models confer a systematic advantage on long-range coherence benchmarks such as LAMBADA?

- **RQ2 — Attention mechanism impact:** Across the size and paradigm matrix, how do MHA, GQA (×4 reduction in KV heads), and MLA (decoupled RoPE with weight absorption) differ in terms of language modelling loss, generation quality, KV-cache memory footprint, and inference throughput? Does the effect size vary between AR and diffusion paradigms?

- **RQ3 — Mixture-of-Experts routing effects:** For matched parameter counts and FLOPs budgets, does MoE (top-2 of 6 experts) improve perplexity relative to Dense FFN? How does MoE interact with attention type and generation paradigm, and what are the per-step latency implications?

- **RQ4 — Confidence-based decoding in masked diffusion:** Does setting a per-token confidence threshold τ during Fast-dLLM DualCache generation improve generation quality metrics (Distinct-2, MAUVE) relative to fixed-step decoding, and what is the throughput–quality Pareto frontier as τ and block size f vary?

---

## Experimental Design

The training matrix is divided into two complementary parts, for a combined total of approximately 180 checkpoints trained across both paradigms.

### Part A — Size × Attention × FFN Matrix

Part A establishes the primary scaling comparison across all combinations of model size, attention type, and feed-forward network type. Each configuration is trained for both AR and diffusion paradigms.

| `dim` | `n_heads` | `head_size` | `num_blocks` | LR | Optimiser | Dense | MoE |
|------:|----------:|------------:|-------------:|----|-----------|:-----:|:---:|
| 128 | 4 | 32 | 12 | 1.2e-3 | Lion | MHA / GQA / MLA | — |
| 192 | 6 | 32 | 12 | 1.2e-3 | Lion | MHA / GQA / MLA | — |
| 256 | 8 | 32 | 8 | 1.2e-3 | Lion | MHA / GQA / MLA | MHA / GQA / MLA |
| 256 | 8 | 32 | 12 | 1.2e-3 | Lion | MHA / GQA / MLA | MHA / GQA / MLA |
| 256 | 8 | 32 | 16 | 1.0e-3 | AdamW | MHA / GQA / MLA | MHA / GQA / MLA |
| 384 | 12 | 32 | 12 | 1.0e-3 | AdamW | MHA / GQA / MLA | — |
| 512 | 16 | 32 | 8 | 8.0e-4 | AdamW | MHA / GQA / MLA | MHA / GQA / MLA |
| 512 | 16 | 32 | 12 | 8.0e-4 | AdamW | MHA / GQA / MLA | MHA / GQA / MLA |
| 512 | 16 | 32 | 16 | 6.0e-4 | AdamW | MHA / GQA / MLA | MHA / GQA / MLA |
| 768 | 12 | 64 | 12 | 6.0e-4 | AdamW | MHA / GQA / MLA | — |

MoE configurations use 6 experts with top-2 routing (`n_experts=6`, `top_k_mlp=2`). GQA uses a 4:1 query-to-KV-head ratio. MLA uses decoupled RoPE with `down_dim_kv = min(head_size × 3, 256)` and `down_dim_q = min(head_size × 6, 256)`. **Total Part A checkpoints: 10 sizes × 3 attention types × Dense + 6 MoE configs × 3 attention types = 48 runs per paradigm → 96 checkpoints combined.**

### Part B — Architecture Ablations on 256d/12b/Dense Baseline

Part B isolates the effect of individual hyperparameter choices by varying one axis at a time relative to the canonical 256-dimensional, 12-block, Dense baseline. Ablations are replicated across all three attention types and both paradigms.

| Code | Ablation | Changed flag vs. baseline |
|:-----|:---------|:--------------------------|
| `RMSNorm` | Normalisation type | `--norm_type rmsnorm` (vs. LayerNorm) |
| `Drop0` | No dropout | `--dropout_rate 0.0` (vs. 0.15) |
| `Drop20` | Higher dropout | `--dropout_rate 0.20` |
| `GELU` | FFN activation | `--use_swiglu false` (vs. SwiGLU) |
| `SlidingWin64` | Local attention | `--sliding_window true --context_window 64` |
| `NoSink` | Disable sink token | `--no_sink true` |
| `SchedWSD` | LR schedule | `--lr_schedule wsd` (vs. cosine) |
| `OptLion` | Optimiser | `--optimizer lion --lr 3e-4` (vs. AdamW) |
| `MoE8exp` | MoE with 8 experts | `--use_moe true --n_experts 8 --top_k_mlp 2` |
| `BS128` | Larger batch size | `--batch_size 128 --grad_accum 8` |
| `Ctx256` | Shorter context | `--max_context 256` (vs. 512) |
| `Ctx1024` | Longer context | `--max_context 1024` |

**Total Part B checkpoints: 12 ablations × 3 attention types × 2 paradigms = 72 checkpoints.** (The diffusion suite also includes noise schedule ablations `SchedLinear`, `SchedSqrt`, `T500`, and `TimeEmb128`, which are diffusion-specific and do not have AR equivalents.)

**Grand total: ~180 checkpoints across both training suites.**

---

## Evaluation Pipeline

After training, the pipeline proceeds through three stages — inference benchmarks on randomly initialised models, trained-model evaluation, and figure generation — organised as labelled stages in `scripts/run_full_emnlp.sh`.

### Stage B — Inference Benchmarks (Architecture, No Training Required)

| Stage | Script | What it measures | Output |
|:------|:-------|:-----------------|:-------|
| **B1** | `benchmarks/inference_sweep.py` | AR throughput across 13 experimental groups (attention type, scale, batch size, context length, dtype, KV cache, MoE, activation, positional encoding, GQA vs. cache, scale × dtype, batch × attention, sampling strategy) on randomly initialised MHA/GQA/MLA models | `results/inference_sweep.csv` + 21 plots |
| **B2** | `benchmarks/diffusion_ar_sweep.py` | AR vs. Diffusion latency and throughput across 13 equivalent groups; isolates paradigm-specific overhead independent of trained weights | `results/diffusion_ar_sweep.csv` + 20 plots |
| **B3** | `benchmarks/confidence_sweep.py` | Confidence threshold τ and block size f sweep for Fast-dLLM DualCache; measures throughput–quality tradeoff surface (50 configurations, 3 attention types) | `results/confidence_sweep.csv` |

B1 and B2 run in parallel on two GPUs; B3 runs in parallel with stage E1.

### Stage E — Trained-Model Evaluation

| Stage | Script | What it measures | Output |
|:------|:-------|:-----------------|:-------|
| **E1** | `benchmarks/trained_analysis.py` | Per-checkpoint latency, throughput (tok/s), and validation perplexity for every `ar_*` and `diff_*` run directory; 20 measurement trials per run | `results/benchmark_results.csv` |
| **E2** | `benchmarks/trained_batch_sweep.py` | Throughput vs. batch size (1–128) at seq_len=512 for each trained checkpoint; uses E1 CSV to select representative runs | `results/batch_sweep_results.csv` |
| **E3** | `benchmarks/perplexity_eval.py` | Sliding-window bits-per-byte evaluation on WikiText-103, Penn Treebank, LAMBADA, and C4 for all trained checkpoints; AR uses standard CE loss, Diffusion uses ELBO at uniform timestep grid | `results/perplexity.csv` |
| **E4** | `benchmarks/generation_quality.py` | Open-ended generation quality for 100 prompts per checkpoint at gen_len=128: Distinct-1, Distinct-2, Self-BLEU, Rep-4 (repetition at 4-gram level), and MAUVE | `results/generation_quality.csv` |

E3 and E4 run in parallel on two GPUs.

### Stage F — Figure Generation

| Stage | Script | What it produces | Output |
|:------|:-------|:-----------------|:-------|
| **F1** | `benchmarks/plot_inference.py` | 21 publication-quality figures from the inference sweep, covering all 13 experimental groups across MHA/GQA/MLA | `results/plots/` |
| **F2** | `benchmarks/plot_diffusion_ar.py` | 20 figures comparing AR and Diffusion latency/throughput curves | `results/plots/` |
| **F3** | `benchmarks/plot_emnlp.py` | 8 paper-ready figures combining trained-model perplexity, throughput, generation quality, and confidence sweep results; produces a combined PDF suitable for camera-ready submission | `results/paper_figures/` + PDF |

---

## Reproducing Results

The complete pipeline — from raw training through to camera-ready figures — is encapsulated in a single shell script. Completed training runs are automatically skipped (checkpoint existence is checked before each run), and benchmark stages that have already produced a CSV are also skipped by default.

```bash
# Full pipeline: training → benchmarks → evaluation → figures
# Estimated wall time: 6–10 hours (training dominates)
# Hardware: 2× NVIDIA A100 40 GB for training; all GPUs for benchmarks
bash scripts/run_full_emnlp.sh

# Skip the training stage — run benchmarks and evaluation only
# (requires existing checkpoints in runs/)
bash scripts/run_full_emnlp.sh --skip-training

# Skip inference benchmarks — training + evaluation only
bash scripts/run_full_emnlp.sh --skip-benchmarks

# Re-generate all plots from existing CSVs without re-running any experiments
bash scripts/run_full_emnlp.sh --only-plots

# Print all commands that would be executed, without running any
bash scripts/run_full_emnlp.sh --dry-run

# Re-run benchmark stages even if output CSVs already exist
bash scripts/run_full_emnlp.sh --skip-training --force

# Restrict training to Part A (size × attention matrix) only
PART=A bash scripts/run_full_emnlp.sh

# Restrict training to a single attention type
ATTN=mla bash scripts/run_full_emnlp.sh

# Override GPU assignment (defaults: TRAIN_DEVICE=0,1, DEVICE=0)
TRAIN_DEVICE=2,3 bash scripts/run_full_emnlp.sh
```

**Disk requirements:** Training requires at least 15 GB free at the start of each run (checked automatically). If disk space runs low, run `python scripts/cleanup_runs.py --execute` to remove intermediate checkpoints while preserving final weights.

After a full pipeline run, outputs are organised as follows:

```
results/
├── inference_sweep.csv        # B1 raw measurements
├── diffusion_ar_sweep.csv     # B2 raw measurements
├── confidence_sweep.csv       # B3 raw measurements
├── benchmark_results.csv      # E1 trained-model throughput/latency
├── batch_sweep_results.csv    # E2 batch-size throughput
├── perplexity.csv             # E3 bpb on WT103/PTB/LAMBADA/C4
├── generation_quality.csv     # E4 Distinct/MAUVE/Rep-4
├── plots/                     # F1 + F2 figures (~41 PNGs)
└── paper_figures/             # F3 camera-ready figures + combined PDF
```

---

## Key Findings

!!! note "Results pending full pipeline run"
    Results are populated after running the full pipeline.
    See the `results/` directory and `results/paper_figures/` after executing:

    ```bash
    bash scripts/run_full_emnlp.sh
    ```

    The `results/perplexity.csv` file provides per-checkpoint bpb across all four evaluation corpora, and `results/generation_quality.csv` provides Distinct-1/2, Self-BLEU, Rep-4, and MAUVE for all trained models. The combined PDF at `results/paper_figures/*.pdf` aggregates the 8 paper figures.

---

## Citation

If you use DantinoX in your research, please cite:

```bibtex
@inproceedings{simoni2026dantinox,
  title     = {{DantinoX}: A Unified {JAX/Flax} Framework for Autoregressive
               and Masked Diffusion Language Models across {MHA}, {GQA},
               and {MLA} Attention Variants},
  author    = {Simoni, Marco},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods
               in Natural Language Processing: System Demonstrations},
  year      = {2026},
  publisher = {Association for Computational Linguistics},
  url       = {https://github.com/winstonsmith1897/DantinoX},
}

@software{dantinox2026,
  author  = {Simoni, Marco},
  title   = {DantinoX: A Research-Grade Transformer Library in {JAX}},
  year    = {2026},
  url     = {https://github.com/winstonsmith1897/DantinoX},
}
```
