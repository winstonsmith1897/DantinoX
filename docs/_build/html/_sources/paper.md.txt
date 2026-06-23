---
title: Experiments & Results
---

# Experiments & Results

<div class="hero-badges" markdown>
[![JAX](https://img.shields.io/badge/JAX-000000?style=flat-square&logo=JAX&logoColor=white)](https://github.com/google/jax)
[![Flax NNX](https://img.shields.io/badge/Flax_NNX-5E17EB?style=flat-square&logoColor=white)](https://github.com/google/flax)
[![License MIT](https://img.shields.io/badge/License-MIT-16a34a?style=flat-square)](https://opensource.org/licenses/MIT)
</div>

DantinoX is a unified, configurable framework for systematically comparing autoregressive (AR), masked discrete diffusion, and continuous flow-matching (ELF) language models under strictly identical training conditions. This page documents the experimental design, training matrix, and evaluation pipeline.

---

## Research Questions

- **RQ1 — Quality–efficiency tradeoff (AR vs. Diffusion):** Under identical architectures and training budgets, does masked diffusion achieve competitive perplexity relative to autoregressive LM, and at what throughput cost?

- **RQ2 — Attention mechanism impact:** Across the size and paradigm matrix, how do MHA, GQA (×4 reduction in KV heads), and MLA (decoupled RoPE with weight absorption) differ in language modelling loss, generation quality, and inference throughput?

- **RQ3 — Mixture-of-Experts routing effects:** For matched parameter counts and FLOPs budgets, does MoE (top-2 of 6 experts) improve perplexity relative to Dense FFN across paradigms?

- **RQ4 — Confidence-based decoding in masked diffusion:** Does a per-token confidence threshold during Fast-dLLM DualCache generation improve quality metrics relative to fixed-step decoding?

---

## Experimental Design

The training matrix is divided into two complementary parts, for a combined total of approximately 180 checkpoints.

### Part A — Size × Attention × FFN Matrix

Each configuration is trained for both AR and diffusion paradigms under identical hyperparameters.

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

MoE configurations use 6 experts with top-2 routing. GQA uses a 4:1 query-to-KV-head ratio. MLA uses decoupled RoPE with `down_dim_kv = min(head_size × 3, 256)` and `down_dim_q = min(head_size × 6, 256)`.

**Total Part A: 10 sizes × 3 attention types × Dense + 6 MoE configs × 3 attention types = 48 runs per paradigm → 96 checkpoints combined.**

### Part B — Architecture Ablations (256d / 12b / Dense baseline)

Part B isolates the effect of individual hyperparameter choices, varying one axis at a time relative to the canonical baseline. Ablations are replicated across all three attention types and both paradigms.

| Code | Ablation | Changed flag vs. baseline |
|:-----|:---------|:--------------------------|
| `RMSNorm` | Normalisation type | `--norm_type rmsnorm` |
| `Drop0` | No dropout | `--dropout_rate 0.0` |
| `Drop20` | Higher dropout | `--dropout_rate 0.20` |
| `GELU` | FFN activation | `--use_swiglu false` |
| `SlidingWin64` | Local attention | `--sliding_window true --context_window 64` |
| `NoSink` | Disable sink token | `--no_sink true` |
| `SchedWSD` | LR schedule | `--lr_schedule wsd` |
| `OptLion` | Optimiser | `--optimizer lion --lr 3e-4` |
| `MoE8exp` | MoE with 8 experts | `--use_moe true --n_experts 8 --top_k_mlp 2` |
| `BS128` | Larger batch size | `--batch_size 128 --grad_accum 8` |
| `Ctx256` | Shorter context | `--max_context 256` |
| `Ctx1024` | Longer context | `--max_context 1024` |

**Total Part B: 12 ablations × 3 attention types × 2 paradigms = 72 checkpoints.**

**Grand total: ~180 checkpoints across both training suites.**

---

## Evaluation Pipeline

After training, the pipeline runs three sequential stages.

### Stage B — Inference Benchmarks

| Stage | Script | What it measures | Output |
|:------|:-------|:-----------------|:-------|
| **B1** | `benchmarks/inference_sweep.py` | AR throughput across 13 experimental groups on randomly initialised MHA/GQA/MLA models | `results/inference_sweep.csv` + 21 plots |
| **B2** | `benchmarks/diffusion_ar_sweep.py` | AR vs. Diffusion latency and throughput across equivalent groups | `results/diffusion_ar_sweep.csv` + 20 plots |
| **B3** | `benchmarks/confidence_sweep.py` | Confidence threshold τ and block size sweep for Fast-dLLM DualCache (50 configurations, 3 attention types) | `results/confidence_sweep.csv` |

### Stage E — Trained-Model Evaluation

| Stage | Script | What it measures | Output |
|:------|:-------|:-----------------|:-------|
| **E1** | `benchmarks/trained_analysis.py` | Per-checkpoint latency, throughput (tok/s), and validation perplexity for every trained checkpoint | `results/benchmark_results.csv` |
| **E2** | `benchmarks/trained_batch_sweep.py` | Throughput vs. batch size (1–128) at seq_len=512 | `results/batch_sweep_results.csv` |
| **E3** | `benchmarks/perplexity_eval.py` | Sliding-window bits-per-byte on WikiText-103, Penn Treebank, LAMBADA, and C4 | `results/perplexity.csv` |
| **E4** | `benchmarks/generation_quality.py` | Open-ended generation quality: Distinct-1/2, Self-BLEU, Rep-4, and MAUVE | `results/generation_quality.csv` |

### Stage F — Figure Generation

| Stage | Script | What it produces | Output |
|:------|:-------|:-----------------|:-------|
| **F1** | `benchmarks/plot_inference.py` | 21 figures from the inference sweep | `results/plots/` |
| **F2** | `benchmarks/plot_diffusion_ar.py` | 20 figures comparing AR and Diffusion throughput curves | `results/plots/` |
| **F3** | `benchmarks/plot_emnlp.py` | 8 summary figures combining perplexity, throughput, generation quality, and confidence sweep | `results/paper_figures/` + PDF |

---

## Running the Full Pipeline

```bash
# Full pipeline: training → benchmarks → evaluation → figures
# Estimated wall time: 6–10 hours (training dominates)
# Hardware: 2× NVIDIA A100 40 GB
bash scripts/run_full_emnlp.sh

# Skip training — run benchmarks on existing checkpoints only
bash scripts/run_full_emnlp.sh --skip-training

# Re-generate all plots from existing CSVs
bash scripts/run_full_emnlp.sh --only-plots

# Dry run — print all commands without executing
bash scripts/run_full_emnlp.sh --dry-run

# Restrict to a single attention type
ATTN=mla bash scripts/run_full_emnlp.sh
```

After a full run, outputs are organised as:

```
results/
├── inference_sweep.csv        # B1 raw measurements
├── diffusion_ar_sweep.csv     # B2 raw measurements
├── confidence_sweep.csv       # B3 raw measurements
├── benchmark_results.csv      # E1 trained-model throughput/latency
├── batch_sweep_results.csv    # E2 batch-size throughput
├── perplexity.csv             # E3 bpb on WT103/PTB/LAMBADA/C4
├── generation_quality.csv     # E4 Distinct/MAUVE/Rep-4
└── plots/                     # F1 + F2 + F3 figures (~49 PNGs + PDF)
```

---

## Citation

If you use DantinoX in your work, please cite:

```bibtex
@software{dantinox2026,
  author  = {Simoni, Marco},
  title   = {DantinoX: A Unified {JAX}/Flax Framework for Autoregressive
             and Masked Diffusion Language Models},
  year    = {2026},
  url     = {https://github.com/winstonsmith1897/DantinoX},
}
```
