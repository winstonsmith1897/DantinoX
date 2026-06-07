---
title: EMNLP Training Suite
---

# EMNLP 2026 Training Suite

This page documents the systematic training suite used to produce the DantinoX EMNLP 2026 System Demonstration results. The suite comprises two symmetric shell scripts — `scripts/train_ar_suite.sh` and `scripts/train_diffusion_suite.sh` — that together train approximately 180 checkpoints under a controlled experimental design. Both scripts are orchestrated by the top-level pipeline driver `scripts/run_full_emnlp.sh`.

---

## Overview

The training suite is designed around a single overarching principle: **the only variable between any AR and Diffusion comparison point is `model_type`**. Architecture dimensions, attention hyperparameters, optimiser settings, dataset, tokeniser, precision, and hardware configuration are kept strictly identical across paradigms. This ensures that any observed difference in perplexity, throughput, or generation quality is attributable to the generation paradigm rather than to confounding training choices.

The suite is divided into two parts:

- **Part A** establishes the primary scaling comparison: a full crossing of model size, attention type (MHA, GQA, MLA), and feed-forward network type (Dense, MoE). This produces 48 checkpoints per paradigm.
- **Part B** isolates the effect of individual hyperparameter axes via controlled ablations, all anchored to the canonical 256-dimensional, 12-block, Dense baseline. This produces an additional 36 checkpoints per paradigm for AR (12 ablations × 3 attention types) and up to 42 for diffusion (which adds diffusion-specific noise schedule and time embedding ablations).

All runs are **idempotent**: if a checkpoint already exists in `runs/<tag>/`, the run is silently skipped. This makes it safe to interrupt and restart the suite at any point, and to add new ablations to the scripts without re-running completed work.

---

## Part A — Size × Attention × FFN Matrix

### Dense Configurations

The following ten model sizes are trained for all three attention types in Dense FFN mode. The learning rate and optimiser were selected by a prior sweep to be appropriate for each width.

| `dim` | `n_heads` | `head_size` | `num_blocks` | LR | Optimiser | Run tag pattern |
|------:|----------:|------------:|-------------:|----|-----------|:----------------|
| 128 | 4 | 32 | 12 | 1.2e-3 | Lion | `ar_{attn}_128d_12b_Dense` |
| 192 | 6 | 32 | 12 | 1.2e-3 | Lion | `ar_{attn}_192d_12b_Dense` |
| 256 | 8 | 32 | 8 | 1.2e-3 | Lion | `ar_{attn}_256d_8b_Dense` |
| 256 | 8 | 32 | 12 | 1.2e-3 | Lion | `ar_{attn}_256d_12b_Dense` |
| 256 | 8 | 32 | 16 | 1.0e-3 | AdamW | `ar_{attn}_256d_16b_Dense` |
| 384 | 12 | 32 | 12 | 1.0e-3 | AdamW | `ar_{attn}_384d_12b_Dense` |
| 512 | 16 | 32 | 8 | 8.0e-4 | AdamW | `ar_{attn}_512d_8b_Dense` |
| 512 | 16 | 32 | 12 | 8.0e-4 | AdamW | `ar_{attn}_512d_12b_Dense` |
| 512 | 16 | 32 | 16 | 6.0e-4 | AdamW | `ar_{attn}_512d_16b_Dense` |
| 768 | 12 | 64 | 12 | 6.0e-4 | AdamW | `ar_{attn}_768d_12b_Dense` |

`{attn}` is one of `mha`, `gqa`, or `mla`. For diffusion runs, replace the `ar_` prefix with `diff_`. The full set of Dense runs is therefore **10 sizes × 3 attention types × 2 paradigms = 60 checkpoints**.

### MoE Configurations

MoE variants are trained at the 256 and 512 width points, across all three depth configurations available at those widths, giving six MoE configurations per paradigm:

| `dim` | `n_heads` | `head_size` | `num_blocks` | LR | Optimiser | Run tag pattern |
|------:|----------:|------------:|-------------:|----|-----------|:----------------|
| 256 | 8 | 32 | 8 | 1.2e-3 | Lion | `ar_{attn}_256d_8b_MoE` |
| 256 | 8 | 32 | 12 | 1.2e-3 | Lion | `ar_{attn}_256d_12b_MoE` |
| 256 | 8 | 32 | 16 | 1.0e-3 | AdamW | `ar_{attn}_256d_16b_MoE` |
| 512 | 16 | 32 | 8 | 8.0e-4 | AdamW | `ar_{attn}_512d_8b_MoE` |
| 512 | 16 | 32 | 12 | 8.0e-4 | AdamW | `ar_{attn}_512d_12b_MoE` |
| 512 | 16 | 32 | 16 | 6.0e-4 | AdamW | `ar_{attn}_512d_16b_MoE` |

All MoE runs use `n_experts=6`, `top_k_mlp=2`. **Total MoE runs: 6 configs × 3 attention types × 2 paradigms = 36 checkpoints.**

### Attention Type Parameterisation

Each attention type has a fixed parameterisation derived from the base `n_heads` and `head_size` of the size configuration:

=== "MHA (Multi-Head Attention)"

    ```bash
    --kv_heads {n_heads} --mla false
    ```

    Standard multi-head attention with one KV head per query head.

=== "GQA (Grouped-Query Attention)"

    ```bash
    --kv_heads {n_heads // 4} --mla false
    ```

    Groups of 4 query heads share a single KV head pair, reducing the KV cache by 4×.

=== "MLA (Multi-head Latent Attention)"

    ```bash
    --kv_heads {n_heads} --mla true --inference false \
    --down_dim_kv {min(head_size * 3, 256)} \
    --down_dim_q  {min(head_size * 6, 256)} \
    --rope_dim    {max(head_size // 2, 16)}
    ```

    Decoupled RoPE with compressed KV latent space. Weight absorption is disabled during training (`--inference false`) and enabled at inference time. The latent dimension grows with head size but is capped at 256 to prevent oversized projections at the 768d configuration.

---

## Part B — Architecture Ablations

All Part B ablations are anchored to the canonical 256-dimensional, 12-block, Dense baseline:

```
dim=256  n_heads=8  head_size=32  num_blocks=12  lr=1.2e-3  optimizer=lion  use_moe=false
```

Each ablation modifies exactly one axis relative to this baseline and is replicated across all three attention types, giving 12 ablations × 3 attention types = 36 run tags per paradigm for the shared ablations. The diffusion suite adds diffusion-specific ablations (noise schedule, time embedding) which do not have AR equivalents.

### Shared Ablations (AR and Diffusion)

| Label | What changes | Flag(s) | Purpose |
|:------|:-------------|:--------|:--------|
| `RMSNorm` | Normalisation type | `--norm_type rmsnorm` | RMSNorm vs. LayerNorm effect on convergence and final loss |
| `Drop0` | Dropout rate | `--dropout_rate 0.0` | No regularisation vs. 15% baseline |
| `Drop20` | Dropout rate | `--dropout_rate 0.20` | Heavier regularisation |
| `GELU` | FFN activation | `--use_swiglu false` | Standard GELU FFN vs. SwiGLU gated FFN |
| `SlidingWin64` | Attention span | `--sliding_window true --context_window 64` | Local-only attention with 64-token window vs. full context |
| `NoSink` | Sink token | `--no_sink true` | Effect of removing the sink (first-position) attention token |
| `SchedWSD` | LR schedule | `--lr_schedule wsd` | Warmup-stable-decay schedule vs. cosine annealing |
| `OptLion` | Optimiser | `--optimizer lion --lr 3e-4` | Lion optimiser (adjusted LR) vs. AdamW baseline |
| `MoE8exp` | FFN type | `--use_moe true --n_experts 8 --top_k_mlp 2` | MoE with 8 experts on the 256d baseline (cf. 6-expert Part A) |
| `BS128` | Batch size | `--batch_size 128 --grad_accum 8` | 2× larger effective batch size (128 × 8 = 1024 tokens per step) |
| `Ctx256` | Context length | `--max_context 256` | Half the default 512-token context window |
| `Ctx1024` | Context length | `--max_context 1024` | Double the default context window |

### Diffusion-Specific Ablations

The following ablations apply only to the diffusion training suite (`train_diffusion_suite.sh`) and do not have corresponding AR runs:

| Label | What changes | Flag(s) | Purpose |
|:------|:-------------|:--------|:--------|
| `SchedLinear` | Noise schedule | `--noise_schedule linear` | Linear corruption schedule vs. cosine |
| `SchedSqrt` | Noise schedule | `--noise_schedule sqrt` | Square-root corruption schedule |
| `T500` | Diffusion steps | `--diffusion_steps 500` | Coarser timestep grid (500 vs. 1000 steps) |
| `TimeEmb128` | Time embedding dim | `--time_emb_dim 128` | Smaller time conditioning MLP (128 vs. 256 dims) |

---

## Run Directory Naming Convention

Every training run produces a self-contained directory under `runs/`. The naming convention encodes all variable axes:

```
runs/{paradigm}_{attn}_{dim}d_{blocks}b_{variant}/
```

| Component | Values | Example |
|:----------|:-------|:--------|
| `{paradigm}` | `ar`, `diff` | `ar` |
| `{attn}` | `mha`, `gqa`, `mla` | `mla` |
| `{dim}d` | `128d`–`768d` | `256d` |
| `{blocks}b` | `8b`–`16b` | `12b` |
| `{variant}` | `Dense`, `MoE`, ablation label | `Dense`, `RMSNorm`, `BS128` |

Full examples:

```
runs/ar_mha_256d_12b_Dense/          # AR, MHA, 256d, 12 blocks, Dense FFN
runs/diff_mla_512d_16b_MoE/          # Diffusion, MLA, 512d, 16 blocks, MoE FFN
runs/ar_gqa_256d_12b_SlidingWin64/   # AR, GQA, 256d, 12 blocks, sliding-window ablation
runs/diff_mha_256d_12b_T500/         # Diffusion, MHA, 256d, 12 blocks, 500-step ablation
```

Each run directory contains:

```
runs/<tag>/
├── config.yaml                  # complete config snapshot for this run
├── tokenizer.json               # character-level tokenizer (shared across runs)
├── model_weights.msgpack        # latest checkpoint
├── best_model_weights.msgpack   # checkpoint with lowest validation loss
├── training_cursor.json         # resume pointer (step, epoch, best loss)
├── model_summary.json           # parameter count and VRAM estimate
└── training_log.csv             # step-by-step train_loss, val_loss, ms/step
```

---

## Common Training Flags

The following flags are fixed across all runs in both suites and are not varied as ablations:

| Flag | Value | Notes |
|:-----|:------|:------|
| `--n_devices` | `2` | Data-parallel across 2 A100s via JAX SPMD |
| `--use_bf16` | `true` | bfloat16 mixed-precision training |
| `--use_flash_attention` | `true` | Flash Attention 2 for O(N) memory attention |
| `--gradient_checkpointing` | `true` | Recompute activations during backward pass to reduce peak VRAM |
| `--dataset_source` | `huggingface` | Dataset loaded via HuggingFace `datasets` |
| `--dataset_name` | `wikitext` | WikiText dataset family |
| `--dataset_config` | `wikitext-103-raw-v1` | Raw character-level WikiText-103 |
| `--dataset_text_field` | `text` | Field name in the HuggingFace dataset |
| `--dataset_split` | `train` | Training split |
| `--streaming` | `false` | Full download and pre-tokenisation cache |

The pre-tokenised dataset is cached to `data/wikitext_wikitext-103-raw-v1_char.npy` after the first run and reused by all subsequent runs, reducing per-run startup from approximately 60 seconds to 2 seconds.

---

## Progress Monitoring

### Live log tailing

Each run writes a log file to `logs/ar_suite/<tag>.log` or `logs/diffusion_suite/<tag>.log`. To monitor a run in real time:

```bash
tail -f logs/ar_suite/ar_mha_256d_12b_Dense.log
```

### Training CSV

The per-step CSV is updated at each validation interval and can be plotted directly:

```bash
# Quick loss curve (requires pandas + matplotlib)
python - <<'EOF'
import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv("runs/ar_mha_256d_12b_Dense/training_log.csv")
df.plot(x="step", y=["train_loss", "val_loss"])
plt.savefig("/tmp/loss_curve.png")
EOF
```

### Suite progress summary

The training scripts print a running count of completed, skipped, and failed runs at the end of each execution:

```
════════════════════════════════════════════════════════════
  Done: trained=12  skipped=36  failed=0  total=48
════════════════════════════════════════════════════════════
```

### W&B integration

If `WANDB_API_KEY` is set in the environment, every run automatically logs to a W&B project named `dantinox`. Sweeps across the full suite can be analysed via the W&B web interface using the run tags as the primary grouping key.

---

## Resuming Failed Runs

The training scripts are fully idempotent. A run is considered complete if either `model_weights.msgpack` or `best_model_weights.msgpack` exists in its run directory. To resume a suite after an interruption:

```bash
# Simply re-run the script — completed runs are skipped automatically
bash scripts/train_ar_suite.sh

# Or resume via the top-level pipeline
bash scripts/run_full_emnlp.sh --skip-benchmarks
```

If a run failed mid-epoch (e.g., due to OOM or preemption), the `training_cursor.json` in the run directory stores the last completed step. The `Trainer` will resume from this cursor rather than restarting from scratch:

```bash
# Force a specific run to restart from its cursor (not from scratch)
dantinox train --config runs/ar_mha_256d_12b_Dense/config.yaml \
               --run_dir runs/ar_mha_256d_12b_Dense
```

To force a full restart of a specific run, delete its run directory:

```bash
rm -rf runs/ar_mha_256d_12b_Dense/
```

---

## Partial Suite Execution

Both training scripts support environment variable filters to run a subset of the matrix:

```bash
# Run only Part A (size × attention × FFN matrix), skip Part B ablations
PART=A bash scripts/train_ar_suite.sh

# Run only Part B ablations, skip Part A
PART=B bash scripts/train_ar_suite.sh

# Run only MLA attention configurations
ATTN=mla bash scripts/train_ar_suite.sh

# Run only 256-dimensional models
DIM=256 bash scripts/train_ar_suite.sh

# Run only Dense FFN models (skip MoE)
MOE=dense bash scripts/train_ar_suite.sh

# Combine filters: MLA attention, 512d models only
ATTN=mla DIM=512 bash scripts/train_ar_suite.sh

# Print all commands without executing (dry run)
bash scripts/train_ar_suite.sh --dry-run
```

The same filters apply to `train_diffusion_suite.sh`.

---

## Hardware Requirements

| Resource | Minimum | Recommended |
|:---------|:--------|:------------|
| GPUs | 1× NVIDIA GPU (CUDA 12+) | 2× NVIDIA A100 40 GB |
| GPU VRAM (per card) | 24 GB (for 256d/12b models) | 40 GB (for 768d/12b models) |
| System RAM | 32 GB | 64 GB |
| Disk (for all checkpoints) | 50 GB | 150 GB |
| Dataset download | ~500 MB (WikiText-103) | — |

If only a single GPU is available, set `--n_devices 1` in the relevant training commands or override via:

```bash
CUDA_VISIBLE_DEVICES=0 PART=A DIM=256 bash scripts/train_ar_suite.sh
```

Note that with a single GPU, `n_devices=1` must also be reflected in the config or passed as an override flag; the default base configs assume 2 devices.

For disk space management, the cleanup script removes intermediate checkpoints while preserving final weights:

```bash
python scripts/cleanup_runs.py --dry-run    # preview what would be deleted
python scripts/cleanup_runs.py --execute    # delete intermediate checkpoints
```
