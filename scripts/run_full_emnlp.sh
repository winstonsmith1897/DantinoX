#!/usr/bin/env bash
# scripts/run_full_emnlp.sh
#
# End-to-end pipeline for the DantinoX EMNLP 2026 System Demo paper.
#
# ══════════════════════════════════════════════════════════════════════════════
# Pipeline stages
# ══════════════════════════════════════════════════════════════════════════════
#
# TRAINING  (run once — skips already-completed runs)
# ─────────
#  T1  AR training suite       (~84 runs, skips existing ones)
#  T2  Diffusion training suite (~96 runs)
#
# INFERENCE BENCHMARKS  (random models, no training required)
# ─────────────────────
#  B1  AR inference sweep      (13 groups × MHA/GQA/MLA → 21 plots)
#  B2  AR vs Diffusion sweep   (13 groups → 20 plots)
#  B3  Confidence sweep        (τ/f × attention → tradeoff curves)
#
# TRAINED-MODEL EVALUATION
# ─────────────────────────
#  E1  Trained analysis        (latency / throughput / val-PPL per run)
#  E2  Batch sweep             (tok/s vs batch size)
#  E3  Perplexity eval         (WikiText-103 / PTB / LAMBADA / C4 — bpb metric)
#  E4  Generation quality      (Distinct-1/2 / Self-BLEU / Rep-4 / MAUVE)
#
# FIGURE GENERATION
# ─────────────────
#  F1  Inference plots (21 figures)
#  F2  AR vs Diffusion plots (20 figures)
#  F3  EMNLP paper figures (8 publication-ready figures + PDF)
#
# ══════════════════════════════════════════════════════════════════════════════
# Hardware:   2 × A100 for training; all GPUs for benchmarks
# Total time: ~6–10 h (training dominates)
# ══════════════════════════════════════════════════════════════════════════════
#
# Usage:
#   bash scripts/run_full_emnlp.sh                     # full pipeline
#   bash scripts/run_full_emnlp.sh --skip-training     # benchmarks only
#   bash scripts/run_full_emnlp.sh --skip-benchmarks   # training only
#   bash scripts/run_full_emnlp.sh --only-plots        # re-plot from existing CSVs
#   bash scripts/run_full_emnlp.sh --dry-run           # print commands only
#   PART=A bash scripts/run_full_emnlp.sh              # filter to Part A runs
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

# ── Parse flags ───────────────────────────────────────────────────────────────
SKIP_TRAINING=false
SKIP_BENCHMARKS=false
ONLY_PLOTS=false
DRY_RUN=false
DEVICE="${DEVICE:-0}"          # CUDA device for inference benchmarks
TRAIN_DEVICE="${TRAIN_DEVICE:-0,1}"  # GPUs for training (2 GPUs)
FORCE=false                    # --force re-runs benchmarks even if CSV exists

for arg in "$@"; do
    case "${arg}" in
        --skip-training)   SKIP_TRAINING=true ;;
        --skip-benchmarks) SKIP_BENCHMARKS=true ;;
        --only-plots)      ONLY_PLOTS=true; SKIP_TRAINING=true ;;
        --dry-run)         DRY_RUN=true ;;
        --device=*)        DEVICE="${arg#--device=}" ;;
        --train-device=*)  TRAIN_DEVICE="${arg#--train-device=}" ;;
        --force)           FORCE=true ;;
    esac
done

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS="results"
PLOTS="${RESULTS}/plots"
PAPER="${RESULTS}/paper_figures"
LOGS="logs"
mkdir -p "${RESULTS}" "${PLOTS}" "${PAPER}" "${LOGS}"

INF_CSV="${RESULTS}/inference_sweep.csv"
DIFF_AR_CSV="${RESULTS}/diffusion_ar_sweep.csv"
TRAINED_CSV="${RESULTS}/benchmark_results.csv"
BATCH_CSV="${RESULTS}/batch_sweep_results.csv"
PPL_CSV="${RESULTS}/perplexity.csv"
CONF_CSV="${RESULTS}/confidence_sweep.csv"
GENQ_CSV="${RESULTS}/generation_quality.csv"

# ── Helpers ───────────────────────────────────────────────────────────────────
TS()   { date +'%H:%M:%S'; }
BANNER() { echo ""; echo "════════════════════════════════════════════════════════════"; echo "  [$(TS)] $*"; echo "════════════════════════════════════════════════════════════"; }
RUN()  {
    echo "  \$ $*"
    [[ "${DRY_RUN}" == "true" ]] && return 0
    "$@"
}

check_disk() {
    local free_kb free_gb
    free_kb=$(df -k "${ROOT}" | awk 'NR==2{print $4}')
    free_gb=$(( free_kb / 1024 / 1024 ))
    echo "  [$(TS)] Disk free: ${free_gb} GB"
    if (( free_gb < 10 )); then
        echo "ERROR: only ${free_gb} GB free — need 10 GB. Run: python scripts/cleanup_runs.py --execute" >&2
        exit 1
    fi
}

# ── Pipeline ──────────────────────────────────────────────────────────────────

# ── T1+T2: Training ───────────────────────────────────────────────────────────
if [[ "${SKIP_TRAINING}" == "false" ]]; then
    check_disk
    BANNER "T1 — AR training suite (~84 runs, GPUs ${TRAIN_DEVICE})"
    RUN env CUDA_VISIBLE_DEVICES="${TRAIN_DEVICE}" bash scripts/train_ar_suite.sh

    check_disk
    BANNER "T2 — Diffusion training suite (~96 runs, GPUs ${TRAIN_DEVICE})"
    RUN env CUDA_VISIBLE_DEVICES="${TRAIN_DEVICE}" bash scripts/train_diffusion_suite.sh
fi

if [[ "${SKIP_BENCHMARKS}" == "true" ]]; then
    BANNER "Skipping benchmarks (--skip-benchmarks)"
    exit 0
fi

# Derive the two GPU IDs from TRAIN_DEVICE (e.g. "0,1" → GPU_A=0, GPU_B=1).
# Falls back to DEVICE (single GPU) if TRAIN_DEVICE has only one entry.
GPU_A=$(echo "${TRAIN_DEVICE}" | cut -d',' -f1)
GPU_B=$(echo "${TRAIN_DEVICE}" | cut -d',' -f2)
[[ -z "${GPU_B}" || "${GPU_B}" == "${GPU_A}" ]] && GPU_B="${GPU_A}"

BG_BENCH_LOG="${LOGS}/bench_gpu_b.log"

# ── B1 + B2: run in parallel on GPU_A / GPU_B ────────────────────────────────
BANNER "B1+B2 — AR inference sweep & AR-vs-Diffusion sweep (parallel)"

if [[ ! -f "${INF_CSV}" || "${FORCE}" == "true" ]]; then
    echo "  [GPU ${GPU_A}] B1 — inference_sweep.py"
    if [[ "${DRY_RUN}" == "false" ]]; then
        env CUDA_VISIBLE_DEVICES="${GPU_A}" python benchmarks/inference_sweep.py \
            --out "${INF_CSV}" --n-warmup 3 --n-trials 10 --verbose &
        B1_PID=$!
    fi
else
    echo "  [SKIP] B1 — ${INF_CSV} exists"; B1_PID=""
fi

if [[ ! -f "${DIFF_AR_CSV}" || "${FORCE}" == "true" ]]; then
    echo "  [GPU ${GPU_B}] B2 — diffusion_ar_sweep.py"
    if [[ "${DRY_RUN}" == "false" ]]; then
        env CUDA_VISIBLE_DEVICES="${GPU_B}" python benchmarks/diffusion_ar_sweep.py \
            --out "${DIFF_AR_CSV}" --n-warmup 3 --n-trials 10 --verbose \
            >> "${BG_BENCH_LOG}" 2>&1 &
        B2_PID=$!
    fi
else
    echo "  [SKIP] B2 — ${DIFF_AR_CSV} exists"; B2_PID=""
fi

[[ -n "${B1_PID:-}" ]] && wait "${B1_PID}" && echo "  [OK] B1"
[[ -n "${B2_PID:-}" ]] && wait "${B2_PID}" && echo "  [OK] B2  (log: ${BG_BENCH_LOG})"

# ── B3 + E1: run in parallel on GPU_A / GPU_B ────────────────────────────────
BANNER "B3+E1 — Confidence sweep & Trained analysis (parallel)"

if [[ ! -f "${CONF_CSV}" || "${FORCE}" == "true" ]]; then
    echo "  [GPU ${GPU_A}] B3 — confidence_sweep.py"
    if [[ "${DRY_RUN}" == "false" ]]; then
        env CUDA_VISIBLE_DEVICES="${GPU_A}" python benchmarks/confidence_sweep.py \
            --out "${CONF_CSV}" --n-runs 50 --n-warmup 3 --n-measure 10 --verbose &
        B3_PID=$!
    fi
else
    echo "  [SKIP] B3 — ${CONF_CSV} exists"; B3_PID=""
fi

echo "  [GPU ${GPU_B}] E1 — trained_analysis.py"
if [[ "${DRY_RUN}" == "false" ]]; then
    env CUDA_VISIBLE_DEVICES="${GPU_B}" python benchmarks/trained_analysis.py \
        --runs-dir runs --run-prefix ar_ diff_ --out-csv "${TRAINED_CSV}" \
        --out-plot "${PLOTS}/trained_analysis.png" \
        --n-warmup 3 --n-trials 20 \
        >> "${BG_BENCH_LOG}" 2>&1 &
    E1_PID=$!
fi

[[ -n "${B3_PID:-}" ]] && wait "${B3_PID}" && echo "  [OK] B3"
[[ -n "${E1_PID:-}" ]] && wait "${E1_PID}" && echo "  [OK] E1  (log: ${BG_BENCH_LOG})"

# ── E2 (needs E1 CSV) ─────────────────────────────────────────────────────────
BANNER "E2 — Batch-size throughput sweep"
RUN env CUDA_VISIBLE_DEVICES="${GPU_A}" python benchmarks/trained_batch_sweep.py \
    --runs-dir runs --run-prefix ar_ diff_ --out-csv "${BATCH_CSV}" \
    --analysis-csv "${TRAINED_CSV}" \
    --batch-sizes 1 2 4 8 16 32 64 128 --seq-len 512

# ── E3 + E4: run in parallel on GPU_A / GPU_B ────────────────────────────────
BANNER "E3+E4 — Perplexity eval & Generation quality (parallel)"

echo "  [GPU ${GPU_A}] E3 — perplexity_eval.py"
if [[ "${DRY_RUN}" == "false" ]]; then
    env CUDA_VISIBLE_DEVICES="${GPU_A}" python benchmarks/perplexity_eval.py \
        --runs-dir runs \
        --run-prefix ar_ diff_ \
        --datasets wikitext-103 ptb lambada c4 dante \
        --max-windows 200 \
        --out "${PPL_CSV}" &
    E3_PID=$!
fi

echo "  [GPU ${GPU_B}] E4 — generation_quality.py"
if [[ "${DRY_RUN}" == "false" ]]; then
    env CUDA_VISIBLE_DEVICES="${GPU_B}" python benchmarks/generation_quality.py \
        --runs-dir runs \
        --run-prefix ar_ diff_ \
        --n-samples 100 --gen-len 128 \
        --out "${GENQ_CSV}" \
        >> "${BG_BENCH_LOG}" 2>&1 &
    E4_PID=$!
fi

[[ -n "${E3_PID:-}" ]] && wait "${E3_PID}" && echo "  [OK] E3"
[[ -n "${E4_PID:-}" ]] && wait "${E4_PID}" && echo "  [OK] E4  (log: ${BG_BENCH_LOG})"

# ── F1: Inference plots ───────────────────────────────────────────────────────
BANNER "F1 — Inference sweep plots (21 figures)"
RUN python benchmarks/plot_inference.py \
    --csv "${INF_CSV}" --out-dir "${PLOTS}"

# ── F2: AR vs Diffusion plots ─────────────────────────────────────────────────
BANNER "F2 — AR vs Diffusion plots (20 figures)"
RUN python benchmarks/plot_diffusion_ar.py \
    --csv "${DIFF_AR_CSV}" --out "${PLOTS}"

# ── F3: EMNLP paper figures ───────────────────────────────────────────────────
BANNER "F3 — EMNLP paper figures (8 figures + PDF)"
RUN python benchmarks/plot_emnlp.py \
    --out-dir          "${PAPER}" \
    --trained-csv      "${TRAINED_CSV}" \
    --ppl-csv          "${PPL_CSV}" \
    --diffusion-ar-csv "${DIFF_AR_CSV}" \
    --confidence-csv   "${CONF_CSV}" \
    --gen-quality-csv  "${GENQ_CSV}" \
    --pdf

# ── Summary ───────────────────────────────────────────────────────────────────
BANNER "COMPLETE"
echo "  Inference CSV    : ${INF_CSV}"
echo "  Diff-AR CSV      : ${DIFF_AR_CSV}"
echo "  Trained CSV      : ${TRAINED_CSV}"
echo "  Perplexity CSV   : ${PPL_CSV}"
echo "  Gen-quality CSV  : ${GENQ_CSV}"
n_plots=$(find "${PLOTS}" "${PAPER}" -name "*.png" 2>/dev/null | wc -l)
echo "  Figures          : ${n_plots} PNGs in ${PLOTS}/ and ${PAPER}/"
echo ""
echo "  → Run 'make benchmark-full' to re-run benchmarks only."
