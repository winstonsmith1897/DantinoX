#!/usr/bin/env bash
# scripts/run_emnlp_parallel.sh
#
# Parallel EMNLP launcher: runs Diffusion and ELF suites simultaneously on 2 GPUs.
#
# Usage:
#   bash scripts/run_emnlp_parallel.sh                    # defaults: GPU_DIFF=0, GPU_ELF=1
#   GPU_DIFF=0 GPU_ELF=4 bash scripts/run_emnlp_parallel.sh
#   GPU_DIFF=0 GPU_ELF=4 PART=A bash scripts/run_emnlp_parallel.sh   # only Part A
#
# Budget overrides (propagated to both suites):
#   TOKENS_A=50000000 EPOCHS_A=30   -- Part A (default)
#   TOKENS_B=20000000 EPOCHS_B=15   -- Part B ablations (default)
#
# GPU assignment:
#   GPU_DIFF  → train_diffusion_suite.sh
#   GPU_ELF   → train_elf_suite.sh
#
# Timing reference (A100 40GB, bf16, flash-attn):
#   Part A (50M tok, 30 ep):  ~8-12 h per GPU depending on model sizes
#   Part B (20M tok, 15 ep):  ~2-3 h per GPU (ablations on 1024d)
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

GPU_DIFF="${GPU_DIFF:-0}"
GPU_ELF="${GPU_ELF:-1}"
PART="${PART:-all}"
DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# Budget vars are inherited from environment or use defaults inside each suite
export TOKENS_A="${TOKENS_A:-50000000}"
export EPOCHS_A="${EPOCHS_A:-30}"
export TOKENS_B="${TOKENS_B:-20000000}"
export EPOCHS_B="${EPOCHS_B:-15}"

mkdir -p logs

echo "════════════════════════════════════════════════════════════"
echo "  DantinoX — EMNLP parallel launcher"
echo "  Diffusion → GPU ${GPU_DIFF}"
echo "  ELF       → GPU ${GPU_ELF}"
echo "  Part      : ${PART}"
echo "  Part A budget: ${TOKENS_A} tokens × ${EPOCHS_A} epochs"
echo "  Part B budget: ${TOKENS_B} tokens × ${EPOCHS_B} epochs"
echo "  Dry-run   : ${DRY_RUN}"
echo "════════════════════════════════════════════════════════════"

if [[ "${DRY_RUN}" == "true" ]]; then
    echo ""
    echo "[dry-run] Would launch:"
    echo "  GPU=${GPU_DIFF} PART=${PART} ... bash scripts/train_diffusion_suite.sh"
    echo "  GPU=${GPU_ELF}  PART=${PART} ... bash scripts/train_elf_suite.sh"
    exit 0
fi

DIFF_LOG="logs/emnlp_diffusion_parallel.log"
ELF_LOG="logs/emnlp_elf_parallel.log"

echo ""
echo "  Launching Diffusion suite on GPU ${GPU_DIFF} → ${DIFF_LOG}"
GPU="${GPU_DIFF}" PART="${PART}" \
    TOKENS_A="${TOKENS_A}" EPOCHS_A="${EPOCHS_A}" \
    TOKENS_B="${TOKENS_B}" EPOCHS_B="${EPOCHS_B}" \
    bash "${SCRIPT_DIR}/train_diffusion_suite.sh" \
    > "${DIFF_LOG}" 2>&1 &
DIFF_PID=$!

echo "  Launching ELF suite on GPU ${GPU_ELF} → ${ELF_LOG}"
GPU="${GPU_ELF}" PART="${PART}" \
    TOKENS_A="${TOKENS_A}" EPOCHS_A="${EPOCHS_A}" \
    TOKENS_B="${TOKENS_B}" EPOCHS_B="${EPOCHS_B}" \
    bash "${SCRIPT_DIR}/train_elf_suite.sh" \
    > "${ELF_LOG}" 2>&1 &
ELF_PID=$!

echo ""
echo "  Both suites running in background."
echo "  Follow progress:"
echo "    tail -f ${DIFF_LOG}"
echo "    tail -f ${ELF_LOG}"
echo ""

# Wait for both and collect exit codes
DIFF_EXIT=0; ELF_EXIT=0
wait "${DIFF_PID}" || DIFF_EXIT=$?
wait "${ELF_PID}"  || ELF_EXIT=$?

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  EMNLP parallel run complete"
echo "  Diffusion exit: ${DIFF_EXIT}  (log: ${DIFF_LOG})"
echo "  ELF       exit: ${ELF_EXIT}   (log: ${ELF_LOG})"
echo "════════════════════════════════════════════════════════════"

[[ ${DIFF_EXIT} -eq 0 && ${ELF_EXIT} -eq 0 ]] || exit 1
