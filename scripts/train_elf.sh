#!/usr/bin/env bash
# scripts/train_elf.sh
#
# Single-GPU ELF (Embedded Language Flows) training run.
#
# Usage
# ─────
#   bash scripts/train_elf.sh                         # default: GPU 0, base config
#   GPU=1 bash scripts/train_elf.sh                   # different GPU
#   TAG=my_run bash scripts/train_elf.sh              # custom run directory name
#   bash scripts/train_elf.sh --dim 768 --num_blocks 16   # CLI overrides
#   bash scripts/train_elf.sh --dry-run               # print command, no exec
#
# Environment variables
# ─────────────────────
#   GPU      GPU index to use (default: 0)
#   TAG      run directory suffix (default: elf_base)
#   CONFIG   path to YAML config (default: configs/elf_base.yaml)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

# ── Settings ──────────────────────────────────────────────────────────────────
GPU="${GPU:-0}"
TAG="${TAG:-elf_base}"
CONFIG="${CONFIG:-configs/elf_base.yaml}"
RUN_DIR="runs/${TAG}"
LOG_DIR="logs/elf"
DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true && shift

# Extra CLI arguments forwarded to the trainer (e.g. --dim 768 --num_blocks 16)
EXTRA_ARGS=("$@")

mkdir -p "${LOG_DIR}" logs

# ── Disk space guard (require at least 10 GB free) ────────────────────────────
free_kb=$(df -k "${ROOT}" | awk 'NR==2 {print $4}')
free_gb=$(( free_kb / 1024 / 1024 ))
if [[ ${free_gb} -lt 10 ]]; then
    echo "ERROR: only ${free_gb} GB free — need at least 10 GB.  Aborting." >&2
    exit 1
fi

# ── Skip if already completed ─────────────────────────────────────────────────
if [[ -f "${RUN_DIR}/model_weights.msgpack" || -f "${RUN_DIR}/best_model_weights.msgpack" ]]; then
    echo "  [SKIP] ${TAG} — weights already exist in ${RUN_DIR}"
    exit 0
fi

# ── Build command ─────────────────────────────────────────────────────────────
CMD=(
    env
    CUDA_VISIBLE_DEVICES="${GPU}"
    PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
    python dantinox/cli.py train
    --config "${CONFIG}"
    --run_dir "${RUN_DIR}"
    --n_devices 1
    --use_bf16 true
    --gradient_checkpointing true
    "${EXTRA_ARGS[@]}"
)

# ── Run ───────────────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "  DantinoX — ELF training"
echo "  Config  : ${CONFIG}"
echo "  Run dir : ${RUN_DIR}"
echo "  GPU     : ${GPU}"
echo "  Disk    : ${free_gb} GB free"
echo "════════════════════════════════════════════════════════════"

if [[ "${DRY_RUN}" == "true" ]]; then
    echo "  [DRY-RUN] ${CMD[*]}"
    exit 0
fi

LOG_FILE="${LOG_DIR}/${TAG}.log"
echo "  Logging to ${LOG_FILE}"
echo ""

if "${CMD[@]}" 2>&1 | tee "${LOG_FILE}"; then
    echo ""
    echo "  [OK] Training complete — weights in ${RUN_DIR}"
else
    echo ""
    echo "  [FAIL] Training failed — see ${LOG_FILE}" >&2
    exit 1
fi
