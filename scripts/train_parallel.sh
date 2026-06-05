#!/usr/bin/env bash
# scripts/train_parallel.sh
#
# Run the AR + Diffusion training suites in parallel across GPU pairs.
#
# With 8× A100 we run 4 concurrent training jobs, each using 2 GPUs:
#   Pair 0 → GPUs 0,1  (jobs: ar  Part A  MHA)
#   Pair 1 → GPUs 2,3  (jobs: ar  Part A  GQA + MLA)
#   Pair 2 → GPUs 4,5  (jobs: ar  Part B ablations)
#   Pair 3 → GPUs 6,7  (jobs: diff Part A + B)
#
# Total wall-clock: ~5–6h (vs ~20h sequential)
#
# Usage:
#   bash scripts/train_parallel.sh             # full parallel run
#   bash scripts/train_parallel.sh --dry-run   # preview commands
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

REPO="${ROOT}"
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

ts() { date +'%H:%M:%S'; }
banner() { echo ""; echo "════════════════════════════════════════════════════════════"; echo "  [$(ts)] $*"; echo "════════════════════════════════════════════════════════════"; }

# ── Disk space guard ──────────────────────────────────────────────────────────
check_disk() {
    local free_gb
    free_gb=$(( $(df -k "${ROOT}" | awk 'NR==2{print $4}') / 1024 / 1024 ))
    echo "  Disk free: ${free_gb} GB"
    if (( free_gb < 15 )); then
        echo "ERROR: only ${free_gb} GB free." >&2; exit 1
    fi
}

[[ "${DRY_RUN}" == "false" ]] && check_disk

banner "Parallel training — 4 jobs × 2 GPUs (MHA | GQA | MLA | Diffusion)"
echo "  GPU pairs:   0,1 (MHA) | 2,3 (GQA) | 4,5 (MLA) | 6,7 (Diffusion)"
echo "  Dry-run:     ${DRY_RUN}"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Job definitions — split by attention type (no overlap, no race conditions)
# ─────────────────────────────────────────────────────────────────────────────
#
#  Job 0: AR  MHA  (Part A + B)   → GPUs 0,1
#  Job 1: AR  GQA  (Part A + B)   → GPUs 2,3
#  Job 2: AR  MLA  (Part A + B)   → GPUs 4,5
#  Job 3: Diffusion (Part A + B)  → GPUs 6,7
#
# ─────────────────────────────────────────────────────────────────────────────

run_job() {
    local job_id="$1" gpu_pair="$2" log_file="$3" script="$4"
    shift 4
    local env_vars=("$@")

    local cmd=(env
        CUDA_VISIBLE_DEVICES="${gpu_pair}"
        PYTHONPATH="${REPO}:${PYTHONPATH:-}"
        "${env_vars[@]}"
        bash "${script}")

    echo "  Job ${job_id}  GPUs=${gpu_pair}  → ${log_file}"
    [[ "${DRY_RUN}" == "true" ]] && echo "     ${cmd[*]}" && return 0

    "${cmd[@]}" > "${log_file}" 2>&1 &
    echo "  Job ${job_id} PID=$!"
}

# Job 0: AR MHA — Part A + B
run_job 0 "0,1" "${LOG_DIR}/job0_ar_mha.log" \
    "scripts/train_ar_suite.sh" \
    ATTN=mha

# Job 1: AR GQA — Part A + B
run_job 1 "2,3" "${LOG_DIR}/job1_ar_gqa.log" \
    "scripts/train_ar_suite.sh" \
    ATTN=gqa

# Job 2: AR MLA — Part A + B
run_job 2 "4,5" "${LOG_DIR}/job2_ar_mla.log" \
    "scripts/train_ar_suite.sh" \
    ATTN=mla

# Job 3: Diffusion all (Part A + B)
run_job 3 "6,7" "${LOG_DIR}/job3_diffusion.log" \
    "scripts/train_diffusion_suite.sh" \
    ""  # no extra env vars

if [[ "${DRY_RUN}" == "true" ]]; then
    echo ""
    echo "  Dry-run complete — no processes started."
    exit 0
fi

echo ""
banner "All 4 jobs launched — waiting for completion"
echo "  Monitor progress:"
echo "    tail -f logs/job0_ar_mha.log"
echo "    tail -f logs/job3_diffusion.log"
echo "    watch 'ls runs/ | grep -E \"^ar_|^diff_\" | wc -l'"
echo ""

# Wait for all background jobs
wait
echo ""
banner "All jobs complete — $(ls runs/ | grep -E '^ar_|^diff_' | wc -l) runs saved"
