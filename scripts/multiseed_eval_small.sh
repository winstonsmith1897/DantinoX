#!/usr/bin/env bash
# scripts/multiseed_eval_small.sh
#
# Multi-seed generation-quality eval for the Small (512d, 12b) Dense runs.
# Seed 42 already exists in results/generation_quality_full.csv; this adds
# seeds 43 and 44 so the paper can report mean ± std over 3 generation seeds.
#
# Usage:
#   GPU=1 WAIT_PID=<pid> bash scripts/multiseed_eval_small.sh
#   (WAIT_PID optional: blocks until that PID exits before starting)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

GPU="${GPU:-1}"
WAIT_PID="${WAIT_PID:-}"
RUNS="${RUNS:-diff_mha_512d_12b_Dense diff_gqa_512d_12b_Dense diff_mla_512d_12b_Dense elf_mha_512d_12b_Dense elf_gqa_512d_12b_Dense elf_mla_512d_12b_Dense}"

if [[ -n "${WAIT_PID}" ]]; then
    echo "Waiting for PID ${WAIT_PID} to exit before starting evals..."
    while kill -0 "${WAIT_PID}" 2>/dev/null; do sleep 60; done
    echo "PID ${WAIT_PID} exited — starting evals."
    sleep 60   # let the GPU driver release memory
fi

for seed in 43 44; do
    out="results/generation_quality_small_seed${seed}.csv"
    if [[ -f "${out}" ]]; then
        echo "[SKIP] seed ${seed} (${out} exists)"; continue
    fi
    echo "── seed ${seed} → ${out}"
    # shellcheck disable=SC2086
    env CUDA_VISIBLE_DEVICES="${GPU}" XLA_PYTHON_CLIENT_PREALLOCATE=false \
        PYTHONPATH="${ROOT}:${PYTHONPATH:-}" \
        python benchmarks/generation_quality.py \
        --runs ${RUNS} \
        --out "${out}" \
        --n-samples 100 --gen-len 128 --seed "${seed}" \
        2>&1 | tee "logs/eval_small_seed${seed}.log"
done
echo "[DONE] multi-seed eval"
