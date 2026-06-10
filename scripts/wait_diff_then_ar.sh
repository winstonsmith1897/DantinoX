#!/usr/bin/env bash
# Aspetta che il diffusion suite termini, poi rilancia l'AR suite.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DIFF_PID="${1:-}"

if [[ -z "${DIFF_PID}" ]]; then
    # Trova il processo "bash scripts/train_diffusion_suite.sh"
    DIFF_PID=$(pgrep -f "bash scripts/train_diffusion_suite.sh" 2>/dev/null | head -1 || true)
fi

if [[ -z "${DIFF_PID}" ]]; then
    echo "[wait_diff_then_ar] Nessun processo diffusion trovato, rilancio AR subito."
else
    echo "[wait_diff_then_ar] Aspetto fine diffusion suite (PID ${DIFF_PID})..."
    while kill -0 "${DIFF_PID}" 2>/dev/null; do
        sleep 60
    done
    echo "[wait_diff_then_ar] Diffusion suite terminata — $(date)"
fi

echo "[wait_diff_then_ar] Avvio AR suite — $(date)"
bash scripts/train_ar_suite.sh 2>&1 | tee -a logs/ar_suite_bpe.log
echo "[wait_diff_then_ar] AR suite completata — $(date)"
