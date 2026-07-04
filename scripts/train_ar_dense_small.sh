#!/usr/bin/env bash
# scripts/train_ar_dense_small.sh
#
# AR Dense Small (512d, 12b) runs for the EMNLP generation-quality table.
# EXACT mirror of scripts/train_diffusion_suite.sh Part A (512d row), with
# model_type=autoregressive. Same tokenizer, budget, optimizer, and attention
# flags as the diff_*/elf_* Dense runs.
#
# Usage:
#   GPU=0 bash scripts/train_ar_dense_small.sh mha [mla ...]
#   GPU=1 bash scripts/train_ar_dense_small.sh gqa

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

GPU="${GPU:-0}"
BASE_CFG="configs/diffusion_base.yaml"
LOG_DIR="logs/ar_dense"
mkdir -p "${LOG_DIR}"

# 512d row from train_diffusion_suite.sh: dim n_heads head_size blocks lr opt
DIM=512; NH=8; HS=64; BLOCKS=12; LR=0.002; OPT=muon

attn_flags() {
    case "$1" in
        mha) echo "--kv_heads ${NH} --mla false" ;;
        gqa) echo "--kv_heads $(( NH / 4 )) --mla false" ;;
        mla) echo "--kv_heads ${NH} --mla true --inference false --down_dim_kv $(( HS * 3 )) --down_dim_q 256 --rope_dim $(( HS / 2 ))" ;;
        *)   echo "unknown attention: $1" >&2; exit 1 ;;
    esac
}

for attn in "$@"; do
    tag="ar_${attn}_${DIM}d_${BLOCKS}b_Dense"
    run_dir="runs/${tag}"
    if [[ -f "${run_dir}/best_model_weights.msgpack" && ! -f "${run_dir}/training_cursor.json" ]]; then
        echo "[SKIP] ${tag}"; continue
    fi
    resume_flag=()
    [[ -f "${run_dir}/training_cursor.json" ]] && resume_flag=(--resume)

    echo "── ${tag} (GPU ${GPU})"
    # shellcheck disable=SC2046
    env CUDA_VISIBLE_DEVICES="${GPU}" XLA_PYTHON_CLIENT_PREALLOCATE=false \
        PYTHONPATH="${ROOT}:${PYTHONPATH:-}" \
        python dantinox/cli.py train \
        --config "${BASE_CFG}" \
        --run_dir "${run_dir}" \
        --model_type autoregressive \
        --use_bf16 true \
        --use_flash_attention true \
        --gradient_checkpointing true \
        --tokenizer_type t5 \
        --vocab_size 32128 \
        --mask_token_id 32099 \
        --max_train_tokens 50000000 \
        --epochs 30 \
        --dim "${DIM}" --n_heads "${NH}" --head_size "${HS}" \
        --num_blocks "${BLOCKS}" --lr "${LR}" --optimizer "${OPT}" \
        --use_moe false \
        "${resume_flag[@]}" \
        $(attn_flags "${attn}") \
        2>&1 | tee "${LOG_DIR}/${tag}.log"
    echo "[DONE] ${tag}"
    sleep 30
done
