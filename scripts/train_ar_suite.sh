#!/usr/bin/env bash
# scripts/train_ar_suite.sh
#
# Full AR (Autoregressive) model training suite for DantinoX — EMNLP 2026.
#
# EXACT MIRROR of scripts/train_diffusion_suite.sh with model_type=autoregressive.
# Having symmetric ablations ensures fair AR vs Diffusion comparison.
#
# ══════════════════════════════════════════════════════════════════════════════
# Matrix (~96 runs):
#   Part A — Size × Attention × FFN  (48 runs)
#   Part B — Architecture ablations on 256d×12b  (48 runs)
# ══════════════════════════════════════════════════════════════════════════════
#
# Usage:
#   bash scripts/train_ar_suite.sh             # all ~96 runs
#   PART=A bash scripts/train_ar_suite.sh      # only Part A
#   ATTN=mha bash scripts/train_ar_suite.sh    # only MHA
#   DIM=256 bash scripts/train_ar_suite.sh     # only dim=256
#   bash scripts/train_ar_suite.sh --dry-run
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

BASE_CFG="configs/default_config.yaml"
LOG_DIR="logs/ar_suite"
mkdir -p "${LOG_DIR}" logs

PART="${PART:-all}"
ATTN_F="${ATTN:-all}"
DIM_F="${DIM:-all}"
MOE_F="${MOE:-all}"
DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

N_TOTAL=0; N_DONE=0; N_SKIP=0; N_FAIL=0

check_disk() {
    local free_kb free_gb
    free_kb=$(df -k "${ROOT}" | awk 'NR==2 {print $4}')
    free_gb=$(( free_kb / 1024 / 1024 ))
    echo "  Disk check OK: ${free_gb} GB free"
    if (( free_gb < 15 )); then
        echo "ERROR: only ${free_gb} GB free — need 15 GB. Run: python scripts/cleanup_runs.py --execute" >&2
        exit 1
    fi
}

train_one() {
    local tag="$1"; shift
    local extra_args=("$@")
    [[ "${ATTN_F}" != "all" ]] && [[ "${tag}" != *"_${ATTN_F}_"* ]] && { ((N_SKIP++)) || true; return 0; }
    [[ "${DIM_F}"  != "all" ]] && [[ "${tag}" != *"_${DIM_F}d_"*  ]] && { ((N_SKIP++)) || true; return 0; }
    if [[ "${MOE_F}" == "dense" ]]; then
        [[ "${tag}" == *"_MoE"* ]] && { ((N_SKIP++)) || true; return 0; }
    elif [[ "${MOE_F}" == "moe" ]]; then
        [[ "${tag}" != *"_MoE"* ]] && { ((N_SKIP++)) || true; return 0; }
    fi

    ((N_TOTAL++)) || true
    local run_dir="runs/${tag}"
    if [[ -f "${run_dir}/model_weights.msgpack" || -f "${run_dir}/best_model_weights.msgpack" ]]; then
        echo "  [SKIP]  ${tag}"; ((N_SKIP++)) || true; return 0
    fi

    local cmd=(env PYTHONPATH="/ssd1/marco.simoni/VULNERABILITY/NETGROUP/DantinoX:${PYTHONPATH:-}" python dantinox/cli.py train
        --config "${BASE_CFG}"
        --run_dir "${run_dir}"
        --n_devices 2
        --use_bf16 true
        --use_flash_attention true
        --gradient_checkpointing true
        --tokenizer_type bpe
        --vocab_size 4096
        --dataset_source huggingface
        --dataset_name "wikitext"
        --dataset_config "wikitext-103-raw-v1"
        --dataset_text_field text
        --dataset_split train
        --streaming false
        "${extra_args[@]}")

    echo ""; echo "  ── ${tag}"
    [[ "${DRY_RUN}" == "true" ]] && echo "     ${cmd[*]}" && return 0

    check_disk
    local log_file="${LOG_DIR}/${tag}.log"
    if "${cmd[@]}" 2>&1 | tee "${log_file}"; then
        ((N_DONE++)) || true; echo "  [OK]   ${tag}"
    else
        ((N_FAIL++)) || true; echo "  [FAIL] ${tag}  log: ${log_file}" >&2
    fi
}

mha_flags() { local nh=$1; echo "--kv_heads ${nh} --mla false"; }
gqa_flags() { local nh=$1; local kv=$(( nh / 4 )); [[ ${kv} -lt 1 ]] && kv=1; echo "--kv_heads ${kv} --mla false"; }
mla_flags() {
    local nh=$1 hs=$2
    local dkv=$(( hs * 3 )); [[ ${dkv} -gt 256 ]] && dkv=256
    local dq=$(( hs * 6  )); [[ ${dq}  -gt 256 ]] && dq=256
    local rd=$(( hs / 2  )); [[ ${rd}  -lt 16  ]] && rd=16
    echo "--kv_heads ${nh} --mla true --inference false --down_dim_kv ${dkv} --down_dim_q ${dq} --rope_dim ${rd}"
}

run_part_a() {
    [[ "${PART}" == "B" ]] && return 0
    local -a DENSE=(
        "128  4  32 12  0.0012 lion"
        "192  6  32 12  0.0012 lion"
        "256  8  32  8  0.0012 lion"
        "256  8  32 12  0.0012 lion"
        "256  8  32 16  0.0010 adamw"
        "384 12  32 12  0.0010 adamw"
        "512 16  32  8  0.0008 adamw"
        "512 16  32 12  0.0008 adamw"
        "512 16  32 16  0.0006 adamw"
        "768 12  64 12  0.0006 adamw"
    )
    for cfg in "${DENSE[@]}"; do
        read -r dim nh hs blocks lr opt <<< "${cfg}"
        # shellcheck disable=SC2046,SC2086
        train_one "ar_mha_${dim}d_${blocks}b_Dense" --dim "${dim}" --n_heads "${nh}" --head_size "${hs}" \
            --num_blocks "${blocks}" --lr "${lr}" --optimizer "${opt}" --use_moe false $(mha_flags "${nh}")
        # shellcheck disable=SC2046,SC2086
        train_one "ar_gqa_${dim}d_${blocks}b_Dense" --dim "${dim}" --n_heads "${nh}" --head_size "${hs}" \
            --num_blocks "${blocks}" --lr "${lr}" --optimizer "${opt}" --use_moe false $(gqa_flags "${nh}")
        # shellcheck disable=SC2046,SC2086
        train_one "ar_mla_${dim}d_${blocks}b_Dense" --dim "${dim}" --n_heads "${nh}" --head_size "${hs}" \
            --num_blocks "${blocks}" --lr "${lr}" --optimizer "${opt}" --use_moe false $(mla_flags "${nh}" "${hs}")
    done

    local -a MOE=(
        "256  8  32  8  0.0012 lion"
        "256  8  32 12  0.0012 lion"
        "256  8  32 16  0.0010 adamw"
        "512 16  32  8  0.0008 adamw"
        "512 16  32 12  0.0008 adamw"
        "512 16  32 16  0.0006 adamw"
    )
    for cfg in "${MOE[@]}"; do
        read -r dim nh hs blocks lr opt <<< "${cfg}"
        # shellcheck disable=SC2046,SC2086
        train_one "ar_mha_${dim}d_${blocks}b_MoE" --dim "${dim}" --n_heads "${nh}" --head_size "${hs}" \
            --num_blocks "${blocks}" --lr "${lr}" --optimizer "${opt}" \
            --use_moe true --n_experts 6 --top_k_mlp 2 $(mha_flags "${nh}")
        # shellcheck disable=SC2046,SC2086
        train_one "ar_gqa_${dim}d_${blocks}b_MoE" --dim "${dim}" --n_heads "${nh}" --head_size "${hs}" \
            --num_blocks "${blocks}" --lr "${lr}" --optimizer "${opt}" \
            --use_moe true --n_experts 6 --top_k_mlp 2 $(gqa_flags "${nh}")
        # shellcheck disable=SC2046,SC2086
        train_one "ar_mla_${dim}d_${blocks}b_MoE" --dim "${dim}" --n_heads "${nh}" --head_size "${hs}" \
            --num_blocks "${blocks}" --lr "${lr}" --optimizer "${opt}" \
            --use_moe true --n_experts 6 --top_k_mlp 2 $(mla_flags "${nh}" "${hs}")
    done
}

run_part_b() {
    [[ "${PART}" == "A" ]] && return 0
    local BASE_DIM=256 BASE_NH=8 BASE_HS=32 BASE_BLOCKS=12 BASE_LR=0.0012 BASE_OPT=lion

    ablation() {
        local suffix="$1" attn_tag="$2" attn_flags="$3"; shift 3; local extra=("$@")
        # shellcheck disable=SC2086
        train_one "ar_${attn_tag}_256d_12b_${suffix}" \
            --dim "${BASE_DIM}" --n_heads "${BASE_NH}" --head_size "${BASE_HS}" \
            --num_blocks "${BASE_BLOCKS}" --lr "${BASE_LR}" --optimizer "${BASE_OPT}" \
            --use_moe false ${attn_flags} "${extra[@]}"
    }

    for attn in mha gqa mla; do
        case ${attn} in
            mha) flags=$(mha_flags "${BASE_NH}") ;;
            gqa) flags=$(gqa_flags "${BASE_NH}") ;;
            mla) flags=$(mla_flags "${BASE_NH}" "${BASE_HS}") ;;
        esac
        ablation "RMSNorm"      "${attn}" "${flags}" --norm_type rmsnorm
        ablation "Drop0"        "${attn}" "${flags}" --dropout_rate 0.0
        ablation "Drop20"       "${attn}" "${flags}" --dropout_rate 0.20
        ablation "GELU"         "${attn}" "${flags}" --use_swiglu false
        ablation "SlidingWin64" "${attn}" "${flags}" --sliding_window true --context_window 64
        ablation "NoSink"       "${attn}" "${flags}" --no_sink true
        ablation "SchedWSD"     "${attn}" "${flags}" --lr_schedule wsd
        ablation "OptLion"      "${attn}" "${flags}" --optimizer lion --lr 0.0003
        ablation "MoE8exp"      "${attn}" "${flags}" --use_moe true --n_experts 8 --top_k_mlp 2
        ablation "BS128"        "${attn}" "${flags}" --batch_size 128 --grad_accum 8
        ablation "Ctx256"       "${attn}" "${flags}" --max_context 256
        ablation "Ctx1024"      "${attn}" "${flags}" --max_context 1024
    done
}

echo "════════════════════════════════════════════════════════════"
echo "  DantinoX — AR model training suite"
echo "  Part: ${PART}  Attn: ${ATTN_F}  Dim: ${DIM_F}  MoE: ${MOE_F}"
echo "  Dry-run: ${DRY_RUN}"
echo "════════════════════════════════════════════════════════════"

if [[ "${DRY_RUN}" == "false" ]]; then check_disk; fi

run_part_a
run_part_b

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Done: trained=${N_DONE}  skipped=${N_SKIP}  failed=${N_FAIL}  total=${N_TOTAL}"
echo "════════════════════════════════════════════════════════════"
