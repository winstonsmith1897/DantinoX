#!/usr/bin/env bash
# scripts/train_elf_suite.sh
#
# Full ELF (Embedded Language Flows) training suite for DantinoX — EMNLP 2026.
#
# Mirrors train_diffusion_suite.sh for fair comparison — LARGE MODELS ONLY.
#
# ══════════════════════════════════════════════════════════════════════════════
# Experiment matrix — LARGE MODELS (≥67M params, up to 40 GB GPU)
# ══════════════════════════════════════════════════════════════════════════════
#
# PART A — Size × Attention × FFN matrix
# ───────────────────────────────────────
#   Attention:   MHA / GQA(×4) / MLA
#   Sizes (dim × blocks × approx params):
#      512×12   ~67M   bottleneck=256
#      768×16  ~176M   bottleneck=256
#     1024×24  ~435M   bottleneck=256
#     1536×24  ~954M   bottleneck=384
#     2048×32  ~2.2B   bottleneck=512  ← max for 1× A100 40 GB
#   T5 oracle: t5-base (embed_dim=768) for all Part A runs.
#   FFN: Dense (all) + MoE top-2/6exp (1024 + 2048 only)
#
# PART B — Architecture ablations on 1024d×24b baseline  (~51 runs)
# ──────────────────────────────────────────────────────────────────
# ONE axis varies at a time vs Dense 1024d×24b MHA/GQA/MLA:
#
#   B1.  norm_type:       rmsnorm                                 × MHA/GQA/MLA
#   B2.  dropout:         0.0, 0.20                               × MHA/GQA/MLA
#   B3.  use_swiglu:      false/GELU                              × MHA/GQA/MLA
#   B4.  sliding_window:  true, ctx=64                            × MHA/GQA/MLA
#   B5.  lr_schedule:     wsd                                     × MHA/GQA/MLA
#   B6.  optimizer:       adamw (3e-4), lion (1e-4)               × MHA/GQA/MLA
#   B7.  MoE 8exp                                                 × MHA/GQA/MLA
#   B8.  batch_size:      128, grad_accum=8                       × MHA/GQA/MLA
#   B9.  context:         256, 1024                               × MHA/GQA/MLA
#   B10. t5_variant:      t5-small (embed_dim=512, bd=256)        × MHA/GQA/MLA
#   B11. bottleneck_dim:  128, 512         (vs 256 default)       × MHA/GQA/MLA
#   B12. denoiser_prob:   0.5, 1.0         (vs 0.8)               × MHA/GQA/MLA
#   B13. self_cond_prob:  0.0, 1.0         (vs 0.5)               × MHA/GQA/MLA
#
# ══════════════════════════════════════════════════════════════════════════════
# Hardware:   1 × A100 40 GB (GPU=3 default)
# Precision:  bf16
# Dataset:    wikitext-103-raw-v1 (HuggingFace)
# Tokenizer:  T5 SentencePiece (vocab_size=32128)
# ══════════════════════════════════════════════════════════════════════════════
#
# Usage:
#   GPU=3 bash scripts/train_elf_suite.sh                  # all ~96 runs on GPU 3
#   GPU=3 PART=A bash scripts/train_elf_suite.sh           # only Part A
#   GPU=3 PART=B bash scripts/train_elf_suite.sh           # only Part B
#   GPU=3 ATTN=mha bash scripts/train_elf_suite.sh         # only MHA
#   GPU=3 DIM=256 bash scripts/train_elf_suite.sh          # only dim=256
#   GPU=3 bash scripts/train_elf_suite.sh --dry-run        # print commands, no exec
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

BASE_CFG="configs/elf_base.yaml"
LOG_DIR="logs/elf_suite"
mkdir -p "${LOG_DIR}" logs

# ── Filters ───────────────────────────────────────────────────────────────────
GPU="${GPU:-3}"
PART="${PART:-all}"
ATTN_F="${ATTN:-all}"
DIM_F="${DIM:-all}"
MOE_F="${MOE:-all}"
DRY_RUN=false

# ── Training budget ────────────────────────────────────────────────────────────
TOKENS_A="${TOKENS_A:-50000000}"
EPOCHS_A="${EPOCHS_A:-30}"
TOKENS_B="${TOKENS_B:-20000000}"
EPOCHS_B="${EPOCHS_B:-15}"
_CUR_TOKENS=${TOKENS_A}
_CUR_EPOCHS=${EPOCHS_A}
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# ── Counters ──────────────────────────────────────────────────────────────────
N_TOTAL=0; N_DONE=0; N_SKIP=0; N_FAIL=0

# ── Disk space guard ──────────────────────────────────────────────────────────
check_disk() {
    local free_kb free_gb
    free_kb=$(df -k "${ROOT}" | awk 'NR==2 {print $4}')
    free_gb=$(( free_kb / 1024 / 1024 ))
    echo "  Disk check OK: ${free_gb} GB free"
    if (( free_gb < 15 )); then
        echo "ERROR: only ${free_gb} GB free — need 15 GB." >&2
        exit 1
    fi
}

# ── Core train_one ────────────────────────────────────────────────────────────
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

    # Skip if completed (best checkpoint exists but no cursor = training finished)
    if [[ -f "${run_dir}/best_model_weights.msgpack" && ! -f "${run_dir}/training_cursor.json" ]]; then
        echo "  [SKIP]  ${tag}"; ((N_SKIP++)) || true; return 0
    fi

    local _gc="true"

    # Resume if interrupted checkpoint exists
    local _resume="false"
    [[ -f "${run_dir}/training_cursor.json" ]] && _resume="true"

    local cmd=(
        env
        CUDA_VISIBLE_DEVICES="${GPU}"
        XLA_PYTHON_CLIENT_PREALLOCATE=false
        PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
        python dantinox/cli.py train
        --config "${BASE_CFG}"
        --run_dir "${run_dir}"
        --n_devices 1
        --use_bf16 true
        --gradient_checkpointing "${_gc}"
        --use_flash_attention true
        --grad_accum 4
        --max_train_tokens "${_CUR_TOKENS}"
        --epochs "${_CUR_EPOCHS}"
        --resume "${_resume}"
        --tokenizer_type t5
        --dataset_source huggingface
        --dataset_name "wikitext"
        --dataset_config "wikitext-103-raw-v1"
        --dataset_text_field text
        --dataset_split train
        --streaming false
        "${extra_args[@]}"
    )

    echo ""; echo "  ── ${tag}"
    [[ "${DRY_RUN}" == "true" ]] && echo "     ${cmd[*]}" && return 0

    check_disk
    local log_file="${LOG_DIR}/${tag}.log"
    if "${cmd[@]}" 2>&1 | tee "${log_file}"; then
        ((N_DONE++)) || true; echo "  [OK]   ${tag}"
    else
        ((N_FAIL++)) || true; echo "  [FAIL] ${tag}  log: ${log_file}" >&2
    fi
    # Let CUDA driver fully release GPU memory before the next run
    sleep 30
}

# ── Attention helpers ─────────────────────────────────────────────────────────
mha_flags() { local nh=$1; echo "--kv_heads ${nh} --mla false"; }
gqa_flags() { local nh=$1; local kv=$(( nh / 4 )); [[ ${kv} -lt 1 ]] && kv=1; echo "--kv_heads ${kv} --mla false"; }
mla_flags() {
    local nh=$1 hs=$2
    local dkv=$(( hs * 3 )); [[ ${dkv} -gt 256 ]] && dkv=256
    local dq=$(( hs * 6  )); [[ ${dq}  -gt 256 ]] && dq=256
    local rd=$(( hs / 2  )); [[ ${rd}  -lt 16  ]] && rd=16
    echo "--kv_heads ${nh} --mla true --inference false --down_dim_kv ${dkv} --down_dim_q ${dq} --rope_dim ${rd}"
}

# ── Part A — Size × Attention × FFN ──────────────────────────────────────────
run_part_a() {
    [[ "${PART}" == "B" ]] && return 0

    # dim  n_heads  head_size  num_blocks  lr     optimizer  bottleneck
    # head_size=64 throughout; bottleneck=max(256, dim//4)
    local -a DENSE=(
        " 512  8 64 12  0.002 muon 256"
        " 768 12 64 16  0.002 muon 256"
    )
    for cfg in "${DENSE[@]}"; do
        read -r dim nh hs blocks lr opt bd <<< "${cfg}"
        # shellcheck disable=SC2046,SC2086
        train_one "elf_mha_${dim}d_${blocks}b_Dense" --dim "${dim}" --n_heads "${nh}" --head_size "${hs}" \
            --num_blocks "${blocks}" --lr "${lr}" --optimizer "${opt}" --use_moe false \
            --embed_dim 768 --bottleneck_dim "${bd}" --t5_model_name t5-base $(mha_flags "${nh}")
        # shellcheck disable=SC2046,SC2086
        train_one "elf_gqa_${dim}d_${blocks}b_Dense" --dim "${dim}" --n_heads "${nh}" --head_size "${hs}" \
            --num_blocks "${blocks}" --lr "${lr}" --optimizer "${opt}" --use_moe false \
            --embed_dim 768 --bottleneck_dim "${bd}" --t5_model_name t5-base $(gqa_flags "${nh}")
        # shellcheck disable=SC2046,SC2086
        train_one "elf_mla_${dim}d_${blocks}b_Dense" --dim "${dim}" --n_heads "${nh}" --head_size "${hs}" \
            --num_blocks "${blocks}" --lr "${lr}" --optimizer "${opt}" --use_moe false \
            --embed_dim 768 --bottleneck_dim "${bd}" --t5_model_name t5-base $(mla_flags "${nh}" "${hs}")
    done
}

# ── Part B — Ablations on 1024d×24b ───────────────────────────────────────────
run_part_b() {
    [[ "${PART}" == "A" ]] && return 0
    _CUR_TOKENS=${TOKENS_B}
    _CUR_EPOCHS=${EPOCHS_B}

    local BASE_DIM=768 BASE_NH=12 BASE_HS=64 BASE_BLOCKS=16 BASE_LR=0.002 BASE_OPT=muon BASE_BD=256

    ablation() {
        local suffix="$1" attn_tag="$2" attn_flags="$3"; shift 3; local extra=("$@")
        # shellcheck disable=SC2086
        train_one "elf_${attn_tag}_768d_16b_${suffix}" \
            --dim "${BASE_DIM}" --n_heads "${BASE_NH}" --head_size "${BASE_HS}" \
            --num_blocks "${BASE_BLOCKS}" --lr "${BASE_LR}" --optimizer "${BASE_OPT}" \
            --use_moe false --embed_dim 768 --bottleneck_dim "${BASE_BD}" \
            --t5_model_name t5-base ${attn_flags} "${extra[@]}"
    }

    for attn in mha gqa mla; do
        case ${attn} in
            mha) flags=$(mha_flags "${BASE_NH}") ;;
            gqa) flags=$(gqa_flags "${BASE_NH}") ;;
            mla) flags=$(mla_flags "${BASE_NH}" "${BASE_HS}") ;;
        esac

        # B1 — norm
        ablation "RMSNorm"       "${attn}" "${flags}" --norm_type rmsnorm
        # B2 — dropout
        ablation "Drop20"        "${attn}" "${flags}" --dropout_rate 0.20
        # B3 — FFN activation
        ablation "GELU"          "${attn}" "${flags}" --use_swiglu false
        # B4 — sliding window
        ablation "SlidingWin64"  "${attn}" "${flags}" --sliding_window true --context_window 64
        # B5 — LR schedule
        ablation "SchedWSD"      "${attn}" "${flags}" --lr_schedule wsd
        # B6 — optimizer (compare Muon vs AdamW vs Lion)
        ablation "OptAdamW"      "${attn}" "${flags}" --optimizer adamw --lr 0.0003
        ablation "OptLion"       "${attn}" "${flags}" --optimizer lion  --lr 0.0001
        # B7 — MoE
        ablation "MoE8exp"       "${attn}" "${flags}" --use_moe true --n_experts 8 --top_k_mlp 2
        # B8 — batch size
        ablation "BS128"         "${attn}" "${flags}" --batch_size 128 --grad_accum 8
        # B9 — context length
        ablation "Ctx256"        "${attn}" "${flags}" --max_context 256
        ablation "Ctx1024"       "${attn}" "${flags}" --max_context 1024
        # B10 — T5 variant (t5-small: embed_dim=512)
        ablation "T5Small"       "${attn}" "${flags}" --t5_model_name t5-small --embed_dim 512 --bottleneck_dim 256
        # B11 — bottleneck size
        ablation "Bottleneck128" "${attn}" "${flags}" --bottleneck_dim 128
        ablation "Bottleneck512" "${attn}" "${flags}" --bottleneck_dim 512
        # B12 — denoiser probability
        ablation "DenProb05"     "${attn}" "${flags}" --denoiser_prob 0.5
        ablation "DenProb10"     "${attn}" "${flags}" --denoiser_prob 1.0
        # B13 — self-conditioning probability
        ablation "SelfCond0"     "${attn}" "${flags}" --self_cond_prob 0.0
        ablation "SelfCond1"     "${attn}" "${flags}" --self_cond_prob 1.0
    done
}

# ── Summary ───────────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "  DantinoX — ELF training suite"
echo "  Base config : ${BASE_CFG}"
echo "  GPU         : ${GPU}"
echo "  Part filter : ${PART}"
echo "  Attn filter : ${ATTN_F}"
echo "  Dim filter  : ${DIM_F}"
echo "  MoE filter  : ${MOE_F}"
echo "  Dry-run     : ${DRY_RUN}"
echo "════════════════════════════════════════════════════════════"

if [[ "${DRY_RUN}" == "false" ]]; then check_disk; fi

run_part_a
run_part_b

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ELF suite complete"
echo "  Trained : ${N_DONE}"
echo "  Skipped : ${N_SKIP}  (already exist)"
echo "  Failed  : ${N_FAIL}"
echo "  Total   : ${N_TOTAL}"
echo "════════════════════════════════════════════════════════════"
