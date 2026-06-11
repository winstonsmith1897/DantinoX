#!/usr/bin/env bash
# scripts/train_diffusion_suite.sh
#
# Full Diffusion model training suite for DantinoX — EMNLP 2026 paper.
#
# ══════════════════════════════════════════════════════════════════════════════
# Experiment matrix — LARGE MODELS ONLY (≥67M params, up to 40 GB GPU)
# ══════════════════════════════════════════════════════════════════════════════
#
# PART A — Size × Attention × FFN matrix
# ───────────────────────────────────────
#   Attention: MHA / GQA(×4) / MLA
#   Sizes (dim × blocks × approx params):
#     512×12   ~67M
#     768×16  ~176M
#    1024×24  ~435M
#    1536×24  ~954M
#    2048×32  ~2.2B  ← max for 1× A100 40 GB with Muon
#   FFN: Dense (all) + MoE top-2/6exp (1024 + 2048 only)
#
# PART B — Architecture ablations on 1024d×24b baseline  (~42 runs)
# ──────────────────────────────────────────────────────────────────
# ONE axis varies at a time vs Dense 1024d×24b MHA/GQA/MLA:
#
#   B1.  norm_type:       rmsnorm                                 × MHA/GQA/MLA
#   B2.  noise_schedule:  cosine, sqrt                            × MHA/GQA/MLA
#   B3.  dropout_rate:    0.0, 0.20                               × MHA/GQA/MLA
#   B4.  use_swiglu:      false/GELU                              × MHA/GQA/MLA
#   B5.  sliding_window:  true, ctx=64                            × MHA/GQA/MLA
#   B6.  no_sink:         true                                    × MHA/GQA/MLA
#   B7.  lr_schedule:     wsd                                     × MHA/GQA/MLA
#   B8.  optimizer:       adamw (lr=3e-4), lion (lr=1e-4)         × MHA/GQA/MLA
#   B9.  MoE 8exp                                                 × MHA/GQA/MLA
#   B10. batch_size:      128, grad_accum=8                       × MHA/GQA/MLA
#   B11. max_context:     256, 1024                               × MHA/GQA/MLA
#
# ══════════════════════════════════════════════════════════════════════════════
# Hardware:   1 × A100 40 GB (GPU=2 default)
# Precision:  bf16
# Dataset:    wikitext-103-raw-v1 (HuggingFace)
# Tokenizer:  T5 SentencePiece (vocab_size=32128)
# ══════════════════════════════════════════════════════════════════════════════
#
# Usage:
#   bash scripts/train_diffusion_suite.sh              # all ~90 runs
#   PART=A bash scripts/train_diffusion_suite.sh       # only Part A
#   PART=B bash scripts/train_diffusion_suite.sh       # only Part B (ablations)
#   ATTN=mla bash scripts/train_diffusion_suite.sh     # only MLA attention
#   DIM=256 bash scripts/train_diffusion_suite.sh      # only dim=256
#   bash scripts/train_diffusion_suite.sh --dry-run    # print commands, no exec
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

GPU="${GPU:-2}"
BASE_CFG="configs/diffusion_base.yaml"
LOG_DIR="logs/diffusion_suite"
mkdir -p "${LOG_DIR}" logs

# ── Training budget ────────────────────────────────────────────────────────────
# Part A (main results):  full budget
# Part B (ablations):     reduced — enough to establish relative ordering
TOKENS_A="${TOKENS_A:-50000000}"
EPOCHS_A="${EPOCHS_A:-30}"
TOKENS_B="${TOKENS_B:-20000000}"
EPOCHS_B="${EPOCHS_B:-15}"
_CUR_TOKENS=${TOKENS_A}
_CUR_EPOCHS=${EPOCHS_A}

# ── Filters ──────────────────────────────────────────────────────────────────
PART="${PART:-all}"       # all | A | B
ATTN_F="${ATTN:-all}"     # all | mha | gqa | mla
DIM_F="${DIM:-all}"       # all | 256 | 512 | ...
MOE_F="${MOE:-all}"       # all | dense | moe
DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# ── Counters ──────────────────────────────────────────────────────────────────
N_TOTAL=0; N_DONE=0; N_SKIP=0; N_FAIL=0

# ── Disk space guard (require at least 15 GB free) ────────────────────────────
check_disk() {
    local free_kb
    free_kb=$(df -k "${ROOT}" | awk 'NR==2 {print $4}')
    local free_gb=$(( free_kb / 1024 / 1024 ))
    if [[ ${free_gb} -lt 15 ]]; then
        echo "ERROR: only ${free_gb} GB free on ${ROOT} — need at least 15 GB. Aborting." >&2
        exit 1
    fi
    echo "  Disk check OK: ${free_gb} GB free"
}

# ── Core train_one function ────────────────────────────────────────────────────
train_one() {
    local tag="$1"; shift
    local extra_args=("$@")

    # Apply filters
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
        echo "  [SKIP]  ${tag}"
        ((N_SKIP++)) || true
        return 0
    fi

    # GC always on: 768d+ needs >40GB without it; on 512d it only costs ~10% speed
    local _gc="true"

    # Resume if interrupted checkpoint exists (--resume is a store_true flag)
    local -a _resume_flag=()
    [[ -f "${run_dir}/training_cursor.json" ]] && _resume_flag=(--resume)

    local cmd=(env CUDA_VISIBLE_DEVICES="${GPU}" XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH="/ssd1/marco.simoni/VULNERABILITY/NETGROUP/DantinoX:${PYTHONPATH:-}" python dantinox/cli.py train
        --config "${BASE_CFG}"
        --run_dir "${run_dir}"
        --use_bf16 true
        --use_flash_attention true
        --gradient_checkpointing "${_gc}"
        --tokenizer_type t5
        --vocab_size 32128
        --mask_token_id 32099
        --max_train_tokens "${_CUR_TOKENS}"
        --epochs "${_CUR_EPOCHS}"
        "${_resume_flag[@]}"
        "${extra_args[@]}")

    echo ""
    echo "  ── ${tag}"
    [[ "${DRY_RUN}" == "true" ]] && echo "     ${cmd[*]}" && return 0

    check_disk
    local log_file="${LOG_DIR}/${tag}.log"
    if "${cmd[@]}" 2>&1 | tee "${log_file}"; then
        ((N_DONE++)) || true
        echo "  [OK]   ${tag}"
    else
        ((N_FAIL++)) || true
        echo "  [FAIL] ${tag}  log: ${log_file}" >&2
    fi
    # Let CUDA driver fully release GPU memory before the next run
    sleep 30
}

# ── Attention config helpers ──────────────────────────────────────────────────
mha_flags() { local nh=$1; echo "--kv_heads ${nh} --mla false"; }
gqa_flags() { local nh=$1; local kv=$(( nh / 4 )); [[ ${kv} -lt 1 ]] && kv=1; echo "--kv_heads ${kv} --mla false"; }
mla_flags() {
    local nh=$1 hs=$2
    local dkv=$(( hs * 3 )); [[ ${dkv} -gt 256 ]] && dkv=256
    local dq=$(( hs * 6  )); [[ ${dq}  -gt 256 ]] && dq=256
    local rd=$(( hs / 2  )); [[ ${rd}  -lt 16  ]] && rd=16
    echo "--kv_heads ${nh} --mla true --inference false --down_dim_kv ${dkv} --down_dim_q ${dq} --rope_dim ${rd}"
}

# ── Part A runner  ────────────────────────────────────────────────────────────
run_part_a() {
    [[ "${PART}" == "B" ]] && return 0

    # ── A1: Dense size × attention matrix ────────────────────────────────────
    # dim  n_heads  head_size  num_blocks  lr     optimizer
    # head_size=64 throughout — clean dims, efficient on A100 flash-attention.
    local -a DENSE=(
        " 512  8 64 12  0.002 muon"
        " 768 12 64 16  0.002 muon"
    )
    for cfg in "${DENSE[@]}"; do
        read -r dim nh hs blocks lr opt <<< "${cfg}"
        # shellcheck disable=SC2046,SC2086
        train_one "diff_mha_${dim}d_${blocks}b_Dense" --dim "${dim}" --n_heads "${nh}" --head_size "${hs}" \
            --num_blocks "${blocks}" --lr "${lr}" --optimizer "${opt}" --use_moe false $(mha_flags "${nh}")
        # shellcheck disable=SC2046,SC2086
        train_one "diff_gqa_${dim}d_${blocks}b_Dense" --dim "${dim}" --n_heads "${nh}" --head_size "${hs}" \
            --num_blocks "${blocks}" --lr "${lr}" --optimizer "${opt}" --use_moe false $(gqa_flags "${nh}")
        # shellcheck disable=SC2046,SC2086
        train_one "diff_mla_${dim}d_${blocks}b_Dense" --dim "${dim}" --n_heads "${nh}" --head_size "${hs}" \
            --num_blocks "${blocks}" --lr "${lr}" --optimizer "${opt}" --use_moe false $(mla_flags "${nh}" "${hs}")
    done
}

# ── Part B: ablations on 1024d×24b baseline ───────────────────────────────────
# All ablations use the same base: dim=1024, n_heads=16, head_size=64, blocks=24,
# optimizer=muon, lr=0.002.  ONE variable changes at a time.
run_part_b() {
    [[ "${PART}" == "A" ]] && return 0
    _CUR_TOKENS=${TOKENS_B}
    _CUR_EPOCHS=${EPOCHS_B}

    local BASE_DIM=768 BASE_NH=12 BASE_HS=64 BASE_BLOCKS=16 BASE_LR=0.002 BASE_OPT=muon

    ablation() {
        # ablation <label_suffix> <attn_tag> <attn_flags> <extra...>
        local suffix="$1"; shift
        local attn_tag="$1"; shift
        local attn_flags="$1"; shift
        local extra=("$@")
        # shellcheck disable=SC2086
        train_one "diff_${attn_tag}_768d_16b_${suffix}" \
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

        # B1 — norm_type: RMSNorm
        ablation "RMSNorm"       "${attn}" "${flags}" --norm_type rmsnorm

        # B2 — noise_schedule: cosine, sqrt (vs linear default)
        ablation "SchedCosine"   "${attn}" "${flags}" --noise_schedule cosine
        ablation "SchedSqrt"     "${attn}" "${flags}" --noise_schedule sqrt

        # B3 — dropout
        ablation "Drop0"         "${attn}" "${flags}" --dropout_rate 0.0
        ablation "Drop20"        "${attn}" "${flags}" --dropout_rate 0.20

        # B4 — FFN activation: GELU (no SwiGLU gate)
        ablation "GELU"          "${attn}" "${flags}" --use_swiglu false

        # B5 — sliding window attention (window=64)
        ablation "SlidingWin64"  "${attn}" "${flags}" --sliding_window true --context_window 64

        # B6 — attention gate (no_sink)
        ablation "NoSink"        "${attn}" "${flags}" --no_sink true

        # B7 — LR schedule: WSD (warmup-stable-decay)
        ablation "SchedWSD"      "${attn}" "${flags}" --lr_schedule wsd

        # B8 — optimizer: AdamW and Lion (compare vs Muon base)
        ablation "OptAdamW"      "${attn}" "${flags}" --optimizer adamw --lr 0.0003
        ablation "OptLion"       "${attn}" "${flags}" --optimizer lion  --lr 0.0001

        # B9 — MoE 8exp (complement to Part A MoE 6exp)
        ablation "MoE8exp"       "${attn}" "${flags}" --use_moe true --n_experts 8 --top_k_mlp 2

        # B10 — batch size 128 (2× larger micro-batch)
        ablation "BS128"         "${attn}" "${flags}" --batch_size 128 --grad_accum 8

        # B11 — context length
        ablation "Ctx256"        "${attn}" "${flags}" --max_context 256
        ablation "Ctx1024"       "${attn}" "${flags}" --max_context 1024
    done
}

# ── Summary ───────────────────────────────────────────────────────────────────
print_summary() {
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  Diffusion suite complete"
    echo "  Trained : ${N_DONE}"
    echo "  Skipped : ${N_SKIP}  (already exist)"
    echo "  Failed  : ${N_FAIL}"
    echo "  Total   : ${N_TOTAL}"
    echo "════════════════════════════════════════════════════════════"
}

# ── Entry point ───────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "  DantinoX — Diffusion model training suite"
echo "  Base config : ${BASE_CFG}"
echo "  Part filter : ${PART}"
echo "  Attn filter : ${ATTN_F}"
echo "  Dim filter  : ${DIM_F}"
echo "  MoE filter  : ${MOE_F}"
echo "  Dry-run     : ${DRY_RUN}"
echo "════════════════════════════════════════════════════════════"

if [[ "${DRY_RUN}" == "false" ]]; then check_disk; fi

run_part_a
run_part_b

print_summary
