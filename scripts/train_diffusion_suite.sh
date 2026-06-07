#!/usr/bin/env bash
# scripts/train_diffusion_suite.sh
#
# Full Diffusion model training suite for DantinoX — EMNLP 2026 paper.
#
# ══════════════════════════════════════════════════════════════════════════════
# Experiment matrix  (~90 runs total)
# ══════════════════════════════════════════════════════════════════════════════
#
# PART A — Size × Attention × FFN matrix  (48 runs)
# ─────────────────────────────────────────────────
#   Attention: MHA / GQA(×4) / MLA
#   Sizes (dim × blocks):
#     128×12, 192×12,
#     256×8,  256×12, 256×16,
#     384×12,
#     512×8,  512×12, 512×16,
#     768×12
#   FFN: Dense (all sizes) + MoE top-2/6exp (256+512 only)
#
# PART B — Architecture ablations on 256d×12b  (~42 runs)
# ────────────────────────────────────────────────────────
# Each ablation varies ONE axis vs the Dense 256d×12b baseline:
#
#   B1. norm_type:       rmsnorm          (vs layernorm)           × MHA/GQA/MLA
#   B2. noise_schedule:  linear, sqrt     (vs cosine)              × MHA/GQA/MLA
#   B3. dropout_rate:    0.0, 0.20        (vs 0.15)                × MHA/GQA/MLA
#   B4. use_swiglu:      false/GELU       (vs SwiGLU)              × MHA/GQA/MLA
#   B5. sliding_window:  true,ctx=64      (vs full attention)      × MHA/GQA/MLA
#   B6. no_sink:         true             (vs false)               × MHA/GQA/MLA
#   B7. lr_schedule:     wsd              (vs cosine)              × MHA/GQA/MLA
#   B8. optimizer:       lion             (vs adamw)               × MHA/GQA/MLA
#
# ══════════════════════════════════════════════════════════════════════════════
# Hardware:   2 × A100 (n_devices=2 in base config)
# Precision:  bf16 (n_devices=2 in base config)
# Dataset:    Daniele/dante-corpus (HuggingFace, streaming)
# Tokenizer:  char-level (re-trained per run)
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

BASE_CFG="configs/diffusion_base.yaml"
LOG_DIR="logs/diffusion_suite"
mkdir -p "${LOG_DIR}" logs

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

    # Skip if already completed
    if [[ -f "${run_dir}/model_weights.msgpack" || -f "${run_dir}/best_model_weights.msgpack" ]]; then
        echo "  [SKIP]  ${tag}"
        ((N_SKIP++)) || true
        return 0
    fi

    local cmd=(env PYTHONPATH="/ssd1/marco.simoni/VULNERABILITY/NETGROUP/DantinoX:${PYTHONPATH:-}" python dantinox/cli.py train
        --config "${BASE_CFG}"
        --run_dir "${run_dir}"
        --use_bf16 true
        --use_flash_attention true
        --gradient_checkpointing true
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
    # dim  n_heads  head_size  num_blocks  lr      optimizer
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
        train_one "diff_mha_${dim}d_${blocks}b_Dense" --dim "${dim}" --n_heads "${nh}" --head_size "${hs}" \
            --num_blocks "${blocks}" --lr "${lr}" --optimizer "${opt}" --use_moe false $(mha_flags "${nh}")
        # shellcheck disable=SC2046,SC2086
        train_one "diff_gqa_${dim}d_${blocks}b_Dense" --dim "${dim}" --n_heads "${nh}" --head_size "${hs}" \
            --num_blocks "${blocks}" --lr "${lr}" --optimizer "${opt}" --use_moe false $(gqa_flags "${nh}")
        # shellcheck disable=SC2046,SC2086
        train_one "diff_mla_${dim}d_${blocks}b_Dense" --dim "${dim}" --n_heads "${nh}" --head_size "${hs}" \
            --num_blocks "${blocks}" --lr "${lr}" --optimizer "${opt}" --use_moe false $(mla_flags "${nh}" "${hs}")
    done

    # ── A2: MoE variants (256+512, select depths) ─────────────────────────────
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
        train_one "diff_mha_${dim}d_${blocks}b_MoE" --dim "${dim}" --n_heads "${nh}" --head_size "${hs}" \
            --num_blocks "${blocks}" --lr "${lr}" --optimizer "${opt}" \
            --use_moe true --n_experts 6 --top_k_mlp 2 $(mha_flags "${nh}")
        # shellcheck disable=SC2046,SC2086
        train_one "diff_gqa_${dim}d_${blocks}b_MoE" --dim "${dim}" --n_heads "${nh}" --head_size "${hs}" \
            --num_blocks "${blocks}" --lr "${lr}" --optimizer "${opt}" \
            --use_moe true --n_experts 6 --top_k_mlp 2 $(gqa_flags "${nh}")
        # shellcheck disable=SC2046,SC2086
        train_one "diff_mla_${dim}d_${blocks}b_MoE" --dim "${dim}" --n_heads "${nh}" --head_size "${hs}" \
            --num_blocks "${blocks}" --lr "${lr}" --optimizer "${opt}" \
            --use_moe true --n_experts 6 --top_k_mlp 2 $(mla_flags "${nh}" "${hs}")
    done
}

# ── Part B: ablations on 256d×12b baseline ────────────────────────────────────
# All ablations use the same base: dim=256, n_heads=8, head_size=32, blocks=12,
# lr=0.0012, optimizer=lion, use_moe=false.  ONE variable changes at a time.
run_part_b() {
    [[ "${PART}" == "A" ]] && return 0

    local BASE_DIM=256 BASE_NH=8 BASE_HS=32 BASE_BLOCKS=12 BASE_LR=0.0012 BASE_OPT=lion

    ablation() {
        # ablation <label_suffix> <attn_tag> <attn_flags> <extra...>
        local suffix="$1"; shift
        local attn_tag="$1"; shift
        local attn_flags="$1"; shift
        local extra=("$@")
        # shellcheck disable=SC2086
        train_one "diff_${attn_tag}_256d_12b_${suffix}" \
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

        # B2 — noise_schedule
        ablation "SchedLinear"   "${attn}" "${flags}" --noise_schedule linear
        ablation "SchedSqrt"     "${attn}" "${flags}" --noise_schedule sqrt

        # B3 — dropout
        ablation "Drop0"         "${attn}" "${flags}" --dropout_rate 0.0
        ablation "Drop20"        "${attn}" "${flags}" --dropout_rate 0.20

        # B4 — FFN activation: GELU (no SwiGLU gate)
        ablation "GELU"          "${attn}" "${flags}" --use_swiglu false

        # B5 — sliding window attention (window=64)
        ablation "SlidingWin64" "${attn}" "${flags}" --sliding_window true --context_window 64

        # B6 — attention gate (no_sink)
        ablation "NoSink"        "${attn}" "${flags}" --no_sink true

        # B7 — LR schedule: WSD (warmup-stable-decay)
        ablation "SchedWSD"      "${attn}" "${flags}" --lr_schedule wsd

        # B8 — optimizer: Lion
        ablation "OptLion"       "${attn}" "${flags}" --optimizer lion --lr 0.0003

        # B9 — deeper diffusion T grid: 500 steps (vs 1000)
        ablation "T500"          "${attn}" "${flags}" --diffusion_steps 500

        # B10 — time_emb_dim: 128 (vs 256)
        ablation "TimeEmb128"    "${attn}" "${flags}" --time_emb_dim 128

        # B11 — MoE 8exp on 256d (complement to Part A MoE 6exp)
        ablation "MoE8exp"       "${attn}" "${flags}" --use_moe true --n_experts 8 --top_k_mlp 2

        # B12 — batch size 128 (2× larger)
        ablation "BS128"         "${attn}" "${flags}" --batch_size 128 --grad_accum 8

        # B13 — max_context 256 (half the default)
        ablation "Ctx256"        "${attn}" "${flags}" --max_context 256

        # B14 — max_context 1024 (double the default)
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
