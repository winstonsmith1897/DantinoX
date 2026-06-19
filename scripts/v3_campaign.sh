#!/bin/bash
# v3 ablation campaign on GPUs 2+3, borrowing them from the user's own
# training runs (SIGSTOP → run → SIGCONT).  The EXIT trap guarantees the
# trainings resume even if the campaign crashes.
set -u
cd "$(dirname "$0")/.."

TRAIN_PIDS="1039299 378132"          # diff_gqa_768d_16b (GPU2), elf_gqa_512d_12b (GPU3)

resume() {
  for p in $TRAIN_PIDS; do kill -CONT "$p" 2>/dev/null; done
  echo "=== $(date '+%H:%M:%S') trainings resumed (SIGCONT $TRAIN_PIDS)"
}
trap resume EXIT

echo "=== $(date '+%H:%M:%S') suspending trainings (SIGSTOP $TRAIN_PIDS)"
for p in $TRAIN_PIDS; do kill -STOP "$p" || echo "WARN: cannot stop $p"; done
sleep 5
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader -i 2,3

export XLA_PYTHON_CLIENT_PREALLOCATE=false   # co-tenant VRAM: allocate on demand

run() {  # run <logname> <ablation> <arch> [extra args...]
  # Up to 3 attempts: the runners write incrementally and resume, so a
  # segfaulted point is skipped on retry and the rest of the sweep completes.
  local name=$1 abl=$2 arch=$3; shift 3
  local attempt rc
  for attempt in 1 2 3; do
    echo "=== $(date '+%H:%M:%S') $abl $arch $* (attempt $attempt)"
    python benchmarks/paradigm_ablations.py "$abl" --arch "$arch" "$@" \
        >> "logs/v3_${name}.log" 2>&1
    rc=$?
    echo "    exit=$rc"
    [ "$rc" -eq 0 ] && break
  done
}

# ── Single-GPU campaign (GPU 2, ~21 GB free) ──────────────────────────────────
export CUDA_VISIBLE_DEVICES=2
for ARCH in 512d12b 768d16b 1024d16b; do
  run "pareto_${ARCH}" pareto "$ARCH" --precision bf16 --out "results/ablation_pareto_${ARCH}.csv"
  run "grid_${ARCH}"   grid   "$ARCH" --precision bf16 --out "results/ablation_grid_${ARCH}.csv"
done
# Precision study (512d12b): true fp32 vs TF32 vs bf16 (bf16 done above)
run pareto_512_tf32 pareto 512d12b --precision tf32 --out results/ablation_pareto_512d12b_tf32.csv
run pareto_512_f32  pareto 512d12b --precision f32  --out results/ablation_pareto_512d12b_f32.csv

# ── Tensor-parallel campaign (GPUs 2+3, TP=2, ~0.95 B params) ────────────────
export CUDA_VISIBLE_DEVICES=2,3
echo "=== $(date '+%H:%M:%S') TP sanity test"
TP_READY=0
if timeout 240 python scripts/tp_test.py 2>&1 | grep -q "TP-OK"; then
  TP_READY=1
elif NCCL_P2P_DISABLE=1 timeout 240 python scripts/tp_test.py 2>&1 | grep -q "TP-OK"; then
  TP_READY=1
  export NCCL_P2P_DISABLE=1
  echo "    (TP works with NCCL_P2P_DISABLE=1)"
fi

if [ "$TP_READY" = 1 ]; then
  run grid_tp   grid   1536d24b --tp 2 --precision bf16 --out results/ablation_grid_1536d24b.csv
  run pareto_tp pareto 1536d24b --tp 2 --precision bf16 --out results/ablation_pareto_1536d24b.csv
  run stack_tp  stack  1536d24b --tp 2 --precision bf16 --out results/ablation_stack_1536d24b.csv
else
  echo "=== TP sanity FAILED on GPUs 2,3 — skipping 1536d24b TP runs"
fi

echo "=== $(date '+%H:%M:%S') campaign complete"
