#!/bin/bash
# Crash-safe ceiling driver: one process per (arch, paradigm, attn) series.
# A segfault during OOM probing only ends that series; probes already
# appended to the CSV survive.
set -u
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=.92
for ARCH in 512d12b 768d16b 1024d16b; do
  rm -f "results/ablation_ceiling_${ARCH}.csv"
  for P in AR Discrete Continuous; do
    for A in mha gqa mla; do
      echo "=== $(date '+%H:%M:%S') ceiling $ARCH $P:$A ==="
      python benchmarks/paradigm_ablations.py ceiling --arch "$ARCH" \
        --series "$P:$A" --out "results/ablation_ceiling_${ARCH}.csv" \
        >> "logs/ablation_ceiling_${ARCH}.log" 2>&1 || echo "    series ended ($?)"
    done
  done
done
echo "=== ceiling driver complete ==="
