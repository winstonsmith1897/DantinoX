#!/bin/bash
# Sequential ablation chain on a single GPU (0): waits for the currently
# running 512d grid, then runs the remaining ablations for all three
# production architectures.  Logs per step in logs/.
set -u
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0

# Wait for the in-flight 512d grid run to finish
while pgrep -f "benchmarks/paradigm_ablations.py grid" >/dev/null; do
  sleep 20
done
# Normalise its output name to the per-arch convention
[ -f results/ablation_grid.csv ] && cp results/ablation_grid.csv results/ablation_grid_512d12b.csv

run() {  # run <ablation> <arch>
  local abl=$1 arch=$2 env=()
  [ "$abl" = ceiling ] && env=(XLA_PYTHON_CLIENT_MEM_FRACTION=.92)
  echo "=== $(date '+%H:%M:%S') $abl $arch ==="
  env "${env[@]}" python benchmarks/paradigm_ablations.py "$abl" --arch "$arch" \
      --out "results/ablation_${abl}_${arch}.csv" \
      > "logs/ablation_${abl}_${arch}.log" 2>&1
  echo "    exit=$?"
}

run stack 512d12b
run ceiling 512d12b

for ARCH in 768d16b 1024d16b; do
  run grid "$ARCH"
  run stack "$ARCH"
  run ceiling "$ARCH"
done

echo "=== $(date '+%H:%M:%S') chain complete ==="
