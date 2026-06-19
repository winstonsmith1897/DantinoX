#!/bin/bash
# OOM retry watcher.
#
# 1. Waits for the v3 campaign and the ceiling driver to finish.
# 2. Then waits until no user other than marco.simoni has processes on any
#    GPU (poll every 5 min, up to 48 h).
# 3. Picks fully-free GPUs (<500 MiB used) and re-runs every grid/pareto
#    point whose last CSV row is an OOM marker (--retry-oom skips completed
#    points, so each pass only attempts what is missing).
# 4. TP retries for 1536d24b need two free GPUs; skipped otherwise.
set -u
cd "$(dirname "$0")/.."
LOG="logs/oom_retry_watcher.log"
exec >> "$LOG" 2>&1

echo "=== $(date '+%F %H:%M:%S') watcher started"

# ── 1. wait for our own campaigns to finish ───────────────────────────────────
while pgrep -f "scripts/v3_campaign.sh" >/dev/null \
   || pgrep -f "scripts/ceiling_driver.sh" >/dev/null; do
  sleep 120
done
echo "=== $(date '+%F %H:%M:%S') campaigns finished — waiting for exclusive GPUs"

# ── 2. wait until only marco.simoni uses the GPUs ─────────────────────────────
other_users() {
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    u=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
    [ -n "$u" ] && [ "$u" != "marco.simoni" ] && [ "$u" != "marco.s+" ] && echo "$u"
  done | sort -u
}

DEADLINE=$(( $(date +%s) + 48*3600 ))
while :; do
  OTHERS=$(other_users)
  if [ -z "$OTHERS" ]; then
    break
  fi
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then
    echo "=== $(date '+%F %H:%M:%S') 48h deadline reached, others still active: $OTHERS — giving up"
    exit 1
  fi
  sleep 300
done
echo "=== $(date '+%F %H:%M:%S') GPUs exclusive to marco.simoni"

# ── 3. pick fully-free GPUs ───────────────────────────────────────────────────
mapfile -t FREE < <(nvidia-smi --query-gpu=index,memory.used \
                      --format=csv,noheader,nounits | awk -F', ' '$2<500{print $1}')
echo "free GPUs: ${FREE[*]:-none}"
if [ "${#FREE[@]}" -eq 0 ]; then
  echo "no fully-free GPU available — aborting"
  exit 1
fi
G0=${FREE[0]}

retry() {  # retry <ablation> <arch> [extra]
  local abl=$1 arch=$2; shift 2
  local csv="results/ablation_${abl}_${arch}.csv"
  [ -f "$csv" ] || return 0
  echo "--- $(date '+%H:%M:%S') retry-oom $abl $arch $*"
  python benchmarks/paradigm_ablations.py "$abl" --arch "$arch" \
      --retry-oom --precision bf16 --out "$csv" "$@" || echo "    exit=$?"
}

export CUDA_VISIBLE_DEVICES=$G0
for ARCH in 512d12b 768d16b 1024d16b; do
  retry pareto "$ARCH"
  retry grid   "$ARCH"
done
# Precision-study CSVs have explicit names → explicit retry calls
[ -f results/ablation_pareto_512d12b_tf32.csv ] && \
  python benchmarks/paradigm_ablations.py pareto --arch 512d12b --retry-oom \
    --precision tf32 --out results/ablation_pareto_512d12b_tf32.csv
[ -f results/ablation_pareto_512d12b_f32.csv ] && \
  python benchmarks/paradigm_ablations.py pareto --arch 512d12b --retry-oom \
    --precision f32 --out results/ablation_pareto_512d12b_f32.csv

# ── 4. TP retries need two free GPUs ─────────────────────────────────────────
if [ "${#FREE[@]}" -ge 2 ]; then
  export CUDA_VISIBLE_DEVICES=${FREE[0]},${FREE[1]}
  retry grid   1536d24b --tp 2
  retry pareto 1536d24b --tp 2
else
  echo "only one free GPU — skipping 1536d24b TP retries"
fi

echo "=== $(date '+%F %H:%M:%S') watcher complete"
