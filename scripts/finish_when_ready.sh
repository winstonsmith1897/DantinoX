#!/bin/bash
# Detached finisher: waits for the gap-filling pass on GPU 0, then
# regenerates all figures (headline + analysis) for every architecture.
set -u
cd "$(dirname "$0")/.."
exec >> logs/finish_when_ready.log 2>&1
echo "=== $(date '+%F %H:%M:%S') finisher started"
while pgrep -f "fill_gaps.sh" >/dev/null || pgrep -f "paradigm_ablations.py (pareto|grid)" >/dev/null; do
  sleep 60
done
echo "=== $(date '+%F %H:%M:%S') data complete — regenerating figures"
python benchmarks/plot_headline.py
python benchmarks/plot_paradigm_ablations.py
echo "=== $(date '+%F %H:%M:%S') ALL FIGURES READY in results/paradigm_bench/"
touch results/paradigm_bench/.figures_complete
