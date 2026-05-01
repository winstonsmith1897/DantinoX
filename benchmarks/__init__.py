"""
DantinoX inference benchmark suite
===================================
Two-stage pipeline:
  1. inference_sweep  — collect metrics across attention types, model sizes, etc.
  2. plot_inference   — render 21 figures from the CSV

Run everything:
  python benchmarks/run_all.py           # full pipeline
  dantinox infbench                      # same via CLI
  make infbench                          # same via Makefile
"""
