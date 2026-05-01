"""
Thin stub — logic lives in benchmarks/trained_batch_sweep.py.

Usage:
  python benchmark_batch_sweep.py [args...]
  python benchmarks/trained_batch_sweep.py [args...]   # preferred
  dantinox infbench --trained                           # full pipeline
"""
from benchmarks.trained_batch_sweep import main

if __name__ == "__main__":
    main()
