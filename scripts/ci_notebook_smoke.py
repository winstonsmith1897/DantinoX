#!/usr/bin/env python3
"""Execute a docs notebook end-to-end on CPU with a shrunk compute budget.

Used by the ``notebooks.yml`` CI workflow to catch notebook rot (stale API
calls, broken caches, renamed kwargs) *before* a user finds it on Colab.

The notebook's code is executed as-is except for smoke transforms that shrink
the compute budget (CI runs on CPU): epochs → 1, model depth capped, batch
capped, generation lengths capped.  Install cells and GPU pins are skipped —
CI installs the working-tree package itself.

Usage::

    python scripts/ci_notebook_smoke.py docs/notebooks/01_quickstart.ipynb
"""
from __future__ import annotations

import re
import sys
import time

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

#: (pattern, replacement) applied to every code cell before execution.
SMOKE_TRANSFORMS: tuple[tuple[str, str], ...] = (
    (r"epochs\s*=\s*\d+", "epochs=1"),
    (r"num_blocks\s*=\s*\d+", "num_blocks=2"),
    (r"batch_size\s*=\s*\d+", "batch_size=8"),
    (r"max_new_tokens\s*=\s*\d+", "max_new_tokens=16"),
    (r"n_steps\s*=\s*\d+", "n_steps=4"),
    (r"CUDA_VISIBLE_DEVICES'\]\s*=\s*'\d+'", "CUDA_VISIBLE_DEVICES'] = ''"),
)


def _should_skip(src: str) -> bool:
    s = src.strip()
    return s.startswith("!pip install") and "dantinox" in s.lower()


def main() -> int:
    """Run one notebook in smoke mode; return a process exit code."""
    path = sys.argv[1]
    timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 1200

    nb = nbformat.read(path, as_version=4)
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for pat, rep in SMOKE_TRANSFORMS:
            cell.source = re.sub(pat, rep, cell.source)

    client = NotebookClient(nb, timeout=timeout, kernel_name="python3")
    n_code = sum(1 for c in nb.cells if c.cell_type == "code")
    print(f"[smoke] {path} ({n_code} code cells)", flush=True)

    with client.setup_kernel():
        for i, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            first = cell.source.strip().splitlines()[0] if cell.source.strip() else "(empty)"
            if _should_skip(cell.source):
                print(f"[skip] cell {i}: {first[:70]}", flush=True)
                continue
            t0 = time.time()
            try:
                client.execute_cell(cell, i)
                print(f"[ok {time.time()-t0:5.1f}s] cell {i}: {first[:70]}", flush=True)
            except CellExecutionError:
                print(f"[FAIL] cell {i}: {first[:70]}", flush=True)
                for out in cell.get("outputs", []):
                    if out.get("output_type") == "error":
                        print("\n".join(out.get("traceback", []))[-3000:], flush=True)
                return 1
    print("[smoke] PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
