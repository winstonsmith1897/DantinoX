#!/usr/bin/env python3
"""
scripts/cleanup_runs.py
========================

Free disk space by removing duplicate training runs.

Strategy
--------
Runs are grouped by their model configuration (dim, num_blocks, attention_type,
use_moe, model_type).  Within each group **only the run with the lowest
validation loss is kept**; the rest are deleted.

Runs with no training log (failed / empty) are always deleted unless
``--keep-failed`` is passed.

Safety
------
  * Prints a detailed report before deleting anything.
  * Requires explicit ``--execute`` flag to actually delete.
  * Writes a manifest of deleted runs to ``logs/cleanup_manifest.json``.

Usage
-----
  # Dry-run: show what would be deleted (no changes)
  python scripts/cleanup_runs.py --runs-dir runs

  # Actually delete
  python scripts/cleanup_runs.py --runs-dir runs --execute

  # Keep failed runs (only delete duplicates)
  python scripts/cleanup_runs.py --runs-dir runs --execute --keep-failed

  # Restrict to a model_type
  python scripts/cleanup_runs.py --runs-dir runs --model-type autoregressive
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def _load_run(run_path: str) -> dict:
    import yaml
    cfg_path = os.path.join(run_path, "config.yaml")
    if not os.path.exists(cfg_path):
        return {}
    with open(cfg_path) as f:
        c = yaml.safe_load(f)
    return c


def _val_loss(run_path: str) -> float | None:
    import pandas as pd
    log_path = os.path.join(run_path, "training_log.csv")
    if not os.path.exists(log_path):
        return None
    try:
        df = pd.read_csv(log_path)
        vals = df["val_loss"].dropna()
        return float(vals.min()) if len(vals) > 0 else None
    except Exception:
        return None


def _dir_size_mb(path: str) -> int:
    import subprocess
    try:
        return int(subprocess.check_output(["du", "-sm", path]).split()[0])
    except Exception:
        return 0


def _run_key(cfg: dict) -> tuple:
    """Group key: what makes two runs 'the same configuration'."""
    mla      = cfg.get("mla", False)
    kv_heads = cfg.get("kv_heads", cfg.get("n_heads", 0))
    n_heads  = cfg.get("n_heads", 0)
    if mla:
        attn = "mla"
    elif kv_heads < n_heads:
        attn = "gqa"
    else:
        attn = "mha"
    return (
        cfg.get("model_type", "autoregressive"),
        cfg.get("dim"),
        cfg.get("num_blocks"),
        attn,
        cfg.get("use_moe", False),
        cfg.get("use_swiglu", True),
        cfg.get("norm_type", "layernorm"),
        cfg.get("noise_schedule", "cosine"),
        cfg.get("dropout_rate", 0.15),
        cfg.get("no_sink", False),
        cfg.get("sliding_window", False),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Clean up duplicate DantinoX training runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--runs-dir",    default="runs")
    parser.add_argument("--execute",     action="store_true",
                        help="Actually delete runs (default: dry-run only)")
    parser.add_argument("--keep-failed", action="store_true",
                        help="Keep runs with no training log / zero steps")
    parser.add_argument("--model-type",  default=None,
                        help="Restrict to 'autoregressive' or 'diffusion'")
    parser.add_argument("--manifest",    default="logs/cleanup_manifest.json",
                        help="JSON file recording deleted runs")
    args = parser.parse_args(argv)

    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_dir():
        print(f"Runs directory not found: {runs_dir}", file=sys.stderr)
        sys.exit(1)

    # ── Collect all runs ──────────────────────────────────────────────────────
    runs: list[dict] = []
    for name in sorted(os.listdir(runs_dir)):
        path = str(runs_dir / name)
        if not os.path.isdir(path):
            continue
        cfg = _load_run(path)
        if not cfg:
            # No config.yaml → skip (not a DantinoX run)
            continue
        if args.model_type and cfg.get("model_type", "autoregressive") != args.model_type:
            continue
        runs.append({
            "name":     name,
            "path":     path,
            "cfg":      cfg,
            "key":      _run_key(cfg),
            "val_loss": _val_loss(path),
            "size_mb":  _dir_size_mb(path),
        })

    # ── Group and identify duplicates ─────────────────────────────────────────
    from collections import defaultdict
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in runs:
        groups[r["key"]].append(r)

    to_delete:  list[dict] = []
    to_keep:    list[dict] = []
    failed:     list[dict] = []

    for key, grp in groups.items():
        # Separate runs with vs without val_loss
        valid   = [r for r in grp if r["val_loss"] is not None]
        invalid = [r for r in grp if r["val_loss"] is None]

        if not valid:
            # All failed
            failed.extend(grp)
            if not args.keep_failed:
                to_delete.extend(grp)
            else:
                to_keep.extend(grp)
            continue

        # Keep run with best (lowest) val_loss
        best = min(valid, key=lambda r: r["val_loss"])
        to_keep.append(best)

        # Delete the rest (including all invalid in this group)
        for r in valid:
            if r["name"] != best["name"]:
                to_delete.append(r)
        to_delete.extend(invalid)

    # ── Report ────────────────────────────────────────────────────────────────
    total_mb  = sum(r["size_mb"] for r in to_delete)
    keep_mb   = sum(r["size_mb"] for r in to_keep)
    failed_mb = sum(r["size_mb"] for r in failed)

    print(f"\nDantinoX run cleanup — {'DRY RUN' if not args.execute else 'EXECUTING'}")
    print(f"  Runs scanned : {len(runs)}")
    print(f"  Runs to keep : {len(to_keep)}  ({keep_mb} MB = {keep_mb/1024:.1f} GB)")
    print(f"  Runs to delete: {len(to_delete)}  ({total_mb} MB = {total_mb/1024:.1f} GB)")
    if failed:
        print(f"  Failed runs  : {len(failed)}  ({failed_mb} MB)")
    print()

    if to_delete:
        print("Runs to DELETE:")
        for r in sorted(to_delete, key=lambda r: r["size_mb"], reverse=True):
            vl = f"{r['val_loss']:.4f}" if r["val_loss"] else "N/A"
            print(f"  [{r['size_mb']:>4} MB]  val_loss={vl:>7}  {r['name']}")

    if to_keep:
        print("\nRuns to KEEP (best per config):")
        for r in sorted(to_keep, key=lambda r: r["key"]):
            vl = f"{r['val_loss']:.4f}" if r["val_loss"] else "N/A"
            print(f"  [{r['size_mb']:>4} MB]  val_loss={vl:>7}  {r['name']}")

    if not args.execute:
        print(f"\n  → Add --execute to actually delete {len(to_delete)} runs "
              f"and free {total_mb} MB ({total_mb/1024:.1f} GB)")
        return

    # ── Execute deletion ──────────────────────────────────────────────────────
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    deleted  = []
    n_errors = 0
    for r in to_delete:
        try:
            shutil.rmtree(r["path"])
            deleted.append(r["name"])
            print(f"  deleted  {r['name']}  ({r['size_mb']} MB)")
        except Exception as exc:
            print(f"  ERROR deleting {r['name']}: {exc}", file=sys.stderr)
            n_errors += 1

    with open(manifest_path, "w") as f:
        json.dump({
            "deleted": deleted,
            "kept": [r["name"] for r in to_keep],
            "freed_mb": total_mb,
        }, f, indent=2)

    print(f"\nDeleted {len(deleted)} runs, freed ~{total_mb} MB")
    print(f"Manifest: {manifest_path}")
    if n_errors:
        print(f"Errors: {n_errors}", file=sys.stderr)


if __name__ == "__main__":
    main()
