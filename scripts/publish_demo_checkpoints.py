#!/usr/bin/env python3
"""Train the three tiny demo checkpoints and (optionally) push them to HF Hub.

These power the README "10 seconds to first output" one-liner:

    import dantinox as dx
    print(dx.quick_generate("<hf-user>/dantinox-tiny-ar", "HAMLET:"))

Usage
-----
Train only (checkpoints land in runs/demo_*)::

    python scripts/publish_demo_checkpoints.py

Train and push (needs `huggingface-cli login` or HF_TOKEN)::

    python scripts/publish_demo_checkpoints.py --push --owner <hf-username>
"""
from __future__ import annotations

import argparse
import os
import urllib.request

CORPUS = "tiny_shakespeare.txt"
CORPUS_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
              "master/data/tinyshakespeare/input.txt")


def _ensure_corpus() -> None:
    """Download tiny_shakespeare.txt next to the script if missing."""
    if not os.path.exists(CORPUS):
        urllib.request.urlretrieve(CORPUS_URL, CORPUS)
        print(f"downloaded {CORPUS}")


def main() -> None:
    """Train ar / discrete / continuous tiny demo models; push with --push."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--push", action="store_true",
                        help="push the trained runs to HuggingFace Hub")
    parser.add_argument("--owner", default=None,
                        help="HF username/org for the repo ids (required with --push)")
    args = parser.parse_args()
    if args.push and not args.owner:
        parser.error("--push requires --owner <hf-username>")

    import dantinox as dx

    _ensure_corpus()

    runs: dict[str, str] = {}

    runs["ar"] = dx.fit(
        "ar", CORPUS, run_dir="runs/demo_tiny_ar",
        dim=256, n_heads=4, num_blocks=4, max_context=256,
        lr=3e-4, epochs=20, batch_size=32, tokenizer_type="char",
    )
    runs["discrete"] = dx.fit(
        "discrete", CORPUS, run_dir="runs/demo_tiny_discrete",
        dim=256, n_heads=4, num_blocks=4, max_context=256,
        noise_schedule="cosine",
        lr=3e-4, epochs=20, batch_size=32, tokenizer_type="char",
    )
    runs["continuous"] = dx.fit(
        "continuous", CORPUS, run_dir="runs/demo_tiny_continuous",
        t5_model_name="t5-small", embed_dim=512, bottleneck_dim=64,
        dim=192, n_heads=4, num_blocks=4, max_context=128,
        flow_n_steps=32, flow_cfg_scale=2.0,
        lr=1e-3, epochs=15, batch_size=16,
    )

    print("\n── sanity: one generation per paradigm ──")
    for name, rd in runs.items():
        if name == "ar":
            print(f"[{name}] {dx.quick_generate(rd, 'HAMLET:', max_new_tokens=60)[:80]!r}")
        else:
            print(f"[{name}] trained → {rd}")

    if args.push:
        from dantinox.hub import push
        for name, rd in runs.items():
            repo = f"{args.owner}/dantinox-tiny-{name}"
            url = push(rd, repo)
            print(f"pushed {rd} → {url}")
        print("\nREADME one-liner is now live:")
        print(f'  dx.quick_generate("{args.owner}/dantinox-tiny-ar", "HAMLET:")')
    else:
        print("\nTrained only. To publish:")
        print("  python scripts/publish_demo_checkpoints.py --push --owner <hf-username>")


if __name__ == "__main__":
    main()
