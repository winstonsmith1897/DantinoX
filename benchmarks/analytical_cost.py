#!/usr/bin/env python3
"""
benchmarks/analytical_cost.py
=============================

Analytical FLOPs / memory-traffic model for the three DantinoX paradigms.

Why analytical: XLA ``compiled.cost_analysis()`` under-counts FLOPs after
fusion — GEMMs lowered to cuBLAS custom-calls may report zero — so measured
values are unreliable lower bounds (observed ~2–4× low on the production
arch).  Here projection FLOPs are derived **exactly** from the linear-layer
parameters of the *built* model (2 FLOPs per parameter per token, FMA=2),
and the quadratic attention-score terms are added in closed form.

Conventions
-----------
* FMA counts as 2 FLOPs.
* Embedding lookup ≈ 0 FLOPs; tied unembedding costs ``2·D·V`` per token.
* Attention scores+values cost ``4·D·S`` per query token attending to S keys
  per block (q·kᵀ and attn·v, summed over heads: H·hs = D).
* Bytes are a traffic lower bound: weights read once per step + KV-cache
  read/write + logits write + activation in/out.  Good enough to place a
  kernel on the roofline; not a cache-hierarchy simulation.

Self-test
---------
  python benchmarks/analytical_cost.py --selftest
prints analytical vs XLA-measured FLOPs and the implied MFU on the
production arch — implied MFU must stay below hardware peak for the
numbers to be credible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
from flax import nnx

# ── Parameter partitioning (exact, from the built model) ─────────────────────

def _path_str(path: tuple) -> str:
    return "/".join(str(getattr(p, "key", p)) for p in path)


def partition_params(model: nnx.Module) -> dict[str, int]:
    """Split parameter counts by role, walking the real module tree.

    Returns dict with:
      blk_linear : linear params inside transformer blocks (attn + FFN + norms*)
      embed      : token-embedding table (lookup, ~0 FLOPs)
      unembed    : unembedding / lm-head params (tied → same table counted here)
      io_extra   : paradigm-specific in/out projections (ELF bottleneck, ctrl)
      total      : everything

    *Norm scales are counted in blk_linear but contribute O(D) FLOPs — noise.
    """
    state = nnx.state(model, nnx.Param)
    flat = jax.tree_util.tree_flatten_with_path(state)[0]

    counts = dict(blk_linear=0, embed=0, unembed=0, io_extra=0, total=0)
    for path, leaf in flat:
        if not hasattr(leaf, "size"):
            continue
        n = int(leaf.size)
        s = _path_str(path)
        counts["total"] += n
        if "blocks" in s:
            counts["blk_linear"] += n
        elif "embed" in s and "embedder" not in s:   # token embedding table
            counts["embed"] += n
        elif "head" in s or "unembed" in s:
            counts["unembed"] += n
        else:                                        # ELF io projections, ctrl,
            counts["io_extra"] += n                  # final norm, pos tables …
    return counts


@dataclass
class CostModel:
    """Closed-form cost model for one architecture instance."""
    D: int                 # model dim
    L: int                 # number of blocks
    V: int                 # vocab size
    blk_linear: int        # linear params in all blocks (exact, from model)
    io_extra: int          # paradigm io projections (ELF)
    kv_heads: int          # for KV-cache traffic
    head_size: int
    weight_tying: bool = True

    @classmethod
    def from_model(cls, model: nnx.Module, cfg: Any) -> CostModel:
        c = partition_params(model)
        kv_heads = getattr(cfg, "kv_heads", None) or getattr(cfg, "n_heads", 8)
        return cls(
            D=getattr(cfg, "dim", getattr(cfg, "model_dim", 0)),
            L=getattr(cfg, "num_blocks", 0),
            V=getattr(cfg, "vocab_size", 0),
            blk_linear=c["blk_linear"],
            io_extra=c["io_extra"],
            kv_heads=kv_heads,
            head_size=getattr(cfg, "head_size", 64),
        )

    # ── FLOPs (returns GFLOPs) ────────────────────────────────────────────────

    def _proj(self, n_tokens: int, with_unembed: bool, with_io: bool = False) -> float:
        f = 2.0 * self.blk_linear * n_tokens
        if with_unembed:
            f += 2.0 * self.D * self.V * n_tokens
        if with_io:
            f += 2.0 * self.io_extra * n_tokens
        return f

    def _scores_causal(self, B: int, T: int, past: int = 0) -> float:
        # Σ_t 4·D·(past + t) per block
        per_seq = 4.0 * self.D * (past * T + T * (T + 1) / 2.0)
        return per_seq * self.L * B

    def _scores_full(self, B: int, T_q: int, T_kv: int) -> float:
        return 4.0 * self.D * T_q * T_kv * self.L * B

    def ar_prefill(self, B: int, T: int) -> float:
        return (self._proj(B * T, with_unembed=True)
                + self._scores_causal(B, T)) / 1e9

    def ar_decode_step(self, B: int, cache_len: int) -> float:
        return (self._proj(B, with_unembed=True)
                + self._scores_full(B, 1, cache_len + 1)) / 1e9

    def ar_generate(self, B: int, P: int, G: int) -> float:
        """Prefill on P + G cached decode steps."""
        steps = sum(self.ar_decode_step(B, P + i) for i in range(G))
        return self.ar_prefill(B, P) + steps

    def disc_step(self, B: int, T: int, prefix: int) -> float:
        """One bidirectional denoise pass over T tokens, attending to prefix+T."""
        return (self._proj(B * T, with_unembed=True)
                + self._scores_full(B, T, prefix + T)) / 1e9

    def disc_generate(self, B: int, T: int, prefix: int, S: int) -> float:
        prefill = (self._proj(B * prefix, with_unembed=False)
                   + self._scores_full(B, prefix, prefix)) / 1e9 if prefix else 0.0
        return prefill + (S + 1) * self.disc_step(B, T, prefix)

    def elf_step(self, B: int, T: int, n_ctrl: int = 12) -> float:
        """Euler denoise step: no unembedding (only x_pred is consumed)."""
        Tc = T + n_ctrl
        return (self._proj(B * Tc, with_unembed=False)
                + 2.0 * self.io_extra * B * T
                + self._scores_full(B, Tc, Tc)) / 1e9

    def elf_decode(self, B: int, T: int, n_ctrl: int = 12) -> float:
        return self.elf_step(B, T, n_ctrl) + 2.0 * self.D * self.V * B * T / 1e9

    def flow_generate(self, B: int, T: int, S: int) -> float:
        return S * self.elf_step(B, T) + self.elf_decode(B, T)

    # ── Bytes (returns GB, traffic lower bound) ───────────────────────────────

    def _weight_bytes(self, with_unembed: bool, bpp: int) -> float:
        n = self.blk_linear + self.io_extra
        if with_unembed:
            n += self.D * self.V
        return float(n) * bpp

    def ar_decode_bytes(self, B: int, cache_len: int, bpp: int = 2) -> float:
        kv = 2.0 * cache_len * self.kv_heads * self.head_size * self.L * B * bpp
        logits = float(B * self.V) * bpp
        act = 4.0 * B * self.D * self.L * bpp
        return (self._weight_bytes(True, bpp) + kv + logits + act) / 1e9

    def disc_step_bytes(self, B: int, T: int, prefix: int, bpp: int = 2) -> float:
        kv = 2.0 * (prefix + T) * self.kv_heads * self.head_size * self.L * B * bpp
        logits = float(B * T * self.V) * bpp
        act = 4.0 * B * T * self.D * self.L * bpp
        return (self._weight_bytes(True, bpp) + kv + logits + act) / 1e9

    def elf_step_bytes(self, B: int, T: int, bpp: int = 2) -> float:
        act = 4.0 * B * T * self.D * self.L * bpp
        return (self._weight_bytes(False, bpp) + act) / 1e9


# ── Self-test against the production architecture ────────────────────────────

def _selftest() -> None:
    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # CPU is fine for counting
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from dantinox.core.config import FlowMatchingConfig, ModelConfig
    from dantinox.core.flow import FlowMatchingTransformer
    from dantinox.core.model import Transformer

    cfg = ModelConfig(dim=512, n_heads=8, head_size=64, num_blocks=12,
                      vocab_size=32128, max_context=1152, causal=False,
                      mask_token_id=32099)
    model = Transformer(cfg, rngs=nnx.Rngs(0))
    cm = CostModel.from_model(model, cfg)
    parts = partition_params(model)
    print("Param partition (512d12b):", {k: f"{v/1e6:.2f}M" for k, v in parts.items()})

    B, T, P = 64, 1024, 64
    print(f"\nDiscrete step  B={B} T={T} P={P}")
    print(f"  analytical : {cm.disc_step(B, T, P):,.0f} GFLOPs")
    print("  (XLA measured on this config was ~2,099 GFLOPs — expected to be lower)")
    print(f"  bytes      : {cm.disc_step_bytes(B, T, P):,.2f} GB")
    ai = cm.disc_step(B, T, P) / cm.disc_step_bytes(B, T, P)
    print(f"  intensity  : {ai:,.0f} FLOP/B")
    # Implied MFU from the measured 267.6 ms step on A100 bf16:
    tflops = cm.disc_step(B, T, P) / 267.55
    print(f"  implied    : {tflops:.1f} TFLOP/s  → {100*tflops/312:.1f}% of bf16 peak")

    print(f"\nAR decode step B={B} cache={P+T}")
    g = cm.ar_decode_step(B, P + T)
    by = cm.ar_decode_bytes(B, P + T)
    print(f"  analytical : {g:.3f} GFLOPs, {by:.4f} GB → intensity {g/by:.1f} FLOP/B")

    ecfg = FlowMatchingConfig(embed_dim=512, bottleneck_dim=128, model_dim=512,
                     n_heads=8, head_size=64, num_blocks=12,
                     vocab_size=32128, max_seq_len=T,
                     gradient_checkpointing=False)
    emodel = FlowMatchingTransformer(ecfg, rngs=nnx.Rngs(0))
    ecm = CostModel.from_model(emodel, ecfg)
    print(f"\nELF step B={B} T={T}: {ecm.elf_step(B, T):,.0f} GFLOPs "
          f"(no unembed) vs decode {ecm.elf_decode(B, T):,.0f} GFLOPs")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    if args.selftest:
        _selftest()
