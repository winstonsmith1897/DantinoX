from __future__ import annotations

from dataclasses import dataclass

from core.config import ModelConfig


@dataclass(frozen=True)
class FLOPsBreakdown:
    """Per-component FLOPs for a single forward pass."""

    attention: int
    ffn: int
    embedding: int
    total: int

    def __str__(self) -> str:
        def _fmt(n: int) -> str:
            if n >= 1e12:
                return f"{n / 1e12:.2f} TFLOPs"
            if n >= 1e9:
                return f"{n / 1e9:.2f} GFLOPs"
            if n >= 1e6:
                return f"{n / 1e6:.2f} MFLOPs"
            return f"{n} FLOPs"

        return (
            f"FLOPs breakdown:\n"
            f"  attention : {_fmt(self.attention)}\n"
            f"  ffn       : {_fmt(self.ffn)}\n"
            f"  embedding : {_fmt(self.embedding)}\n"
            f"  total     : {_fmt(self.total)}"
        )


def count_flops(
    config: ModelConfig,
    seq_len: int,
    batch_size: int = 1,
) -> FLOPsBreakdown:
    """Estimate FLOPs for one forward pass using standard approximations.

    Attention (per layer):
        QKV projection  : 3 × 2BTD²  (two ops per multiply-add)
        output proj     :     2BTD²
        attention score : 2BT²D
    FFN (per layer, with optional SwiGLU gate doubling):
        up   : 2BT·D·(E·D) × swiglu_factor
        down : 2BT·(E·D)·D
    Logit projection (unembed):
        2BT·V·D
    """
    B = batch_size
    T = seq_len
    D = config.dim
    L = config.num_blocks
    E = config.expansion
    swiglu = 2 if config.use_swiglu else 1

    attn_proj  = 4 * 2 * B * T * D * D  # QKV (×3) + O (×1) folded as ×4
    attn_score = 2 * B * T * T * D
    attn_flops = (attn_proj + attn_score) * L

    ffn_hidden = D * E
    ffn_up     = 2 * B * T * D * ffn_hidden * swiglu
    ffn_down   = 2 * B * T * ffn_hidden * D
    ffn_flops  = (ffn_up + ffn_down) * L

    embed_flops = 2 * B * T * config.vocab_size * D

    total = attn_flops + ffn_flops + embed_flops
    return FLOPsBreakdown(
        attention=attn_flops,
        ffn=ffn_flops,
        embedding=embed_flops,
        total=total,
    )
