from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

import yaml


@dataclass
class Config:
    # ── Model Architecture ───────────────────────────────────────────────────
    dim: int = 512
    n_heads: int = 16
    head_size: int = 32
    num_blocks: int = 20
    vocab_size: int = 200
    max_context: int = 512
    kv_heads: int = 4
    weight_tying: bool = True
    activation: str = "gelu"
    gradient_checkpointing: bool = True
    dropout_rate: float = 0.15
    use_swiglu: bool = True

    # ── MoE ─────────────────────────────────────────────────────────────────
    use_moe: bool = False
    n_experts: int = 4
    top_k_mlp: int = 2
    expansion: int = 4
    alpha_balance: float = 0.1

    # ── Attention & Positional ───────────────────────────────────────────────
    use_rotary_pos: bool = True
    trainable_pos: bool = False
    absolute_pos: bool = False
    sliding_window: bool = False
    context_window: int = 4
    no_sink: bool = True
    use_flash_attention: bool = False

    # ── Multi-Head Latent Attention (MLA) ────────────────────────────────────
    mla: bool = False
    inference: bool = False
    down_dim_q: int = 256
    down_dim_kv: int = 256
    rope_dim: int = 32

    # ── Tokenizer ───────────────────────────────────────────────────────────
    tokenizer_type: str = "char"
    tokenizer_path: str | None = None

    # ── Dataset ─────────────────────────────────────────────────────────────
    dataset_source: str = "local"
    dataset_name: str = ""
    streaming: bool = False

    # ── Training & Optimisation ──────────────────────────────────────────────
    lr: float = 0.005
    batch_size: int = 128
    grad_accum: int = 16
    seed: int = 42
    optimizer: str = "adamw"
    epochs: int = 1000
    warmup_steps: int = 420
    grad_clip: float = 1.0
    patience: int = 0
    use_bf16: bool = False

    # ── Normalisation ────────────────────────────────────────────────────────
    norm_type: str = "layernorm"     # "layernorm" | "rmsnorm"

    # ── RoPE scaling ─────────────────────────────────────────────────────────
    rope_scale_factor: float = 1.0   # >1 compresses frequencies for long-ctx (NTK-aware)

    # ── LR schedule ──────────────────────────────────────────────────────────
    lr_schedule: str = "cosine"      # "cosine" | "linear" | "constant" | "wsd"

    # ── Logging & Metrics ───────────────────────────────────────────────────
    eval_iters: int = 20
    log_file: str = "training_log.csv"
    summary_file: str = "model_summary.json"

    def __post_init__(self) -> None:
        if self.kv_heads is None:
            self.kv_heads = self.n_heads // 4

        if self.dim != self.n_heads * self.head_size:
            raise ValueError(
                f"dim ({self.dim}) must equal n_heads * head_size "
                f"({self.n_heads} * {self.head_size} = {self.n_heads * self.head_size})"
            )
        if self.n_heads % self.kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by kv_heads ({self.kv_heads})"
            )
        if self.mla and self.use_rotary_pos and self.rope_dim > self.head_size:
            raise ValueError(
                f"rope_dim ({self.rope_dim}) must be <= head_size ({self.head_size}) "
                "when using MLA with rotary positional encoding"
            )
        if self.norm_type not in ("layernorm", "rmsnorm"):
            raise ValueError(
                f"norm_type must be 'layernorm' or 'rmsnorm', got {self.norm_type!r}"
            )
        if self.lr_schedule not in ("cosine", "linear", "constant", "wsd"):
            raise ValueError(
                f"lr_schedule must be 'cosine', 'linear', 'constant', or 'wsd', "
                f"got {self.lr_schedule!r}"
            )
        if self.rope_scale_factor <= 0:
            raise ValueError(
                f"rope_scale_factor must be > 0, got {self.rope_scale_factor}"
            )

    def __repr__(self) -> str:
        attn = "MLA" if self.mla else ("GQA" if self.kv_heads < self.n_heads else "MHA")
        moe = "+MoE" if self.use_moe else ""
        return (
            f"Config(dim={self.dim}, heads={self.n_heads}, blocks={self.num_blocks}, "
            f"ctx={self.max_context}, attn={attn}{moe})"
        )

    # ── Serialisation ────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict of all config fields."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Config:
        """Construct a Config from a plain dict, ignoring unknown keys."""
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})

    @classmethod
    def from_yaml(cls, path: str) -> Config:
        """Load a Config from a YAML file (flat or sectioned)."""
        with open(path) as f:
            raw = yaml.safe_load(f)

        flat: dict[str, Any] = {}
        for v in raw.values():
            if isinstance(v, dict):
                flat.update(v)
        if not flat:
            flat = raw

        return cls.from_dict(flat)

    def save_yaml(self, path: str) -> None:
        """Write the config to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f)
