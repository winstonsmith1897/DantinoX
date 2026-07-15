"""All serialisable configuration for DantinoX models and training runs.

This module owns configuration and nothing else — it deliberately imports no
JAX so that reading/writing configs stays cheap.

Public classes
--------------
``ModelConfig``
    Architecture of the unified ``Transformer`` (model.py).  The ``paradigm``
    field ("ar" | "discrete" | "continuous" | "embedder") selects the
    training/inference strategy; everything else describes the backbone.
``TrainingConfig``
    Optimisation, dataset, tokenizer, and multi-GPU settings — *how* to train,
    never *what* the model is.
``FlowMatchingConfig``
    Architecture of the continuous flow-matching model (flow.py).
``Config`` (deprecated)
    Monolithic legacy config kept for old checkpoints, YAML files, and the
    CLI.  Bridged losslessly to the split configs via ``Config.from_parts`` /
    ``to_model_config`` / ``to_flow_config``; the bridge is driven by the
    module-level rename tables so a new field can never be silently dropped.

Old serialized key names (e.g. ``elf_n_steps``) are remapped on load via
``_LEGACY_KEY_RENAMES`` so historical ``config.yaml`` files keep working.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from typing import Any

import yaml

# ── ANSI color helpers (used by ModelConfig.__repr__) ────────────────────────
# Self-contained; no import from _ui.py to keep config.py import-cost low.

def _ansi_ok() -> bool:
    return not os.environ.get("NO_COLOR") and os.environ.get("TERM") != "dumb"

def _C(n: int, s: str) -> str:     # 256-color wrap
    return f"\033[38;5;{n}m{s}\033[0m" if _ansi_ok() else s

def _Cmgn(s: str) -> str: return _C(201, s)  # magenta  — paradigm value
def _Cprp(s: str) -> str: return _C(141, s)  # lavender — section headers
def _Cblu(s: str) -> str: return _C(69,  s)  # cornflower — labels (attn:, pos:…)
def _Cazr(s: str) -> str: return _C(39,  s)  # azure    — mode / secondary
def _Ccyn(s: str) -> str: return _C(51,  s)  # cyan     — numbers
def _Cwht(s: str) -> str: return (f"\033[97m{s}\033[0m" if _ansi_ok() else s)
def _Cdim(s: str) -> str: return (f"\033[2m{s}\033[0m"  if _ansi_ok() else s)


# ── Legacy serialized-key renames ─────────────────────────────────────────────
# Field names that changed across library versions.  Applied in every
# ``from_dict`` so config.yaml files written by older versions keep loading.
_LEGACY_KEY_RENAMES: dict[str, str] = {
    "elf_n_steps": "flow_n_steps",
    "elf_cfg_scale": "flow_cfg_scale",
}


def _remap_legacy_keys(d: dict[str, Any]) -> dict[str, Any]:
    return {_LEGACY_KEY_RENAMES.get(k, k): v for k, v in d.items()}


def _install_legacy_ctor_aliases(cls: type) -> None:
    """Let *cls* accept legacy constructor kwargs, warning on each use.

    The read side of renames has always stayed compatible via properties
    (``cfg.elf_n_steps`` still answers), but constructor calls with old
    names crashed — an asymmetry that broke several user notebooks.  This
    wraps ``cls.__init__`` to remap:

    * every entry of ``_LEGACY_KEY_RENAMES`` (e.g. ``elf_n_steps`` →
      ``flow_n_steps``), and
    * ``use_moe=<bool>`` → ``ffn='moe'|'mlp'`` (a value transform, not a
      rename, so it can't live in the table).

    Each remap emits a :class:`DeprecationWarning`; passing both the old
    and the new name raises ``TypeError``.
    """
    import functools
    import warnings
    from dataclasses import fields as _flds

    orig_init = cls.__init__  # type: ignore[misc]
    cls_fields = {f.name for f in _flds(cls)}  # type: ignore[arg-type]

    @functools.wraps(orig_init)
    def _init(self: Any, *args: Any, **kwargs: Any) -> None:
        for old, new in _LEGACY_KEY_RENAMES.items():
            if old in kwargs and new in cls_fields:
                if new in kwargs:
                    raise TypeError(
                        f"{cls.__name__} got both legacy {old!r} and {new!r}"
                    )
                warnings.warn(
                    f"{cls.__name__}({old}=…) is deprecated — use {new}=…",
                    DeprecationWarning, stacklevel=2,
                )
                kwargs[new] = kwargs.pop(old)
        if "use_moe" in kwargs and "use_moe" not in cls_fields and "ffn" in cls_fields:
            if "ffn" in kwargs:
                raise TypeError(f"{cls.__name__} got both legacy 'use_moe' and 'ffn'")
            warnings.warn(
                f"{cls.__name__}(use_moe=…) is deprecated — use ffn='moe' | 'mlp'",
                DeprecationWarning, stacklevel=2,
            )
            kwargs["ffn"] = "moe" if kwargs.pop("use_moe") else "mlp"
        orig_init(self, *args, **kwargs)

    cls.__init__ = _init  # type: ignore[method-assign, misc]


# ── ModelConfig ────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """Model architecture specification — nothing but *what the model is*.

    Pass this to ``Transformer(config, rngs=nnx.Rngs(42))`` or use the
    class-based builder ``Transformer.build(...)``.

    Fields are grouped by concern; for each clean name the old name is
    available as a read-only property so existing component code continues
    to work unchanged.
    """

    # ── Core dimensions ───────────────────────────────────────────────────────
    dim: int = 512
    n_heads: int = 16
    head_size: int | None = None   # auto-computed as dim // n_heads when None
    num_blocks: int = 12
    vocab_size: int | None = None  # auto-set from tokenizer when using Trainer.fit()
    max_context: int = 512

    # ── Architecture choices ──────────────────────────────────────────────────
    attention: str = "mha"        # "mha" | "gqa" | "mla"
    ffn: str = "mlp"              # "mlp" | "moe"
    norm: str = "rmsnorm"         # "rmsnorm" | "layernorm"
    pos_encoding: str = "rotary"  # "rotary" | "absolute" | "learned" | "none"
    causal: bool = True           # True = autoregressive; False = bidirectional / diffusion

    # ── Shared regularisation ─────────────────────────────────────────────────
    dropout: float = 0.0
    weight_tying: bool = True
    tp_size: int = 1

    # ── Attention ─────────────────────────────────────────────────────────────
    kv_heads: int | None = None   # GQA: number of KV heads; None → same as n_heads
    use_flash: bool = False
    rope_scale: float = 1.0
    sliding_window: bool = False
    context_window: int = 4
    no_sink: bool = False
    differential: bool = False    # differential attention (Ye et al., 2025)
    lambda_init: float = 0.8      # λ_init for differential attention
    use_linear_attention: bool = False  # linear attention (elu+1 kernel) for bidirectional (is_causal=False) calls

    # ── MLA-specific ──────────────────────────────────────────────────────────
    down_dim_q: int = 256
    down_dim_kv: int = 256
    rope_dim: int = 32
    inference_mode: bool = False   # MLA absorbed KV projection at inference

    # ── FFN ───────────────────────────────────────────────────────────────────
    expansion: int = 4
    use_swiglu: bool = True
    activation: str = "gelu"

    # ── MoE ───────────────────────────────────────────────────────────────────
    n_experts: int = 4
    top_k: int = 2
    moe_balance_coeff: float = 0.1
    moe_latent: bool = False
    moe_latent_dim: int = 64

    # ── Paradigm selector ────────────────────────────────────────────────────
    # Explicit key that tells Paradigm() which training/inference strategy to
    # use.  When set, `causal` is auto-configured so you don't need to repeat
    # it.  When None, Paradigm() falls back to inspecting `causal`/`embed_dim`.
    paradigm: str | None = None   # "ar" | "discrete" | "continuous" | "embedder"

    # ── Diffusion (discrete) ─────────────────────────────────────────────────
    mask_token_id: int = -1  # sentinel: auto-synced from the tokenizer by Trainer.fit()
    noise_schedule: str = "linear"  # "linear" | "cosine" | "sqrt"

    # ── Continuous flow-matching (ELF recipe) ────────────────────────────────
    # Set embed_dim > 0 to enable flow-matching mode; must match the frozen
    # T5 encoder.
    embed_dim: int = 0              # 0 = standard transformer; 768 for t5-base
    bottleneck_dim: int = 128       # projection between embed space and model_dim
    flow_n_steps: int = 32           # ODE denoising steps at inference
    flow_cfg_scale: float = 1.5      # classifier-free guidance scale
    sde_gamma: float = 0.0          # SDE noise re-injection (0 = pure ODE)
    t5_model_name: str = "t5-base"  # frozen T5 encoder variant

    # ── Embedder (contrastive) ───────────────────────────────────────────────
    embed_pooling: str = "auto"     # "auto" | "mean" | "last" | "cls"
    embed_temperature: float = 0.05  # InfoNCE temperature

    # ── LoRA ──────────────────────────────────────────────────────────────────
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_targets: str = "attention"  # "attention" | "ffn" | "all"

    # ── Validation ────────────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        _valid_paradigms = ("ar", "discrete", "continuous", "embedder")
        if self.paradigm is not None and self.paradigm not in _valid_paradigms:
            raise ValueError(
                f"paradigm must be one of {_valid_paradigms} or None; "
                f"got {self.paradigm!r}"
            )
        # Auto-configure causal from the paradigm key so users don't repeat it
        if self.paradigm in ("ar", "embedder"):
            self.causal = True
        elif self.paradigm in ("discrete", "continuous"):
            self.causal = False

        # For the continuous flow-matching paradigm, vocab_size is fixed by the T5 tokenizer.
        # All T5 variants share the same 32100-token SentencePiece vocabulary.
        if self.paradigm == "continuous" and self.vocab_size is None:
            self.vocab_size = 32100

        if self.head_size is None:
            if self.dim % self.n_heads != 0:
                raise ValueError(
                    f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads}) "
                    "when head_size is not specified"
                )
            self.head_size = self.dim // self.n_heads
        if self.kv_heads is None:
            self.kv_heads = self.n_heads
        if self.dim != self.n_heads * self.head_size:
            raise ValueError(
                f"dim ({self.dim}) must equal n_heads × head_size "
                f"({self.n_heads} × {self.head_size} = {self.n_heads * self.head_size})"
            )
        if self.n_heads % self.kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by kv_heads ({self.kv_heads})"
            )
        if self.attention not in ("mha", "gqa", "mla"):
            raise ValueError(f"attention must be 'mha', 'gqa', or 'mla'; got {self.attention!r}")
        if self.attention == "mla":
            # Decoupled RoPE is part of the MLA architecture: the attention
            # module always builds and applies the rope projections.
            if self.pos_encoding != "rotary":
                raise ValueError(
                    f"attention='mla' requires pos_encoding='rotary'; got {self.pos_encoding!r}"
                )
            if self.rope_dim > self.head_size:
                raise ValueError(
                    f"rope_dim ({self.rope_dim}) must be <= head_size ({self.head_size}) "
                    "when using MLA with rotary positional encoding"
                )
        if self.ffn not in ("mlp", "moe"):
            raise ValueError(f"ffn must be 'mlp' or 'moe'; got {self.ffn!r}")
        if self.norm not in ("rmsnorm", "layernorm"):
            raise ValueError(f"norm must be 'rmsnorm' or 'layernorm'; got {self.norm!r}")
        if self.pos_encoding not in ("rotary", "absolute", "learned", "none"):
            raise ValueError(
                f"pos_encoding must be 'rotary', 'absolute', 'learned', or 'none'; "
                f"got {self.pos_encoding!r}"
            )
        if self.noise_schedule not in ("linear", "cosine", "sqrt"):
            raise ValueError(
                f"noise_schedule must be 'linear', 'cosine', or 'sqrt'; "
                f"got {self.noise_schedule!r}"
            )
        if self.tp_size < 1:
            raise ValueError(f"tp_size must be >= 1; got {self.tp_size}")
        if self.lora_targets not in ("attention", "ffn", "all"):
            raise ValueError("lora_targets must be 'attention', 'ffn', or 'all'")
        if self.lora_rank < 1:
            raise ValueError("lora_rank must be >= 1")
        if self.embed_pooling not in ("auto", "mean", "last", "cls"):
            raise ValueError(
                f"embed_pooling must be 'auto', 'mean', 'last', or 'cls'; "
                f"got {self.embed_pooling!r}"
            )
        if self.moe_latent and not (0 < self.moe_latent_dim < self.dim):
            raise ValueError(
                f"moe_latent_dim ({self.moe_latent_dim}) must be in (0, dim={self.dim})"
            )

    # ── Backward-compat property shims ────────────────────────────────────────
    # All existing component code (attention.py, mlp.py, …) uses old field
    # names.  These read-only properties expose the old names so components
    # work with ModelConfig without any changes.

    @property
    def dropout_rate(self) -> float:
        return self.dropout

    @property
    def use_flash_attention(self) -> bool:
        return self.use_flash

    @property
    def rope_scale_factor(self) -> float:
        return self.rope_scale

    @property
    def inference(self) -> bool:
        return self.inference_mode

    @property
    def top_k_mlp(self) -> int:
        return self.top_k

    @property
    def alpha_balance(self) -> float:
        return self.moe_balance_coeff

    @property
    def use_moe(self) -> bool:
        return self.ffn == "moe"

    @property
    def use_rotary_pos(self) -> bool:
        return self.pos_encoding == "rotary"

    @property
    def trainable_pos(self) -> bool:
        return self.pos_encoding == "learned"

    @property
    def absolute_pos(self) -> bool:
        return self.pos_encoding == "absolute"

    @property
    def norm_type(self) -> str:
        return self.norm

    @property
    def attention_type(self) -> str:
        return self.attention

    @property
    def mla(self) -> bool:
        return self.attention == "mla"

    @property
    def model_type(self) -> str:
        return "autoregressive" if self.causal else "diffusion"

    # Deprecated ELF-branded names (paradigm-level names are flow_*):
    @property
    def elf_n_steps(self) -> int:
        return self.flow_n_steps

    @property
    def elf_cfg_scale(self) -> float:
        return self.flow_cfg_scale

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelConfig:
        d = _remap_legacy_keys(d)
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})

    @classmethod
    def from_yaml(cls, path: str) -> ModelConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        flat: dict[str, Any] = {}
        for v in raw.values():
            if isinstance(v, dict):
                flat.update(v)
        return cls.from_dict(flat if flat else raw)

    def save_yaml(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f)

    def replace(self, **kwargs: Any) -> ModelConfig:
        """Return a new ModelConfig with selected fields overridden."""
        from dataclasses import replace as _replace
        return _replace(self, **kwargs)

    def __repr__(self) -> str:
        _W = 46   # section rule width

        def _sec(title: str) -> str:
            prefix = f"─── {title} "
            return "  " + _Cprp(prefix + "─" * max(2, _W - len(prefix)))

        # ── Attention ──
        if self.attention == "gqa" and self.kv_heads is not None and self.kv_heads != self.n_heads:
            attn_s = f"GQA  ({self.n_heads}q / {self.kv_heads}kv)"
        elif self.attention == "mla":
            attn_s = f"MLA  (dq={self.down_dim_q}, dkv={self.down_dim_kv}, rope_dim={self.rope_dim})"
        else:
            attn_s = "MHA"

        # ── Positional encoding ──
        _pos = {"rotary": "RoPE", "absolute": "sinusoidal", "learned": "learned", "none": "none"}
        pos_s = _pos.get(self.pos_encoding, self.pos_encoding)
        if self.pos_encoding == "rotary" and self.rope_scale != 1.0:
            pos_s += f"(×{self.rope_scale})"

        # ── Norm / FFN ──
        norm_s = {"rmsnorm": "RMSNorm", "layernorm": "LayerNorm"}.get(self.norm, self.norm)
        act_s  = "SwiGLU" if self.use_swiglu else self.activation.upper()
        if self.use_moe:
            lat_s = f"  latent={self.moe_latent_dim}" if self.moe_latent else ""
            ffn_s = f"MoE  ({self.n_experts} experts, top-{self.top_k}, α={self.moe_balance_coeff}{lat_s})"
        else:
            ffn_s = f"MLP  (×{self.expansion}, {act_s})"

        hs    = self.head_size or (self.dim // self.n_heads)
        vocab = f"{self.vocab_size:,}" if self.vocab_size is not None else "?"
        mode  = "causal" if self.causal else "bidirectional"

        L: list[str] = ["ModelConfig("]

        # paradigm / mode header
        if self.paradigm is not None:
            L.append(
                f"  {_Cdim('paradigm')} = {_Cmgn(repr(self.paradigm))}"
                f"  ·  {_Cdim('mode')} = {_Cazr(mode)}"
            )
        else:
            L.append(f"  {_Cdim('mode')} = {_Cazr(mode)}")

        # ── core ──
        L.append(_sec("core"))
        L.append(
            f"  dim={_Ccyn(str(self.dim))}"
            f"  ·  {_Ccyn(str(self.n_heads))} heads × {_Ccyn(str(hs))}"
            f"  ·  {_Ccyn(str(self.num_blocks))} blocks"
        )
        L.append(
            f"  vocab={_Ccyn(vocab)}"
            f"  ·  ctx={_Ccyn(str(self.max_context))}"
        )

        # ── architecture ──
        L.append(_sec("architecture"))
        L.append(f"  {_Cblu('attn:')}  {_Cwht(attn_s)}")
        L.append(
            f"  {_Cblu('pos:')}   {_Cwht(pos_s)}"
            f"  ·  {_Cblu('norm:')} {_Cwht(norm_s)}"
        )
        L.append(f"  {_Cblu('ffn:')}   {_Cwht(ffn_s)}")

        # ── regularization ──
        L.append(_sec("regularization"))
        L.append(
            f"  dropout={_Ccyn(str(self.dropout))}"
            f"  ·  weight_tying={_Ccyn(str(self.weight_tying))}"
        )
        extras: list[str] = []
        if self.use_flash:
            extras.append("flash_attn")
        if self.sliding_window:
            extras.append(f"sliding_window(k={self.context_window})")
        if self.tp_size > 1:
            extras.append(f"tp={self.tp_size}")
        if extras:
            L.append(f"  {_Cdim(' · '.join(extras))}")

        # ── paradigm-specific sections ──
        if self.paradigm == "discrete":
            L.append(_sec("discrete diffusion"))
            L.append(
                f"  schedule={_Cwht(self.noise_schedule)}"
                f"  ·  mask_id={_Ccyn(str(self.mask_token_id))}"
            )

        elif self.paradigm == "continuous":
            L.append(_sec("continuous flow-matching"))
            L.append(
                f"  embed_dim={_Ccyn(str(self.embed_dim))}"
                f"  ·  bottleneck={_Ccyn(str(self.bottleneck_dim))}"
                f"  ·  T5={_Cwht(self.t5_model_name)}"
            )
            L.append(
                f"  n_steps={_Ccyn(str(self.flow_n_steps))}"
                f"  ·  cfg_scale={_Ccyn(str(self.flow_cfg_scale))}"
                f"  ·  sde_γ={_Ccyn(str(self.sde_gamma))}"
            )

        elif self.paradigm == "embedder":
            L.append(_sec("embedder"))
            L.append(
                f"  pooling={_Cwht(self.embed_pooling)}"
                f"  ·  temperature={_Ccyn(str(self.embed_temperature))}"
            )

        # ── LoRA ──
        if self.use_lora:
            L.append(_sec("LoRA"))
            L.append(
                f"  rank={_Ccyn(str(self.lora_rank))}"
                f"  ·  alpha={_Ccyn(str(self.lora_alpha))}"
                f"  ·  targets={_Cwht(self.lora_targets)}"
                f"  ·  dropout={_Ccyn(str(self.lora_dropout))}"
            )

        L.append(")")
        return "\n".join(L)


_install_legacy_ctor_aliases(ModelConfig)


# ── TrainingConfig ─────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    """Training, dataset, and infrastructure configuration.

    Everything that controls *how the model is trained*, not *what it is*.
    """

    # ── Optimisation ──────────────────────────────────────────────────────────
    lr: float = 3e-4
    batch_size: int = 32
    grad_accum: int = 1
    epochs: float = 100   # int → full passes; float < 1 → fraction of one pass (e.g. 0.1 = 10%)
    warmup_steps: int = 400
    lr_schedule: str = "cosine"    # "cosine" | "linear" | "constant" | "wsd"
    optimizer: str = "adamw"       # "adamw" | "adafactor" | "lion" | "adam" | "muon"
    grad_clip: float = 1.0
    seed: int = 42
    use_bf16: bool = False
    gradient_checkpointing: bool = False  # recompute block activations on backward (saves VRAM)
    patience: int = 0              # early stopping: epochs without val improvement (0 = off)
    eval_iters: int = 20
    val_frac: float = 0.1          # fraction of tokens held out for validation
    val_every: int = 1             # run validation every N epochs (0 = only at the end)

    # ── Multi-GPU ─────────────────────────────────────────────────────────────
    n_devices: int = 0             # 0 = all available; with tp_size>1 this is the DP replica count
    tp_size: int = 1               # TP shards per replica; total GPUs = n_devices * tp_size

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset_source: str = "local"  # "local" | "huggingface"
    dataset_name: str = ""
    dataset_config: str = ""
    dataset_text_field: str = "text"
    dataset_split: str = "train"
    max_train_tokens: int = 10_000_000
    streaming: bool = False

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer_type: str = "char"   # "char" | "bpe" | "t5"
    tokenizer_path: str | None = None

    # ── Warm start ────────────────────────────────────────────────────────────
    init_from: str = ""            # run dir whose best checkpoint initialises the model

    # ── Logging ───────────────────────────────────────────────────────────────
    log_file: str = "training_log.csv"

    def __post_init__(self) -> None:
        if self.lr_schedule not in ("cosine", "linear", "constant", "wsd"):
            raise ValueError("lr_schedule must be 'cosine', 'linear', 'constant', or 'wsd'")
        if self.n_devices < 0:
            raise ValueError("n_devices must be >= 0")
        if self.tp_size < 1:
            raise ValueError(f"tp_size must be >= 1; got {self.tp_size}")
        if not 0.0 <= self.val_frac < 1.0:
            raise ValueError(f"val_frac must be in [0, 1); got {self.val_frac}")
        if self.grad_accum < 1:
            raise ValueError(f"grad_accum must be >= 1; got {self.grad_accum}")

    def replace(self, **kwargs: Any) -> TrainingConfig:
        """Return a new TrainingConfig with selected fields overridden."""
        from dataclasses import replace as _replace
        return _replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainingConfig:
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})

    @classmethod
    def from_yaml(cls, path: str) -> TrainingConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        flat: dict[str, Any] = {}
        for v in raw.values():
            if isinstance(v, dict):
                flat.update(v)
        return cls.from_dict(flat if flat else raw)

    def save_yaml(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f)


# ── Config (backward-compat monolithic config) ────────────────────────────────
# Trainer, CLI, and existing YAML configs use this.
# New experiments should use ModelConfig + TrainingConfig separately.

# Field-name bridge between the split configs and the legacy Config.
# ``Config.from_parts`` / ``Config.to_model_config`` / ``Config.to_flow_config``
# copy every dataclass field automatically: same-named fields directly, the
# renames below via these tables.  Only fields listed in the *_SPECIAL sets are
# bridged by dedicated logic; adding a new field to ModelConfig/FlowMatchingConfig with
# a matching (or mapped) name in Config is therefore sufficient — it can no
# longer be silently dropped by the conversion.

# ModelConfig field → legacy Config field (renames only)
_MODEL_TO_LEGACY_RENAMES: dict[str, str] = {
    "dropout": "dropout_rate",
    "use_flash": "use_flash_attention",
    "rope_scale": "rope_scale_factor",
    "inference_mode": "inference",
    "top_k": "top_k_mlp",
    "moe_balance_coeff": "alpha_balance",
    "norm": "norm_type",
    "attention": "attention_type",
}
# Bridged by dedicated logic (paradigm/causal ↔ model_type, ffn ↔ use_moe,
# pos_encoding ↔ three booleans, lora_targets "ffn" ↔ "mlp"):
_MODEL_SPECIAL_FIELDS = frozenset({"paradigm", "causal", "ffn", "pos_encoding", "lora_targets"})
# New-API-only fields with no legacy counterpart (embedder paradigm):
_MODEL_ONLY_FIELDS = frozenset({"embed_pooling", "embed_temperature"})

# FlowMatchingConfig field → legacy Config field (renames only)
_FLOW_TO_LEGACY_RENAMES: dict[str, str] = {
    "model_dim": "dim",
    "max_seq_len": "max_context",
    "attention": "attention_type",
    "norm": "norm_type",
    "dropout": "dropout_rate",
    "top_k": "top_k_mlp",
    "moe_balance_coeff": "alpha_balance",
}
_FLOW_SPECIAL_FIELDS = frozenset({"ffn", "pos_encoding", "kv_heads"})

_MODEL_TYPE_FROM_PARADIGM: dict[str, str] = {
    "ar": "autoregressive",
    "discrete": "diffusion",
    "continuous": "elf",
    "embedder": "autoregressive",
}
_PARADIGM_FROM_MODEL_TYPE: dict[str, str] = {
    "autoregressive": "ar",
    "diffusion": "discrete",
    "elf": "continuous",
}


@dataclass
class Config:
    """Monolithic config kept for backward compatibility.

    Combines all ModelConfig and TrainingConfig fields in a single flat
    dataclass so existing code (trainer, CLI, YAML files) works unchanged.

    New code should use ``ModelConfig`` for architecture and
    ``TrainingConfig`` for training hyper-parameters.
    """

    # ── Architecture ──────────────────────────────────────────────────────────
    # Defaults are kept identical to ModelConfig so the legacy and split-config
    # APIs build the same model when a field is not specified.
    dim: int = 512
    n_heads: int = 16
    head_size: int = 32
    num_blocks: int = 12
    vocab_size: int | None = None
    max_context: int = 512
    kv_heads: int | None = None
    weight_tying: bool = True
    activation: str = "gelu"
    gradient_checkpointing: bool = False
    dropout_rate: float = 0.0
    use_swiglu: bool = True

    # ── Model type & attention variant ────────────────────────────────────────
    model_type: str = "autoregressive"   # "autoregressive" | "diffusion"
    attention_type: str = "auto"         # "mha" | "gqa" | "mla" | "auto" (derived)

    # ── Diffusion (discrete — LLaDA / MDLM) ─────────────────────────────────
    diffusion_steps: int = 1000
    noise_schedule: str = "linear"
    mask_token_id: int = 4
    num_sampling_steps: int = 50
    time_emb_dim: int = 256

    # ── Continuous flow-matching (ELF recipe) ─────────────────────────────────
    # Used when model_type = "elf".  dim/n_heads/head_size/num_blocks/
    # max_context/vocab_size/dropout_rate are shared with the existing fields.
    embed_dim:            int   = 512    # token embedding & flow-space dimension
    bottleneck_dim:       int   = 128    # bottleneck dim (embed_dim → bottleneck → dim)
    num_time_tokens:      int   = 4      # control tokens for timestep t
    num_cfg_tokens:       int   = 4      # control tokens for CFG scale w
    num_mode_tokens:      int   = 4      # control tokens for denoiser / decode mode
    denoiser_pmean:       float = -1.5   # logit-normal t ~ sigmoid(N(pmean, pstd²))
    denoiser_pstd:        float =  0.8
    denoiser_noise_scale: float =  2.0   # ε scale for denoiser corruption
    decoder_pmean:        float =  0.8   # logit-normal p per token
    decoder_pstd:         float =  0.8
    decoder_noise_scale:  float =  5.0   # ε scale for decoder corruption
    denoiser_prob:        float =  0.8   # fraction of steps in denoiser branch
    self_cond_prob:       float =  0.5   # probability of using self-conditioning
    cfg_scale_min:        float =  0.5   # CFG scale w ~ power-dist in [min, max]
    cfg_scale_max:        float =  5.0
    flow_cfg_scale:        float =  1.0   # CFG scale at inference time
    flow_n_steps:          int   = 64     # denoising steps at inference time
    sde_gamma:            float =  0.0   # SDE noise re-injection (0 = pure ODE)
    t5_model_name:        str   = "t5-base"  # frozen T5 embedder (ELF recipe §3.1)

    # ── MoE ───────────────────────────────────────────────────────────────────
    use_moe: bool = False
    n_experts: int = 4
    top_k_mlp: int = 2
    expansion: int = 4
    alpha_balance: float = 0.1
    moe_latent: bool = False
    moe_latent_dim: int = 64

    # ── Attention & Positional ────────────────────────────────────────────────
    use_rotary_pos: bool = True
    trainable_pos: bool = False
    absolute_pos: bool = False
    sliding_window: bool = False
    context_window: int = 4
    no_sink: bool = False
    differential: bool = False
    lambda_init: float = 0.8
    use_flash_attention: bool = False
    use_linear_attention: bool = False

    # ── MLA ───────────────────────────────────────────────────────────────────
    mla: bool = False
    inference: bool = False
    down_dim_q: int = 256
    down_dim_kv: int = 256
    rope_dim: int = 32

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer_type: str = "char"
    tokenizer_path: str | None = None

    # ── Warm start ────────────────────────────────────────────────────────────
    init_from: str = ""            # run dir whose best checkpoint initialises the model

    # ── Dataset ───────────────────────────────────────────────────────────────
    max_train_tokens: int = 10_000_000
    dataset_source: str = "local"
    dataset_name: str = ""
    dataset_config: str = ""
    dataset_text_field: str = "text"
    dataset_split: str = "train"
    streaming: bool = False

    # ── Training ──────────────────────────────────────────────────────────────
    lr: float = 3e-4
    batch_size: int = 32
    grad_accum: int = 1
    seed: int = 42
    optimizer: str = "adamw"
    epochs: float = 100
    warmup_steps: int = 400
    grad_clip: float = 1.0
    patience: int = 0
    use_bf16: bool = False

    # ── Normalisation ─────────────────────────────────────────────────────────
    norm_type: str = "rmsnorm"

    # ── RoPE ──────────────────────────────────────────────────────────────────
    rope_scale_factor: float = 1.0

    # ── LR schedule ───────────────────────────────────────────────────────────
    lr_schedule: str = "cosine"

    # ── LoRA ──────────────────────────────────────────────────────────────────
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_targets: str = "attention"

    # ── Multi-GPU ─────────────────────────────────────────────────────────────
    n_devices: int = 0
    tp_size: int = 1

    # ── Logging ───────────────────────────────────────────────────────────────
    eval_iters: int = 20
    log_file: str = "training_log.csv"
    summary_file: str = "model_summary.json"

    def __post_init__(self) -> None:
        if self.kv_heads is None:
            self.kv_heads = self.n_heads   # same as ModelConfig: default is MHA

        if self.attention_type == "auto":
            if self.mla:
                self.attention_type = "mla"
            elif self.kv_heads < self.n_heads:
                self.attention_type = "gqa"
            else:
                self.attention_type = "mha"
        self.mla = (self.attention_type == "mla")

        # For MHA and MLA, kv_heads == n_heads (GQA grouping is irrelevant)
        if self.attention_type in ("mha", "mla"):
            self.kv_heads = self.n_heads

        if self.dim != self.n_heads * self.head_size:
            raise ValueError(
                f"dim ({self.dim}) must equal n_heads * head_size "
                f"({self.n_heads} * {self.head_size} = {self.n_heads * self.head_size})"
            )
        if self.n_heads % self.kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by kv_heads ({self.kv_heads})"
            )
        if self.mla:
            # Decoupled RoPE is part of the MLA architecture (see ModelConfig).
            if not self.use_rotary_pos:
                raise ValueError("attention_type='mla' requires use_rotary_pos=True")
            if self.rope_dim > self.head_size:
                raise ValueError(
                    f"rope_dim ({self.rope_dim}) must be <= head_size ({self.head_size}) "
                    "when using MLA with rotary positional encoding"
                )
        if self.model_type not in ("autoregressive", "diffusion", "elf"):
            raise ValueError("model_type must be 'autoregressive', 'diffusion', or 'elf'")
        if self.attention_type not in ("mha", "gqa", "mla"):
            raise ValueError("attention_type must be 'mha', 'gqa', or 'mla'")
        if self.noise_schedule not in ("cosine", "linear", "sqrt"):
            raise ValueError("noise_schedule must be 'cosine', 'linear', or 'sqrt'")
        if self.diffusion_steps < 1:
            raise ValueError("diffusion_steps must be >= 1")
        if self.num_sampling_steps < 1:
            raise ValueError("num_sampling_steps must be >= 1")
        if self.norm_type not in ("layernorm", "rmsnorm"):
            raise ValueError("norm_type must be 'layernorm' or 'rmsnorm'")
        if self.lr_schedule not in ("cosine", "linear", "constant", "wsd"):
            raise ValueError("lr_schedule must be 'cosine', 'linear', 'constant', or 'wsd'")
        if self.rope_scale_factor <= 0:
            raise ValueError("rope_scale_factor must be > 0")
        if self.lora_targets not in ("attention", "mlp", "all"):
            raise ValueError("lora_targets must be 'attention', 'mlp', or 'all'")
        if self.lora_rank < 1:
            raise ValueError("lora_rank must be >= 1")
        if self.n_devices < 0:
            raise ValueError("n_devices must be >= 0")
        if self.tp_size < 1:
            raise ValueError("tp_size must be >= 1")
        if self.moe_latent and not (0 < self.moe_latent_dim < self.dim):
            raise ValueError(
                f"moe_latent_dim ({self.moe_latent_dim}) must be in (0, dim={self.dim})"
            )

    @property
    def causal(self) -> bool:
        return self.model_type == "autoregressive"

    # Deprecated ELF-branded names (paradigm-level names are flow_*):
    @property
    def elf_n_steps(self) -> int:
        return self.flow_n_steps

    @property
    def elf_cfg_scale(self) -> float:
        return self.flow_cfg_scale

    @classmethod
    def from_parts(
        cls,
        model_config: ModelConfig | FlowMatchingConfig,
        training_config: TrainingConfig | None = None,
        **overrides: Any,
    ) -> Config:
        """Merge a ``ModelConfig``/``FlowMatchingConfig`` and a ``TrainingConfig`` into a
        monolithic ``Config`` understood by the Trainer.

        This is the bridge between the new split-config API and the
        full-featured training engine.  *overrides* win over both parts
        (e.g. ``model_type="diffusion"``, ``noise_schedule="cosine"``).
        """
        kw: dict[str, Any] = {}

        if isinstance(model_config, FlowMatchingConfig):
            e = model_config
            for f in fields(FlowMatchingConfig):
                if f.name in _FLOW_SPECIAL_FIELDS:
                    continue
                kw[_FLOW_TO_LEGACY_RENAMES.get(f.name, f.name)] = getattr(e, f.name)
            kw.update(
                model_type="elf",
                kv_heads=e.kv_heads if e.kv_heads is not None else e.n_heads,
                use_moe=(e.ffn == "moe"),
                use_rotary_pos=(e.pos_encoding == "rotary"),
                trainable_pos=(e.pos_encoding == "learned"),
                absolute_pos=(e.pos_encoding == "absolute"),
                weight_tying=False,
                tokenizer_type="t5",
            )
        else:
            m = model_config
            for f in fields(ModelConfig):
                if f.name in _MODEL_SPECIAL_FIELDS or f.name in _MODEL_ONLY_FIELDS:
                    continue
                kw[_MODEL_TO_LEGACY_RENAMES.get(f.name, f.name)] = getattr(m, f.name)
            if m.paradigm is not None:
                model_type = _MODEL_TYPE_FROM_PARADIGM[m.paradigm]
            else:
                model_type = "autoregressive" if m.causal else "diffusion"
            kw.update(
                model_type=model_type,
                use_moe=(m.ffn == "moe"),
                use_rotary_pos=(m.pos_encoding == "rotary"),
                trainable_pos=(m.pos_encoding == "learned"),
                absolute_pos=(m.pos_encoding == "absolute"),
                lora_targets="mlp" if m.lora_targets == "ffn" else m.lora_targets,
            )

        if training_config is not None:
            t_valid = {f.name for f in fields(cls)}
            kw.update({
                k: v for k, v in asdict(training_config).items() if k in t_valid
            })

        if isinstance(model_config, FlowMatchingConfig):
            # The frozen T5 encoder consumes T5 token IDs — any other tokenizer
            # would feed it garbage. Override whatever the TrainingConfig says.
            kw["tokenizer_type"] = "t5"

        kw.update(overrides)
        return cls(**kw)

    def _pos_encoding_str(self) -> str:
        """Collapse the three legacy positional booleans into the canonical string."""
        if self.trainable_pos:
            return "learned"
        if self.absolute_pos:
            return "absolute"
        if self.use_rotary_pos:
            return "rotary"
        return "none"

    def to_flow_config(self) -> FlowMatchingConfig:
        """Convert to ``FlowMatchingConfig`` for use with ``FlowMatchingTransformer``.

        Every ``FlowMatchingConfig`` field is populated from its legacy counterpart via
        the module-level rename table, so a field added to ``FlowMatchingConfig`` (with
        a matching Config field) round-trips automatically.
        """
        kw: dict[str, Any] = {}
        for f in fields(FlowMatchingConfig):
            if f.name in _FLOW_SPECIAL_FIELDS:
                continue
            kw[f.name] = getattr(self, _FLOW_TO_LEGACY_RENAMES.get(f.name, f.name))
        return FlowMatchingConfig(
            **kw,
            ffn="moe" if self.use_moe else "mlp",
            pos_encoding=self._pos_encoding_str(),
            kv_heads=self.kv_heads if self.attention_type == "gqa" else None,
        )

    def to_elf_config(self) -> FlowMatchingConfig:
        """Deprecated alias of :meth:`to_flow_config` (removed in v1.0)."""
        import warnings

        warnings.warn(
            "Config.to_elf_config() is deprecated; use Config.to_flow_config().",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.to_flow_config()

    def to_model_config(self) -> ModelConfig:
        """Convert to a ``ModelConfig`` for use with the new ``Transformer`` API.

        The inverse of ``Config.from_parts``: every ``ModelConfig`` field is
        read from its legacy counterpart via the module-level rename table, so
        ``ModelConfig → Config → ModelConfig`` is the identity on all fields.
        """
        kw: dict[str, Any] = {}
        for f in fields(ModelConfig):
            if f.name in _MODEL_SPECIAL_FIELDS or f.name in _MODEL_ONLY_FIELDS:
                continue
            kw[f.name] = getattr(self, _MODEL_TO_LEGACY_RENAMES.get(f.name, f.name))
        return ModelConfig(
            **kw,
            paradigm=_PARADIGM_FROM_MODEL_TYPE.get(self.model_type),
            causal=self.causal,
            ffn="moe" if self.use_moe else "mlp",
            pos_encoding=self._pos_encoding_str(),
            lora_targets="ffn" if self.lora_targets == "mlp" else self.lora_targets,
        )

    def __repr__(self) -> str:
        attn = self.attention_type.upper()
        mode = {"autoregressive": "AR", "diffusion": "Diffusion", "elf": "Flow-Matching"}.get(
            self.model_type, self.model_type
        )
        moe  = "+MoE" if self.use_moe else ""
        return (
            f"Config(dim={self.dim}, heads={self.n_heads}, blocks={self.num_blocks}, "
            f"ctx={self.max_context}, mode={mode}, attn={attn}{moe})"
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Config:
        d = _remap_legacy_keys(d)
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})

    @classmethod
    def from_yaml(cls, path: str) -> Config:
        with open(path) as f:
            raw = yaml.safe_load(f)
        flat: dict[str, Any] = {}
        for v in raw.values():
            if isinstance(v, dict):
                flat.update(v)
        return cls.from_dict(flat if flat else raw)

    def save_yaml(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f)


# ── FlowMatchingConfig ─────────────────────────────────────────────────────────────────

@dataclass
class FlowMatchingConfig:
    """Architecture config for the continuous flow-matching paradigm.

    Analogous to ModelConfig but for continuous flow-matching.  The current
    implementation follows the ELF recipe (Hu et al., 2026); ``ELFConfig``
    remains available as a deprecated alias of this class.
    The forward process is ``z_t = t·x + (1−t)·ε`` (t=0→noise, t=1→data).
    The network predicts clean embeddings x̂ (x-prediction).

    Quick-start
    -----------
    ::

        config = FlowMatchingConfig(embed_dim=512, bottleneck_dim=128,
                           model_dim=768, n_heads=12, head_size=64,
                           num_blocks=12, vocab_size=32_000)
        model  = FlowMatchingTransformer(config, rngs=nnx.Rngs(42))
    """

    # ── Embedding / bottleneck space ──────────────────────────────────────────
    embed_dim:      int   = 512    # token embedding and flow-space dim
    bottleneck_dim: int   = 128    # bottleneck between embed and model space

    # ── Transformer backbone ──────────────────────────────────────────────────
    model_dim:  int   = 768        # transformer hidden dim (= n_heads × head_size)
    n_heads:    int   = 12
    head_size:  int | None = None  # auto-computed as model_dim // n_heads when None
    num_blocks: int   = 12
    vocab_size: int | None = None  # auto-set from tokenizer when using Trainer.fit()
    max_seq_len: int  = 1024       # sequence length (excluding control tokens)

    # ── Architecture ──────────────────────────────────────────────────────────
    pos_encoding:           str   = "rotary"  # "rotary" | "absolute" | "learned" | "none"
    norm:                   str   = "rmsnorm" # "rmsnorm" | "layernorm"
    dropout:                float = 0.0

    # ── Backbone attention (mirrors ModelConfig) ──────────────────────────────
    attention: str        = "mha"   # "mha" | "gqa" | "mla"
    kv_heads:  int | None = None    # GQA: number of KV heads; None → n_heads
    down_dim_q:  int = 256          # MLA latent dims
    down_dim_kv: int = 256
    rope_dim:    int = 32

    # ── In-context control tokens ─────────────────────────────────────────────
    time_emb_dim:    int = 256      # sinusoidal embedding dim for t and w
    num_time_tokens: int = 4        # control tokens encoding timestep t
    num_cfg_tokens:  int = 4        # control tokens encoding CFG scale w
    num_mode_tokens: int = 4        # control tokens encoding denoiser / decode mode

    # ── Denoiser-branch training ──────────────────────────────────────────────
    denoiser_pmean:       float = -1.5  # logit-normal time sampling parameters
    denoiser_pstd:        float =  0.8
    denoiser_noise_scale: float =  2.0  # noise scale applied to ε

    # ── Decoder-branch training ───────────────────────────────────────────────
    decoder_pmean:       float = 0.8    # logit-normal per-token p parameters
    decoder_pstd:        float = 0.8
    decoder_noise_scale: float = 5.0

    # ── Shared training ───────────────────────────────────────────────────────
    denoiser_prob:  float = 0.8   # fraction of steps in denoiser mode
    self_cond_prob: float = 0.5   # probability of using self-conditioning
    cfg_scale_min:  float = 0.5
    cfg_scale_max:  float = 5.0

    # ── Inference ─────────────────────────────────────────────────────────────
    sde_gamma:     float = 1.0   # SDE noise re-injection scale (0 = ODE)
    flow_n_steps:   int   = 32    # number of flow-matching integration steps
    flow_cfg_scale: float = 1.5   # default CFG guidance scale for generation

    # ── FFN / MoE ─────────────────────────────────────────────────────────────
    ffn:              str   = "mlp"   # "mlp" | "moe"
    n_experts:        int   = 4
    top_k:            int   = 2
    expansion:        int   = 4
    moe_balance_coeff: float = 0.1
    moe_latent:       bool  = False
    moe_latent_dim:   int   = 64

    # ── Pretrained embedder ───────────────────────────────────────────────────
    # T5 variant used as the frozen embedding oracle (ELF §3.1).
    # vocab_size and embed_dim must match the chosen variant:
    #   t5-small  → vocab=32128, embed_dim=512
    #   t5-base   → vocab=32128, embed_dim=768
    #   t5-large  → vocab=32128, embed_dim=1024
    t5_model_name: str = "t5-base"

    def __post_init__(self) -> None:
        if self.head_size is None:
            if self.model_dim % self.n_heads != 0:
                raise ValueError(
                    f"model_dim ({self.model_dim}) must be divisible by n_heads ({self.n_heads}) "
                    "when head_size is not specified"
                )
            self.head_size = self.model_dim // self.n_heads
        if self.model_dim != self.n_heads * self.head_size:
            raise ValueError(
                f"model_dim ({self.model_dim}) must equal n_heads × head_size "
                f"({self.n_heads} × {self.head_size} = {self.n_heads * self.head_size})"
            )
        if self.pos_encoding not in ("rotary", "absolute", "learned", "none"):
            raise ValueError(
                f"pos_encoding must be 'rotary', 'absolute', 'learned', or 'none'; "
                f"got {self.pos_encoding!r}"
            )
        if self.norm not in ("rmsnorm", "layernorm"):
            raise ValueError(f"norm must be 'rmsnorm' or 'layernorm'; got {self.norm!r}")
        if self.ffn not in ("mlp", "moe"):
            raise ValueError(f"ffn must be 'mlp' or 'moe'; got {self.ffn!r}")
        if self.moe_latent and not (0 < self.moe_latent_dim < self.model_dim):
            raise ValueError(
                f"moe_latent_dim ({self.moe_latent_dim}) must be in (0, model_dim={self.model_dim})"
            )

    @property
    def num_ctrl(self) -> int:
        """Total number of control tokens prepended to each sequence."""
        return self.num_time_tokens + self.num_cfg_tokens + self.num_mode_tokens

    # Deprecated ELF-branded names (paradigm-level names are flow_*):
    @property
    def elf_n_steps(self) -> int:
        return self.flow_n_steps

    @property
    def elf_cfg_scale(self) -> float:
        return self.flow_cfg_scale

    def to_model_config(self) -> ModelConfig:
        """Return the inner transformer's ModelConfig (used internally by FlowMatchingTransformer)."""
        return ModelConfig(
            dim=self.model_dim,
            n_heads=self.n_heads,
            head_size=self.head_size,
            num_blocks=self.num_blocks,
            vocab_size=self.vocab_size,
            max_context=self.max_seq_len + self.num_ctrl,
            attention=self.attention,
            kv_heads=self.kv_heads,
            down_dim_q=self.down_dim_q,
            down_dim_kv=self.down_dim_kv,
            rope_dim=self.rope_dim,
            ffn=self.ffn,
            n_experts=self.n_experts,
            top_k=self.top_k,
            expansion=self.expansion,
            moe_balance_coeff=self.moe_balance_coeff,
            moe_latent=self.moe_latent,
            moe_latent_dim=self.moe_latent_dim,
            norm=self.norm,
            pos_encoding=self.pos_encoding,
            causal=False,
            dropout=self.dropout,
            weight_tying=False,
            use_swiglu=True,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FlowMatchingConfig:
        d = _remap_legacy_keys(d)
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})

    @classmethod
    def from_yaml(cls, path: str) -> FlowMatchingConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        flat: dict[str, Any] = {}
        for v in raw.values():
            if isinstance(v, dict):
                flat.update(v)
        return cls.from_dict(flat if flat else raw)

    def save_yaml(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f)

    def __repr__(self) -> str:
        return (
            f"FlowMatchingConfig(embed={self.embed_dim}, bottleneck={self.bottleneck_dim}, "
            f"dim={self.model_dim}, heads={self.n_heads}, blocks={self.num_blocks}, "
            f"vocab={self.vocab_size}, seq={self.max_seq_len})"
        )


_install_legacy_ctor_aliases(FlowMatchingConfig)


# ── Deprecated alias ──────────────────────────────────────────────────────────
# The paradigm is "continuous flow-matching"; ELF (Hu et al., 2026) is the
# recipe currently implemented.  Kept so old imports and pickles keep working;
# removed in v1.0.
ELFConfig = FlowMatchingConfig
