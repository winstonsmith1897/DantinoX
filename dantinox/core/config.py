from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

import yaml


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
    head_size: int = 32
    num_blocks: int = 12
    vocab_size: int = 200
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
    gradient_checkpointing: bool = False
    tp_size: int = 1

    # ── Attention ─────────────────────────────────────────────────────────────
    kv_heads: int | None = None   # GQA: number of KV heads; None → same as n_heads
    use_flash: bool = False
    rope_scale: float = 1.0
    sliding_window: bool = False
    context_window: int = 4
    no_sink: bool = False

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

    # ── Diffusion ─────────────────────────────────────────────────────────────
    mask_token_id: int = 4

    # ── LoRA ──────────────────────────────────────────────────────────────────
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_targets: str = "attention"  # "attention" | "ffn" | "all"

    # ── Validation ────────────────────────────────────────────────────────────

    def __post_init__(self) -> None:
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
        if self.ffn not in ("mlp", "moe"):
            raise ValueError(f"ffn must be 'mlp' or 'moe'; got {self.ffn!r}")
        if self.norm not in ("rmsnorm", "layernorm"):
            raise ValueError(f"norm must be 'rmsnorm' or 'layernorm'; got {self.norm!r}")
        if self.pos_encoding not in ("rotary", "absolute", "learned", "none"):
            raise ValueError(
                f"pos_encoding must be 'rotary', 'absolute', 'learned', or 'none'; "
                f"got {self.pos_encoding!r}"
            )
        if self.tp_size < 1:
            raise ValueError(f"tp_size must be >= 1; got {self.tp_size}")
        if self.lora_targets not in ("attention", "ffn", "all"):
            raise ValueError(f"lora_targets must be 'attention', 'ffn', or 'all'")
        if self.lora_rank < 1:
            raise ValueError(f"lora_rank must be >= 1")

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

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelConfig:
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

    def __repr__(self) -> str:
        mode = "AR" if self.causal else "Diffusion"
        extra = "+MoE" if self.use_moe else ""
        return (
            f"ModelConfig(dim={self.dim}, heads={self.n_heads}, blocks={self.num_blocks}, "
            f"ctx={self.max_context}, mode={mode}, attn={self.attention.upper()}{extra})"
        )


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
    epochs: int = 100
    warmup_steps: int = 400
    lr_schedule: str = "cosine"    # "cosine" | "linear" | "constant" | "wsd"
    optimizer: str = "adamw"       # "adamw" | "adafactor" | "lion" | "adam"
    grad_clip: float = 1.0
    seed: int = 42
    use_bf16: bool = False
    patience: int = 0              # early stopping: epochs without val improvement (0 = off)
    eval_iters: int = 20
    val_frac: float = 0.1          # fraction of tokens held out for validation

    # ── Multi-GPU ─────────────────────────────────────────────────────────────
    n_devices: int = 0             # 0 = all available devices

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

    # ── Diffusion training ────────────────────────────────────────────────────
    noise_schedule: str = "linear" # "linear" | "cosine" | "sqrt"

    # ── Warm start ────────────────────────────────────────────────────────────
    init_from: str = ""            # run dir whose best checkpoint initialises the model

    # ── Logging ───────────────────────────────────────────────────────────────
    log_file: str = "training_log.csv"

    def __post_init__(self) -> None:
        if self.lr_schedule not in ("cosine", "linear", "constant", "wsd"):
            raise ValueError(f"lr_schedule must be 'cosine', 'linear', 'constant', or 'wsd'")
        if self.noise_schedule not in ("linear", "cosine", "sqrt"):
            raise ValueError(f"noise_schedule must be 'linear', 'cosine', or 'sqrt'")
        if self.n_devices < 0:
            raise ValueError(f"n_devices must be >= 0")
        if not 0.0 <= self.val_frac < 1.0:
            raise ValueError(f"val_frac must be in [0, 1); got {self.val_frac}")
        if self.grad_accum < 1:
            raise ValueError(f"grad_accum must be >= 1; got {self.grad_accum}")

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

@dataclass
class Config:
    """Monolithic config kept for backward compatibility.

    Combines all ModelConfig and TrainingConfig fields in a single flat
    dataclass so existing code (trainer, CLI, YAML files) works unchanged.

    New code should use ``ModelConfig`` for architecture and
    ``TrainingConfig`` for training hyper-parameters.
    """

    # ── Architecture ──────────────────────────────────────────────────────────
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

    # ── Model type & attention variant ────────────────────────────────────────
    model_type: str = "autoregressive"   # "autoregressive" | "diffusion"
    attention_type: str = "auto"         # "mha" | "gqa" | "mla" | "auto" (derived)

    # ── Diffusion (discrete — LLaDA / MDLM) ─────────────────────────────────
    diffusion_steps: int = 1000
    noise_schedule: str = "cosine"
    mask_token_id: int = 4
    num_sampling_steps: int = 50
    time_emb_dim: int = 256

    # ── ELF (continuous flow-matching) ────────────────────────────────────────
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
    elf_cfg_scale:        float =  1.0   # CFG scale at inference time
    elf_n_steps:          int   = 64     # denoising steps at inference time
    t5_model_name:        str   = "t5-base"  # frozen T5 embedder (ELF §3.1)

    # ── MoE ───────────────────────────────────────────────────────────────────
    use_moe: bool = False
    n_experts: int = 4
    top_k_mlp: int = 2
    expansion: int = 4
    alpha_balance: float = 0.1

    # ── Attention & Positional ────────────────────────────────────────────────
    use_rotary_pos: bool = True
    trainable_pos: bool = False
    absolute_pos: bool = False
    sliding_window: bool = False
    context_window: int = 4
    no_sink: bool = True
    use_flash_attention: bool = False

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

    # ── Normalisation ─────────────────────────────────────────────────────────
    norm_type: str = "layernorm"

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
            self.kv_heads = self.n_heads // 4

        if self.attention_type == "auto":
            if self.mla:
                self.attention_type = "mla"
            elif self.kv_heads < self.n_heads:
                self.attention_type = "gqa"
            else:
                self.attention_type = "mha"
        self.mla = (self.attention_type == "mla")

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
        if self.model_type not in ("autoregressive", "diffusion", "elf"):
            raise ValueError(f"model_type must be 'autoregressive', 'diffusion', or 'elf'")
        if self.attention_type not in ("mha", "gqa", "mla"):
            raise ValueError(f"attention_type must be 'mha', 'gqa', or 'mla'")
        if self.noise_schedule not in ("cosine", "linear", "sqrt"):
            raise ValueError(f"noise_schedule must be 'cosine', 'linear', or 'sqrt'")
        if self.diffusion_steps < 1:
            raise ValueError(f"diffusion_steps must be >= 1")
        if self.num_sampling_steps < 1:
            raise ValueError(f"num_sampling_steps must be >= 1")
        if self.norm_type not in ("layernorm", "rmsnorm"):
            raise ValueError(f"norm_type must be 'layernorm' or 'rmsnorm'")
        if self.lr_schedule not in ("cosine", "linear", "constant", "wsd"):
            raise ValueError(f"lr_schedule must be 'cosine', 'linear', 'constant', or 'wsd'")
        if self.rope_scale_factor <= 0:
            raise ValueError(f"rope_scale_factor must be > 0")
        if self.lora_targets not in ("attention", "mlp", "all"):
            raise ValueError(f"lora_targets must be 'attention', 'mlp', or 'all'")
        if self.lora_rank < 1:
            raise ValueError(f"lora_rank must be >= 1")
        if self.n_devices < 0:
            raise ValueError(f"n_devices must be >= 0")
        if self.tp_size < 1:
            raise ValueError(f"tp_size must be >= 1")

    @property
    def causal(self) -> bool:
        return self.model_type == "autoregressive"

    @classmethod
    def from_parts(
        cls,
        model_config: "ModelConfig | ELFConfig",
        training_config: "TrainingConfig | None" = None,
        **overrides: Any,
    ) -> "Config":
        """Merge a ``ModelConfig``/``ELFConfig`` and a ``TrainingConfig`` into a
        monolithic ``Config`` understood by the Trainer.

        This is the bridge between the new split-config API and the
        full-featured training engine.  *overrides* win over both parts
        (e.g. ``model_type="diffusion"``, ``noise_schedule="cosine"``).
        """
        kw: dict[str, Any] = {}

        if isinstance(model_config, ELFConfig):
            e = model_config
            kw.update(
                model_type="elf",
                dim=e.model_dim,
                n_heads=e.n_heads,
                head_size=e.head_size,
                num_blocks=e.num_blocks,
                vocab_size=e.vocab_size,
                max_context=e.max_seq_len,
                kv_heads=e.n_heads,
                attention_type="mha",
                norm_type=e.norm,
                dropout_rate=e.dropout,
                gradient_checkpointing=e.gradient_checkpointing,
                use_rotary_pos=(e.pos_encoding == "rotary"),
                trainable_pos=(e.pos_encoding == "learned"),
                absolute_pos=(e.pos_encoding == "absolute"),
                weight_tying=False,
                embed_dim=e.embed_dim,
                bottleneck_dim=e.bottleneck_dim,
                time_emb_dim=e.time_emb_dim,
                num_time_tokens=e.num_time_tokens,
                num_cfg_tokens=e.num_cfg_tokens,
                num_mode_tokens=e.num_mode_tokens,
                denoiser_pmean=e.denoiser_pmean,
                denoiser_pstd=e.denoiser_pstd,
                denoiser_noise_scale=e.denoiser_noise_scale,
                decoder_pmean=e.decoder_pmean,
                decoder_pstd=e.decoder_pstd,
                decoder_noise_scale=e.decoder_noise_scale,
                denoiser_prob=e.denoiser_prob,
                self_cond_prob=e.self_cond_prob,
                cfg_scale_min=e.cfg_scale_min,
                cfg_scale_max=e.cfg_scale_max,
                t5_model_name=e.t5_model_name,
                tokenizer_type="t5",
            )
        else:
            m = model_config
            kw.update(
                model_type="autoregressive" if m.causal else "diffusion",
                dim=m.dim,
                n_heads=m.n_heads,
                head_size=m.head_size,
                num_blocks=m.num_blocks,
                vocab_size=m.vocab_size,
                max_context=m.max_context,
                kv_heads=m.kv_heads if m.kv_heads is not None else m.n_heads,
                attention_type=m.attention,
                use_moe=(m.ffn == "moe"),
                norm_type=m.norm,
                use_rotary_pos=(m.pos_encoding == "rotary"),
                trainable_pos=(m.pos_encoding == "learned"),
                absolute_pos=(m.pos_encoding == "absolute"),
                dropout_rate=m.dropout,
                weight_tying=m.weight_tying,
                gradient_checkpointing=m.gradient_checkpointing,
                tp_size=m.tp_size,
                use_flash_attention=m.use_flash,
                rope_scale_factor=m.rope_scale,
                sliding_window=m.sliding_window,
                context_window=m.context_window,
                no_sink=m.no_sink,
                down_dim_q=m.down_dim_q,
                down_dim_kv=m.down_dim_kv,
                rope_dim=m.rope_dim,
                inference=m.inference_mode,
                expansion=m.expansion,
                use_swiglu=m.use_swiglu,
                activation=m.activation,
                n_experts=m.n_experts,
                top_k_mlp=m.top_k,
                alpha_balance=m.moe_balance_coeff,
                mask_token_id=m.mask_token_id,
                use_lora=m.use_lora,
                lora_rank=m.lora_rank,
                lora_alpha=m.lora_alpha,
                lora_dropout=m.lora_dropout,
                lora_targets="mlp" if m.lora_targets == "ffn" else m.lora_targets,
            )

        if training_config is not None:
            t_valid = {f.name for f in fields(cls)}
            kw.update({
                k: v for k, v in asdict(training_config).items() if k in t_valid
            })

        if isinstance(model_config, ELFConfig):
            # The frozen T5 encoder consumes T5 token IDs — any other tokenizer
            # would feed it garbage. Override whatever the TrainingConfig says.
            kw["tokenizer_type"] = "t5"

        kw.update(overrides)
        return cls(**kw)

    def to_elf_config(self) -> "ELFConfig":
        """Convert to ``ELFConfig`` for use with ``ELFTransformer``.

        Shared fields (dim, n_heads, head_size, …) map directly; ELF-specific
        fields (embed_dim, bottleneck_dim, …) are read from the ELF section.
        """
        if self.use_rotary_pos:
            pos = "rotary"
        elif self.trainable_pos:
            pos = "learned"
        elif self.absolute_pos:
            pos = "absolute"
        else:
            pos = "none"

        return ELFConfig(
            embed_dim=self.embed_dim,
            bottleneck_dim=self.bottleneck_dim,
            model_dim=self.dim,
            n_heads=self.n_heads,
            head_size=self.head_size,
            num_blocks=self.num_blocks,
            vocab_size=self.vocab_size,
            max_seq_len=self.max_context,
            pos_encoding=pos,
            norm=self.norm_type,
            dropout=self.dropout_rate,
            time_emb_dim=self.time_emb_dim,
            num_time_tokens=self.num_time_tokens,
            num_cfg_tokens=self.num_cfg_tokens,
            num_mode_tokens=self.num_mode_tokens,
            denoiser_pmean=self.denoiser_pmean,
            denoiser_pstd=self.denoiser_pstd,
            denoiser_noise_scale=self.denoiser_noise_scale,
            decoder_pmean=self.decoder_pmean,
            decoder_pstd=self.decoder_pstd,
            decoder_noise_scale=self.decoder_noise_scale,
            denoiser_prob=self.denoiser_prob,
            self_cond_prob=self.self_cond_prob,
            cfg_scale_min=self.cfg_scale_min,
            cfg_scale_max=self.cfg_scale_max,
            t5_model_name=self.t5_model_name,
            gradient_checkpointing=self.gradient_checkpointing,
        )

    def to_model_config(self) -> ModelConfig:
        """Convert to a ``ModelConfig`` for use with the new ``Transformer`` API."""
        if self.trainable_pos:
            pos = "learned"
        elif self.absolute_pos:
            pos = "absolute"
        elif self.use_rotary_pos:
            pos = "rotary"
        else:
            pos = "none"

        return ModelConfig(
            dim=self.dim,
            n_heads=self.n_heads,
            head_size=self.head_size,
            num_blocks=self.num_blocks,
            vocab_size=self.vocab_size,
            max_context=self.max_context,
            attention=self.attention_type,
            ffn="moe" if self.use_moe else "mlp",
            norm=self.norm_type,
            pos_encoding=pos,
            causal=(self.model_type == "autoregressive"),
            dropout=self.dropout_rate,
            weight_tying=self.weight_tying,
            gradient_checkpointing=self.gradient_checkpointing,
            tp_size=self.tp_size,
            kv_heads=self.kv_heads,
            use_flash=self.use_flash_attention,
            rope_scale=self.rope_scale_factor,
            sliding_window=self.sliding_window,
            context_window=self.context_window,
            no_sink=self.no_sink,
            down_dim_q=self.down_dim_q,
            down_dim_kv=self.down_dim_kv,
            rope_dim=self.rope_dim,
            inference_mode=self.inference,
            expansion=self.expansion,
            use_swiglu=self.use_swiglu,
            activation=self.activation,
            n_experts=self.n_experts,
            top_k=self.top_k_mlp,
            moe_balance_coeff=self.alpha_balance,
            mask_token_id=self.mask_token_id,
            use_lora=self.use_lora,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            lora_targets="ffn" if self.lora_targets == "mlp" else self.lora_targets,
        )

    def __repr__(self) -> str:
        attn = self.attention_type.upper()
        mode = {"autoregressive": "AR", "diffusion": "Diffusion", "elf": "ELF"}.get(
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


# ── ELFConfig ─────────────────────────────────────────────────────────────────

@dataclass
class ELFConfig:
    """Architecture config for ELF (Embedded Language Flows).

    Analogous to ModelConfig but for the continuous flow-matching paradigm.
    The forward process is ``z_t = t·x + (1−t)·ε`` (t=0→noise, t=1→data).
    The network predicts clean embeddings x̂ (x-prediction).

    Quick-start
    -----------
    ::

        config = ELFConfig(embed_dim=512, bottleneck_dim=128,
                           model_dim=768, n_heads=12, head_size=64,
                           num_blocks=12, vocab_size=32_000)
        model  = ELFTransformer(config, rngs=nnx.Rngs(42))
    """

    # ── Embedding / bottleneck space ──────────────────────────────────────────
    embed_dim:      int   = 512    # token embedding and flow-space dim
    bottleneck_dim: int   = 128    # bottleneck between embed and model space

    # ── Transformer backbone ──────────────────────────────────────────────────
    model_dim:  int   = 768        # transformer hidden dim (= n_heads × head_size)
    n_heads:    int   = 12
    head_size:  int   = 64
    num_blocks: int   = 12
    vocab_size: int   = 32_000
    max_seq_len: int  = 1024       # sequence length (excluding control tokens)

    # ── Architecture ──────────────────────────────────────────────────────────
    pos_encoding:           str   = "rotary"  # "rotary" | "absolute" | "learned" | "none"
    norm:                   str   = "rmsnorm" # "rmsnorm" | "layernorm"
    dropout:                float = 0.0
    gradient_checkpointing: bool  = True

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
    sde_gamma: float = 1.0  # SDE noise re-injection scale (0 = ODE)

    # ── Pretrained embedder ───────────────────────────────────────────────────
    # T5 variant used as the frozen embedding oracle (ELF §3.1).
    # vocab_size and embed_dim must match the chosen variant:
    #   t5-small  → vocab=32128, embed_dim=512
    #   t5-base   → vocab=32128, embed_dim=768
    #   t5-large  → vocab=32128, embed_dim=1024
    t5_model_name: str = "t5-base"

    def __post_init__(self) -> None:
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

    @property
    def num_ctrl(self) -> int:
        """Total number of control tokens prepended to each sequence."""
        return self.num_time_tokens + self.num_cfg_tokens + self.num_mode_tokens

    def to_model_config(self) -> ModelConfig:
        """Return the inner transformer's ModelConfig (used internally by ELFTransformer)."""
        return ModelConfig(
            dim=self.model_dim,
            n_heads=self.n_heads,
            head_size=self.head_size,
            num_blocks=self.num_blocks,
            vocab_size=self.vocab_size,
            max_context=self.max_seq_len + self.num_ctrl,
            attention="mha",
            ffn="mlp",
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
    def from_dict(cls, d: dict[str, Any]) -> "ELFConfig":
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})

    @classmethod
    def from_yaml(cls, path: str) -> "ELFConfig":
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
            f"ELFConfig(embed={self.embed_dim}, bottleneck={self.bottleneck_dim}, "
            f"dim={self.model_dim}, heads={self.n_heads}, blocks={self.num_blocks}, "
            f"vocab={self.vocab_size}, seq={self.max_seq_len})"
        )
