"""Training console UI — ANSI-colored, paradigm-aware output for DantinoX runs.

Pure stdlib (+ tqdm which is already a dependency).
Respects ``NO_COLOR`` and ``TERM=dumb`` environment variables.
Auto-detects Jupyter and routes output to stdout so it appears in cell output.
"""
from __future__ import annotations

import os
import sys
from typing import Any

# ── Environment detection ─────────────────────────────────────────────────────

def _use_color() -> bool:
    return not os.environ.get("NO_COLOR") and os.environ.get("TERM") != "dumb"


def _is_jupyter() -> bool:
    try:
        from IPython import get_ipython  # type: ignore[import]
        ip = get_ipython()
        return ip is not None and hasattr(ip, "kernel")
    except ImportError:
        return False


def _out() -> Any:
    """Output stream: stdout in Jupyter (cell output), stderr in terminal."""
    return sys.stdout if _is_jupyter() else sys.stderr


# ── 256-color palette (matches banner gradient: magenta → cyan) ───────────────

def _c(n: int, s: str) -> str:
    """Wrap *s* in 256-color ANSI code *n*, or return *s* unchanged if colors off."""
    return f"\033[38;5;{n}m{s}\033[0m" if _use_color() else s

def _mgn(s: str) -> str: return _c(201, s)   # magenta   — paradigm name
def _prp(s: str) -> str: return _c(141, s)   # lavender  — section headers, epoch label
def _blu(s: str) -> str: return _c(69,  s)   # cornflower — row labels
def _azr(s: str) -> str: return _c(39,  s)   # azure      — secondary values
def _cyn(s: str) -> str: return _c(51,  s)   # cyan       — numbers / highlights
def _grn(s: str) -> str: return _c(82,  s)   # green      — new-best metrics
def _yel(s: str) -> str: return _c(226, s)   # yellow     — ★ best marker
def _wht(s: str) -> str: return (f"\033[97m{s}\033[0m" if _use_color() else s)
def _dim(s: str) -> str: return (f"\033[2m{s}\033[0m"  if _use_color() else s)

_W = 64   # visual width of the header panel

# ── Layout primitives ─────────────────────────────────────────────────────────

def _hr() -> str:
    """Full-width horizontal rule."""
    return "  " + _dim("─" * (_W - 2))


def _sec(title: str) -> str:
    """Section sub-rule with a colored title."""
    prefix = f"── {title} "
    dashes = "─" * max(2, _W - 2 - len(prefix))
    return "  " + _prp(prefix + dashes)


def _row(label: str, value: str, lw: int = 12) -> str:
    """Labeled row: '  cornflower-label   value'."""
    return f"  {_blu(label.ljust(lw))}  {value}"


# ── Compact model architecture summary ───────────────────────────────────────

def _cfg_dim(cfg: Any) -> int:
    """Return the transformer backbone width from either ModelConfig or ELFConfig."""
    return getattr(cfg, "dim", None) or getattr(cfg, "model_dim", 512)


def _cfg_ctx(cfg: Any) -> int:
    """Return max context length from either ModelConfig or ELFConfig."""
    return getattr(cfg, "max_context", None) or getattr(cfg, "max_seq_len", 512)


def _cfg_use_moe(cfg: Any) -> bool:
    use_moe = getattr(cfg, "use_moe", None)
    if use_moe is not None:
        return bool(use_moe)
    return getattr(cfg, "ffn", "mlp") == "moe"


def _model_lines(cfg: Any) -> list[str]:
    """Return two lines summarizing the model architecture.

    Accepts both ``ModelConfig`` and the legacy ``ELFConfig``.
    """
    dim   = _cfg_dim(cfg)
    ctx   = _cfg_ctx(cfg)
    hs    = getattr(cfg, "head_size", None) or (dim // cfg.n_heads)
    vocab = f"{cfg.vocab_size:,}" if cfg.vocab_size is not None else "?"
    mode  = "causal" if getattr(cfg, "causal", False) else "bidirectional"

    if cfg.attention == "gqa" and getattr(cfg, "kv_heads", None) not in (None, cfg.n_heads):
        attn_s = f"GQA({cfg.n_heads}q/{cfg.kv_heads}kv)"
    elif cfg.attention == "mla":
        attn_s = f"MLA(dq={cfg.down_dim_q},dkv={cfg.down_dim_kv})"
    else:
        attn_s = "MHA"

    pos_map = {"rotary": "RoPE", "absolute": "sin", "learned": "learned", "none": "none"}
    pos_s   = pos_map.get(cfg.pos_encoding, cfg.pos_encoding)
    norm_s  = {"rmsnorm": "RMSNorm", "layernorm": "LayerNorm"}.get(cfg.norm, cfg.norm)

    act_s = "SwiGLU" if getattr(cfg, "use_swiglu", True) else getattr(cfg, "activation", "gelu").upper()
    if _cfg_use_moe(cfg):
        ffn_s = f"MoE({cfg.n_experts}×top{cfg.top_k})"
    else:
        ffn_s = f"MLP(×{cfg.expansion},{act_s})"

    line1 = (
        f"  {_cyn(str(dim))}-dim  ·  "
        f"{_cyn(str(cfg.n_heads))}h×{_cyn(str(hs))}  ·  "
        f"{_cyn(str(cfg.num_blocks))} blocks  ·  "
        f"vocab={_cyn(vocab)}  ·  "
        f"ctx={_cyn(str(ctx))}"
    )
    line2 = (
        f"  {_wht(attn_s)}  ·  {_wht(pos_s)}  ·  "
        f"{_wht(norm_s)}  ·  {_azr(mode)}  ·  {_wht(ffn_s)}"
    )
    return [line1, line2]


# ── Public API ────────────────────────────────────────────────────────────────

def print_run_header(
    paradigm_type:   str,
    model_cfg:       Any,
    cfg:             Any,          # TrainingConfig
    data_source:     str,
    tokenizer_type:  str,
    tok_vocab:       int,
    n_train:         int,
    n_val:           int,
    n_params:        int,
    run_dir:         str,
    n_epochs:        int,
    steps_per_epoch: int,
    steps_this_epoch: int,
    total_updates:   int,
    n_dev:           int,
    tp_size:         int,
) -> None:
    """Print a rich colored run-configuration panel before training starts."""
    import jax

    stream = _out()
    L: list[str] = [""]

    # ── paradigm bar ──────────────────────────────────────────────────────────
    mode   = "causal" if getattr(model_cfg, "causal", False) else "bidirectional"
    L.append(_hr())
    L.append(f"  {_mgn(paradigm_type)}  ·  {_azr(mode)}")
    L.append(_hr())

    # ── run dir + params ──────────────────────────────────────────────────────
    param_str = (
        f"{_cyn(f'{n_params / 1e6:.1f}')} M"
        f"  {_dim(f'({n_params:,})')}"
    )
    L.append(_row("run dir",    _dim(run_dir)))
    L.append(_row("parameters", param_str))
    L.append("")

    # ── model ─────────────────────────────────────────────────────────────────
    L.append(_sec("model"))
    L.extend(_model_lines(model_cfg))
    extras: list[str] = []
    if getattr(model_cfg, "use_lora", False):
        extras.append(f"LoRA(r={model_cfg.lora_rank}, α={model_cfg.lora_alpha})")
    if getattr(model_cfg, "use_flash", False):
        extras.append("flash-attn")
    if getattr(model_cfg, "gradient_checkpointing", False):
        extras.append("grad-ckpt")
    if getattr(model_cfg, "tp_size", 1) > 1:
        extras.append(f"tp={model_cfg.tp_size}")
    if extras:
        L.append(f"  {_dim(' · '.join(extras))}")
    L.append("")

    # ── data ──────────────────────────────────────────────────────────────────
    L.append(_sec("data"))
    L.append(_row("source",    _dim(data_source)))
    L.append(_row("tokenizer", f"{_wht(tokenizer_type)}  ·  {_cyn(str(tok_vocab))} vocab"))
    total_tok = n_train + n_val
    if n_val > 0:
        tok_str = (
            f"{_cyn(f'{total_tok:,}')}  "
            f"{_dim('(')}train {_cyn(f'{n_train:,}')}  "
            f"·  val {_cyn(f'{n_val:,}')}{_dim(')')}"
        )
    else:
        tok_str = _cyn(f"{n_train:,}")
    L.append(_row("tokens", tok_str))
    L.append("")

    # ── training ──────────────────────────────────────────────────────────────
    L.append(_sec("training"))

    lr_val = cfg.lr
    lr_str = f"{lr_val:.0e}" if lr_val < 0.01 else str(lr_val)
    opt_str = (
        f"{_wht(cfg.optimizer)}  ·  lr={_cyn(lr_str)}"
        f"  ·  {_wht(cfg.lr_schedule)}  ·  warmup={_cyn(str(cfg.warmup_steps))}"
    )
    L.append(_row("optimizer", opt_str))

    eff_batch = cfg.batch_size * cfg.grad_accum
    batch_str = _cyn(str(cfg.batch_size))
    if cfg.grad_accum > 1:
        batch_str += (
            f"  ·  acc={_cyn(str(cfg.grad_accum))}"
            f"  ·  effective={_cyn(str(eff_batch))}"
        )
    L.append(_row("batch", batch_str))

    epoch_disp = f"{cfg.epochs:g}"
    sched_str = (
        f"{_cyn(epoch_disp)} epoch{'s' if n_epochs != 1 else ''}"
        f"  ·  {_cyn(str(steps_this_epoch))} steps/epoch"
        f"  ·  {_cyn(str(total_updates))} updates"
    )
    L.append(_row("schedule", sched_str))

    prec_str = _wht("bf16") if cfg.use_bf16 else _dim("fp32")
    L.append(_row("precision", prec_str))

    dev_platform = jax.devices()[0].platform.upper()
    if tp_size > 1:
        dev_str = (
            f"{_cyn(str(n_dev * tp_size))}× {_wht(dev_platform)}"
            f"  {_dim(f'({n_dev} DP × {tp_size} TP)')}"
        )
    elif n_dev > 1:
        dev_str = f"{_cyn(str(n_dev))}× {_wht(dev_platform)}  {_dim('(data-parallel)')}"
    else:
        dev_str = f"1× {_wht(dev_platform)}"
    L.append(_row("devices", dev_str))

    L.append("")
    L.append(_hr())
    L.append("")

    print("\n".join(L), file=stream, flush=True)


def print_compile_hint(epoch: int) -> None:
    """Print a one-time hint that the first step triggers JIT compilation."""
    if epoch == 1:
        print(
            f"  {_dim('step 1: JIT compiling (may take 1-3 min on first run)...')}",
            file=_out(), flush=True,
        )


def make_epoch_bar(epoch: int, n_epochs: int, total: int):
    """Return a plain tqdm progress bar for one training epoch.

    ``tqdm.auto`` picks the Jupyter widget in notebooks and the plain
    text bar in terminals.  Colors are intentionally omitted here;
    ``print_epoch_result`` provides the colored per-epoch summary.
    """
    from tqdm.auto import tqdm as _tqdm  # type: ignore[import]
    return _tqdm(range(total), desc=f"Epoch {epoch}/{n_epochs}", leave=False)


def print_epoch_result(
    epoch:       int,
    n_epochs:    int,
    train_loss:  float,
    val_loss:    float,
    is_best:     bool,
    elapsed:     float,
) -> None:
    """Print a one-line summary after each epoch's progress bar closes."""
    ep_tag  = _prp(f"Epoch {epoch}/{n_epochs}")
    tr_tag  = f"train={_cyn(f'{train_loss:.4f}')}"
    va_val  = _grn(f"{val_loss:.4f}") if is_best else _azr(f"{val_loss:.4f}")
    va_tag  = f"val={va_val}"
    best    = f"  {_yel('★ best')}" if is_best else ""
    t_tag   = _dim(f"{elapsed:.1f}s")
    print(f"  {ep_tag}  {tr_tag}  {va_tag}{best}  {t_tag}",
          file=_out(), flush=True)


def print_training_done(best_loss: float, run_dir: str) -> None:
    """Print a final colored summary when training finishes."""
    L = [
        "",
        _hr(),
        (
            f"  {_grn('training complete')}"
            f"  ·  best val loss = {_cyn(f'{best_loss:.4f}')}"
        ),
        f"  {_dim('saved')} → {_dim(run_dir)}",
        _hr(),
        "",
    ]
    print("\n".join(L), file=_out(), flush=True)
