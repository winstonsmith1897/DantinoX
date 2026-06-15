from __future__ import annotations

import csv
import datetime
import json
import logging
import os
import warnings
from pathlib import Path
from typing import Any

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from tqdm import tqdm

from dantinox.core.config import Config, TrainingConfig
from dantinox.core.sharding import make_mesh, num_devices, replicate, shard_batch
from dantinox.paradigms.base import Paradigm
from dantinox.training.optimizer import build_optimizer

log = logging.getLogger(__name__)

_TRAIN_STATE_FILE = "train_state.msgpack"
_TRAIN_META_FILE  = "train_state.json"

# Checkpoints exclude RNG key state: typed PRNG keys are not serialisable and
# per-epoch keys are re-derived from the seed anyway.
_WEIGHTS = nnx.Not(nnx.RngState)


class Trainer:
    """Paradigm-agnostic training harness.

    The Trainer is decoupled from model type: paradigm-specific behaviour
    (masking, noise schedules, ELF branches, T5 embeddings) lives in the
    Paradigm, which the Trainer drives through ``loss_fn`` plus the optional
    ``on_train_start`` / ``prepare_batch`` hooks.

    Features
    --------
    * train/validation split (``val_frac``) with the best checkpoint chosen
      by **validation** loss
    * gradient accumulation (``grad_accum``) via ``optax.MultiSteps``
    * early stopping (``patience`` epochs without val improvement)
    * bf16 parameter casting (``use_bf16``)
    * full train-state checkpointing (model + optimizer + epoch) with
      ``fit(..., resume=True)``
    * memory-mapped token cache — corpora are tokenised once and re-read
      via ``np.load(mmap_mode="r")``

    Quick-start::

        from dantinox import ARParadigm, ModelConfig, TrainingConfig, Trainer

        paradigm = ARParadigm(ModelConfig(dim=512, n_heads=8, head_size=64,
                                          num_blocks=12, vocab_size=32_000))
        trainer  = Trainer(paradigm, TrainingConfig(lr=3e-4, epochs=5))
        run_dir  = trainer.fit("data/corpus.txt")
    """

    def __init__(
        self,
        paradigm: Paradigm | Config,
        config: TrainingConfig | None = None,
    ) -> None:
        if isinstance(paradigm, Config):
            # Legacy bridge: a monolithic Config carries both the architecture
            # and the training hyper-parameters.
            warnings.warn(
                "Passing a monolithic Config to Trainer is deprecated — "
                "build a Paradigm and a TrainingConfig instead, e.g. "
                "Trainer(ARParadigm(ModelConfig(...)), TrainingConfig(...)). ",
                DeprecationWarning,
                stacklevel=2,
            )
            paradigm, legacy_train_cfg = _paradigm_from_legacy_config(paradigm)
            config = config or legacy_train_cfg

        if not isinstance(paradigm, Paradigm):
            raise TypeError(
                f"Trainer expects a Paradigm as its first argument, got "
                f"{type(paradigm).__name__}. Build one first, e.g.:\n"
                "    import dantinox as dx\n"
                "    paradigm = dx.build('ar', dim=256, n_heads=4, head_size=64,\n"
                "                        num_blocks=4, vocab_size=200)\n"
                "    Trainer(paradigm, dx.TrainingConfig(lr=1e-3, epochs=1)).fit('corpus.txt')\n"
                "Architecture fields (model_type, dim, embed_dim, num_blocks, …) "
                "belong to ModelConfig / ELFConfig — TrainingConfig only holds "
                "optimisation and dataset settings."
            )
        self.paradigm = paradigm
        self.config   = config or TrainingConfig()

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(
        self,
        data_source: str | None = None,
        *,
        run_dir: str | None = None,
        rngs: nnx.Rngs | None = None,
        resume: bool = False,
    ) -> str:
        """Train the paradigm on *data_source* and return the checkpoint directory.

        Args:
            data_source : Path to a text file.  May be omitted when the
                          TrainingConfig points at a HuggingFace dataset
                          (``dataset_source="huggingface"``).
            run_dir     : Where to write checkpoints and logs.
                          Defaults to ``runs/<timestamp>``.
            rngs        : Flax NNX random state.  Auto-created from
                          ``config.seed`` when omitted.
            resume      : Continue from the full train state saved in
                          *run_dir* (model, optimizer, epoch).

        Returns:
            Absolute path to the run directory containing the best checkpoint.
        """
        cfg      = self.config
        paradigm = self.paradigm

        if data_source is None:
            if cfg.dataset_source == "huggingface" and cfg.dataset_name:
                data_source = cfg.dataset_name  # cache label; data comes from HF
            else:
                raise ValueError(
                    "fit() needs a data_source path, or a TrainingConfig with "
                    "dataset_source='huggingface' and dataset_name set."
                )
        run_dir  = _make_run_dir(run_dir)
        rngs     = rngs or nnx.Rngs(cfg.seed)

        model_cfg  = _paradigm_config(paradigm)
        seq_len    = _paradigm_seq_len(paradigm)
        sample_len = seq_len + (1 if paradigm.requires_shifted_targets else 0)

        # ── Data (memmapped token cache) ──────────────────────────────────────
        tokenizer, tokens = _load_tokens(data_source, cfg, model_cfg)

        # Sync model vocab_size down to the actual tokenizer vocabulary.
        # _check_vocab raises if tok_vocab > model_vocab; when it is smaller
        # (e.g. char tokenizer with vocab_size=65 but model config says 200),
        # the extra output logits are never trained and cause OOV KeyErrors at
        # decode time.  Clamping here ensures the built model matches the real
        # tokenizer so generation stays within valid token IDs.
        tok_vocab = int(tokenizer.vocab_size)
        if hasattr(model_cfg, "vocab_size") and int(model_cfg.vocab_size) > tok_vocab:
            log.info(
                "Clamping model vocab_size from %d to tokenizer vocab_size %d.",
                model_cfg.vocab_size, tok_vocab,
            )
            model_cfg.vocab_size = tok_vocab  # model_cfg IS paradigm.config

        n_val = int(len(tokens) * cfg.val_frac)
        if n_val < sample_len + 1:
            if cfg.val_frac > 0:
                log.warning(
                    "Validation split too small (%d tokens < %d) — "
                    "falling back to train loss for checkpoint selection.",
                    n_val, sample_len + 1,
                )
            n_val = 0
        train_tokens = tokens[: len(tokens) - n_val]
        val_tokens   = tokens[len(tokens) - n_val:] if n_val else None

        steps_per_epoch = max(len(train_tokens) // (cfg.batch_size * seq_len), 1)
        total_updates   = max(steps_per_epoch * cfg.epochs // cfg.grad_accum, 1)
        log.info("tokens=%d (train=%d, val=%d)  steps/epoch=%d  updates=%d",
                 len(tokens), len(train_tokens), n_val, steps_per_epoch, total_updates)

        # ── Model & optimizer ─────────────────────────────────────────────────
        model = paradigm.build_model(rngs)
        if cfg.use_bf16:
            _cast_params(model, jnp.bfloat16)

        mesh  = make_mesh(cfg.n_devices)
        n_dev = num_devices(mesh)
        if n_dev > 1 and cfg.batch_size % n_dev != 0:
            usable = n_dev
            while cfg.batch_size % usable:
                usable -= 1
            log.warning(
                "batch_size=%d is not divisible across %d devices — "
                "using %d device(s). Pick batch_size as a multiple of the "
                "device count to use them all.",
                cfg.batch_size, n_dev, usable,
            )
            mesh  = make_mesh(usable)
            n_dev = usable
        if n_dev > 1:
            # replicate() operates on array pytrees — push the module state
            # through it and write the replicated arrays back.
            nnx.update(model, replicate(nnx.state(model), mesh))

        optimizer = build_optimizer(model, cfg, total_updates)
        log.info("Parameters: %s", _fmt_params(paradigm.num_parameters(model)))

        # ── Run metadata (config.yaml + tokenizer.json for Generator) ────────
        _save_run_metadata(run_dir, paradigm, cfg, tokenizer, data_source)

        # ── Resume ────────────────────────────────────────────────────────────
        start_epoch, best_loss, no_improve = 1, float("inf"), 0
        state_path = os.path.join(run_dir, _TRAIN_STATE_FILE)
        meta_path  = os.path.join(run_dir, _TRAIN_META_FILE)
        if resume and os.path.exists(state_path) and os.path.exists(meta_path):
            _restore_train_state(state_path, model, optimizer)
            with open(meta_path) as f:
                meta = json.load(f)
            start_epoch = meta["epoch"] + 1
            best_loss   = meta["best_loss"]
            no_improve  = meta.get("no_improve", 0)
            log.info("Resumed from %s — continuing at epoch %d (best=%.4f)",
                     state_path, start_epoch, best_loss)

        # ── Paradigm data hook (e.g. ELF T5 norm stats) ───────────────────────
        np_rng = np.random.default_rng(cfg.seed)
        paradigm.on_train_start(
            model,
            [_sample_batch(train_tokens, cfg.batch_size, sample_len, np_rng)
             for _ in range(4)],
        )

        # ── JIT-compiled steps ────────────────────────────────────────────────
        has_extras = paradigm.provides_batch_extras

        @nnx.jit
        def _step(model: Any, optimizer: nnx.Optimizer, batch: jnp.ndarray,
                  rng: jax.Array) -> jnp.ndarray:
            def _loss(m):
                return paradigm.loss_fn(m, batch, rng)
            (loss, _), grads = nnx.value_and_grad(_loss, has_aux=True)(model)
            optimizer.update(model, grads)
            return loss

        @nnx.jit
        def _step_extras(model: Any, optimizer: nnx.Optimizer, batch: jnp.ndarray,
                         extras: jnp.ndarray, rng: jax.Array) -> jnp.ndarray:
            def _loss(m):
                return paradigm.loss_fn(m, batch, rng, embeddings=extras)
            (loss, _), grads = nnx.value_and_grad(_loss, has_aux=True)(model)
            optimizer.update(model, grads)
            return loss

        @nnx.jit
        def _eval_step(model: Any, batch: jnp.ndarray, rng: jax.Array) -> jnp.ndarray:
            return paradigm.loss_fn(model, batch, rng)[0]

        @nnx.jit
        def _eval_step_extras(model: Any, batch: jnp.ndarray, extras: jnp.ndarray,
                              rng: jax.Array) -> jnp.ndarray:
            return paradigm.loss_fn(model, batch, rng, embeddings=extras)[0]

        def _evaluate() -> float:
            """Average loss over ``eval_iters`` deterministic validation batches."""
            source = val_tokens if val_tokens is not None else train_tokens
            eval_rng_np = np.random.default_rng(cfg.seed + 99_991)
            eval_key    = jax.random.PRNGKey(cfg.seed + 99_991)
            losses = []
            for i in range(cfg.eval_iters):
                batch = jnp.asarray(
                    _sample_batch(source, cfg.batch_size, sample_len, eval_rng_np))
                if n_dev > 1:
                    batch = shard_batch(batch, mesh)
                key = jax.random.fold_in(eval_key, i)
                if has_extras:
                    losses.append(_eval_step_extras(
                        model, batch, paradigm.prepare_batch(batch), key))
                else:
                    losses.append(_eval_step(model, batch, key))
            return float(jnp.mean(jnp.stack(losses)))

        # ── Training loop ─────────────────────────────────────────────────────
        base_key = jax.random.PRNGKey(cfg.seed)
        log_rows: list[dict] = []
        log_every = 10  # host syncs for the progress bar, once per N steps

        for epoch in range(start_epoch, cfg.epochs + 1):
            # Per-epoch derived RNGs make resume deterministic without
            # serialising key state.
            epoch_key = jax.random.fold_in(base_key, epoch)
            np_rng    = np.random.default_rng(cfg.seed + epoch)

            step_losses: list[jnp.ndarray] = []
            pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch}/{cfg.epochs}")
            for step in pbar:
                batch = jnp.asarray(
                    _sample_batch(train_tokens, cfg.batch_size, sample_len, np_rng))
                if n_dev > 1:
                    batch = shard_batch(batch, mesh)
                step_key = jax.random.fold_in(epoch_key, step)

                if has_extras:
                    extras = paradigm.prepare_batch(batch)
                    loss = _step_extras(model, optimizer, batch, extras, step_key)
                else:
                    loss = _step(model, optimizer, batch, step_key)

                step_losses.append(loss)
                if step % log_every == 0:
                    pbar.set_postfix(loss=f"{float(loss):.4f}")

            train_loss = float(jnp.mean(jnp.stack(step_losses)))
            val_loss   = _evaluate()
            log.info("Epoch %d  train_loss=%.4f  val_loss=%.4f",
                     epoch, train_loss, val_loss)
            log_rows.append({"epoch": epoch,
                             "train_loss": train_loss,
                             "val_loss": val_loss})

            if val_loss < best_loss:
                best_loss  = val_loss
                no_improve = 0
                _save_checkpoint(model, run_dir, tag="best")
            else:
                no_improve += 1

            _save_checkpoint(model, run_dir, tag="latest")
            _save_train_state(state_path, model, optimizer)
            with open(meta_path, "w") as f:
                json.dump({"epoch": epoch, "best_loss": best_loss,
                           "no_improve": no_improve}, f)
            _write_log(log_rows, run_dir, cfg.log_file)

            if cfg.patience > 0 and no_improve >= cfg.patience:
                log.info("Early stopping at epoch %d — no val improvement "
                         "for %d epochs.", epoch, cfg.patience)
                break

        log.info("Training complete.  Best val loss: %.4f  Run dir: %s",
                 best_loss, run_dir)
        return run_dir


# ── Legacy Config bridge ──────────────────────────────────────────────────────


def _paradigm_from_legacy_config(cfg: Config) -> tuple[Paradigm, TrainingConfig]:
    """Split a monolithic legacy ``Config`` into (Paradigm, TrainingConfig)."""
    from dantinox.paradigms import (
        ARParadigm,
        ContinuousParadigm,
        DiscreteConfig,
        DiscreteParadigm,
    )

    train_cfg = TrainingConfig.from_dict(cfg.to_dict())
    if cfg.model_type == "elf":
        train_cfg.tokenizer_type = "t5"  # the frozen T5 encoder needs T5 IDs
        return ContinuousParadigm(cfg.to_elf_config()), train_cfg
    if cfg.model_type == "diffusion":
        schedule = (cfg.noise_schedule
                    if cfg.noise_schedule in ("linear", "cosine", "sqrt")
                    else "linear")
        paradigm = DiscreteParadigm(
            cfg.to_model_config(),
            DiscreteConfig(noise_schedule=schedule, mask_token_id=cfg.mask_token_id),
        )
        return paradigm, train_cfg
    return ARParadigm(cfg.to_model_config()), train_cfg


# ── Paradigm introspection helpers ────────────────────────────────────────────


def _paradigm_config(paradigm: Paradigm) -> Any:
    """Return the architecture config of *paradigm* (ModelConfig or ELFConfig)."""
    cfg = getattr(paradigm, "config", None)
    if cfg is None:
        cfg = getattr(paradigm, "model_config", None)
    if cfg is None:
        raise AttributeError(
            f"{type(paradigm).__name__} exposes neither '.config' nor "
            "'.model_config' — the Trainer needs one to read seq_len/vocab_size."
        )
    return cfg


def _paradigm_seq_len(paradigm: Paradigm) -> int:
    cfg = _paradigm_config(paradigm)
    seq = getattr(cfg, "max_context", None)
    if seq is None:
        seq = getattr(cfg, "max_seq_len", None)
    return int(seq) if seq else 512


# ── Run directory & metadata ──────────────────────────────────────────────────


def _make_run_dir(run_dir: str | None) -> str:
    if run_dir is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("runs", ts)
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_run_metadata(
    run_dir: str,
    paradigm: Paradigm,
    cfg: TrainingConfig,
    tokenizer: Any,
    data_source: str,
) -> None:
    """Write config.yaml + tokenizer.json so Generator/quick_generate can
    reload the run without the original corpus."""
    try:
        model_cfg = _paradigm_config(paradigm)
        overrides: dict[str, Any] = {"dataset_name": str(data_source)}
        diff_cfg = getattr(paradigm, "diffusion_config", None)
        if diff_cfg is not None:
            overrides.update(noise_schedule=diff_cfg.noise_schedule,
                             mask_token_id=diff_cfg.mask_token_id)
        mono = Config.from_parts(model_cfg, cfg, **overrides)
        mono.save_yaml(os.path.join(run_dir, "config.yaml"))
    except Exception as exc:  # metadata must never kill a training run
        log.warning("Could not write config.yaml: %s", exc)
    try:
        tokenizer.save(os.path.join(run_dir, "tokenizer.json"))
    except Exception as exc:
        log.warning("Could not write tokenizer.json: %s", exc)


# ── Data loading (tokenise once, then memmap) ─────────────────────────────────


def _token_cache_paths(data_source: str, cfg: TrainingConfig) -> tuple[str, str]:
    """Return (tokens.npy, tokenizer.json) cache paths for *data_source*."""
    if cfg.dataset_source == "huggingface":
        os.makedirs("data", exist_ok=True)
        slug = "_".join(filter(None, (
            cfg.dataset_name.replace("/", "_"),
            cfg.dataset_config,
            cfg.dataset_split,
            cfg.tokenizer_type,
        )))
        base = os.path.join("data", slug)
    else:
        base = f"{data_source}.{cfg.tokenizer_type}"
    return f"{base}.tokens.npy", f"{base}.tokenizer.json"


def _read_corpus(data_source: str, cfg: TrainingConfig) -> str:
    if cfg.dataset_source == "huggingface":
        from datasets import load_dataset
        ds = load_dataset(
            cfg.dataset_name,
            cfg.dataset_config or None,
            split=cfg.dataset_split,
            streaming=cfg.streaming,
        )
        # Rough character budget: tokenisers emit ≲1 token per character, so
        # max_train_tokens * 8 chars comfortably covers BPE/T5 compression.
        char_budget = cfg.max_train_tokens * 8
        parts: list[str] = []
        size = 0
        for row in ds:
            text = row[cfg.dataset_text_field]
            parts.append(text)
            size += len(text)
            if size >= char_budget:
                break
        return "\n".join(parts)

    with open(data_source, encoding="utf-8") as f:
        return f.read()


def _load_tokens(
    data_source: str,
    cfg: TrainingConfig,
    model_cfg: Any,
) -> tuple[Any, np.ndarray]:
    """Tokenise *data_source* once, cache as .npy, and return a memmap.

    Returns:
        (tokenizer, tokens) where tokens is an ``np.memmap``-backed int32
        array — large corpora are never fully resident in RAM.
    """
    from dantinox.utils.tokenizer import get_tokenizer, load_tokenizer_from_file

    npy_path, tok_path = _token_cache_paths(data_source, cfg)
    vocab_size = int(getattr(model_cfg, "vocab_size", 200))

    src_mtime = (os.path.getmtime(data_source)
                 if cfg.dataset_source != "huggingface" and os.path.exists(data_source)
                 else 0.0)
    cache_ok = (
        os.path.exists(npy_path)
        and os.path.exists(tok_path)
        and os.path.getmtime(npy_path) >= src_mtime
        and not cfg.tokenizer_path
    )
    if cache_ok:
        tokenizer = load_tokenizer_from_file(tok_path)
        tokens    = np.load(npy_path, mmap_mode="r")
        log.info("Loaded %d cached tokens from %s", len(tokens), npy_path)
        _check_vocab(tokenizer, model_cfg, cfg)
        return tokenizer, tokens

    if cfg.tokenizer_path:
        tokenizer = load_tokenizer_from_file(cfg.tokenizer_path)
    else:
        t5_name   = getattr(model_cfg, "t5_model_name", "t5-base")
        tokenizer = get_tokenizer(cfg.tokenizer_type, model_name=t5_name)

    text = _read_corpus(data_source, cfg)
    if cfg.tokenizer_type == "char" and not cfg.tokenizer_path:
        tokenizer.train_from_text(text)
    elif cfg.tokenizer_type == "bpe" and not cfg.tokenizer_path:
        tokenizer.train_from_text(text, vocab_size=vocab_size)

    token_ids = np.asarray(
        tokenizer.encode(text)[: cfg.max_train_tokens], dtype=np.int32)
    np.save(npy_path, token_ids)
    tokenizer.save(tok_path)
    log.info("Tokenised %d tokens → %s", len(token_ids), npy_path)

    _check_vocab(tokenizer, model_cfg, cfg)
    return tokenizer, np.load(npy_path, mmap_mode="r")


def _check_vocab(tokenizer: Any, model_cfg: Any, cfg: TrainingConfig) -> None:
    """Reconcile tokenizer vocab with the model's vocab_size."""
    tok_vocab   = int(tokenizer.vocab_size)
    model_vocab = int(getattr(model_cfg, "vocab_size", 0))
    if tok_vocab == model_vocab:
        return
    if tok_vocab > model_vocab:
        if cfg.tokenizer_type == "t5":
            raise ValueError(
                f"Model vocab_size ({model_vocab}) is smaller than the T5 "
                f"tokenizer vocabulary ({tok_vocab}). Use vocab_size=32128 "
                "(t5-small/base/large all pad to 32128)."
            )
        raise ValueError(
            f"Tokenizer produced {tok_vocab} tokens but the model vocab_size "
            f"is {model_vocab}. Increase vocab_size to at least {tok_vocab}."
        )
    log.info("Tokenizer vocab (%d) < model vocab_size (%d) — extra rows "
             "stay unused.", tok_vocab, model_vocab)


def _sample_batch(
    tokens: np.ndarray,
    batch_size: int,
    sample_len: int,
    np_rng: np.random.Generator,
) -> np.ndarray:
    """Draw ``batch_size`` random windows of ``sample_len`` tokens."""
    max_start = max(len(tokens) - sample_len, 1)
    starts = np_rng.integers(0, max_start, size=batch_size)
    rows = np.stack([np.asarray(tokens[s: s + sample_len]) for s in starts])
    return rows.astype(np.int32)


# ── Checkpointing ─────────────────────────────────────────────────────────────


def _cast_params(model: Any, dtype: Any) -> None:
    """Cast all floating-point parameters of *model* to *dtype* in place."""
    state = nnx.state(model, nnx.Param)
    state = jax.tree_util.tree_map(
        lambda x: x.astype(dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x,
        state,
    )
    nnx.update(model, state)


def _save_checkpoint(model: Any, run_dir: str, tag: str) -> None:
    """Model-weights-only checkpoint (portable, hub-compatible)."""
    path = os.path.join(run_dir, f"checkpoint_{tag}.msgpack")
    pure = nnx.state(model, _WEIGHTS).to_pure_dict()
    with open(path, "wb") as f:
        f.write(flax.serialization.msgpack_serialize(pure))


def _save_train_state(path: str, model: Any, optimizer: nnx.Optimizer) -> None:
    """Full train state (model + optimizer) for exact resume."""
    payload = {
        "model": nnx.state(model, _WEIGHTS).to_pure_dict(),
        "opt":   nnx.state(optimizer, _WEIGHTS).to_pure_dict(),
    }
    with open(path, "wb") as f:
        f.write(flax.serialization.msgpack_serialize(payload))


def _msgpack_load(path: str) -> Any:
    """Restore a msgpack checkpoint, allowing the integer map keys that
    ``State.to_pure_dict`` produces for layer lists."""
    import msgpack
    from flax.serialization import _msgpack_ext_unpack
    with open(path, "rb") as f:
        return msgpack.unpackb(
            f.read(), ext_hook=_msgpack_ext_unpack, strict_map_key=False)


def _restore_train_state(path: str, model: Any, optimizer: nnx.Optimizer) -> None:
    raw = _msgpack_load(path)

    model_state = nnx.state(model, _WEIGHTS)
    model_state.replace_by_pure_dict(raw["model"])
    nnx.update(model, model_state)

    opt_state = nnx.state(optimizer, _WEIGHTS)
    opt_state.replace_by_pure_dict(raw["opt"])
    nnx.update(optimizer, opt_state)


# ── Logging helpers ───────────────────────────────────────────────────────────


def _write_log(rows: list[dict], run_dir: str, filename: str) -> None:
    path = os.path.join(run_dir, filename)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _fmt_params(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    return f"{n:,}"
