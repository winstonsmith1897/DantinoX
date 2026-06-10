from __future__ import annotations

import csv
import datetime
import logging
import os
import time
from pathlib import Path
from typing import Any

import flax.serialization
import jax
import jax.numpy as jnp
import msgpack
from flax import nnx
from tqdm import tqdm

from core.config import TrainingConfig
from core.sharding import make_mesh, num_devices, replicate, shard_batch
from dantinox.paradigms.base import Paradigm
from dantinox.training.optimizer import build_optimizer

log = logging.getLogger(__name__)


class Trainer:
    """Paradigm-agnostic training harness.

    The Trainer is completely decoupled from model type.  It calls
    ``paradigm.loss_fn(model, batch, rng)`` — nothing else.  All
    paradigm-specific behaviour (masking, noise schedules, ELF branches,
    CFG) lives in the Paradigm, not here.

    Quick-start::

        from dantinox import ARParadigm, ModelConfig, TrainingConfig, Trainer

        paradigm = ARParadigm(ModelConfig(dim=512, n_heads=8, head_size=64,
                                          num_blocks=12, vocab_size=32_000))
        trainer  = Trainer(paradigm, TrainingConfig(lr=3e-4, epochs=5))
        run_dir  = trainer.fit("data/corpus.txt")
    """

    def __init__(
        self,
        paradigm: Paradigm,
        config: TrainingConfig,
    ) -> None:
        self.paradigm = paradigm
        self.config   = config

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(
        self,
        data_source: str,
        *,
        run_dir: str | None = None,
        rngs: nnx.Rngs | None = None,
    ) -> str:
        """Train the paradigm on *data_source* and return the checkpoint directory.

        Args:
            data_source : Path to a text file, or a HuggingFace dataset name.
            run_dir     : Where to write checkpoints and logs.
                          Defaults to ``runs/<timestamp>``.
            rngs        : Flax NNX random state.  Auto-created from
                          ``config.seed`` when omitted.

        Returns:
            Absolute path to the run directory containing the best checkpoint.
        """
        cfg      = self.config
        run_dir  = _make_run_dir(run_dir)
        rngs     = rngs or nnx.Rngs(cfg.seed)

        # ── Data ──────────────────────────────────────────────────────────────
        tokens = _load_tokens(data_source, cfg)
        total_tokens = len(tokens)
        steps_per_epoch = max(total_tokens // (cfg.batch_size * _seq_len(cfg)), 1)
        total_steps = steps_per_epoch * cfg.epochs
        log.info("tokens=%d  steps/epoch=%d  total_steps=%d",
                 total_tokens, steps_per_epoch, total_steps)

        # ── Model & optimizer ─────────────────────────────────────────────────
        model     = self.paradigm.build_model(rngs)
        optimizer = build_optimizer(model, cfg, total_steps)
        n_params  = self.paradigm.num_parameters(model)
        log.info("Parameters: %s", _fmt_params(n_params))

        # ── Multi-device ──────────────────────────────────────────────────────
        mesh    = make_mesh(cfg.n_devices)
        n_dev   = num_devices(mesh)
        if n_dev > 1:
            model = replicate(model, mesh)

        # ── JIT-compiled step ─────────────────────────────────────────────────
        @nnx.jit
        def _step(
            model: Any,
            optimizer: nnx.Optimizer,
            batch: jnp.ndarray,
            rng: jax.random.KeyArray,
        ) -> tuple[jnp.ndarray, dict[str, Any]]:
            def _loss(m):
                return self.paradigm.loss_fn(m, batch, rng)

            (loss, metrics), grads = nnx.value_and_grad(_loss, has_aux=True)(model)
            optimizer.update(grads)
            return loss, metrics

        # ── Training loop ─────────────────────────────────────────────────────
        rng        = jax.random.PRNGKey(cfg.seed)
        best_loss  = float("inf")
        log_rows: list[dict] = []
        seq_len    = _seq_len(cfg)

        for epoch in range(1, cfg.epochs + 1):
            epoch_loss = 0.0
            pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch}/{cfg.epochs}")

            for step in pbar:
                rng, rng_batch, rng_step = jax.random.split(rng, 3)

                # sample a batch
                batch = _sample_batch(tokens, cfg.batch_size, seq_len, rng_batch)
                if n_dev > 1:
                    batch = shard_batch(batch, mesh)

                loss, metrics = _step(model, optimizer, batch, rng_step)
                loss_val = float(loss)
                epoch_loss += loss_val
                pbar.set_postfix(loss=f"{loss_val:.4f}")

            avg_loss = epoch_loss / steps_per_epoch
            log.info("Epoch %d  avg_loss=%.4f", epoch, avg_loss)
            log_rows.append({"epoch": epoch, "loss": avg_loss})

            if avg_loss < best_loss:
                best_loss = avg_loss
                _save_checkpoint(model, run_dir, tag="best")

            _save_checkpoint(model, run_dir, tag="latest")

        _write_log(log_rows, run_dir, cfg.log_file)
        log.info("Training complete.  Best loss: %.4f  Run dir: %s", best_loss, run_dir)
        return run_dir


# ── Internal helpers ──────────────────────────────────────────────────────────


def _make_run_dir(run_dir: str | None) -> str:
    if run_dir is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("runs", ts)
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    return run_dir


def _seq_len(cfg: TrainingConfig) -> int:
    return getattr(cfg, "max_context", 512)


def _load_tokens(data_source: str, cfg: TrainingConfig) -> list[int]:
    """Load and tokenize *data_source* into a flat list of token IDs."""
    from utils.tokenizer import get_tokenizer, load_tokenizer_from_file

    if cfg.tokenizer_path:
        tokenizer = load_tokenizer_from_file(cfg.tokenizer_path)
    else:
        tokenizer = get_tokenizer(cfg.tokenizer_type, vocab_size=getattr(cfg, "vocab_size", 200))

    if cfg.dataset_source == "huggingface":
        from datasets import load_dataset
        ds = load_dataset(
            cfg.dataset_name,
            cfg.dataset_config or None,
            split=cfg.dataset_split,
            streaming=cfg.streaming,
        )
        texts = (row[cfg.dataset_text_field] for row in ds)
    else:
        with open(data_source, encoding="utf-8") as f:
            texts = [f.read()]

    tokens: list[int] = []
    for text in texts:
        tokens.extend(tokenizer.encode(text))
        if len(tokens) >= cfg.max_train_tokens:
            break
    return tokens[: cfg.max_train_tokens]


def _sample_batch(
    tokens: list[int],
    batch_size: int,
    seq_len: int,
    rng: jax.random.KeyArray,
) -> jnp.ndarray:
    max_start = max(len(tokens) - seq_len - 1, 1)
    starts = jax.random.randint(rng, (batch_size,), 0, max_start)
    rows = [tokens[s : s + seq_len + 1] for s in starts.tolist()]
    return jnp.array(rows, dtype=jnp.int32)


def _save_checkpoint(model: Any, run_dir: str, tag: str) -> None:
    path = os.path.join(run_dir, f"checkpoint_{tag}.msgpack")
    state = nnx.state(model)
    flat  = flax.serialization.to_bytes(state)
    with open(path, "wb") as f:
        f.write(flat)


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
