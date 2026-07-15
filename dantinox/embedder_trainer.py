"""EmbedderTrainer — supervised contrastive training on (anchor, positive) pairs.

Complements ``EmbedderParadigm`` (unsupervised SimCSE with the stock Trainer).
Use this class when you have labelled pairs — it builds batches of anchor/
positive token sequences and optimises InfoNCE.

Data formats accepted by ``fit_pairs``
---------------------------------------
* **List of tuples**: ``[("anchor text", "positive text"), ...]``
* **JSONL file**: one JSON object per line with keys ``"anchor"`` and
  ``"positive"`` (and optionally ``"negative"``)::

      {"anchor": "What is JAX?", "positive": "JAX is a NumPy-like library..."}

* **TSV file**: two columns separated by ``\\t``, first = anchor, second = positive.

Saves a run directory compatible with ``Embedder.from_run()`` — the same
checkpoint format as the stock Trainer.

Quick-start::

    import dantinox as dx

    cfg       = dx.ModelConfig(dim=256, n_heads=4, head_size=64, num_blocks=4,
                               vocab_size=32_000, causal=False)
    paradigm  = dx.EmbedderParadigm(cfg)
    tokenizer = ...   # any dantinox tokenizer

    trainer = dx.EmbedderTrainer(paradigm, tokenizer,
                                  dx.TrainingConfig(lr=3e-4, epochs=5))
    run_dir = trainer.fit_pairs("data/pairs.jsonl")

    embedder = dx.Embedder.from_run(run_dir)
    vecs = embedder.embed(["hello world"])
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any

import flax.serialization
import jax.numpy as jnp
import numpy as np
from flax import nnx
from tqdm import tqdm

from dantinox.core.config import TrainingConfig
from dantinox.paradigms.embedder import EmbedderParadigm, info_nce_loss
from dantinox.training.optimizer import build_optimizer

log = logging.getLogger(__name__)

_WEIGHTS = nnx.Not(nnx.RngState)

PairList = list[tuple[str, str]]


class EmbedderTrainer:
    """Supervised contrastive trainer for (anchor, positive) text pairs.

    Parameters
    ----------
    paradigm:    An ``EmbedderParadigm`` that defines the model architecture
                 and pooling strategy.
    tokenizer:   Any object with ``encode(str) -> list[int]``.
    config:      ``TrainingConfig`` — ``lr``, ``epochs``, ``batch_size``,
                 ``seed``, and ``grad_clip`` are used.  Other fields are ignored.
    """

    def __init__(
        self,
        paradigm: EmbedderParadigm,
        tokenizer: Any,
        config: TrainingConfig | None = None,
    ) -> None:
        self.paradigm  = paradigm
        self.tokenizer = tokenizer
        self.config    = config or TrainingConfig()

    # ── Public API ────────────────────────────────────────────────────────────

    def fit_pairs(
        self,
        data: str | PairList,
        *,
        run_dir: str | None = None,
        hard_negatives: bool = False,
        model: Any | None = None,
    ) -> str:
        """Train on (anchor, positive) pairs and return the run directory.

        Parameters
        ----------
        data:            Path to a JSONL / TSV file **or** a list of
                         ``(anchor, positive)`` string tuples.
        run_dir:         Output directory.  Auto-generated when ``None``.
        hard_negatives:  When ``True`` and the JSONL has a ``"negative"`` key,
                         appends hard negatives to the batch.
        model:           Pre-built ``Transformer`` to fine-tune.  When ``None``
                         a fresh model is initialised from the paradigm config.
                         Pass ``dx.load(run_dir)`` to fine-tune a pretrained model.

        Returns
        -------
        Absolute path to the run directory containing the best checkpoint.
        """
        cfg     = self.config
        pairs   = _load_pairs(data)
        run_dir = run_dir or _make_run_dir()
        os.makedirs(run_dir, exist_ok=True)

        log.info("EmbedderTrainer: %d pairs  run_dir=%s", len(pairs), run_dir)

        # ── Build model + optimizer ───────────────────────────────────────────
        if model is None:
            rngs  = nnx.Rngs(cfg.seed)
            model = self.paradigm.build_model(rngs)
        model.gradient_checkpointing = cfg.gradient_checkpointing

        steps_per_epoch = max(len(pairs) // cfg.batch_size, 1)
        # cfg.epochs may be a genuine float (fractional epochs); step counts
        # and the epoch range below must stay integers.
        total_steps     = int(steps_per_epoch * cfg.epochs)
        optimizer       = build_optimizer(model, cfg, total_steps)

        # ── JIT-compiled step ─────────────────────────────────────────────────
        pooling     = self.paradigm.pooling
        temperature = self.paradigm.temperature

        @nnx.jit
        def _step(
            model: Any,
            optimizer: nnx.Optimizer,
            anchor_ids: jnp.ndarray,
            pos_ids: jnp.ndarray,
        ) -> jnp.ndarray:
            def _loss(m: Any) -> Any:
                out_a = m.encode_hidden(anchor_ids, pooling=pooling,
                                        normalize=True, deterministic=False)
                out_p = m.encode_hidden(pos_ids,    pooling=pooling,
                                        normalize=True, deterministic=False)
                loss = info_nce_loss(out_a.embeddings, out_p.embeddings, temperature)
                return loss, {}
            (loss, _), grads = nnx.value_and_grad(_loss, has_aux=True)(model)
            optimizer.update(model, grads)
            return loss

        # ── Save tokenizer so Embedder.from_run() works ───────────────────────
        _save_tokenizer(self.tokenizer, run_dir)

        # ── Save config ───────────────────────────────────────────────────────
        _save_config(self.paradigm.config, run_dir)

        # ── Training loop ─────────────────────────────────────────────────────
        rng  = np.random.default_rng(cfg.seed)
        best_loss = float("inf")
        max_len   = self.paradigm.config.max_context

        for epoch in range(1, int(cfg.epochs) + 1):
            rng.shuffle(pairs)   # type: ignore[arg-type]
            epoch_losses: list[float] = []

            pbar = tqdm(
                range(0, len(pairs), cfg.batch_size),
                desc=f"Epoch {epoch}/{cfg.epochs}",
            )
            for start in pbar:
                chunk  = pairs[start : start + cfg.batch_size]
                if len(chunk) < 2:
                    continue

                anchors   = [p[0] for p in chunk]
                positives = [p[1] for p in chunk]

                a_ids, _ = _tokenize_pad(anchors,   self.tokenizer, max_len)
                p_ids, _ = _tokenize_pad(positives, self.tokenizer, max_len)

                loss = _step(model, optimizer, a_ids, p_ids)
                epoch_losses.append(float(loss))
                pbar.set_postfix(loss=f"{float(loss):.4f}")

            mean_loss = float(np.mean(epoch_losses))
            log.info("Epoch %d  loss=%.4f", epoch, mean_loss)

            _save_checkpoint(model, run_dir, "latest")
            if mean_loss < best_loss:
                best_loss = mean_loss
                _save_checkpoint(model, run_dir, "best")
                log.info("  → new best (%.4f)", best_loss)

        log.info("Training done — best loss %.4f  run_dir=%s", best_loss, run_dir)
        return os.path.abspath(run_dir)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_pairs(data: str | PairList) -> list[tuple[str, str]]:
    if not isinstance(data, str):
        return list(data)

    path = data
    pairs: list[tuple[str, str]] = []

    if path.endswith(".jsonl") or path.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                pairs.append((obj["anchor"], obj["positive"]))
    else:
        # TSV: anchor \t positive
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    pairs.append((parts[0], parts[1]))

    if not pairs:
        raise ValueError(f"No pairs found in {path!r}")
    return pairs


def _tokenize_pad(
    texts: list[str],
    tokenizer: Any,
    max_len: int,
    pad_id: int = 0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    # Pad every batch to the fixed `max_len` (not the batch's own longest
    # sequence) so `_step`'s @nnx.jit trace is reused across steps instead of
    # recompiling on every shape change.
    encoded = [tokenizer.encode(t)[:max_len] for t in texts]
    T    = max_len
    ids  = np.full((len(encoded), T), pad_id, dtype=np.int32)
    mask = np.zeros((len(encoded), T), dtype=bool)
    for i, e in enumerate(encoded):
        ids[i, : len(e)] = e
        mask[i, : len(e)] = True
    return jnp.asarray(ids), jnp.asarray(mask)


def _make_run_dir() -> str:
    """Return a unique run directory (``_N`` suffix on same-second collisions)."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join("runs", f"embedder_{ts}")
    candidate = base
    for i in range(1, 1000):
        try:
            os.makedirs(candidate, exist_ok=False)
            return candidate
        except FileExistsError:
            candidate = f"{base}_{i}"
    raise RuntimeError(f"Could not create a unique run dir under {base!r}")


def _save_checkpoint(model: Any, run_dir: str, tag: str) -> None:
    path = os.path.join(run_dir, f"checkpoint_{tag}.msgpack")
    pure = nnx.state(model, _WEIGHTS).to_pure_dict()
    with open(path, "wb") as f:
        f.write(flax.serialization.msgpack_serialize(pure))


def _save_tokenizer(tokenizer: Any, run_dir: str) -> None:
    tok_path = os.path.join(run_dir, "tokenizer.json")
    if hasattr(tokenizer, "save"):
        tokenizer.save(tok_path)
    else:
        log.warning(
            "Tokenizer has no .save() method — tokenizer.json not written. "
            "Pass tokenizer= explicitly to Embedder() when loading."
        )


def _save_config(config: Any, run_dir: str) -> None:
    cfg_path = os.path.join(run_dir, "config.yaml")
    if hasattr(config, "save_yaml"):
        config.save_yaml(cfg_path)
    else:
        log.warning(
            "Config has no save_yaml() method — config.yaml not written. "
            "Embedder.from_run() / dx.load() will fail to reload this run."
        )
