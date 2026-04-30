from __future__ import annotations

import csv
import datetime
import json
import logging
import os
import time
from typing import Optional

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from tqdm import tqdm

from core.config import Config
from core.model import Transformer
from dantinox.exceptions import ConfigError
from utils.helpers import compute_loss, get_batch
from utils.tokenizer import get_tokenizer

log = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _build_optimizer(config: Config, total_steps: int) -> optax.GradientTransformation:
    warmup_steps = min(
        getattr(config, "warmup_steps", int(total_steps * 0.1)),
        int(total_steps * 0.3),
    )
    safe_total = max(total_steps, warmup_steps + 1)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.lr,
        warmup_steps=warmup_steps,
        decay_steps=safe_total,
        end_value=config.lr * 0.01,
    )
    name = config.optimizer.lower()
    if name == "adamw":
        return optax.adamw(learning_rate=schedule)
    if name == "adafactor":
        return optax.adafactor(learning_rate=schedule)
    if name == "lion":
        return optax.lion(learning_rate=schedule)
    return optax.adam(learning_rate=schedule)


def _load_text(config: Config, data_path: Optional[str]) -> str:
    if config.dataset_source == "huggingface":
        from datasets import load_dataset
        raw = load_dataset(config.dataset_name, split="train")
        return " ".join(raw["text"])
    path = data_path or config.dataset_name
    if not path:
        raise ConfigError(
            "No data_path provided and config.dataset_name is empty. "
            "Pass data_path to Trainer.fit() or set dataset_name in the config."
        )
    if not os.path.exists(path):
        raise ConfigError(f"Data file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _format_text(text: str) -> str:
    lines = [l.rstrip() for l in text.split("\n") if l.strip()]
    blocks = ["\n".join(lines[i : i + 3]) for i in range(0, len(lines), 3)]
    return "\n\n".join(blocks) + "\n"


def _model_summary(model: Transformer, config: Config, optimizer: nnx.Optimizer) -> dict:
    params = nnx.state(model, nnx.Param)
    total = sum(x.size for x in jax.tree_util.tree_leaves(params))
    opt_state = nnx.state(optimizer)
    opt_params = sum(
        x.size for x in jax.tree_util.tree_leaves(opt_state) if isinstance(x, jax.Array)
    )
    act = config.batch_size * config.max_context * config.dim * config.num_blocks * 8 * 4
    return {
        "total_params_M": round(total / 1e6, 2),
        "weights_mem_MB": round(total * 4 / 1e6, 2),
        "optimizer_mem_MB": round(opt_params * 4 / 1e6, 2),
        "est_activations_MB": round(act / 1e6, 2),
    }


# ── Trainer ───────────────────────────────────────────────────────────────────

class Trainer:
    """
    High-level training interface for DantinoX.

    Parameters
    ----------
    config : Config
        Model and training configuration.

    Examples
    --------
    >>> trainer = Trainer(Config.from_yaml("configs/default_config.yaml"))
    >>> run_dir = trainer.fit("data/corpus.txt")
    """

    def __init__(self, config: Config) -> None:
        self.config = config

    def __repr__(self) -> str:
        return f"Trainer(config={self.config!r})"

    def fit(
        self,
        data_path: Optional[str] = None,
        *,
        run_dir: Optional[str] = None,
        wandb_project: Optional[str] = None,
    ) -> str:
        """
        Train a model and save the checkpoint.

        Parameters
        ----------
        data_path : str, optional
            Path to the training corpus. Falls back to ``config.dataset_name``.
        run_dir : str, optional
            Directory to write the checkpoint and logs. Auto-generated if omitted.
        wandb_project : str, optional
            If provided, metrics are logged to Weights & Biases.

        Returns
        -------
        str
            Path to the run directory containing the saved checkpoint.

        Raises
        ------
        ConfigError
            If the data path is missing or the file cannot be found.
        """
        config = self.config

        if run_dir is None:
            run_dir = os.path.join(
                "runs", datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
            )
        os.makedirs(run_dir, exist_ok=True)
        config.save_yaml(os.path.join(run_dir, "config.yaml"))

        log.info("Run directory: %s", run_dir)

        text = _format_text(_load_text(config, data_path))
        tokenizer = get_tokenizer(config.tokenizer_type)
        if config.tokenizer_type == "char":
            tokenizer.train_from_text(text)
        elif config.tokenizer_type == "bpe":
            tokenizer.train_from_text(text, vocab_size=config.vocab_size)

        config.vocab_size = tokenizer.vocab_size
        full_data = jnp.array(tokenizer.encode(text), dtype=jnp.int32)
        n = int(0.9 * len(full_data))
        train_data, val_data = full_data[:n], full_data[n:]

        tokens_per_step = config.batch_size * config.max_context
        steps_per_epoch = max(1, len(train_data) // tokens_per_step)
        total_steps = steps_per_epoch * config.epochs

        tx = _build_optimizer(config, total_steps)
        rngs = nnx.Rngs(config.seed)
        model = Transformer(config, rngs=rngs)
        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

        summary = _model_summary(model, config, optimizer)
        with open(os.path.join(run_dir, "model_summary.json"), "w") as f:
            json.dump(summary, f, indent=4)
        log.info(
            "Model: %sM params | Est. VRAM: %sMB",
            summary["total_params_M"],
            summary["weights_mem_MB"] + summary["optimizer_mem_MB"] + summary["est_activations_MB"],
        )

        if wandb_project is not None:
            import wandb
            wandb.init(project=wandb_project, config=config.to_dict())

        micro_bs = config.batch_size // config.grad_accum

        @jax.jit
        def train_step(graphdef, state, full_x, full_y):
            model, opt, metrics = nnx.merge(graphdef, state)
            xs = full_x.reshape(config.grad_accum, micro_bs, -1)
            ys = full_y.reshape(config.grad_accum, micro_bs, -1)

            def _loss(model, x, y):
                logits, _, bal = model(x, use_cache=False, kv_caches=None, cache_index=0)
                loss = compute_loss(logits, y)
                if getattr(model, "use_moe", False):
                    loss = loss + model.alpha_balance * bal
                return loss, bal

            grad_fn = nnx.value_and_grad(_loss, has_aux=True)
            acc = jax.tree_util.tree_map(jnp.zeros_like, nnx.state(model, nnx.Param))
            total_loss = jnp.array(0.0)
            total_bal = jnp.array(0.0)
            for i in range(config.grad_accum):
                (loss, bal), grads = grad_fn(model, xs[i], ys[i])
                acc = jax.tree_util.tree_map(
                    lambda a, g: a + g / config.grad_accum, acc, grads
                )
                total_loss += loss / config.grad_accum
                total_bal += bal / config.grad_accum
            opt.update(model, acc)
            metrics.update(loss=total_loss)
            return total_loss, total_bal, nnx.state((model, opt, metrics))

        @nnx.jit
        def eval_step(model, x, y):
            logits, _, bal = model(x, use_cache=False, kv_caches=None, cache_index=0)
            loss = compute_loss(logits, y)
            if getattr(model, "use_moe", False):
                loss = loss + model.alpha_balance * bal
            return loss, bal

        def estimate_loss(key):
            out: dict[str, float] = {}
            for split, d in [("train", train_data), ("val", val_data)]:
                losses, bals = [], []
                for _ in range(config.eval_iters):
                    key, sub = jax.random.split(key)
                    x, y = get_batch(d, 1, config.max_context, sub)
                    l, b = eval_step(model, x, y)
                    losses.append(float(l))
                    bals.append(float(b))
                out[split] = sum(losses) / len(losses)
                out[f"{split}_bal"] = sum(bals) / len(bals)
            return out, key

        log_path = os.path.join(run_dir, "training_log.csv")
        log_f = open(log_path, "a", newline="")
        log_w = csv.writer(log_f)
        if os.path.getsize(log_path) == 0:
            log_w.writerow(
                ["step", "train_loss", "val_loss", "train_bal", "val_bal", "ms_per_step"]
            )

        key = jax.random.PRNGKey(config.seed)
        metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))
        pbar = tqdm(range(total_steps), desc="Training", unit="step", dynamic_ncols=True)
        t0 = time.time()
        try:
            for step in pbar:
                key, sub = jax.random.split(key)
                x, y = get_batch(train_data, config.batch_size, config.max_context, sub)
                graphdef, state = nnx.split((model, optimizer, metrics))
                _, _, new_state = train_step(graphdef, state, x, y)
                nnx.update((model, optimizer, metrics), new_state)

                if step % 50 == 0:
                    t1 = time.time()
                    dt = (t1 - t0) * 1000 / 50
                    t0 = t1
                    losses, key = estimate_loss(key)
                    pbar.set_postfix(
                        train=f"{losses['train']:.4f}",
                        val=f"{losses['val']:.4f}",
                    )
                    log.info(
                        "step %d/%d | train=%.4f val=%.4f bal=%.4f",
                        step, total_steps,
                        losses["train"], losses["val"], losses["train_bal"],
                    )
                    log_w.writerow(
                        [
                            step,
                            float(losses["train"]),
                            float(losses["val"]),
                            float(losses["train_bal"]),
                            float(losses["val_bal"]),
                            round(dt, 2),
                        ]
                    )
                    log_f.flush()
                    if wandb_project is not None:
                        import wandb
                        wandb.log({"train_loss": losses["train"], "val_loss": losses["val"], "step": step})
        finally:
            pbar.close()
            log_f.close()
            if wandb_project is not None:
                import wandb
                wandb.finish()

        weights_path = os.path.join(run_dir, "model_weights.msgpack")
        state_dict = nnx.state(model, nnx.Param).to_pure_dict()
        import flax.serialization
        with open(weights_path, "wb") as f:
            f.write(flax.serialization.msgpack_serialize(state_dict))
        log.info("Checkpoint saved: %s", weights_path)
        return run_dir
