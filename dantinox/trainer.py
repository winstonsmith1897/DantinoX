from __future__ import annotations

import csv
import datetime
import json
import logging
import os
import time

import flax.serialization
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from tqdm import tqdm

from core.config import Config
from core.model import Transformer
from dantinox.exceptions import ConfigError
from utils.helpers import compute_loss, get_batch
from utils.tokenizer import get_tokenizer, load_tokenizer_from_file

log = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _build_schedule(config: Config, total_steps: int) -> optax.Schedule:
    """Return an optax schedule for the requested ``config.lr_schedule``."""
    warmup_steps = min(
        getattr(config, "warmup_steps", int(total_steps * 0.1)),
        int(total_steps * 0.3),
    )
    safe_total = max(total_steps, warmup_steps + 1)
    peak = config.lr
    end  = peak * 0.01

    kind = getattr(config, "lr_schedule", "cosine")

    if kind == "cosine":
        return optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=peak,
            warmup_steps=warmup_steps,
            decay_steps=safe_total,
            end_value=end,
        )

    if kind == "linear":
        warmup = optax.linear_schedule(init_value=0.0, end_value=peak, transition_steps=warmup_steps)
        decay  = optax.linear_schedule(init_value=peak, end_value=end,
                                        transition_steps=safe_total - warmup_steps)
        return optax.join_schedules([warmup, decay], boundaries=[warmup_steps])

    if kind == "constant":
        warmup   = optax.linear_schedule(init_value=0.0, end_value=peak, transition_steps=warmup_steps)
        constant = optax.constant_schedule(peak)
        return optax.join_schedules([warmup, constant], boundaries=[warmup_steps])

    # wsd: warmup → stable (40 % of budget) → cosine decay to end
    stable_steps = int(safe_total * 0.4)
    decay_steps  = safe_total - warmup_steps - stable_steps
    warmup   = optax.linear_schedule(init_value=0.0, end_value=peak, transition_steps=warmup_steps)
    stable   = optax.constant_schedule(peak)
    decay    = optax.cosine_decay_schedule(init_value=peak, decay_steps=max(decay_steps, 1), alpha=end / peak)
    return optax.join_schedules(
        [warmup, stable, decay],
        boundaries=[warmup_steps, warmup_steps + stable_steps],
    )


def _build_optimizer(config: Config, total_steps: int) -> optax.GradientTransformation:
    schedule = _build_schedule(config, total_steps)

    name = config.optimizer.lower()
    if name == "adamw":
        base_opt: optax.GradientTransformation = optax.adamw(learning_rate=schedule)
    elif name == "adafactor":
        base_opt = optax.adafactor(learning_rate=schedule)
    elif name == "lion":
        base_opt = optax.lion(learning_rate=schedule)
    else:
        base_opt = optax.adam(learning_rate=schedule)

    grad_clip = getattr(config, "grad_clip", 0.0)
    if grad_clip > 0:
        return optax.chain(optax.clip_by_global_norm(grad_clip), base_opt)
    return base_opt


def _load_text(config: Config, data_path: str | None) -> str:
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
    with open(path, encoding="utf-8") as f:
        return f.read()


def _format_text(text: str) -> str:
    lines = [line.rstrip() for line in text.split("\n") if line.strip()]
    blocks = ["\n".join(lines[i : i + 3]) for i in range(0, len(lines), 3)]
    return "\n\n".join(blocks) + "\n"


def _model_summary(model: Transformer, config: Config, optimizer: nnx.Optimizer) -> dict:
    params = nnx.state(model, nnx.Param)
    total = sum(x.size for x in jax.tree_util.tree_leaves(params))
    opt_state = nnx.state(optimizer)
    opt_params = sum(
        x.size for x in jax.tree_util.tree_leaves(opt_state) if isinstance(x, jax.Array)
    )
    bpp = 2 if getattr(config, "use_bf16", False) else 4
    act = config.batch_size * config.max_context * config.dim * config.num_blocks * 8 * bpp
    return {
        "total_params_M": round(total / 1e6, 2),
        "dtype": "bfloat16" if bpp == 2 else "float32",
        "weights_mem_MB": round(total * bpp / 1e6, 2),
        "optimizer_mem_MB": round(opt_params * bpp / 1e6, 2),
        "est_activations_MB": round(act / 1e6, 2),
    }


def _cast_params(model: Transformer, dtype: jnp.dtype) -> None:
    params = nnx.state(model, nnx.Param)
    nnx.update(
        model,
        jax.tree_util.tree_map(
            lambda x: x.astype(dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x,
            params,
        ),
    )


def _save_weights(model: Transformer, path: str) -> None:
    state_dict = nnx.state(model, nnx.Param).to_pure_dict()
    with open(path, "wb") as f:
        f.write(flax.serialization.msgpack_serialize(state_dict))


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
        data_path: str | None = None,
        *,
        run_dir: str | None = None,
        wandb_project: str | None = None,
        resume: bool = False,
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
        resume : bool
            If ``True`` and a previous checkpoint exists in ``run_dir``, training
            resumes from the saved step. Optimizer state is not preserved.

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

        tok_path = os.path.join(run_dir, "tokenizer.json")
        if resume and os.path.exists(tok_path):
            tokenizer = load_tokenizer_from_file(tok_path)
            log.info("Resumed tokenizer from %s", tok_path)
        else:
            tokenizer = get_tokenizer(config.tokenizer_type)
            if config.tokenizer_type == "char":
                tokenizer.train_from_text(text)
            elif config.tokenizer_type == "bpe":
                tokenizer.train_from_text(text, vocab_size=config.vocab_size)
            tokenizer.save(tok_path)
            log.info("Tokenizer saved to %s", tok_path)

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
        if getattr(config, "use_bf16", False):
            _cast_params(model, jnp.bfloat16)
            log.info("Model cast to bfloat16")
        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

        start_step = 0
        cursor_path = os.path.join(run_dir, "training_cursor.json")
        resume_weights = os.path.join(run_dir, "model_weights.msgpack")
        if resume and os.path.exists(cursor_path) and os.path.exists(resume_weights):
            with open(cursor_path) as cursor_f:
                cursor = json.load(cursor_f)
            start_step = int(cursor.get("step", 0)) + 1
            import msgpack
            from flax.serialization import _msgpack_ext_unpack
            with open(resume_weights, "rb") as weights_f:
                state_dict = msgpack.unpackb(
                    weights_f.read(), ext_hook=_msgpack_ext_unpack, strict_map_key=False
                )
            nnx.update(model, state_dict)
            log.info("Resumed training from step %d", start_step)

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
            wandb.init(project=wandb_project, config=config.to_dict())  # type: ignore[attr-defined]

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
                    loss_val, b = eval_step(model, x, y)
                    losses.append(float(loss_val))
                    bals.append(float(b))
                out[split] = sum(losses) / len(losses)
                out[f"{split}_bal"] = sum(bals) / len(bals)
            return out, key

        log_path = os.path.join(run_dir, "training_log.csv")
        weights_path = os.path.join(run_dir, "model_weights.msgpack")
        best_weights_path = os.path.join(run_dir, "best_model_weights.msgpack")

        key = jax.random.PRNGKey(config.seed)
        metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))
        pbar = tqdm(
            range(start_step, total_steps),
            desc="Training",
            unit="step",
            dynamic_ncols=True,
            initial=start_step,
            total=total_steps,
        )
        t0 = time.time()

        patience = getattr(config, "patience", 0)
        best_val_loss = float("inf")
        no_improve = 0

        with open(log_path, "a", newline="") as log_f:
            log_w = csv.writer(log_f)
            if os.path.getsize(log_path) == 0:
                log_w.writerow(
                    ["step", "train_loss", "val_loss", "train_bal", "val_bal", "ms_per_step"]
                )
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
                        val_loss = losses["val"]
                        pbar.set_postfix(
                            train=f"{losses['train']:.4f}",
                            val=f"{val_loss:.4f}",
                        )
                        log.info(
                            "step %d/%d | train=%.4f val=%.4f bal=%.4f",
                            step, total_steps,
                            losses["train"], val_loss, losses["train_bal"],
                        )
                        log_w.writerow(
                            [
                                step,
                                float(losses["train"]),
                                float(val_loss),
                                float(losses["train_bal"]),
                                float(losses["val_bal"]),
                                round(dt, 2),
                            ]
                        )
                        log_f.flush()

                        # Periodic checkpoint for resume
                        _save_weights(model, weights_path)
                        with open(cursor_path, "w") as cf:
                            json.dump({"step": step}, cf)

                        # Best checkpoint tracking
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            no_improve = 0
                            _save_weights(model, best_weights_path)
                            log.info("New best val loss %.4f — saved best checkpoint", best_val_loss)
                        else:
                            no_improve += 1
                            if patience > 0 and no_improve >= patience:
                                log.info(
                                    "Early stopping at step %d (no improvement for %d evals)",
                                    step, patience,
                                )
                                break

                        if wandb_project is not None:
                            import wandb
                            wandb.log({"train_loss": losses["train"], "val_loss": val_loss, "step": step})  # type: ignore[attr-defined]
            finally:
                pbar.close()
                if wandb_project is not None:
                    import wandb
                    wandb.finish()  # type: ignore[attr-defined]

        _save_weights(model, weights_path)
        log.info("Checkpoint saved: %s", weights_path)
        return run_dir

    def find_lr(
        self,
        data_path: str | None = None,
        *,
        min_lr: float = 1e-7,
        max_lr: float = 1.0,
        num_steps: int = 100,
        smoothing: float = 0.9,
    ) -> tuple[float, list[float], list[float]]:
        """
        LR range test (Smith 2015).

        Trains for ``num_steps`` steps while exponentially increasing the
        learning rate from ``min_lr`` to ``max_lr``.  Returns a tuple of
        ``(suggested_lr, lr_history, loss_history)``.

        Parameters
        ----------
        data_path : str, optional
            Path to the training corpus.
        min_lr : float
            Starting learning rate (default 1e-7).
        max_lr : float
            Maximum learning rate (default 1.0).
        num_steps : int
            Number of steps in the sweep (default 100).
        smoothing : float
            Exponential smoothing factor for the loss curve (default 0.9).

        Returns
        -------
        tuple[float, list[float], list[float]]
            ``(suggested_lr, lr_history, loss_history)``
        """
        import math

        config = self.config
        text = _format_text(_load_text(config, data_path))

        tokenizer = get_tokenizer(config.tokenizer_type)
        if config.tokenizer_type == "char":
            tokenizer.train_from_text(text)
        elif config.tokenizer_type == "bpe":
            tokenizer.train_from_text(text, vocab_size=config.vocab_size)

        config.vocab_size = tokenizer.vocab_size
        full_data = jnp.array(tokenizer.encode(text), dtype=jnp.int32)
        train_data = full_data[: int(0.9 * len(full_data))]

        rngs = nnx.Rngs(config.seed)
        model = Transformer(config, rngs=rngs)
        if getattr(config, "use_bf16", False):
            _cast_params(model, jnp.bfloat16)

        log_multiplier = math.log(max_lr / min_lr) / max(1, num_steps - 1)

        def _lr_fn(step: jnp.ndarray) -> jnp.ndarray:
            return jnp.array(min_lr, jnp.float32) * jnp.exp(
                step.astype(jnp.float32) * jnp.array(log_multiplier, jnp.float32)
            )

        tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=_lr_fn))
        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

        @nnx.jit
        def _step(model, opt, x, y):
            def loss_fn(m):
                logits, _, _ = m(x, use_cache=False, kv_caches=None, cache_index=0)
                return compute_loss(logits, y)

            loss, grads = nnx.value_and_grad(loss_fn)(model)
            opt.update(model, grads)
            return loss

        key = jax.random.PRNGKey(config.seed)
        lr_history: list[float] = []
        loss_history: list[float] = []
        smooth_loss = 0.0
        best_loss = float("inf")

        pbar = tqdm(range(num_steps), desc="LR finder", unit="step", dynamic_ncols=True)
        for step in pbar:
            key, sub = jax.random.split(key)
            x, y = get_batch(train_data, config.batch_size, config.max_context, sub)
            loss_val = float(_step(model, optimizer, x, y))

            smooth_loss = (
                loss_val if step == 0
                else smoothing * smooth_loss + (1 - smoothing) * loss_val
            )
            debiased = smooth_loss / (1 - smoothing ** (step + 1))

            current_lr = float(_lr_fn(jnp.array(step)))
            lr_history.append(current_lr)
            loss_history.append(debiased)

            pbar.set_postfix(lr=f"{current_lr:.2e}", loss=f"{debiased:.4f}")

            if debiased < best_loss:
                best_loss = debiased
            if debiased > 4 * best_loss:
                log.info("Loss diverging at step %d — stopping sweep early", step)
                break

        pbar.close()

        if len(loss_history) > 2:
            slopes = [loss_history[i + 1] - loss_history[i] for i in range(len(loss_history) - 1)]
            suggested_lr = lr_history[min(range(len(slopes)), key=lambda i: slopes[i])]
        else:
            suggested_lr = min_lr

        log.info(
            "LR finder done — suggested lr=%.2e (sweep range [%.2e, %.2e])",
            suggested_lr, min_lr, max_lr,
        )
        return suggested_lr, lr_history, loss_history
