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
from flax.nnx.transforms.autodiff import DiffState
from tqdm import tqdm

from core.config import Config
from core.diffusion import NoiseSchedule, corrupt, make_noise_schedule, masked_cross_entropy
from core.elf import ELFTransformer, elf_loss
from core.lora import LoRAParam
from core.model import DiffusionTransformer, Transformer
from core.sharding import make_mesh, num_devices, replicate, shard_batch
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
    elif name == "muon":
        # Muon (Momentum Orthogonalized by Newton-Schulz): applies Newton-Schulz
        # orthogonalization to 2-D param gradients; falls back to Adam for biases
        # and norms.  optax.contrib.muon, available since optax 0.2.6.
        base_opt = optax.contrib.muon(learning_rate=schedule)
    else:
        base_opt = optax.adam(learning_rate=schedule)

    grad_clip = getattr(config, "grad_clip", 0.0)
    if grad_clip > 0:
        return optax.chain(optax.clip_by_global_norm(grad_clip), base_opt)
    return base_opt


def _load_text(config: Config, data_path: str | None) -> str:
    if config.dataset_source == "huggingface":
        from datasets import load_dataset  # type: ignore[import]

        name = config.dataset_name
        subset = getattr(config, "dataset_config", "") or None
        split  = getattr(config, "dataset_split", "train") or "train"
        field  = getattr(config, "dataset_text_field", "text") or "text"

        load_kw: dict = {"split": split, "streaming": config.streaming}
        if subset:
            load_kw["name"] = subset

        raw = load_dataset(name, **load_kw)

        if config.streaming:
            # IterableDataset — materialise to a single string
            parts = [ex[field] for ex in raw if field in ex and ex[field]]
        else:
            parts = [t for t in raw[field] if t]

        if not parts:
            raise ConfigError(
                f"No text found in column '{field}' of '{name}'. "
                f"Set dataset_text_field to the correct column name."
            )
        return " ".join(parts)

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


def _create_model(
    config: Config, rngs: nnx.Rngs
) -> Transformer | DiffusionTransformer | ELFTransformer:
    """Instantiate the right model class from config.model_type."""
    if config.model_type == "elf":
        return ELFTransformer(config.to_elf_config(), rngs=rngs)
    if config.model_type == "diffusion":
        return DiffusionTransformer(config, rngs=rngs)
    return Transformer(config, rngs=rngs)


def _model_summary(model: Transformer | DiffusionTransformer | ELFTransformer, config: Config, optimizer: nnx.Optimizer) -> dict:
    params = nnx.state(model, nnx.Param)
    total = sum(x.size for x in jax.tree_util.tree_leaves(params))
    opt_state = nnx.state(optimizer)
    opt_params = sum(
        x.size for x in jax.tree_util.tree_leaves(opt_state) if isinstance(x, jax.Array)
    )
    bpp = 2 if getattr(config, "use_bf16", False) else 4
    grad_accum = getattr(config, "grad_accum", 1)
    micro_bs = max(1, config.batch_size // grad_accum)
    # activations × grad_accum because the Python for loop inside @nnx.jit is
    # fully unrolled by XLA, so all micro-batch activation graphs coexist in
    # the compiled program simultaneously. gradient_checkpointing=True mitigates
    # this by recomputing block internals on the backward pass.
    act = micro_bs * config.max_context * config.dim * config.num_blocks * 8 * bpp * grad_accum
    return {
        "total_params_M": round(total / 1e6, 2),
        "dtype": "bfloat16" if bpp == 2 else "float32",
        "weights_mem_MB": round(total * bpp / 1e6, 2),
        "optimizer_mem_MB": round(opt_params * bpp / 1e6, 2),
        "est_activations_MB": round(act / 1e6, 2),
    }


def _cast_params(model: Transformer | ELFTransformer, dtype: jnp.dtype) -> None:
    params = nnx.state(model, nnx.Param)
    nnx.update(
        model,
        jax.tree_util.tree_map(
            lambda x: x.astype(dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x,
            params,
        ),
    )


def _save_weights(model: Transformer | DiffusionTransformer | ELFTransformer, path: str) -> None:
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

        # ── Dataset loading with tokenised-array cache ────────────────────────
        # The first run for a given (dataset, tokenizer_type) downloads and
        # tokenises the text, then saves:
        #   data/<dataset_slug>_<tok_type>.npy    ← token ID array (int32)
        #   data/<dataset_slug>_<tok_type>.json   ← shared tokenizer
        # Subsequent runs load directly from these files — no HF download,
        # no re-tokenisation. Reduces per-run overhead from ~60s to ~2s.
        import hashlib, numpy as _np

        _data_dir = os.path.join(os.path.dirname(run_dir) if run_dir else ".", "..", "data")
        _data_dir = os.path.normpath(_data_dir)
        os.makedirs(_data_dir, exist_ok=True)

        # Cache key: dataset name + tokenizer type
        _ds_key  = f"{config.dataset_name.replace('/', '_')}_{config.dataset_config or 'default'}_{config.tokenizer_type}"
        _arr_cache = os.path.join(_data_dir, f"{_ds_key}.npy")
        _tok_cache = os.path.join(_data_dir, f"{_ds_key}.json")

        tok_path = os.path.join(run_dir, "tokenizer.json")

        os.makedirs(run_dir, exist_ok=True)

        if os.path.exists(_arr_cache) and os.path.exists(_tok_cache):
            # Fast path: load pre-tokenised array and shared tokenizer
            log.info("Loading tokenised data from cache: %s", _arr_cache)
            tokenizer   = load_tokenizer_from_file(_tok_cache)
            full_data   = jnp.array(_np.load(_arr_cache))
            config.vocab_size = tokenizer.vocab_size
            # Copy shared tokenizer to run dir for inference
            import shutil as _shutil
            _shutil.copy2(_tok_cache, tok_path)
        else:
            # Slow path: download, tokenise, save cache
            text = _format_text(_load_text(config, data_path))
            if resume and os.path.exists(tok_path):
                tokenizer = load_tokenizer_from_file(tok_path)
                log.info("Resumed tokenizer from %s", tok_path)
            else:
                _t5_name = getattr(config, "t5_model_name", "t5-base")
                tokenizer = get_tokenizer(config.tokenizer_type, model_name=_t5_name)
                if config.tokenizer_type == "char":
                    tokenizer.train_from_text(text)
                elif config.tokenizer_type == "bpe":
                    tokenizer.train_from_text(text, vocab_size=config.vocab_size)
                # t5: pre-trained, no training needed
                tokenizer.save(tok_path)
                log.info("Tokenizer saved to %s", tok_path)
            config.vocab_size = tokenizer.vocab_size
            ids = tokenizer.encode(text)
            full_data = jnp.array(ids, dtype=jnp.int32)
            # Persist cache for all subsequent runs
            _np.save(_arr_cache, _np.array(ids, dtype=_np.int32))
            tokenizer.save(_tok_cache)
            log.info("Tokenised data cached → %s  (%d tokens)", _arr_cache, len(ids))

        # Optional cap: use only the first max_train_tokens tokens.
        # Keeps each run to a fixed compute budget regardless of corpus size.
        # Default 10_000_000 → ~822 steps per run on WikiText-103 (≈7 min on 2×A100).
        _max_tok = getattr(config, "max_train_tokens", 10_000_000)
        if _max_tok > 0 and len(full_data) > _max_tok:
            log.info("Capping dataset to %d tokens (full: %d)", _max_tok, len(full_data))
            full_data = full_data[:_max_tok]

        n = int(0.9 * len(full_data))
        train_data, val_data = full_data[:n], full_data[n:]

        tokens_per_step = config.batch_size * config.max_context
        steps_per_epoch = max(1, len(train_data) // tokens_per_step)
        total_steps = steps_per_epoch * config.epochs

        tx = _build_optimizer(config, total_steps)
        rngs = nnx.Rngs(config.seed)
        model = _create_model(config, rngs)
        if getattr(config, "use_bf16", False):
            _cast_params(model, jnp.bfloat16)
            log.info("Model cast to bfloat16")

        is_diffusion = config.model_type == "diffusion"
        is_elf       = config.model_type == "elf"
        schedule: NoiseSchedule | None = make_noise_schedule(config) if is_diffusion else None

        # LoRA: only train adapter params; base weights are frozen nnx.Param
        wrt_type = LoRAParam if getattr(config, "use_lora", False) else nnx.Param
        optimizer = nnx.Optimizer(model, tx, wrt=wrt_type)

        # Multi-GPU: build a data-parallel mesh when more than one device is requested
        n_dev_cfg = getattr(config, "n_devices", 0)
        import jax as _jax
        n_local = len(_jax.local_devices())
        use_multi_gpu = (n_dev_cfg != 1) and n_local > 1
        mesh = make_mesh(n_dev_cfg) if use_multi_gpu else None
        if use_multi_gpu:
            n_dev = num_devices(mesh)  # type: ignore[arg-type]
            if config.batch_size % n_dev != 0:
                raise ConfigError(
                    f"batch_size ({config.batch_size}) must be divisible by "
                    f"n_devices ({n_dev}) for data-parallel training."
                )
            log.info("Data-parallel training on %d devices", n_dev)

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
        _wrt = wrt_type  # captured in closure for JIT

        if is_elf:
            # ── ELF continuous flow-matching training step ────────────────────
            # Each step:
            #   1. T5 contextual encoder runs OUTSIDE JIT → embeddings [B, L, E]
            #   2. JIT-compiled step normalizes embeddings and runs elf_loss
            #   3. elf_loss routes to denoiser (MSE) or decoder (CE) branch
            #
            # Keeping T5 encoder outside JIT avoids retracing its large graph
            # and lets it run on its own XLA computation.  At inference the
            # ELF model generates from Gaussian noise — T5 is never called.
            _elf_config = config.to_elf_config()

            # Initialize contextual T5 encoder (not a JAX module, never updated)
            from utils.t5_encoder import T5ContextualEncoder
            log.info("Loading T5 contextual encoder: %s", config.t5_model_name)
            _t5_encoder = T5ContextualEncoder(config.t5_model_name)
            log.info("T5 encoder loaded — hidden_dim=%d", _t5_encoder.hidden_dim)

            # Compute channel-wise norm stats from a few training batches and
            # store them in the model's ELFEmbedder so JIT-compiled steps can
            # normalize consistently.
            log.info("Computing T5 embedding normalization statistics …")
            _stat_key = jax.random.PRNGKey(0)
            _stat_batches: list[jnp.ndarray] = []
            for _ in range(4):
                _stat_key, _sub = jax.random.split(_stat_key)
                _xb, _ = get_batch(train_data, config.batch_size, config.max_context, _sub)
                _stat_batches.append(_xb)
            _emb_mean, _emb_std = _t5_encoder.compute_norm_stats(_stat_batches)
            model.embedder.emb_mean.value = _emb_mean
            model.embedder.emb_std.value  = _emb_std
            log.info(
                "Norm stats set — mean |μ|=%.4f  mean σ=%.4f",
                float(jnp.abs(_emb_mean).mean()),
                float(_emb_std.mean()),
            )

            @nnx.jit
            def train_step(model, opt, metrics, full_emb, full_x, key):  # type: ignore[misc]
                E    = full_emb.shape[-1]
                embs = full_emb.reshape(config.grad_accum, micro_bs, -1, E)
                xs   = full_x.reshape(config.grad_accum, micro_bs, -1)

                def _loss(model, emb_i, x_i, key):
                    embeddings = model.encode(emb_i)  # normalize: [B, L, E]
                    loss, aux  = elf_loss(model, embeddings, x_i, key, _elf_config)
                    return loss, aux["den_loss"]

                grad_fn    = nnx.value_and_grad(_loss, argnums=DiffState(0, _wrt), has_aux=True)
                acc        = jax.tree_util.tree_map(jnp.zeros_like, nnx.state(model, _wrt))
                total_loss = jnp.array(0.0)
                total_bal  = jnp.array(0.0)
                for i in range(config.grad_accum):
                    key, sub = jax.random.split(key)
                    (loss, bal), grads = grad_fn(model, embs[i], xs[i], sub)
                    acc = jax.tree_util.tree_map(
                        lambda a, g: a + g / config.grad_accum, acc, grads
                    )
                    total_loss += loss / config.grad_accum
                    total_bal  += bal  / config.grad_accum
                opt.update(model, acc)
                metrics.update(loss=total_loss)
                return total_loss, total_bal, key

            @nnx.jit
            def eval_step(model, emb, x, key):  # type: ignore[misc]
                embeddings = model.encode(emb)  # normalize
                loss, aux  = elf_loss(model, embeddings, x, key, _elf_config)
                return loss, aux["den_loss"], key

            def estimate_loss(key: jax.Array) -> tuple[dict[str, float], jax.Array]:
                out: dict[str, float] = {}
                for split, d in [("train", train_data), ("val", val_data)]:
                    losses, bals = [], []
                    for _ in range(config.eval_iters):
                        key, sub_b = jax.random.split(key)
                        x, _ = get_batch(d, config.batch_size, config.max_context, sub_b)
                        emb = _t5_encoder.encode(x)  # outside JIT
                        key, sub_l = jax.random.split(key)
                        loss_val, b, _ = eval_step(model, emb, x, sub_l)
                        losses.append(float(loss_val))
                        bals.append(float(b))
                    out[split]          = sum(losses) / len(losses)
                    out[f"{split}_bal"] = sum(bals)   / len(bals)
                return out, key

        elif is_diffusion:
            # ── LLaDA-style diffusion training step ───────────────────────────
            # Follows arXiv:2502.09992:
            #   • t ~ U[t_min, 1] continuous — mask rate directly
            #   • p_mask(t) depends on noise_schedule (linear: p_mask=t)
            #   • Loss = (1/t) * Σ_masked nll / L  (ELBO weight, Eq. 3)
            #   • Model is time-free — no t passed to model.__call__
            _noise_schedule = getattr(config, "noise_schedule", "linear")
            # t_min=0.05 prevents extreme gradient variance: at t=1/L≈0.002 a
            # single masked token gets a 1/t≈500 weight that dominates the step.
            # Floor at 0.05 (≈26 masked tokens per seq) keeps the gradient stable
            # while still covering the full denoising range.
            _t_min = max(1.0 / max(config.max_context, 1), 0.05)

            @nnx.jit
            def train_step(model, opt, metrics, full_x, _unused_y, key):  # type: ignore[misc]
                xs = full_x.reshape(config.grad_accum, micro_bs, -1)

                # Sample t independently per sequence (LLaDA §3): each of the
                # micro_bs sequences gets its own noise level.  With the 1/t loss
                # all noise levels contribute equally in expectation, so mixing
                # multiple t values per micro-batch is safe and reduces gradient
                # variance relative to using one t for the whole optimizer step.
                def _loss(model, x, key):
                    key, sub_t = jax.random.split(key)
                    t_batch = jax.random.uniform(sub_t, (micro_bs,), minval=_t_min, maxval=1.0)
                    key, sub_c = jax.random.split(key)
                    x_t = corrupt(x, t_batch, sub_c, _noise_schedule, config.mask_token_id)
                    out = model(x_t, deterministic=False)  # time-free: no t arg
                    loss = masked_cross_entropy(
                        out.logits, x, x_t, config.mask_token_id,
                        t_float=t_batch,
                        aux_loss=out.aux_loss,
                        alpha_balance=model.alpha_balance,
                    )
                    return loss, out.aux_loss

                grad_fn = nnx.value_and_grad(_loss, argnums=DiffState(0, _wrt), has_aux=True)
                acc        = jax.tree_util.tree_map(jnp.zeros_like, nnx.state(model, _wrt))
                total_loss = jnp.array(0.0)
                total_bal  = jnp.array(0.0)
                for i in range(config.grad_accum):
                    key, sub = jax.random.split(key)
                    (loss, bal), grads = grad_fn(model, xs[i], sub)
                    acc = jax.tree_util.tree_map(
                        lambda a, g: a + g / config.grad_accum, acc, grads
                    )
                    total_loss += loss / config.grad_accum
                    total_bal  += bal  / config.grad_accum
                opt.update(model, acc)
                metrics.update(loss=total_loss)
                return total_loss, total_bal, key

            @nnx.jit
            def eval_step(model, x, t_float, key):  # type: ignore[misc]
                key, sub_c = jax.random.split(key)
                x_t = corrupt(x, t_float, sub_c, _noise_schedule, config.mask_token_id)
                out = model(x_t, deterministic=True)
                loss = masked_cross_entropy(
                    out.logits, x, x_t, config.mask_token_id, t_float=t_float,
                )
                return loss, out.aux_loss, key

            def estimate_loss(key: jax.Array) -> tuple[dict[str, float], jax.Array]:
                # Stratified t in [t_min, 1]: same lower bound as training so that
                # val and train losses are on the same scale.  Previously starting
                # strata at t=0.5/n (below t_min=0.05) inflated val > train because
                # the model was never trained at t<0.05 AND 1/t amplified those strata.
                n = config.eval_iters
                eval_bs = 8
                t_low = _t_min
                t_strata = [
                    jnp.full((eval_bs,), t_low + (1.0 - t_low) * (i + 0.5) / n)
                    for i in range(n)
                ]
                out: dict[str, float] = {}
                for split, d in [("train", train_data), ("val", val_data)]:
                    losses, bals = [], []
                    for i in range(n):
                        key, sub = jax.random.split(key)
                        x, _ = get_batch(d, eval_bs, config.max_context, sub)
                        loss_val, b, key = eval_step(model, x, t_strata[i], key)
                        losses.append(float(loss_val))
                        bals.append(float(b))
                    out[split] = sum(losses) / len(losses)
                    out[f"{split}_bal"] = sum(bals) / len(bals)
                return out, key

        else:
            # ── Autoregressive training step (original) ───────────────────────

            @nnx.jit
            def train_step(model, opt, metrics, full_x, full_y, _unused_key=None):  # type: ignore[misc]
                xs = full_x.reshape(config.grad_accum, micro_bs, -1)
                ys = full_y.reshape(config.grad_accum, micro_bs, -1)

                def _loss(model, x, y):
                    out = model(x)
                    loss = compute_loss(out.logits, y)
                    if getattr(model, "use_moe", False):
                        loss = loss + model.alpha_balance * out.aux_loss
                    return loss, out.aux_loss

                grad_fn = nnx.value_and_grad(_loss, argnums=DiffState(0, _wrt), has_aux=True)
                acc        = jax.tree_util.tree_map(jnp.zeros_like, nnx.state(model, _wrt))
                total_loss = jnp.array(0.0)
                total_bal  = jnp.array(0.0)
                for i in range(config.grad_accum):
                    (loss, bal), grads = grad_fn(model, xs[i], ys[i])
                    acc = jax.tree_util.tree_map(
                        lambda a, g: a + g / config.grad_accum, acc, grads
                    )
                    total_loss += loss / config.grad_accum
                    total_bal  += bal  / config.grad_accum
                opt.update(model, acc)
                metrics.update(loss=total_loss)
                return total_loss, total_bal, _unused_key  # pass key through unchanged

            @nnx.jit
            def eval_step(model, x, y, _unused_key=None):  # type: ignore[misc]
                out = model(x, deterministic=True)
                loss = compute_loss(out.logits, y)
                if getattr(model, "use_moe", False):
                    loss = loss + model.alpha_balance * out.aux_loss
                return loss, out.aux_loss, None

            def estimate_loss(key: jax.Array) -> tuple[dict[str, float], jax.Array]:
                out: dict[str, float] = {}
                for split, d in [("train", train_data), ("val", val_data)]:
                    losses, bals = [], []
                    for _ in range(config.eval_iters):
                        key, sub = jax.random.split(key)
                        x, y = get_batch(d, 1, config.max_context, sub)
                        loss_val, b, _ = eval_step(model, x, y)
                        losses.append(float(loss_val))
                        bals.append(float(b))
                    out[split] = sum(losses) / len(losses)
                    out[f"{split}_bal"] = sum(bals) / len(bals)
                return out, key

        log_path = os.path.join(run_dir, "training_log.csv")
        best_weights_path = os.path.join(run_dir, "best_model_weights.msgpack")

        key = jax.random.PRNGKey(config.seed)
        metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))

        # Multi-GPU: replicate model/optimizer/metrics to all devices once
        if use_multi_gpu:
            assert mesh is not None
            state = replicate(nnx.state((model, optimizer, metrics)), mesh)
            nnx.update((model, optimizer, metrics), state)

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
                    if use_multi_gpu:
                        assert mesh is not None
                        x = shard_batch(x, mesh)
                        y = shard_batch(y, mesh)
                    if is_elf:
                        emb = _t5_encoder.encode(x)  # contextual embeddings outside JIT
                        _, _, key = train_step(model, optimizer, metrics, emb, x, key)
                    else:
                        _, _, key = train_step(model, optimizer, metrics, x, y, key)

                    if step % 500 == 0:
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

                        # Only the best checkpoint is saved to disk
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

                        # Periodic resume checkpoint (every checkpoint_every steps)
                        _ckpt_every = getattr(config, "checkpoint_every", 2000)
                        if _ckpt_every > 0 and step > 0 and step % _ckpt_every == 0:
                            _save_weights(model, resume_weights)
                            with open(cursor_path, "w") as _cf:
                                json.dump({"step": step, "best_val_loss": best_val_loss}, _cf)
                            log.info("Resume checkpoint saved at step %d", step)

                        if wandb_project is not None:
                            import wandb
                            wandb.log({"train_loss": losses["train"], "val_loss": val_loss, "step": step})  # type: ignore[attr-defined]
            finally:
                pbar.close()
                # Remove cursor so a completed run is not mistaken for an interrupted one
                if os.path.exists(cursor_path):
                    os.remove(cursor_path)
                if wandb_project is not None:
                    import wandb
                    wandb.finish()  # type: ignore[attr-defined]

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

        _t5_name = getattr(config, "t5_model_name", "t5-base")
        tokenizer = get_tokenizer(config.tokenizer_type, model_name=_t5_name)
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
                return compute_loss(m(x).logits, y)

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
