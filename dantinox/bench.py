from __future__ import annotations

import dataclasses
import logging
import os
import time
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import msgpack
import numpy as np
from flax import nnx
from flax.serialization import _msgpack_ext_unpack

from core.config import Config
from core.model import Transformer
from dantinox.exceptions import BenchmarkError

log = logging.getLogger(__name__)

SEQ_LENS    = [64, 128, 256, 512]
BATCH_SIZES = [1, 4, 16, 64, 128, 256]
FIXED_SEQ   = 256
N_WARMUP    = 3
N_MEASURE   = 20


@nnx.jit
def _decode_step(model: Transformer, tok: jnp.ndarray, cache, idx: int):
    return model(tok, use_cache=True, kv_caches=cache, cache_index=idx)


@nnx.jit
def _prefill_step(model: Transformer, prompt: jnp.ndarray):
    return model(prompt, use_cache=False, kv_caches=None, cache_index=0)


def _load_config(run_path: str) -> Config:
    import yaml
    cfg_path = os.path.join(run_path, "config.yaml")
    if not os.path.exists(cfg_path):
        raise BenchmarkError(f"Config not found: {cfg_path}")
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    flat: dict = {}
    for section in raw.values():
        if isinstance(section, dict):
            flat.update(section)
    if not flat:
        flat = raw
    return Config.from_dict(flat)


def _detect_vocab(state_dict: dict, dim: int) -> int | None:
    def _get(d, key):
        if not isinstance(d, dict):
            return None
        return d.get(key) or d.get(key.encode() if isinstance(key, str) else key)

    def _unwrap(obj):
        if isinstance(obj, dict):
            for k in ("value", "raw_value", b"value", b"raw_value"):
                if k in obj:
                    return obj[k]
        return obj

    wte = _get(state_dict, "wte")
    if wte is None:
        return None
    emb = _unwrap(_get(wte, "embedding"))
    if emb is None or not hasattr(emb, "shape") or emb.ndim != 2:
        return None
    return int(emb.shape[0]) if emb.shape[1] == dim else (
        int(emb.shape[1]) if emb.shape[0] == dim else None
    )


def _load_model(run_path: str, config: Config) -> Transformer:
    weights_path = os.path.join(run_path, "model_weights.msgpack")
    if not os.path.exists(weights_path):
        raise BenchmarkError(f"Weights not found: {weights_path}")
    with open(weights_path, "rb") as f:
        state_dict = msgpack.unpackb(
            f.read(), ext_hook=_msgpack_ext_unpack, strict_map_key=False
        )
    actual_vocab = _detect_vocab(state_dict, config.dim)
    if actual_vocab is not None and actual_vocab != config.vocab_size:
        config = dataclasses.replace(config, vocab_size=actual_vocab)
    model = Transformer(config, rngs=nnx.Rngs(42))
    nnx.update(model, state_dict)
    return model


def _attn_type(config: Config) -> str:
    if getattr(config, "mla", False):
        return "MLA"
    if getattr(config, "kv_heads", config.n_heads) < config.n_heads:
        return "GQA"
    return "MHA"


def _theoretical_cache_mb(config: Config) -> float:
    S = config.max_context
    if getattr(config, "mla", False):
        per_layer = S * (getattr(config, "down_dim_kv", 0) + getattr(config, "rope_dim", 0)) * 4
    else:
        per_layer = 2 * S * getattr(config, "kv_heads", config.n_heads) * (config.dim // config.n_heads) * 4
    return per_layer * config.num_blocks / 1e6


def _val_loss(run_path: str) -> float | None:
    import pandas as pd
    log_csv = os.path.join(run_path, "training_log.csv")
    if not os.path.exists(log_csv):
        return None
    try:
        df = pd.read_csv(log_csv)
        return float(df["val_loss"].dropna().iloc[-1])
    except Exception:
        return None


def _xla_costs(fn, *args) -> tuple[float, float]:
    try:
        costs = fn.lower(*args).cost_analysis()
        if isinstance(costs, list):
            flops = sum(c.get("flops", 0) for c in costs)
            mem   = sum(c.get("bytes accessed", 0) for c in costs)
        else:
            flops = costs.get("flops", float("nan"))
            mem   = costs.get("bytes accessed", float("nan"))
        return float(flops), float(mem)
    except Exception:
        return float("nan"), float("nan")


def benchmark_run(run_path: str) -> dict:
    """Benchmark a single run directory, returning a metrics dict."""
    config = _load_config(run_path)
    model  = _load_model(run_path, config)

    prompt_len = min(config.max_context, 256)
    prompt = jnp.ones((1, prompt_len), dtype=jnp.int32)
    tok    = jnp.ones((1, 1), dtype=jnp.int32)

    # Prefill latency
    jax.block_until_ready(_prefill_step(model, prompt))
    t0 = time.perf_counter()
    jax.block_until_ready(_prefill_step(model, prompt))
    prefill_ms = (time.perf_counter() - t0) * 1000

    # Decode throughput — sequence length scaling @ BS=1
    tps_by_seqlen: dict[int, float] = {}
    for seq in SEQ_LENS:
        if seq > config.max_context:
            tps_by_seqlen[seq] = float("nan")
            continue
        try:
            for _ in range(N_WARMUP):
                _decode_step(model, tok, None, seq)
            jax.block_until_ready(_decode_step(model, tok, None, seq))
            t0 = time.perf_counter()
            for _ in range(N_MEASURE):
                _decode_step(model, tok, None, seq)
            jax.block_until_ready(_decode_step(model, tok, None, seq))
            tps_by_seqlen[seq] = N_MEASURE / (time.perf_counter() - t0)
            log.debug("seq=%d: %.1f tok/s", seq, tps_by_seqlen[seq])
        except Exception as exc:
            log.warning("seq=%d failed: %s", seq, exc)
            tps_by_seqlen[seq] = float("nan")

    # Decode throughput — batch scaling @ FIXED_SEQ
    tps_by_batch: dict[int, float] = {}
    max_batch = 0
    for bs in BATCH_SIZES:
        if config.max_context < FIXED_SEQ:
            tps_by_batch[bs] = float("nan")
            continue
        tok_b = jnp.ones((bs, 1), dtype=jnp.int32)
        try:
            for _ in range(N_WARMUP):
                _decode_step(model, tok_b, None, FIXED_SEQ)
            jax.block_until_ready(_decode_step(model, tok_b, None, FIXED_SEQ))
            t0 = time.perf_counter()
            for _ in range(N_MEASURE):
                _decode_step(model, tok_b, None, FIXED_SEQ)
            jax.block_until_ready(_decode_step(model, tok_b, None, FIXED_SEQ))
            tps_by_batch[bs] = N_MEASURE * bs / (time.perf_counter() - t0)
            max_batch = bs
            log.debug("bs=%d: %.1f tok/s", bs, tps_by_batch[bs])
        except Exception as exc:
            log.warning("bs=%d OOM/failed: %s", bs, exc)
            tps_by_batch[bs] = float("nan")
            break
    for bs in BATCH_SIZES:
        tps_by_batch.setdefault(bs, float("nan"))

    # FLOPs via XLA cost analysis
    mid_idx = min(config.max_context // 2, config.max_context - 1)
    decode_flops, decode_bytes   = _xla_costs(_decode_step, model, tok, None, mid_idx)
    prefill_flops, prefill_bytes = _xla_costs(_prefill_step, model, prompt)

    def _safe_div(a: float, b: float) -> float:
        return round(a / max(b, 1), 4) if not (np.isnan(a) or np.isnan(b)) else float("nan")

    def _safe_round(v: float, n: float) -> float:
        return round(v / n, 4) if not np.isnan(v) else float("nan")

    best_tps = tps_by_seqlen.get(max(s for s in SEQ_LENS if s <= config.max_context), float("nan"))
    decode_gflops  = _safe_round(decode_flops, 1e9)
    prefill_gflops = _safe_round(prefill_flops, 1e9)

    _, model_state = nnx.split(model)
    params_m = sum(
        x.size for x in jax.tree_util.tree_leaves(model_state) if hasattr(x, "size")
    ) / 1e6

    return {
        "run":                   os.path.basename(run_path),
        "type":                  _attn_type(config),
        "params_m":              params_m,
        "moe":                   getattr(config, "use_moe", False),
        "num_blocks":            config.num_blocks,
        "dim":                   config.dim,
        "n_heads":               config.n_heads,
        "kv_heads":              getattr(config, "kv_heads", config.n_heads),
        "max_context":           config.max_context,
        "down_dim_kv":           getattr(config, "down_dim_kv", None),
        "theoretical_cache_mb":  round(_theoretical_cache_mb(config), 2),
        "prefill_ms":            prefill_ms,
        "val_loss":              _val_loss(run_path),
        "decode_gflops":         decode_gflops,
        "prefill_gflops":        prefill_gflops,
        "decode_arith_int":      _safe_div(decode_flops, decode_bytes),
        "prefill_arith_int":     _safe_div(prefill_flops, prefill_bytes),
        "decode_tflops_s":       (
            round(decode_gflops * best_tps / 1e3, 4)
            if not (np.isnan(decode_gflops) or np.isnan(best_tps))
            else float("nan")
        ),
        "max_batch_survived":    max_batch,
        **{f"tps_{s}": tps_by_seqlen.get(s, float("nan")) for s in SEQ_LENS},
        **{f"tps_bs{b}": tps_by_batch.get(b, float("nan")) for b in BATCH_SIZES},
    }


class BenchmarkRunner:
    """
    Benchmarks one or more DantinoX run directories.

    Parameters
    ----------
    runs_dir : str
        Directory containing run sub-directories (default ``"runs"``).
    seq_lens : list[int], optional
        Sequence lengths to test for throughput scaling.
    batch_sizes : list[int], optional
        Batch sizes to test for memory/throughput scaling.

    Examples
    --------
    >>> runner = BenchmarkRunner("runs")
    >>> df = runner.run()
    >>> df.to_csv("results.csv", index=False)
    """

    def __init__(
        self,
        runs_dir: str = "runs",
        *,
        seq_lens: Sequence[int] | None = None,
        batch_sizes: Sequence[int] | None = None,
    ) -> None:
        self.runs_dir = runs_dir
        if seq_lens is not None:
            global SEQ_LENS
            SEQ_LENS = list(seq_lens)
        if batch_sizes is not None:
            global BATCH_SIZES
            BATCH_SIZES = list(batch_sizes)

    def __repr__(self) -> str:
        return f"BenchmarkRunner(runs_dir={self.runs_dir!r})"

    def run(
        self,
        run_names: Sequence[str] | None = None,
        *,
        out_csv: str | None = None,
    ):
        """
        Run benchmarks and return a DataFrame.

        Parameters
        ----------
        run_names : list[str], optional
            Subset of run names to evaluate. Benchmarks all runs if omitted.
        out_csv : str, optional
            Write results to this CSV path.

        Returns
        -------
        pandas.DataFrame

        Raises
        ------
        BenchmarkError
            If the runs directory does not exist.
        """
        import pandas as pd

        if not os.path.isdir(self.runs_dir):
            raise BenchmarkError(f"Runs directory not found: {self.runs_dir}")

        if run_names is None:
            run_names = [
                d for d in os.listdir(self.runs_dir)
                if os.path.isdir(os.path.join(self.runs_dir, d))
            ]

        results = []
        for name in run_names:
            path = os.path.join(self.runs_dir, name)
            log.info("Benchmarking: %s", name)
            try:
                results.append(benchmark_run(path))
            except BenchmarkError as exc:
                log.error("  Skipped %s: %s", name, exc)
            except Exception as exc:
                log.error("  Unexpected error for %s: %s", name, exc)

        df = pd.DataFrame(results)
        if out_csv:
            df.to_csv(out_csv, index=False)
            log.info("Saved benchmark results to %s", out_csv)
        return df
