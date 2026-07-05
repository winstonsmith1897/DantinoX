from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from typing import Any

import jax
import jax.numpy as jnp
import msgpack
import yaml
from flax import nnx
from flax.serialization import _msgpack_ext_unpack

from dantinox.core.config import Config
from dantinox.core.diffusion import make_noise_schedule
from dantinox.core.generation import (
    diffusion_generate as _diffusion_generate,
)
from dantinox.core.generation import (
    fast_dllm_generate as _fast_dllm_generate,
)
from dantinox.core.generation import (
    flow_generate as _flow_generate,
)
from dantinox.core.generation import (
    generate as _generate,
)
from dantinox.core.generation import (
    stream_diffusion_generate as _stream_diffusion_generate,
)
from dantinox.core.generation import (
    stream_fast_dllm_generate as _stream_fast_dllm_generate,
)
from dantinox.core.generation import (
    stream_flow_generate as _stream_flow_generate,
)
from dantinox.core.model import Transformer
from dantinox.exceptions import CheckpointError
from dantinox.utils.tokenizer import Tokenizer, get_tokenizer, load_tokenizer_from_file

log = logging.getLogger(__name__)

_BPE_REPLACEMENTS = [
    (" ", ""),
    ("Ġ", " "),
    ("âĢĻ", "'"),
    ("Ã¹", "ù"),
    ("Ã¬", "ì"),
    ("Ã©", "é"),
    ("Ã¨", "è"),
    ("Ã²", "ò"),
    ("Ã", "à"),
]


# ── JIT-compiled streaming step functions ────────────────────────────────────

@nnx.jit
def _stream_prefill(model: nnx.Module, x: jnp.ndarray, kv_cache: tuple) -> tuple:
    """Full prompt forward pass. Returns (logits [B,T,V], filled_kv_cache)."""
    out = model(x, caches=kv_cache, cache_index=0, deterministic=True)
    return out.logits, out.kv_caches


@nnx.jit
def _stream_no_cache_step(model: nnx.Module, x: jnp.ndarray) -> jnp.ndarray:
    """Full forward pass (no KV cache). Returns logits [B,T,V].

    Kept as a separate JIT function so it compiles once for a fixed (B, max_ctx)
    shape and is reused every token in the no-cache streaming loop.
    """
    out = model(x, deterministic=True)
    return out.logits


@nnx.jit
def _stream_decode(model: nnx.Module, tok: jnp.ndarray, kv_cache: tuple, pos: jax.Array) -> tuple:
    """Single-token decode step. Returns (logits [B,1,V], new_kv_cache)."""
    out = model(tok, caches=kv_cache, cache_index=pos, deterministic=True)
    return out.logits, out.kv_caches


def _sample_logit(
    logits: jnp.ndarray,
    key: jax.Array,
    greedy: bool,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
) -> tuple[int, jax.Array]:
    """Sample one token id from a [V] logit vector. Returns (token_id, new_key)."""
    log_probs = jax.nn.log_softmax(logits[0].astype(jnp.float32) / temperature)

    if greedy:
        return int(jnp.argmax(log_probs)), key

    if top_k is not None:
        top_k_vals, top_k_idx = jax.lax.top_k(jnp.exp(log_probs), top_k)
        filtered = jnp.full_like(log_probs, -jnp.inf)
        filtered = filtered.at[top_k_idx].set(
            jnp.log(top_k_vals / top_k_vals.sum() + 1e-10)
        )
        log_probs = filtered

    if top_p is not None:
        probs = jnp.exp(log_probs)
        sorted_idx = jnp.argsort(probs)[::-1]
        sorted_probs = probs[sorted_idx]
        cum = jnp.cumsum(sorted_probs)
        mask = (cum - sorted_probs) >= top_p
        filtered_p = jnp.where(mask, 0.0, sorted_probs)
        filtered_p = filtered_p / (filtered_p.sum() + 1e-10)
        filtered_lp = jnp.full_like(log_probs, -jnp.inf)
        filtered_lp = filtered_lp.at[sorted_idx].set(jnp.log(filtered_p + 1e-10))
        log_probs = filtered_lp

    new_key, subkey = jax.random.split(key)
    tok_id = int(jax.random.categorical(subkey, log_probs))
    return tok_id, new_key


def _remap_expert_keys(obj: Any, _parent_key: Any = None) -> Any:
    """Fix old checkpoints where MoE experts were stored as {0: ..., 1: ...}.

    _ExpertList now uses named attributes (e0, e1, …). Only remap dicts with
    integer keys that are nested under the 'experts' key, leaving block indices
    and other integer-keyed dicts untouched.
    """
    if isinstance(obj, dict):
        if _parent_key == "experts" and obj and all(isinstance(k, int) for k in obj):
            return {f"e{k}": _remap_expert_keys(v) for k, v in obj.items()}
        return {k: _remap_expert_keys(v, _parent_key=k) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_remap_expert_keys(v) for v in obj]
    return obj


def _prune_stale_keys(pure: dict, ref: dict, _path: tuple = ()) -> dict:
    """Drop checkpoint entries that no longer exist in the model's state tree.

    Older checkpoints may carry deterministic buffers (e.g. the causal-mask
    ``tril`` removed by an attention refactor); they are recomputed by the
    model itself, so skipping them is safe. Handles int/str key mismatches
    from msgpack round-tripping.
    """
    def _match(key: Any) -> Any:
        if key in ref:
            return key
        try:
            alt = int(key) if isinstance(key, str) else str(key)
        except (ValueError, TypeError):
            return None
        return alt if alt in ref else None

    out = {}
    for k, v in pure.items():
        rk = _match(k)
        if rk is None:
            log.debug("Skipping stale checkpoint key: %s", "/".join(map(str, _path + (k,))))
            continue
        if isinstance(v, dict) and isinstance(ref[rk], dict):
            out[k] = _prune_stale_keys(v, ref[rk], _path + (k,))
        else:
            out[k] = v
    return out


# ── Checkpoint loader ─────────────────────────────────────────────────────────

def _load_checkpoint(run_dir: str, seed: int) -> tuple[Config, Any, Tokenizer]:
    """Return (config, model, tokenizer) loaded from a local run directory."""
    config_path = os.path.join(run_dir, "config.yaml")

    if not os.path.isdir(run_dir):
        raise CheckpointError(f"Run directory not found: {run_dir}")
    if not os.path.exists(config_path):
        raise CheckpointError(f"Config file not found: {config_path}")

    # Legacy-trainer weights first, then the paradigm Trainer's checkpoints.
    weights_path = None
    for fname in ("best_model_weights.msgpack", "model_weights.msgpack",
                  "checkpoint_best.msgpack", "checkpoint_latest.msgpack"):
        candidate = os.path.join(run_dir, fname)
        if os.path.exists(candidate):
            weights_path = candidate
            break
    # The legacy trainer rebuilt the model with the tokenizer's vocab, so its
    # weights expect vocab_size == tokenizer.vocab_size; the paradigm Trainer
    # keeps the configured vocab_size (which may exceed the tokenizer's).
    is_legacy_weights = weights_path is not None and not os.path.basename(
        weights_path).startswith("checkpoint_")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    flat: dict = {}
    for section in raw.values():
        if isinstance(section, dict):
            flat.update(section)
    if not flat:
        flat = raw

    config = Config.from_dict(flat)

    if config.mla:
        config.inference = True

    tok_path = os.path.join(run_dir, "tokenizer.json")
    if os.path.exists(tok_path):
        tokenizer = load_tokenizer_from_file(tok_path)
        log.info("Loaded tokenizer from %s", tok_path)
    else:
        log.warning(
            "tokenizer.json not found in %r — rebuilding from original corpus "
            "(this only happens once; the file will be saved for future calls).",
            run_dir,
        )
        if config.dataset_source == "huggingface":
            import logging as _logging
            # Silence the noisy httpx / datasets HTTP logs during the one-time download.
            for _noisy in ("httpx", "datasets", "huggingface_hub"):
                _logging.getLogger(_noisy).setLevel(_logging.WARNING)
            from datasets import load_dataset
            raw_dataset = load_dataset(config.dataset_name, split="train")
            text = " ".join(raw_dataset["text"])
        else:
            if not os.path.exists(config.dataset_name):
                raise CheckpointError(
                    f"tokenizer.json not found and dataset file {config.dataset_name!r} "
                    "is also missing. Cannot rebuild the tokenizer vocabulary."
                )
            with open(config.dataset_name, encoding="utf-8") as f:
                text = f.read()
        lines = [line.rstrip() for line in text.split("\n") if line.strip()]
        blocks = ["\n".join(lines[i : i + 3]) for i in range(0, len(lines), 3)]
        text = "\n\n".join(blocks) + "\n"
        tokenizer = get_tokenizer(config.tokenizer_type)
        if config.tokenizer_type == "char":
            tokenizer.train_from_text(text)
        elif config.tokenizer_type == "bpe":
            tokenizer.train_from_text(text, vocab_size=config.vocab_size)
        # Persist so the next call loads instantly without touching the corpus.
        tokenizer.save(tok_path)
        log.warning("Saved tokenizer to %s — subsequent calls will skip the download.", tok_path)

    if is_legacy_weights or weights_path is None:
        config.vocab_size = tokenizer.vocab_size

    rngs = nnx.Rngs(seed)
    # Declared once as Any: the two branches build unrelated model classes.
    model: Any
    if config.model_type == "elf":
        from dantinox.core.flow import FlowMatchingTransformer
        model = FlowMatchingTransformer(config.to_flow_config(), rngs=rngs)
    else:
        model = Transformer(config, rngs=rngs)

    if weights_path is not None:
        log.info("Loading weights from %s", weights_path)
        with open(weights_path, "rb") as f:
            state_dict = msgpack.unpackb(
                f.read(), ext_hook=_msgpack_ext_unpack, strict_map_key=False
            )
        state_dict = _remap_expert_keys(state_dict)
        if is_legacy_weights:
            nnx.update(model, state_dict)
        else:
            state = nnx.state(model, nnx.Not(nnx.RngState))
            state_dict = _prune_stale_keys(state_dict, state.to_pure_dict())
            state.replace_by_pure_dict(state_dict)
            nnx.update(model, state)
    else:
        log.warning("No weights file found in %s — using random initialisation", run_dir)

    return config, model, tokenizer


# ── Generator ─────────────────────────────────────────────────────────────────

class Generator:
    """
    Loads a trained DantinoX checkpoint and generates text.

    Accepts either a **local run directory** or a **HuggingFace Hub repo ID**
    — the checkpoint is downloaded automatically when needed.

    Parameters
    ----------
    run_dir : str
        Local path produced by ``Trainer.fit()`` **or** a Hub repo ID such
        as ``"my-org/dantinox-dante"``.
    seed : int
        RNG seed used for sampling (default 42).
    token : str, optional
        HuggingFace access token for private repositories.
    revision : str, optional
        Branch, tag, or commit SHA to download from the Hub.

    Raises
    ------
    CheckpointError
        If the checkpoint cannot be found locally or downloaded from the Hub.

    Examples
    --------
    >>> gen = Generator("runs/run_20260101_120000")          # local
    >>> gen = Generator("my-org/dantinox-dante")             # HF Hub
    >>> gen = Generator("my-org/private-model", token="hf_…")  # private Hub
    >>> text = gen.generate("Nel mezzo del cammin ")
    >>> print(text)
    """

    def __init__(
        self,
        run_dir: str,
        *,
        seed: int = 42,
        token: str | None = None,
        revision: str | None = None,
    ) -> None:
        from dantinox.hub import resolve_checkpoint

        self.seed = seed
        # Resolve once: download from Hub if needed, then use the local path
        local_dir = resolve_checkpoint(run_dir, token=token, revision=revision)
        self.run_dir = local_dir
        self.config, self.model, self.tokenizer = _load_checkpoint(local_dir, seed)

    def __repr__(self) -> str:
        attn = "MLA" if self.config.mla else (
            "GQA" if (self.config.kv_heads or self.config.n_heads) < self.config.n_heads else "MHA"
        )
        return f"Generator(run_dir={self.run_dir!r}, attn={attn}, seed={self.seed})"

    def _bpe_fix(self, text: str) -> str:
        if self.config.tokenizer_type == "bpe":
            for old, new in _BPE_REPLACEMENTS:
                text = text.replace(old, new)
        return text

    # ── Single-prompt generation ──────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 150,
        greedy: bool = False,
        top_k: int | None = None,
        top_p: float | None = None,
        temperature: float = 1.0,
        use_cache: bool = True,
        # diffusion-only params
        n_steps: int = 50,
        decoding_strategy: str = "sample",
        confidence_threshold: float = 0.9,
        factor: float = 1.5,
        # block-wise (Fast-dLLM) params
        use_blocks: bool = False,
        block_size: int = 32,
        steps_per_block: int = 50,
        use_dual_cache: bool = True,
    ) -> str:
        """
        Generate text continuing from ``prompt``.

        Parameters
        ----------
        prompt : str
            The input prefix.
        max_new_tokens : int
            Number of tokens to generate (default 150).
        greedy : bool
            Use greedy decoding instead of sampling (default False).
        top_k : int, optional
            Keep only the top-k logits before sampling.
        top_p : float, optional
            Nucleus sampling threshold.
        temperature : float
            Softmax temperature (default 1.0).
        use_cache : bool
            Enable KV-cache for faster generation (default True).

        Returns
        -------
        str
            The full generated string (prompt + continuation).
        """
        if not self.config.causal:
            if self.config.model_type == "elf":
                flow_tokens = _flow_generate(
                    self.model, gen_len=max_new_tokens,
                    n_steps=n_steps,
                    cfg_scale=getattr(self.config, "flow_cfg_scale", 1.0),
                    gamma=getattr(self.config, "sde_gamma", 0.0),
                    seed=self.seed,
                )
                return self._bpe_fix(self.tokenizer.decode(flow_tokens[0].tolist()))
            if use_blocks:
                return self._fast_dllm_generate(
                    prompt, max_new_tokens=max_new_tokens,
                    block_size=block_size, steps_per_block=steps_per_block,
                    decoding_strategy=decoding_strategy,
                    confidence_threshold=confidence_threshold, factor=factor,
                    use_dual_cache=use_dual_cache,
                )
            return self._diffusion_generate(
                prompt, max_new_tokens=max_new_tokens, temperature=temperature,
                n_steps=n_steps, decoding_strategy=decoding_strategy,
                confidence_threshold=confidence_threshold, factor=factor,
            )

        tokens = self.tokenizer.encode(prompt)
        x = jnp.array([tokens], dtype=jnp.int32)

        log.debug(
            "Generating %d tokens from prompt of %d tokens (greedy=%s, cache=%s)",
            max_new_tokens, len(tokens), greedy, use_cache,
        )

        output = _generate(
            model=self.model,
            x=x,
            max_generations=max_new_tokens,
            greedy=greedy,
            seed=self.seed,
            use_cache=use_cache,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
        output.block_until_ready()

        return self._bpe_fix(self.tokenizer.decode(output[0].tolist()))

    # ── Batched generation ────────────────────────────────────────────────────

    def generate_batch(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int = 150,
        greedy: bool = False,
        top_k: int | None = None,
        top_p: float | None = None,
        temperature: float = 1.0,
        use_cache: bool = True,
    ) -> list[str]:
        """
        Generate text for multiple prompts in a single batched forward pass.

        Shorter prompts are left-padded with zeros so all share the same
        sequence length.  This runs a true batch through the model, so
        throughput scales with GPU parallelism.

        Parameters
        ----------
        prompts : list[str]
            Input prefixes to generate from.
        max_new_tokens : int
            Tokens to generate per prompt (default 150).
        greedy : bool
            Greedy decoding (default False).
        top_k : int, optional
            Top-k filtering before sampling.
        top_p : float, optional
            Nucleus sampling threshold.
        temperature : float
            Softmax temperature (default 1.0).
        use_cache : bool
            Enable KV-cache (default True).

        Returns
        -------
        list[str]
            Generated strings (prompt + continuation) in the same order as
            ``prompts``.
        """
        if not prompts:
            return []

        encoded = [self.tokenizer.encode(p) for p in prompts]
        max_len = max(len(e) for e in encoded)

        # Left-pad shorter prompts with zeros so all share the same start position.
        padded = [([0] * (max_len - len(e))) + e for e in encoded]
        x = jnp.array(padded, dtype=jnp.int32)  # [B, max_len]

        log.debug("Batch generating: B=%d max_prompt_len=%d max_new=%d", len(prompts), max_len, max_new_tokens)

        output = _generate(
            model=self.model,
            x=x,
            max_generations=max_new_tokens,
            greedy=greedy,
            seed=self.seed,
            use_cache=use_cache,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
        output.block_until_ready()

        results = []
        for i, enc in enumerate(encoded):
            # Strip the left-padding: prompt starts at (max_len - len(enc))
            start = max_len - len(enc)
            tokens_out = output[i, start:].tolist()
            results.append(self._bpe_fix(self.tokenizer.decode(tokens_out)))
        return results

    # ── Streaming generation ──────────────────────────────────────────────────

    def stream(
        self,
        prompt: str = "",
        *,
        max_new_tokens: int = 150,
        greedy: bool = False,
        top_k: int | None = None,
        top_p: float | None = None,
        temperature: float = 1.0,
        use_cache: bool = True,
        # diffusion-only
        n_steps: int = 50,
        decoding_strategy: str = "sample",
        confidence_threshold: float = 0.9,
        factor: float = 1.5,
        # block-wise (Fast-dLLM)
        use_blocks: bool = False,
        block_size: int = 32,
        steps_per_block: int = 50,
        use_dual_cache: bool = True,
    ) -> Iterator[str]:
        """
        Stream generated tokens one at a time as they are produced.

        Uses the KV-cache path: the prompt is prefilled in one forward pass,
        then each new token is decoded individually.  Each ``yield`` returns
        the string for one generated token (may be a character or a BPE
        subword).

        Parameters
        ----------
        prompt : str
            The input prefix.
        max_new_tokens : int
            Maximum number of tokens to generate (default 150).
        greedy : bool
            Greedy decoding (default False).
        top_k : int, optional
            Top-k filtering.
        top_p : float, optional
            Nucleus sampling threshold.
        temperature : float
            Softmax temperature (default 1.0).

        Yields
        ------
        str
            Decoded string for each generated token.

        Examples
        --------
        >>> gen = Generator("runs/my_run")
        >>> for chunk in gen.stream("Nel mezzo", max_new_tokens=50):
        ...     print(chunk, end="", flush=True)
        """
        if not self.config.causal:
            if self.config.model_type == "elf":
                yield from self._stream_elf(max_new_tokens=max_new_tokens, n_steps=n_steps)
                return
            if use_blocks:
                yield from self._stream_fast_dllm(
                    prompt, max_new_tokens=max_new_tokens, temperature=temperature,
                    block_size=block_size, steps_per_block=steps_per_block,
                    decoding_strategy=decoding_strategy,
                    confidence_threshold=confidence_threshold, factor=factor,
                    use_dual_cache=use_dual_cache,
                )
            else:
                yield from self._stream_discrete(
                    prompt, max_new_tokens=max_new_tokens, temperature=temperature,
                    n_steps=n_steps, decoding_strategy=decoding_strategy,
                    confidence_threshold=confidence_threshold, factor=factor,
                )
            return

        if not use_cache:
            yield from self._stream_no_cache(prompt, max_new_tokens=max_new_tokens,
                                              greedy=greedy, top_k=top_k, top_p=top_p,
                                              temperature=temperature)
            return

        tokens = self.tokenizer.encode(prompt)
        T = len(tokens)
        max_ctx = self.config.max_context  # type: ignore[attr-defined]
        num_blocks = self.config.num_blocks  # type: ignore[attr-defined]

        # Build full-context input with prompt at the start.
        x = jnp.zeros((1, max_ctx), dtype=jnp.int32)
        x = x.at[0, :T].set(jnp.array(tokens, dtype=jnp.int32))

        init_kv_cache = tuple((None, None) for _ in range(num_blocks))
        key = jax.random.key(self.seed)

        # Prefill: one pass over the entire prompt, populate KV cache.
        logits, kv_cache = _stream_prefill(self.model, x, init_kv_cache)

        # Sample the first generated token from the last prompt position.
        tok_id, key = _sample_logit(logits[:, T - 1, :], key, greedy, temperature, top_k, top_p)
        yield self._bpe_fix(self.tokenizer.decode([tok_id]))

        # Autoregressive decode loop.
        for pos in range(T, T + max_new_tokens - 1):
            if pos >= max_ctx:
                break
            tok = jnp.array([[tok_id]], dtype=jnp.int32)
            logits, kv_cache = _stream_decode(self.model, tok, kv_cache, jnp.array(pos))
            tok_id, key = _sample_logit(logits[:, 0, :], key, greedy, temperature, top_k, top_p)
            yield self._bpe_fix(self.tokenizer.decode([tok_id]))

    # ── Diffusion (discrete) generation ──────────────────────────────────────

    def _diffusion_generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 150,
        temperature: float = 1.0,
        n_steps: int = 50,
        decoding_strategy: str = "sample",
        confidence_threshold: float = 0.9,
        factor: float = 1.5,
    ) -> str:
        """Full diffusion reverse-process for discrete (masked) models."""
        tokens = self.tokenizer.encode(prompt)
        prefix = jnp.array([tokens], dtype=jnp.int32) if tokens else None
        schedule = make_noise_schedule(self.config.noise_schedule)
        mask_id = self.config.mask_token_id

        result = _diffusion_generate(
            self.model, prefix, gen_len=max_new_tokens,
            schedule=schedule, mask_token_id=mask_id,
            seed=self.seed, num_sampling_steps=n_steps,
            temperature=temperature,
            decoding_strategy=decoding_strategy,
            confidence_threshold=confidence_threshold,
            factor=factor,
        )
        prefix_text = self.tokenizer.decode(tokens)
        gen_text = self._bpe_fix(self.tokenizer.decode(result[0].tolist()))
        return prefix_text + gen_text

    def _stream_discrete(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 150,
        temperature: float = 1.0,
        n_steps: int | None = None,
        decoding_strategy: str = "sample",
        confidence_threshold: float = 0.9,
        factor: float = 1.5,
    ) -> Iterator[str]:
        """Stream discrete diffusion denoising — yields in-place rewrite strings.

        Each yielded chunk is ``\\r<current decoded state>`` so the caller can
        print it with ``end=""`` to see the token sequence unmask in real-time.
        Masked positions are shown as ``░``.
        """
        tokens = self.tokenizer.encode(prompt)
        prefix = jnp.array([tokens], dtype=jnp.int32) if tokens else None
        schedule = make_noise_schedule(self.config.noise_schedule)
        mask_id = self.config.mask_token_id
        prefix_text = self.tokenizer.decode(tokens)
        # Replace newlines with a visible symbol so \r stays on one line.
        prefix_display = prefix_text.replace("\n", "↵").replace("\r", "")

        prev_len = 0
        if not n_steps:
            n_steps = max_new_tokens
        for _step, _total, x_t in _stream_diffusion_generate(
            self.model, prefix, gen_len=max_new_tokens,
            schedule=schedule, mask_token_id=mask_id,
            seed=self.seed, num_sampling_steps=n_steps,
            temperature=temperature,
            decoding_strategy=decoding_strategy,
            confidence_threshold=confidence_threshold,
            factor=factor,
        ):
            ids = x_t[0].tolist()
            gen_text = self._decode_masked(ids, mask_id).replace("\n", "↵").replace("\r", "")
            line = f"{prefix_display}{gen_text}"
            padding = max(0, prev_len - len(line))
            prev_len = len(line)
            yield f"\r{line}{' ' * padding}"

    def _stream_no_cache(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 150,
        greedy: bool = False,
        top_k: int | None = None,
        top_p: float | None = None,
        temperature: float = 1.0,
    ) -> Iterator[str]:
        """Stream AR generation without KV cache — full attention recomputed each step.

        The input is padded to ``max_context`` so the XLA kernel compiles once
        and is reused every iteration.  Slower than the cached path (O(T) per
        token vs O(1)) but demonstrates full-attention inference.
        """
        tokens = self.tokenizer.encode(prompt)
        T = len(tokens)
        max_ctx = self.config.max_context  # type: ignore[attr-defined]
        key = jax.random.key(self.seed)

        # Pad to max_ctx so the shape is static → single JIT compilation.
        x = jnp.zeros((1, max_ctx), dtype=jnp.int32)
        x = x.at[0, :T].set(jnp.array(tokens, dtype=jnp.int32))

        for pos in range(T, T + max_new_tokens):
            if pos >= max_ctx:
                break
            logits = _stream_no_cache_step(self.model, x)
            tok_id, key = _sample_logit(logits[:, pos - 1, :], key, greedy, temperature, top_k, top_p)
            x = x.at[0, pos].set(tok_id)
            yield self._bpe_fix(self.tokenizer.decode([tok_id]))

    # ── Block-wise Fast-dLLM generation ──────────────────────────────────────

    def _fast_dllm_generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 150,
        block_size: int = 32,
        steps_per_block: int = 50,
        decoding_strategy: str = "threshold",
        confidence_threshold: float = 0.9,
        factor: float = 1.5,
        use_dual_cache: bool = True,
    ) -> str:
        """Block-wise masked-diffusion via Fast-dLLM DualCache (Wu et al., 2025)."""
        tokens = self.tokenizer.encode(prompt)
        # fast_dllm_generate requires a real (possibly zero-length) array —
        # unlike diffusion_generate it does not accept prefix=None.
        prefix = jnp.array([tokens], dtype=jnp.int32) if tokens else jnp.zeros((1, 0), dtype=jnp.int32)
        schedule = make_noise_schedule(self.config.noise_schedule)
        mask_id = self.config.mask_token_id

        result = _fast_dllm_generate(
            self.model, prefix, gen_len=max_new_tokens,
            schedule=schedule, mask_token_id=mask_id,
            block_size=block_size, steps_per_block=steps_per_block,
            decoding_strategy=decoding_strategy,
            confidence_threshold=confidence_threshold, factor=factor,
            use_dual_cache=use_dual_cache, seed=self.seed,
        )
        prefix_text = self.tokenizer.decode(tokens)
        gen_text = self._bpe_fix(self.tokenizer.decode(result[0].tolist()))
        return prefix_text + gen_text

    def _stream_fast_dllm(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 150,
        block_size: int = 32,
        steps_per_block: int = 50,
        decoding_strategy: str = "threshold",
        confidence_threshold: float = 0.9,
        factor: float = 1.5,
        use_dual_cache: bool = True,
        temperature: float = 1.0,  # unused, kept for API symmetry
    ) -> Iterator[str]:
        """Stream block-wise Fast-dLLM — yields in-place rewrite strings.

        Visualises how each block is denoised left-to-right before the next
        block starts.  Masked positions shown as ``░``.
        """
        tokens = self.tokenizer.encode(prompt)
        # stream_fast_dllm_generate requires a real (possibly zero-length)
        # array — unlike diffusion_generate it does not accept prefix=None.
        prefix = jnp.array([tokens], dtype=jnp.int32) if tokens else jnp.zeros((1, 0), dtype=jnp.int32)
        schedule = make_noise_schedule(self.config.noise_schedule)
        mask_id = self.config.mask_token_id
        prefix_text = self.tokenizer.decode(tokens)
        prefix_display = prefix_text.replace("\n", "↵").replace("\r", "")

        prev_len = 0
        for _step, _total, x_gen in _stream_fast_dllm_generate(
            self.model, prefix, gen_len=max_new_tokens,
            schedule=schedule, mask_token_id=mask_id,
            block_size=block_size, steps_per_block=steps_per_block,
            decoding_strategy=decoding_strategy,
            confidence_threshold=confidence_threshold, factor=factor,
            use_dual_cache=use_dual_cache, seed=self.seed,
        ):
            ids = x_gen[0].tolist()
            gen_text = self._decode_masked(ids, mask_id).replace("\n", "↵").replace("\r", "")
            line = f"{prefix_display}{gen_text}"
            padding = max(0, prev_len - len(line))
            prev_len = len(line)
            yield f"\r{line}{' ' * padding}"

    def _decode_masked(self, ids: list[int], mask_id: int, mask_symbol: str = "░") -> str:
        """Decode ids rendering masked positions as ``mask_symbol``.

        The model masks with ``config.mask_token_id``, which may differ from
        (or be absent in) the tokenizer's own vocabulary — e.g. char tokenizers
        saved without the mask char report ``mask_token_id = None``.  Decode
        runs of non-mask ids together to preserve BPE/SentencePiece spacing.
        """
        tok_mask = getattr(self.tokenizer, "mask_token_id", None)
        if tok_mask is not None:
            if mask_id != tok_mask:
                ids = [tok_mask if t == mask_id else t for t in ids]
            # decode_display is specific to masking-aware tokenizers (Char/BPE),
            # not part of the generic Tokenizer protocol; guarded by the
            # tok_mask check above (only mask-token tokenizers reach here).
            return self.tokenizer.decode_display(ids, mask_symbol=mask_symbol)  # type: ignore[attr-defined]
        out: list[str] = []
        run: list[int] = []
        for t in ids:
            if t == mask_id:
                if run:
                    out.append(self.tokenizer.decode(run))
                    run = []
                out.append(mask_symbol)
            else:
                run.append(t)
        if run:
            out.append(self.tokenizer.decode(run))
        return "".join(out)

    def _stream_elf(
        self,
        max_new_tokens: int = 64,
        n_steps: int | None = None,
    ) -> Iterator[str]:
        """Stream continuous flow-matching — yields in-place rewrite strings.

        Each yielded chunk is ``\\r[step/total] <current decoded sequence>`` showing
        all token positions evolving simultaneously through the ODE steps.
        """
        steps: int = n_steps or self.config.flow_n_steps
        cfg_w = self.config.flow_cfg_scale
        gamma = self.config.sde_gamma

        prev_len = 0
        for step, total, cur_tokens in _stream_flow_generate(
            self.model, gen_len=max_new_tokens, batch_size=1,
            n_steps=steps, cfg_scale=cfg_w, gamma=gamma, seed=self.seed,
        ):
            gen_text = self._bpe_fix(
                self.tokenizer.decode(cur_tokens[0].tolist())
            ).replace("\n", "↵").replace("\r", "")
            step_tag = f"[{step + 1:02d}/{total}] "
            line = step_tag + gen_text
            padding = max(0, prev_len - len(line))
            prev_len = len(line)
            yield f"\r{line}{' ' * padding}"
