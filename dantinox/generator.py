from __future__ import annotations

import logging
import os
from collections.abc import Iterator

import jax
import jax.numpy as jnp
import msgpack
import yaml
from flax import nnx
from flax.serialization import _msgpack_ext_unpack

from dantinox.core.config import Config
from dantinox.core.generation import generate as _generate
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
    logits, new_cache, _ = model(x, True, kv_cache, 0, deterministic=True)
    return logits, new_cache


@nnx.jit
def _stream_decode(model: nnx.Module, tok: jnp.ndarray, kv_cache: tuple, pos: jax.Array) -> tuple:
    """Single-token decode step. Returns (logits [B,1,V], new_kv_cache)."""
    logits, new_cache, _ = model(tok, True, kv_cache, pos, deterministic=True)
    return logits, new_cache


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


# ── Checkpoint loader ─────────────────────────────────────────────────────────

def _load_checkpoint(run_dir: str, seed: int) -> tuple[Config, Transformer, Tokenizer]:
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
    model = Transformer(config, rngs=rngs)

    if weights_path is not None:
        log.info("Loading weights from %s", weights_path)
        with open(weights_path, "rb") as f:
            state_dict = msgpack.unpackb(
                f.read(), ext_hook=_msgpack_ext_unpack, strict_map_key=False
            )
        if is_legacy_weights:
            nnx.update(model, state_dict)
        else:
            state = nnx.state(model, nnx.Not(nnx.RngState))
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
        attn = "MLA" if self.config.mla else ("GQA" if self.config.kv_heads < self.config.n_heads else "MHA")
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
        prompt: str,
        *,
        max_new_tokens: int = 150,
        greedy: bool = False,
        top_k: int | None = None,
        top_p: float | None = None,
        temperature: float = 1.0,
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
