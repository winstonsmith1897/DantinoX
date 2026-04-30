from __future__ import annotations

import os
from typing import Optional

import jax.numpy as jnp
import msgpack
import yaml
from flax import nnx
from flax.serialization import _msgpack_ext_unpack

from core.config import Config
from core.model import Transformer
from core.generation import generate as _generate
from utils.tokenizer import get_tokenizer


def _load_checkpoint(run_dir: str, seed: int):
    """Return (config, model, tokenizer) loaded from a run directory."""
    config_path = os.path.join(run_dir, "config.yaml")
    weights_path = os.path.join(run_dir, "model_weights.msgpack")

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    flat = {}
    for section in raw.values():
        if isinstance(section, dict):
            flat.update(section)
    if not flat:
        flat = raw

    config = Config(**{k: v for k, v in flat.items() if k in Config.__dataclass_fields__})

    if config.mla:
        config.inference = True

    # Rebuild the tokenizer vocabulary from the original corpus
    if config.dataset_source == "huggingface":
        from datasets import load_dataset
        raw_dataset = load_dataset(config.dataset_name, split="train")
        text = " ".join(raw_dataset["text"])
    else:
        with open(config.dataset_name, "r", encoding="utf-8") as f:
            text = f.read()

    lines = [l.rstrip() for l in text.split("\n") if l.strip()]
    blocks = ["\n".join(lines[i : i + 3]) for i in range(0, len(lines), 3)]
    text = "\n\n".join(blocks) + "\n"

    tokenizer = get_tokenizer(config.tokenizer_type)
    if config.tokenizer_type == "char":
        tokenizer.train_from_text(text)
    elif config.tokenizer_type == "bpe":
        tokenizer.train_from_text(text, vocab_size=config.vocab_size)

    config.vocab_size = tokenizer.vocab_size

    rngs = nnx.Rngs(seed)
    model = Transformer(config, rngs=rngs)

    if os.path.exists(weights_path):
        with open(weights_path, "rb") as f:
            state_dict = msgpack.unpackb(
                f.read(), ext_hook=_msgpack_ext_unpack, strict_map_key=False
            )
        nnx.update(model, state_dict)

    return config, model, tokenizer


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


class Generator:
    """
    Loads a trained DantinoX checkpoint and generates text.

    Parameters
    ----------
    run_dir : str
        Path to a training run directory containing ``config.yaml`` and
        ``model_weights.msgpack``.
    seed : int
        RNG seed used for sampling (default 42).

    Examples
    --------
    >>> gen = Generator("runs/run_20260101_120000")
    >>> text = gen.generate("Nel mezzo del cammin ")
    >>> print(text)
    """

    def __init__(self, run_dir: str, *, seed: int = 42) -> None:
        self.run_dir = run_dir
        self.seed = seed
        self.config, self.model, self.tokenizer = _load_checkpoint(run_dir, seed)

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 150,
        greedy: bool = False,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
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

        text = self.tokenizer.decode(output[0].tolist())
        if self.config.tokenizer_type == "bpe":
            for old, new in _BPE_REPLACEMENTS:
                text = text.replace(old, new)
        return text
