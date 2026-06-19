"""Streaming data pipeline for memory-efficient pretraining.

Wraps HuggingFace ``datasets`` in streaming mode so large corpora are never
loaded entirely into RAM.  Texts are tokenised on the fly, concatenated into a
flat buffer, and emitted as non-overlapping AR-formatted windows.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator

import numpy as np

log = logging.getLogger(__name__)


class AutoDataPipeline:
    """Infinite streaming pipeline that yields next-token-prediction batches.

    Texts from a HuggingFace dataset are streamed one-by-one, tokenised, and
    appended to an in-memory buffer.  Once the buffer holds at least
    ``max_context + 1`` tokens, a non-overlapping window is emitted as an
    ``(inputs, targets)`` pair where ``targets = inputs`` shifted right by one
    position — the standard autoregressive objective.

    The iterator loops the stream automatically when it reaches the end of the
    dataset, making it suitable for multi-epoch training.

    Args:
        dataset_path: HuggingFace dataset name (e.g. ``"wikitext"``).
        subset: Dataset configuration / subset name
            (e.g. ``"wikitext-2-raw-v1"``).
        split: Split to use: ``"train"``, ``"validation"``, or ``"test"``.
        text_column: Name of the text field inside each example dict.
        tokenizer: Object exposing ``.encode(text: str) -> list[int]``.
        max_context: Sequence length for each emitted chunk.  Both
            ``inputs`` and ``targets`` will have exactly this length.

    Yields:
        ``{"inputs": np.ndarray[int32, shape=(max_context,)],``
        ``"targets": np.ndarray[int32, shape=(max_context,)]}``

    Example::

        from dantinox.utils.data_pipeline import AutoDataPipeline
        from dantinox.utils.tokenizer import get_tokenizer

        tok  = get_tokenizer("char")
        pipe = AutoDataPipeline(
            "wikitext", "wikitext-2-raw-v1", "train", "text", tok, 512
        )
        for batch in pipe:
            # batch["inputs"].shape == (512,)
            train_step(batch["inputs"], batch["targets"])
    """

    def __init__(
        self,
        dataset_path: str,
        subset: str,
        split: str,
        text_column: str,
        tokenizer: Any,
        max_context: int,
    ) -> None:
        self._dataset_path = dataset_path
        self._subset = subset or None
        self._split = split
        self._text_column = text_column
        self._tokenizer = tokenizer
        self._max_context = max_context
        log.info(
            "AutoDataPipeline: dataset=%s subset=%s split=%s ctx=%d",
            dataset_path, subset, split, max_context,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
        """Yield ``{"inputs": ..., "targets": ...}`` batches indefinitely.

        A new streaming pass over the dataset is started each time the stream
        is exhausted, ensuring training never blocks waiting for data.
        """
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' package is required for AutoDataPipeline. "
                "Install it with:  pip install datasets"
            ) from exc

        ctx = self._max_context
        buffer: list[int] = []
        n_yielded = 0

        while True:  # outer loop: restart stream each epoch
            ds = load_dataset(
                self._dataset_path,
                self._subset,
                split=self._split,
                streaming=True,
            )

            for example in ds:
                text: str = example.get(self._text_column, "") or ""
                if not text.strip():
                    continue

                ids: list[int] = self._tokenizer.encode(text)
                if not ids:
                    continue

                buffer.extend(ids)

                # Drain non-overlapping windows of ctx + 1 tokens.
                #   inputs  = buffer[0 : ctx]        → token positions 0..ctx-1
                #   targets = buffer[1 : ctx + 1]    → next-token labels
                while len(buffer) >= ctx + 1:
                    chunk   = buffer[: ctx + 1]
                    buffer  = buffer[ctx:]          # advance by ctx (no overlap)
                    inputs  = np.array(chunk[:ctx],      dtype=np.int32)
                    targets = np.array(chunk[1: ctx + 1], dtype=np.int32)
                    n_yielded += 1
                    yield {"inputs": inputs, "targets": targets}

            log.info(
                "AutoDataPipeline: stream exhausted after %d chunks — restarting",
                n_yielded,
            )

    def __repr__(self) -> str:
        return (
            f"AutoDataPipeline(dataset={self._dataset_path!r}, "
            f"subset={self._subset!r}, split={self._split!r}, "
            f"ctx={self._max_context})"
        )
