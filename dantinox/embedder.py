"""dantinox.embedder — turn any DantinoX model into a sentence encoder for RAG.

Usage
-----
**Quick — from a run directory**::

    import dantinox as dx

    embedder = dx.Embedder.from_run("runs/my_run")
    vecs = embedder.embed(["hello world", "foo bar"])   # np.ndarray [2, D]

**Manual — bring your own model + tokenizer**::

    embedder = dx.Embedder(model, tokenizer, pooling="mean")
    vecs = embedder.embed(texts, batch_size=64)

**With a paradigm**::

    paradigm = dx.ARParadigm(config)
    embedder = dx.Embedder.from_run("runs/my_run", paradigm=paradigm)

Integration with vector stores
-------------------------------
*FAISS*::

    import faiss, numpy as np
    vecs = embedder.embed(corpus)
    index = faiss.IndexFlatIP(embedder.dim)   # inner product = cosine when normalized
    index.add(vecs)

*ChromaDB*::

    import chromadb
    client = chromadb.Client()
    col = client.create_collection("docs", embedding_function=embedder.as_chroma_fn())
    col.add(documents=corpus, ids=[str(i) for i in range(len(corpus))])

*LangChain*::

    from langchain.vectorstores import FAISS as LC_FAISS
    vs = LC_FAISS.from_texts(corpus, embedding=embedder.as_langchain_embeddings())
"""

from __future__ import annotations

import os
from typing import Any

import jax.numpy as jnp
import numpy as np

from dantinox.core.model import Transformer


class Embedder:
    """Sentence encoder built on top of any DantinoX model.

    Parameters
    ----------
    model:        A trained ``Transformer`` (AR, Discrete diffusion).
    tokenizer:    Any object with ``encode(str) -> list[int]``.
    pooling:      ``"auto"`` | ``"mean"`` | ``"last"`` | ``"cls"``.
                  ``"auto"`` picks ``"last"`` for causal models, ``"mean"`` otherwise.
    normalize:    L2-normalize output vectors (recommended for cosine similarity).
    max_length:   Truncate / pad sequences to this length.  Defaults to the
                  model's ``max_context``.
    pad_id:       Token ID used for padding.  Defaults to ``0``.
    """

    def __init__(
        self,
        model: Transformer,
        tokenizer: Any,
        *,
        pooling: str = "auto",
        normalize: bool = True,
        max_length: int | None = None,
        pad_id: int = 0,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.pooling = pooling
        self.normalize = normalize
        self.max_length = max_length or model.max_context
        self.pad_id = pad_id

        # cache the embedding dimension
        self._dim: int = model.embed.features

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def dim(self) -> int:
        """Embedding dimension."""
        return self._dim

    def embed(
        self,
        texts: list[str] | str,
        *,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Encode *texts* and return L2-normalised embeddings ``[N, D]``.

        Parameters
        ----------
        texts:      A single string or a list of strings.
        batch_size: Number of samples processed per JAX call.

        Returns
        -------
        ``np.ndarray`` with shape ``[len(texts), dim]``, dtype ``float32``.
        """
        if isinstance(texts, str):
            texts = [texts]

        all_vecs: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            vecs = self._encode_batch(chunk)
            all_vecs.append(np.array(vecs))

        return np.concatenate(all_vecs, axis=0)

    def embed_tokens(
        self,
        token_ids: jnp.ndarray,
        *,
        attention_mask: jnp.ndarray | None = None,
    ) -> np.ndarray:
        """Encode pre-tokenized integer arrays ``[B, T]`` → ``[B, D]``."""
        out = self.model.encode_hidden(
            token_ids,
            pooling=self.pooling,
            normalize=self.normalize,
            attention_mask=attention_mask,
        )
        return np.array(out.embeddings)

    # ── Integration helpers ───────────────────────────────────────────────────

    def as_langchain_embeddings(self):
        """Return a LangChain-compatible ``Embeddings`` object.

        Usage::

            from langchain.vectorstores import FAISS
            store = FAISS.from_texts(docs, embedding=embedder.as_langchain_embeddings())
        """
        try:
            from langchain_core.embeddings import Embeddings  # type: ignore[import]
        except ImportError:
            from langchain.embeddings.base import Embeddings  # type: ignore[import]

        outer = self

        class _LCEmbeddings(Embeddings):
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return outer.embed(texts).tolist()

            def embed_query(self, text: str) -> list[float]:
                return outer.embed([text])[0].tolist()

        return _LCEmbeddings()

    def as_chroma_fn(self):
        """Return a ChromaDB ``EmbeddingFunction``.

        Usage::

            col = client.create_collection("docs", embedding_function=embedder.as_chroma_fn())
        """
        try:
            from chromadb import Documents, EmbeddingFunction  # type: ignore[import]
            from chromadb import Embeddings as ChromaEmbs
        except ImportError as exc:
            raise ImportError("Install chromadb: pip install chromadb") from exc

        outer = self

        class _ChromaFn(EmbeddingFunction):
            def __call__(self, input: Documents) -> ChromaEmbs:
                return outer.embed(list(input)).tolist()

        return _ChromaFn()

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_run(
        cls,
        run_dir: str,
        *,
        paradigm=None,
        pooling: str = "auto",
        normalize: bool = True,
        pad_id: int = 0,
    ) -> Embedder:
        """Build an Embedder from a DantinoX run directory.

        Loads the best checkpoint and the saved tokenizer automatically.

        Parameters
        ----------
        run_dir:  Path returned by ``dx.fit()`` or ``dx.train()``.
        paradigm: Optional paradigm (used to restore weights via its
                  ``build_model``).  When ``None`` falls back to
                  ``Transformer.from_pretrained``.
        """
        import dantinox as dx
        from dantinox.utils.tokenizer import load_tokenizer_from_file

        tok_path = os.path.join(run_dir, "tokenizer.json")
        if not os.path.exists(tok_path):
            raise FileNotFoundError(
                f"No tokenizer.json found in {run_dir!r}. "
                "Pass tokenizer= explicitly to Embedder()."
            )
        tokenizer = load_tokenizer_from_file(tok_path)
        model = dx.load(run_dir, paradigm=paradigm)

        return cls(model, tokenizer, pooling=pooling, normalize=normalize, pad_id=pad_id)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _tokenize_and_pad(self, texts: list[str]) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Tokenize, truncate, and pad a batch of strings.

        Returns
        -------
        ids:  ``[B, T]`` int32
        mask: ``[B, T]`` bool (True = real token)
        """
        encoded = [self.tokenizer.encode(t)[: self.max_length] for t in texts]
        T = max(len(e) for e in encoded)

        ids  = np.full((len(encoded), T), self.pad_id, dtype=np.int32)
        mask = np.zeros((len(encoded), T), dtype=bool)
        for i, e in enumerate(encoded):
            ids[i, : len(e)] = e
            mask[i, : len(e)] = True

        return jnp.asarray(ids), jnp.asarray(mask)

    def _encode_batch(self, texts: list[str]) -> jnp.ndarray:
        ids, mask = self._tokenize_and_pad(texts)
        out = self.model.encode_hidden(
            ids,
            pooling=self.pooling,
            normalize=self.normalize,
            attention_mask=mask,
        )
        return out.embeddings
