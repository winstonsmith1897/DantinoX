---
title: Retriever (Embedder)
---

# Retriever — Sentence Embeddings for RAG

DantinoX can turn **any of its transformer architectures** into a dense sentence encoder for Retrieval-Augmented Generation (RAG) and semantic search. A single new paradigm — `EmbedderParadigm` — adds contrastive training on top of the same `Transformer` backbone used for AR and discrete diffusion.

---

## At a Glance

| | **EmbedderParadigm** |
|---|---|
| Model backbone | `Transformer` (causal or bidirectional) |
| Recommended directionality | Bidirectional (`causal=False`) |
| Training loss | Symmetric InfoNCE (NT-Xent) |
| Unsupervised mode | SimCSE — two dropout views of the same batch |
| Supervised mode | `EmbedderTrainer.fit_pairs()` on (anchor, positive) pairs |
| Output | `[B, D]` L2-normalised embeddings |
| Pooling | `mean` (default for bidir) · `last` (default for causal) · `cls` |
| Integration | FAISS · ChromaDB · LangChain · LlamaIndex |

---

## Architecture

The embedder re-uses the `Transformer` backbone unchanged. The only difference from generative use is **where the output is taken**: instead of projecting hidden states to token logits, `encode_hidden()` pools the final hidden states and returns a fixed-size vector.

```
Token IDs [B, T]
     │
  Embedding + Positional Encoding
     │
  Transformer Blocks (shared with AR / Diffusion)
     │
  Final LayerNorm
     │
  Hidden States [B, T, D]
     │
  ┌──────────────────────────┐
  │  Pooling (mean / last / cls) │  ← encode_hidden()
  └──────────────────────────┘
     │
  Embedding [B, D]
     │
  L2 Normalisation
     │
  Sentence Vector [B, D]
```

The logit projection head (`Linear(D, V)`) is never called during encoding — it remains in the checkpoint but is simply not executed.

---

## Why Bidirectional?

Bidirectional attention (`causal=False`) lets every token attend to the full sequence before pooling. This is the same architecture used by BERT, E5, and BGE. For sentence embeddings it consistently outperforms causal models because:

- **mean pooling** averages information from all positions, not just the last.
- Rare tokens in the middle of the sequence receive gradients from both directions.
- There is no positional bias from the left-boundary.

Causal models can still be used (`causal=True`, `pooling="last"`), but expect lower retrieval quality at equal model size.

---

## Training Objective — InfoNCE

The training loss is **symmetric InfoNCE** (also called NT-Xent or MNRL):

$$
\mathcal{L} = -\frac{1}{2B} \left[
\sum_{i=1}^{B} \log \frac{e^{\mathbf{a}_i \cdot \mathbf{p}_i / \tau}}{\sum_{j=1}^{B} e^{\mathbf{a}_i \cdot \mathbf{p}_j / \tau}}
+
\sum_{i=1}^{B} \log \frac{e^{\mathbf{p}_i \cdot \mathbf{a}_i / \tau}}{\sum_{j=1}^{B} e^{\mathbf{p}_j \cdot \mathbf{a}_j / \tau}}
\right]
$$

where $\mathbf{a}_i, \mathbf{p}_i \in \mathbb{R}^D$ are L2-normalised anchor and positive embeddings, $\tau$ is the temperature, and the denominator sums over all $B$ samples in the batch (in-batch negatives).

The symmetry (both directions averaged) doubles the number of contrastive signals per batch at zero additional cost.

### Temperature $\tau$

Temperature controls how peaked the similarity distribution is.

| $\tau$ | Effect |
|--------|--------|
| `0.02–0.05` | Sharp — strong signal but gradients saturate quickly if the model is already good |
| `0.07–0.10` | Recommended starting point — balances signal strength and stability |
| `0.20–0.50` | Soft — stable but slow convergence |

The default in DantinoX is `temperature=0.05`, consistent with SimCSE and E5.

---

## Unsupervised Mode — SimCSE

When no labelled pairs are available, `EmbedderParadigm` generates its own positives using **dropout as augmentation** (SimCSE, Gao et al., 2021).

The same token window `x` is encoded **twice** in the same forward pass. Because NNX advances the dropout RNG state between calls, each pass sees a different set of zeroed-out attention weights — producing slightly different representations of the same sequence. These two views serve as the anchor/positive pair.

```python
# Inside loss_fn — two forward passes, different dropout masks
out_a = model.encode_hidden(batch, deterministic=False)   # view 1
out_p = model.encode_hidden(batch, deterministic=False)   # view 2 (different mask)
loss  = info_nce_loss(out_a.embeddings, out_p.embeddings, temperature)
```

!!! warning "Dropout must be > 0"
    SimCSE requires `dropout > 0` in `ModelConfig`. With `dropout=0.0` both forward passes produce identical representations, the InfoNCE diagonal is always 1, and the loss provides no gradient.  
    `EmbedderParadigm` will warn you at construction time if `dropout=0.0`.

This mode integrates with the standard `Trainer` and any flat text corpus — no pair preparation needed.

---

## Supervised Mode — Labelled Pairs

When you have (anchor, positive) pairs — FAQ-answer, query-document, NLI premise-hypothesis, paraphrases — use `EmbedderTrainer.fit_pairs()`. Each pair is encoded independently (different sequences, not just different dropout), so the positive signal is stronger and more semantically precise.

Accepted data formats:

| Format | Example |
|--------|---------|
| List of tuples | `[("anchor", "positive"), ...]` |
| JSONL | `{"anchor": "...", "positive": "..."}` |
| TSV | `anchor\tpositive` |

---

## Pooling Strategies

`encode_hidden()` supports three pooling modes:

| `pooling` | Description | When to use |
|-----------|-------------|-------------|
| `"mean"` | Average of all token hidden states (mask-aware) | Bidirectional models — default |
| `"last"` | Hidden state of the last non-padding token | Causal (AR) models — default |
| `"cls"` | Hidden state of the first token (position 0) | When a dedicated `[CLS]` token is prepended |
| `"auto"` | Selects `"last"` for causal, `"mean"` otherwise | Recommended — safe for both |

---

## `encode_hidden` — Low-level API

```python
out = model.encode_hidden(
    x,                      # [B, T] token IDs
    pooling="mean",         # "auto" | "mean" | "last" | "cls"
    normalize=True,         # L2-normalize (required for cosine similarity)
    attention_mask=mask,    # [B, T] bool, True = valid token (optional)
    deterministic=True,     # False during training for SimCSE augmentation
)

out.embeddings    # [B, D]  — pooled, normalised sentence vectors
out.hidden_states # [B, T, D] — per-token representations
```

---

## Quick Reference

```python
import dantinox as dx

# ── Build the paradigm ────────────────────────────────────────────────────────
cfg = dx.ModelConfig(
    dim=256, n_heads=4, head_size=64, num_blocks=6,
    vocab_size=32_000,
    causal=False,   # bidirectional encoder
    dropout=0.1,    # required for SimCSE
)
paradigm = dx.EmbedderParadigm(cfg, pooling="mean", temperature=0.05)

# ── Unsupervised training (flat corpus) ───────────────────────────────────────
run_dir = dx.fit(
    "embedder", "data/corpus.txt",
    dim=256, n_heads=4, head_size=64, num_blocks=6,
    vocab_size=32_000, dropout=0.1,
    lr=3e-4, epochs=10, batch_size=64,
)

# ── Inference ─────────────────────────────────────────────────────────────────
embedder = dx.Embedder.from_run(run_dir)
vecs = embedder.embed(["hello world", "foo bar"])  # np.ndarray [2, D]
```

---

## See Also

- [Embedder Training Guide](../training/retriever.md) — unsupervised, supervised, fine-tuning
- [RAG Tutorial](../tutorials/retriever-rag.md) — end-to-end with FAISS and LangChain
- [API Reference — Embedder](../api/paradigms.md#sentence-embedder)
