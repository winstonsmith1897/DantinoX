---
title: "Tutorial: RAG with DantinoX Embedders"
---

# RAG with DantinoX Embedders

This tutorial trains a sentence encoder from scratch and builds a complete Retrieval-Augmented Generation (RAG) pipeline. No external embedding API required.

**What you will build:**

1. Train an `EmbedderParadigm` on `wikitext-2` (unsupervised, SimCSE)
2. Fine-tune it on sentence pairs extracted from the same corpus
3. Encode a document corpus into a FAISS index
4. Retrieve relevant passages for a query
5. Connect the retriever to LangChain and ChromaDB

**Prerequisites:** DantinoX installed, one GPU, `datasets` and `faiss-gpu` (or `faiss-cpu`) available.

---

## Step 1 — Train the Embedder (Unsupervised)

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # single GPU

import dantinox as dx

train_cfg = dx.TrainingConfig(
    dataset_source="huggingface",
    dataset_name="wikitext",
    dataset_config="wikitext-2-raw-v1",
    dataset_text_field="text",
    tokenizer_type="bpe",
    lr=3e-4,
    epochs=10,
    batch_size=64,
    warmup_steps=200,
    max_train_tokens=2_000_000,
)

cfg = dx.ModelConfig(
    dim=256, n_heads=4, head_size=64, num_blocks=6,
    vocab_size=8_000,   # overridden by the tokenizer at fit time
    causal=False,       # bidirectional encoder
    dropout=0.1,        # required for SimCSE
    max_context=256,
)
paradigm = dx.EmbedderParadigm(cfg, pooling="mean", temperature=0.05)

run_dir = dx.train(paradigm, training_config=train_cfg)
print("Unsupervised run:", run_dir)
```

Training ~10 epochs on wikitext-2 takes roughly **15 minutes** on a single A100.

---

## Step 2 — Fine-Tune on Sentence Pairs (Optional but Recommended)

Consecutive sentences in the same paragraph are semantically related and make good positive pairs.

```python
from datasets import load_dataset
from dantinox.utils.tokenizer import load_tokenizer_from_file

wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

def extract_sentence_pairs(dataset, max_pairs: int = 5_000):
    pairs = []
    for row in dataset:
        text = row["text"].strip()
        if not text or text.startswith(" ="):
            continue
        sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 40]
        for i in range(len(sentences) - 1):
            pairs.append((sentences[i], sentences[i + 1]))
            if len(pairs) >= max_pairs:
                return pairs
    return pairs

pairs = extract_sentence_pairs(wiki)
print(f"{len(pairs)} pairs extracted")

# Reuse the tokenizer from the unsupervised run
tok = load_tokenizer_from_file(f"{run_dir}/tokenizer.json")

# Use the same architecture
cfg_ft = dx.ModelConfig.from_yaml(f"{run_dir}/config.yaml")
paradigm_ft = dx.EmbedderParadigm(cfg_ft, pooling="mean", temperature=0.05)
pretrained_model = dx.load(run_dir, paradigm=paradigm_ft)

trainer = dx.EmbedderTrainer(
    paradigm_ft, tok,
    dx.TrainingConfig(lr=5e-5, epochs=5, batch_size=32),
)
run_dir_ft = trainer.fit_pairs(pairs, model=pretrained_model,
                               run_dir="runs/embedder_finetuned")
print("Fine-tuned run:", run_dir_ft)
```

---

## Step 3 — Build the Embedder

`Embedder.from_run()` loads the checkpoint and tokenizer automatically:

```python
embedder = dx.Embedder.from_run(run_dir_ft)
print(f"Embedding dimension: {embedder.dim}")
```

Test it:

```python
import numpy as np

vecs = embedder.embed(["JAX uses XLA to compile numerical code.",
                        "XLA is a domain-specific compiler for linear algebra."])
# Cosine similarity — vectors are already L2-normalised
sim = float(vecs[0] @ vecs[1])
print(f"Cosine similarity: {sim:.3f}")   # should be > 0.8
```

---

## Step 4 — Index Documents with FAISS

```python
import faiss

wiki_test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
documents = [
    row["text"].strip()
    for row in wiki_test
    if len(row["text"].strip()) > 80 and not row["text"].strip().startswith(" =")
][:1_000]

print(f"Indexing {len(documents)} documents...")
doc_vecs = embedder.embed(documents, batch_size=64)   # [N, D] np.float32

# IndexFlatIP = inner product; for L2-normalised vectors this equals cosine similarity
index = faiss.IndexFlatIP(embedder.dim)
index.add(doc_vecs.astype(np.float32))
print(f"Index built: {index.ntotal} vectors")
```

### Retrieve top-k

```python
def retrieve(query: str, k: int = 5) -> list[str]:
    q_vec = embedder.embed([query]).astype(np.float32)
    scores, indices = index.search(q_vec, k)
    return [(documents[i], float(s)) for i, s in zip(indices[0], scores[0])]

results = retrieve("How did the Roman Empire fall?")
for doc, score in results:
    print(f"[{score:.3f}] {doc[:120]}")
```

### Persist the index

```python
faiss.write_index(index, "runs/embedder_finetuned/faiss.index")

# Load later
index = faiss.read_index("runs/embedder_finetuned/faiss.index")
```

---

## Step 5a — LangChain Integration

`Embedder.as_langchain_embeddings()` returns a LangChain-compatible `Embeddings` object that can be used with any LangChain vector store.

```python
# pip install langchain langchain-community
from langchain_community.vectorstores import FAISS as LC_FAISS
from langchain_core.documents import Document

lc_embed = embedder.as_langchain_embeddings()

lc_docs = [Document(page_content=d) for d in documents]
store = LC_FAISS.from_documents(lc_docs, embedding=lc_embed)

# Semantic search
results = store.similarity_search("Roman Empire military campaigns", k=3)
for r in results:
    print(r.page_content[:120])
```

#### RAG with a generator

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Use any LangChain-compatible LLM
from langchain_community.llms import HuggingFacePipeline

retriever = store.as_retriever(search_kwargs={"k": 4})

prompt = ChatPromptTemplate.from_template(
    "Answer using only the context below.\n\nContext:\n{context}\n\nQuestion: {question}"
)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("What caused the decline of the Roman Empire?")
print(answer)
```

---

## Step 5b — ChromaDB Integration

`Embedder.as_chroma_fn()` returns a ChromaDB `EmbeddingFunction`:

```python
# pip install chromadb
import chromadb

client = chromadb.PersistentClient(path="chroma_db/")
col = client.get_or_create_collection(
    "wiki",
    embedding_function=embedder.as_chroma_fn(),
)

# Add documents in batches
batch_size = 100
for start in range(0, len(documents), batch_size):
    chunk = documents[start : start + batch_size]
    col.add(
        documents=chunk,
        ids=[str(i) for i in range(start, start + len(chunk))],
    )
print(f"Collection has {col.count()} documents")

# Query
results = col.query(query_texts=["Roman Empire"], n_results=3)
for doc in results["documents"][0]:
    print(doc[:120])
```

---

## Step 6 — Low-Level Access

For full control, use `encode_hidden()` and `embed_tokens()` directly:

```python
import jax.numpy as jnp

# From text
vecs = embedder.embed(["some text"], batch_size=32)   # np.ndarray [N, D]

# From pre-tokenized IDs (skip tokenizer overhead in tight loops)
ids  = jnp.array([[1, 2, 3, 4, 0, 0]])               # [B, T]
mask = jnp.array([[True, True, True, True, False, False]])
vecs = embedder.embed_tokens(ids, attention_mask=mask) # np.ndarray [B, D]

# From model directly (JAX array in, JAX array out)
out = model.encode_hidden(ids, pooling="mean", normalize=True, attention_mask=mask)
out.embeddings    # jnp.ndarray [B, D]
out.hidden_states # jnp.ndarray [B, T, D] — token-level features for re-ranking
```

---

## Performance Tips

| Tip | Effect |
|-----|--------|
| Larger `batch_size` during training | More in-batch negatives → stronger InfoNCE signal |
| `max_context=128` for short queries | Halves memory and compute vs `max_context=256` |
| `faiss.IndexIVFFlat` instead of `IndexFlatIP` | Approximate NN — 10–100× faster search for large corpora |
| `embedder.embed(docs, batch_size=128)` | Tune batch size to saturate GPU during offline indexing |
| `normalize=True` (default) | Required for correct cosine similarity with `IndexFlatIP` |

---

## See Also

- [Retriever Paradigm](../paradigms/retriever.md) — architecture and InfoNCE math
- [Embedder Training Guide](../training/retriever.md) — all training modes
- [Notebook 08](../../docs/notebooks/08_retrievers_training.ipynb) — runnable Colab notebook
