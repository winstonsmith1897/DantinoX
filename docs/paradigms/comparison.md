---
title: AR vs Diffusion Comparison
---

# Autoregressive vs Masked Diffusion — Head-to-Head

Both paradigms are trained on the same corpus with the same architecture.
This page summarises the expected tradeoffs based on DantinoX's systematic
benchmark suite across three attention types (MHA · GQA · MLA).

The quantitative results reported here are produced by the DantinoX systematic training pipeline. For full experimental details — including the training matrix, ablation design, and evaluation methodology — see the [Experiments & Results](../paper.md) page. All numerical entries marked with "—" are populated by running `bash scripts/run_full_emnlp.sh`; the resulting CSVs are written to `results/perplexity.csv`, `results/generation_quality.csv`, and `results/benchmark_results.csv`.

---

## Research Design

The comparison between AR and Masked Diffusion in DantinoX is structured as a controlled experiment: **the only variable between any pair of compared checkpoints is `model_type`**. All other configuration axes are held constant:

- **Corpus:** WikiText-103 raw (v1), character-level tokenisation, pre-tokenised and cached identically for all runs.
- **Architecture base:** identical `dim`, `n_heads`, `head_size`, `num_blocks`, `kv_heads`, MLA parameters, MoE configuration, normalisation type, and positional encoding.
- **Training regime:** same optimiser, learning rate, schedule, batch size, gradient accumulation, bfloat16 precision, flash attention, and gradient checkpointing.
- **Hardware:** 2× NVIDIA A100 40 GB with JAX SPMD data parallelism across both GPUs.

This design ensures that any measured difference in perplexity, throughput, or generation quality is attributable to the generation paradigm rather than to confounding factors. The 12-ablation Part B suite further isolates the effect of individual hyperparameter choices (normalisation, dropout, activation, attention span, LR schedule, optimiser, batch size, context length) for each attention type independently.

The primary controlled variable at inference time is the decoding procedure: AR generates tokens left-to-right with a static KV cache, while Masked Diffusion generates all positions in parallel over K denoising steps with Fast-dLLM DualCache (prefix KV + suffix KV for remaining MASK positions). The confidence threshold τ and block size f are additional inference-time hyperparameters for diffusion that have no AR counterpart.

---

## Architecture Differences

| | AR (`Transformer`) | Diffusion (`DiffusionTransformer`) |
|---|---|---|
| Class | `core.model.Transformer` | `core.model.DiffusionTransformer` (alias of `Transformer` — a single unified class serves both paradigms) |
| Block type | `Block` | `DiffusionBlock` (alias of `Block`) |
| Attention mask | Causal | Full (bidirectional) |
| Time conditioning | — | **None.** DantinoX's discrete diffusion does not condition on `t` at all — no `AdaLayerNorm`, no time-embedding MLP. (`AdaLayerNorm` exists in `core/block.py` but is unused dead code for this paradigm; see [Discrete Diffusion](diffusion.md).) |
| Extra parameters | — | None beyond the shared backbone |
| KV-cache type | Static KV | DualCache (prefix + suffix) |
| Decode step cost | $O(T_{\text{gen}})$ | $O(T_{\text{gen}})$ (block-wise) |
| Total decode cost | $O(T_{\text{gen}}^2)$ | $O(K \cdot B \cdot N_{\text{steps}})$ |

`Transformer` and `DiffusionTransformer` are literally the same class
(`DiffusionTransformer = Transformer` in `core/model.py`), and likewise
`DiffusionBlock = Block` in `core/block.py` — this is the point of DantinoX's
"single backbone" design: the paradigm only changes which attention mask and
loss function are used, not which classes are instantiated.

---

## Quality

### Language Modelling (bpb on WikiText-103)

Evaluated with sliding-window perplexity; AR uses standard cross-entropy loss, Diffusion uses ELBO at a uniform timestep grid across 1000 (or 500, in the `T500` ablation) noise levels. Results are reported in bits-per-byte (bpb = log₂(e) × NLL / sequence_length), which is tokeniser-agnostic and directly comparable across character-level models.

| Model | AR bpb ↓ | Diffusion ELBO-bpb ↓ | AR − Diff |
|:------|:--------:|:--------------------:|:---------:|
| MHA 256d 12b Dense | — | — | — |
| GQA 256d 12b Dense | — | — | — |
| MLA 256d 12b Dense | — | — | — |
| MHA 512d 12b Dense | — | — | — |
| GQA 512d 12b Dense | — | — | — |
| MLA 512d 12b Dense | — | — | — |

:::{admonition} Populating this table
:class: note

Run the full pipeline to populate perplexity results:

```bash
bash scripts/run_full_emnlp.sh --skip-benchmarks
# or for evaluation only (requires trained checkpoints):
python benchmarks/perplexity_eval.py \
    --runs-dir runs --run-prefix ar_ diff_ \
    --datasets wikitext-103 ptb lambada c4 \
    --out results/perplexity.csv
```

Results are written to `results/perplexity.csv`. See the [Experiments & Results](../paper.md) page for stage E3 documentation.
:::

### Cross-Dataset Generalisation (bpb)

The same trained checkpoints are also evaluated on Penn Treebank (PTB), LAMBADA, and C4 to assess out-of-domain generalisation. All models are trained on WikiText-103 only.

| Dataset | Domain | AR (MHA 256d) | Diff (MHA 256d) | AR (MLA 256d) | Diff (MLA 256d) |
|:--------|:-------|:-------------:|:---------------:|:-------------:|:---------------:|
| WikiText-103 | Wikipedia (in-domain) | — | — | — | — |
| Penn Treebank | Wall Street Journal | — | — | — | — |
| LAMBADA | Long-range coherence | — | — | — | — |
| C4 | Web crawl (coarse) | — | — | — | — |

:::{admonition} Populating this table
:class: note

Generated by stage E3 of `bash scripts/run_full_emnlp.sh`. Source: `results/perplexity.csv`.
:::

### Long-Range Coherence (LAMBADA)

LAMBADA tests last-word prediction accuracy given a long paragraph of context. The bidirectional attention of masked diffusion models allows the decoder to condition on tokens to the right of the target position, giving it a structural advantage over strictly causal AR models, which cannot attend to future tokens.

From theory (Austin et al., 2021; Nie et al., 2023), masked diffusion is expected to achieve lower bpb on LAMBADA than AR at matched parameter counts, with the advantage increasing as context length grows. The `Ctx1024` ablation is specifically designed to test whether this gap widens with longer sequences.

---

## Efficiency

### Throughput (tok/s, BS=1, 256-token generation)

Throughput is measured on bfloat16 models on a single A100 40 GB. AR throughput uses the static KV cache; Diffusion throughput is measured both with naive full-forward per denoising step and with Fast-dLLM DualCache (prefix KV + suffix KV).

Numbers below are from the DantinoX systematic inference sweep (`dantinox infbench`) using representative model sizes. Run `dantinox infbench --trained` on your own checkpoints to get numbers specific to your trained models.

| Model | AR (tok/s) | Diff simple (tok/s) | Diff DualCache (tok/s) | DualCache speedup |
|:------|:----------:|:-------------------:|:----------------------:|:-----------------:|
| MHA 256d 12b | ~106 | ~52 | ~94 | ~1.8× |
| GQA 256d 12b | ~63 | ~50 | ~87 | ~1.7× |
| MLA 256d 12b | ~63 | ~40 | ~72 | ~1.8× |
| MHA 512d 12b | ~64 | ~31 | ~57 | ~1.8× |
| GQA 512d 12b | ~62 | ~30 | ~54 | ~1.8× |
| MLA 512d 12b | ~49 | ~24 | ~44 | ~1.8× |

:::{admonition} Running the benchmark on trained checkpoints
:class: note

To get throughput for your trained models specifically:

```bash
python benchmarks/trained_analysis.py \
    --runs-dir runs --run-prefix ar_ diff_ \
    --out-csv results/benchmark_results.csv \
    --n-warmup 3 --n-trials 20
```

Source: `results/benchmark_results.csv`.
:::

### Throughput vs. Batch Size

At larger batch sizes the parallel decode advantage of masked diffusion — which decodes all positions simultaneously per denoising step — is expected to compound relative to the strictly sequential AR decode. Stage E2 sweeps batch sizes from 1 to 128 at sequence length 512 for all trained checkpoints.

:::{admonition} Populating this table
:class: note

Generated by stage E2: `results/batch_sweep_results.csv`, plotted as part of figure set F3.
See `results/paper_figures/` after running `bash scripts/run_full_emnlp.sh`.
:::

### KV-Cache Memory

AR and Diffusion share the same attention mechanism, so KV-cache footprint
is **identical** for the same attention type at inference time.

| Attention | KV-MB @ 512 tok, 12L |
|---|---|
| MHA | 384 KB |
| GQA (×4) | 96 KB |
| MLA | ~23 KB |

For Diffusion with DualCache, the **suffix KV** adds overhead proportional to
the number of remaining MASK blocks.  Averaged across a full generation this
adds ~20–40% to the peak cache size.

---

## Generation Quality

### Open-Ended Generation Metrics

Beyond perplexity, which measures the model's confidence over held-out reference text, open-ended generation quality is evaluated using metrics that assess diversity, repetition, and distributional similarity to human-written text. All metrics are computed over 100 unconditional generations of 128 tokens per checkpoint.

| Metric | Description | Direction |
|:-------|:------------|:---------:|
| Distinct-1 | Fraction of unique unigrams in the output | ↑ |
| Distinct-2 | Fraction of unique bigrams in the output | ↑ |
| Self-BLEU | Average BLEU of each generation against the rest of the sample | ↓ |
| Rep-4 | Fraction of 4-gram repetitions within a single generation | ↓ |
| MAUVE | KL-based distributional distance to human text (higher = closer) | ↑ |

| Model | Distinct-1 ↑ | Distinct-2 ↑ | Rep-4 ↓ | Notes |
|:------|:-----------:|:-----------:|:-------:|:------|
| Diff MHA 512d 12b (28K steps) | 0.467 | 0.808 | 0.004 | Intermediate checkpoint snapshot, mixed EN/FR/DE, Wikipedia structure — **not** the paper's reported Table 2 numbers (different training budget/seed count) |
| Flow MHA 512d 12b (9.5K steps) | 0.365 | 0.851 | 0.002 | Intermediate checkpoint snapshot, early training — **not** the paper's reported Table 2 numbers |
| AR MHA 256d 12b | — | — | — | Run `generation_quality.py` to populate |
| Diff MHA 256d 12b | — | — | — | |
| AR GQA 256d 12b | — | — | — | |
| Diff GQA 256d 12b | — | — | — | |

:::{admonition} These two populated rows are ad-hoc snapshots, not the paper's results
:class: warning

The `Diff MHA 512d 12b` and `Flow MHA 512d 12b` rows above were captured
from one-off intermediate checkpoints (28K and 9.5K steps respectively)
during development, and use a different metric set (Distinct-1/2, Rep-4
only, no MAUVE/BLEU-4cond, no seed averaging). They should **not** be
read as reproducing the paper's Small-scale generation-quality table —
see the authoritative numbers below.
:::

:::{admonition} Populating this table
:class: note

Run the generation quality evaluation for your trained checkpoints:

```bash
python benchmarks/generation_quality.py \
    --runs-dir runs --run-prefix ar_ diff_ elf_ \
    --n-samples 100 --gen-len 128 \
    --out results/generation_quality.csv
```

Or use the qualitative test script for a quick read:

```bash
python scripts/test_generation_quality.py runs/diff_mha_512d_12b_Dense
```

Source: `results/generation_quality.csv`.
:::

From theory, masked diffusion models with bidirectional context are expected to exhibit higher Distinct-2 and MAUVE scores at matched parameter counts, at the cost of requiring multiple denoising steps. The confidence sweep (stage B3) explores the Pareto frontier between generation quality and throughput by varying the per-token confidence threshold τ.

### Paper's reported results (authoritative)

The EMNLP System Demo paper reports all nine paradigm × attention
combinations trained under one identical recipe (WikiText-103, Muon
optimizer, SentencePiece/T5 tokenizer, effective batch of 256 sequences of
512 tokens) at Small scale (512-d, 12-layer, ~65–82M params). Metrics are
mean ± std over 3 generation seeds, 100 unconditional 128-token samples,
matched inference budget of 64 steps per paradigm:

| Arch (Params) | MAUVE ↑ | PPL<sub>GPT-2</sub> ↓ | Distinct-2 ↑ | Rep-4 ↓ | BLEU-4<sub>cond</sub> ↑ |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Autoregressive** | | | | | |
| MHA (67M) | 0.17±.03 | **1216**±5 | **0.697**±.000 | 0.007±.000 | 0.052±.000 |
| GQA (62M) | 0.18±.06 | 1233±3 | 0.688±.001 | 0.005±.000 | 0.050±.004 |
| MLA (65M) | 0.07±.01 | 1860±3 | 0.682±.001 | **0.003**±.000 | 0.020±.002 |
| **Discrete Diffusion** | | | | | |
| MHA (70M) | 0.20±.04 | 1834±43 | 0.728±.006 | 0.019±.010 | 0.029±.000 |
| GQA (65M) | 0.12±.00 | 1777±57 | 0.733±.003 | **0.006**±.000 | 0.030±.003 |
| MLA (70M) | 0.15±.05 | 1803±1 | 0.720±.008 | 0.020±.009 | 0.033±.003 |
| **Continuous Flow-Matching** | | | | | |
| MHA (82M) | 0.80±.04 | 234.6±2.0 | **0.627**±.001 | **0.007**±.000 | — |
| GQA (77M) | 0.69±.09 | 188.0±0.4 | 0.562±.001 | 0.027±.000 | — |
| MLA (79M) | 0.78±.07 | **156.3**±0.3 | 0.538±.001 | 0.107±.000 | — |

BLEU-4<sub>cond</sub> is "—" for Flow-Matching: the ELF formulation supports
prefix conditioning in principle, but DantinoX's current implementation does
not yet expose it (planned future work). At this scale, Flow-Matching is the
most fluent (lowest PPL, highest MAUVE), Diffusion the most lexically
diverse, and AR gives the most accurate/least-repetitive conditional
continuations — the paper makes no claim that these rankings persist at
larger scale or on other corpora.

---

## When to Use Which

The following guidance is grounded in the theoretical properties of each paradigm and is intended to be refined by the empirical findings once the full evaluation pipeline has been run.

### Use Autoregressive when:

- **Latency-critical streaming** generation is required (token-by-token output, first-token latency matters): AR produces tokens incrementally, while diffusion must complete at least one full denoising pass before any output is available.
- Tasks where **strict left-to-right causal structure** is essential (code completion with execution-time feedback, prefix-constrained generation): the causal mask provides a principled inductive bias for sequential generation.
- **Deployment simplicity**: AR inference requires only the forward pass and a KV cache; diffusion inference requires a noise schedule, a mask token, a denoising loop, and (optionally) a confidence estimator.
- **Integration with existing AR ecosystems**: sampling libraries (nucleus sampling, beam search, speculative decoding) and evaluation harnesses (lm-eval-harness) are designed for AR models.

### Use Masked Diffusion when:

- **Infilling and constrained editing**: native `[MASK]` tokens provide a direct mechanism for targeted completion of arbitrary spans, without the approximations required by AR-based infilling (e.g., fill-in-the-middle reordering).
- **Long-range coherence on bidirectional tasks**: by design, diffusion models attend to both left and right context at every denoising step, which theory predicts should yield lower bpb on LAMBADA-style last-word prediction tasks.
- **Throughput-optimised batch serving**: when all positions are decoded in parallel, the decode cost is $O(K \cdot B \cdot N_{\text{steps}})$ (where $K$ is block size and $N_{\text{steps}}$ is the number of denoising steps), which can be amortised over large batches more efficiently than the $O(T_{\text{gen}}^2)$ cost of AR decoding.
- **Research into discrete non-autoregressive generation**: the masked diffusion framework provides a principled probabilistic foundation (Austin et al., 2021) for studying discrete-space generative models beyond the autoregressive factorisation.

---

## Side-by-Side Code

```python
from dantinox.core.config import Config
from dantinox.core.model import Transformer, DiffusionTransformer
import flax.nnx as nnx

base = dict(
    dim=256, n_heads=8, head_size=32, num_blocks=12,
    max_context=512, kv_heads=2, vocab_size=100,
)

# ── Autoregressive ────────────────────────────────────────────────────────
ar_cfg   = Config(**base, model_type="autoregressive")
ar_model = Transformer(ar_cfg, rngs=nnx.Rngs(0))

from dantinox.core.generation import generate
tokens_ar = generate(ar_model, prompt_ids, max_generations=128,
                     top_p=0.9, use_cache=True)

# ── Masked Diffusion ──────────────────────────────────────────────────────
diff_cfg   = Config(**base, model_type="diffusion",
                    diffusion_steps=1000, noise_schedule="cosine")
diff_model = DiffusionTransformer(diff_cfg, rngs=nnx.Rngs(0))

from dantinox.core.generation import fast_dllm_generate
from dantinox.core.diffusion import make_noise_schedule

schedule    = make_noise_schedule(diff_cfg)
tokens_diff = fast_dllm_generate(
    diff_model, prompt_ids, gen_len=128,
    schedule=schedule, mask_token_id=4,
    block_size=32, steps_per_block=20,
    confidence_threshold=0.9,
)
```

---

## Training Configuration Differences

The only config changes needed to switch paradigms:

```yaml
# Autoregressive (default_config.yaml)
model:
  model_type: "autoregressive"

# Masked Diffusion (diffusion_base.yaml)
model:
  model_type: "diffusion"
diffusion:
  diffusion_steps: 1000
  noise_schedule: "cosine"
  mask_token_id: 4
  num_sampling_steps: 50
```

(`time_emb_dim` is not used by diffusion — it belongs to the flow-matching fields; see [Configuration Reference](../configuration.md).)

All other fields — `dim`, `n_heads`, `num_blocks`, `kv_heads`, `mla`, `use_moe` —
are shared.  This enables **controlled comparisons** where the only variable
is the paradigm.
