# Benchmarks

DantinoX ships two complementary benchmark suites that together cover the full
lifecycle of a model — from raw inference primitives on randomly-initialised
networks to end-to-end quality/throughput measurements on real trained
checkpoints.

| Suite | What it measures | Entry point |
|---|---|---|
| **Inference sweep** | Latency, throughput, KV-cache, FLOPs across 13 sweep groups and 3 attention variants (MHA / GQA / MLA) on randomly-initialised models | `make infbench` |
| **Trained-model analysis** | Decode throughput, prefill latency, measured VRAM, XLA FLOPs, and validation loss on real trained checkpoints | `make trained-bench` |

---

## Running Benchmarks

### Full inference pipeline

```bash
# Sweep + 21 plots (default)
make infbench

# Quick smoke-test (fewer warm-up / trial iterations)
python benchmarks/run_all.py --n-warmup 1 --n-trials 3

# Restrict to a subset of sweep groups
python benchmarks/run_all.py --groups attention_type scale batch_size

# Re-plot from an existing CSV without re-running the sweep
python benchmarks/run_all.py --plot-only

# Via the CLI
dantinox infbench --groups scale --n-trials 5 --device 1
```

### Trained-model pipeline

```bash
# Run analysis + batch sweep on checkpoints in runs/
make trained-bench

# Both pipelines in one command
python benchmarks/run_all.py --trained

# Trained pipeline only (skip inference sweep)
python benchmarks/run_all.py --trained --inference-off

# Via the CLI
dantinox infbench --trained --runs-dir runs/
```

### Pipeline stages

```
Stage 1  benchmarks/inference_sweep.py    →  results/inference_sweep.csv
Stage 2  benchmarks/plot_inference.py     →  results/plots/*.png   (21 figures)
Stage 3  benchmarks/trained_analysis.py   →  results/benchmark_results.csv
Stage 4  benchmarks/trained_batch_sweep.py→  results/batch_sweep_results.csv
```

Each stage runs in its own subprocess so JAX state and compiled functions
never conflict between stages. Stages 3–4 only execute when `--trained` is
passed.

---

## Inference Sweep

Systematic performance comparison of **MHA**, **GQA**, and **MLA** attention
variants across 13 orthogonal sweep groups. All models are randomly initialised
— results are pure infrastructure benchmarks, independent of training quality.

Figures are produced by `benchmarks/plot_inference.py`. Each panel shows the
three attention variants as grouped bars (or scatter points) so crossover
effects are immediately visible.

### Attention Type

Overall latency and throughput comparison across the three attention families at
fixed model size. MLA's extra projection steps (`W_DKV`, `W_UV`, `W_UK`)
produce higher prefill latency but a proportionally smaller KV cache.

![01 Attention Type](assets/plots/inference/01_attention_type.png)

### Scale

Prefill latency and decode throughput as model dimension (`dim`) and depth
(`num_blocks`) scale from small to large. MLA latency grows faster than MHA/GQA
because its low-rank projections add compute that is not amortised across heads.

![02 Scale](assets/plots/inference/02_scale.png)

### Batch Size

Throughput (tok/s) at batch sizes 1 → 32. At small batch sizes all three
variants are weight-bandwidth-bound and behave similarly. At large batches the
KV-cache bottleneck surfaces: MLA's compact cache keeps VRAM pressure lower.

![03 Batch Size](assets/plots/inference/03_batch_size.png)

### Context Length

Latency and cache growth across prompt lengths 64 → 2048. The quadratic
attention cost is visible for all variants; MLA's cache slope is 5–10× flatter
than MHA.

![04 Context Length](assets/plots/inference/04_context_len.png)

### Dtype

`float32` vs `bfloat16` latency and memory. `bfloat16` halves activation memory
and accelerates matrix operations, giving a consistent 30–50% latency reduction
without quality loss in practice.

![05 Dtype](assets/plots/inference/05_dtype.png)

### KV Cache

Theoretical KV cache footprint (MB) per variant at different model sizes.
MLA's `down_dim_kv` dimension directly controls cache independent of model
width — the only variant where cache and model size are decoupled.

![06 KV Cache](assets/plots/inference/06_kv_cache.png)

### MoE vs. Dense

Mixture-of-Experts (`use_moe=True`) vs. Dense FFN at matched parameter counts.
MoE adds routing overhead but keeps active FLOPs constant, making it
particularly attractive paired with MLA's compact cache.

![07 MoE](assets/plots/inference/07_moe.png)

### Activation Function

SwiGLU vs. GELU latency. SwiGLU requires an extra gate projection but the
fused kernel keeps overhead minimal; the difference is negligible vs. attention
cost at long sequences.

![08 Activation](assets/plots/inference/08_activation.png)

### Positional Encoding

RoPE vs. ALiBi vs. learned positional biases. RoPE adds negligible overhead;
ALiBi's per-head slope arithmetic is marginal at the sequence lengths tested.

![09 Positional Encoding](assets/plots/inference/09_pos_encoding.png)

### GQA Heads vs. Cache

GQA key/value head count (`kv_heads`) sweep: `n_heads/8` → `n_heads`. Shows
the continuous cache–quality trade-off. At `kv_heads = n_heads` GQA degenerates
to MHA; MLA achieves lower cache at any GQA grouping ratio.

![10 GQA vs Cache](assets/plots/inference/10_gqa_vs_cache.png)

### Scale × Dtype

Joint effect of model scale and dtype on throughput. `bfloat16` advantage is
largest for big models where memory bandwidth is the dominant bottleneck.

![11 Scale × Dtype](assets/plots/inference/11_scale_dtype.png)

### Batch × Attention

Throughput heatmap over batch size × attention type. The crossover where MLA
starts matching or exceeding MHA/GQA throughput moves to smaller batch sizes
as model size grows.

![12 Batch × Attention](assets/plots/inference/12_batch_attn.png)

### Sampling Strategy

Greedy vs. top-k vs. top-p sampling latency. Sampling overhead is dominated
by the softmax and argmax operations, which are identical across attention
variants; per-step latency differences reflect pure attention cost.

![13 Sampling](assets/plots/inference/13_sampling.png)

---

### 3D Relationships

Three-dimensional visualisations linking FLOPs, latency, throughput, batch
size, params, sequence length, and KV-cache across the three attention variants.

#### Params × Sequence → Latency

How prefill latency scales jointly with model parameters and sequence length.
The MLA surface sits highest because its extra projections add a constant
per-step cost on top of the quadratic attention term.

![14 3D Params × Seq → Latency](assets/plots/inference/14_3d_params_seq_latency.png)

#### 2D Projections — Params × Seq → Latency

Pairwise scatter projections of the above 3D surface: Params vs. Latency,
Sequence vs. Latency, and Params vs. Sequence (size ∝ latency).

![18 2D Params × Seq → Latency](assets/plots/inference/18_2d_params_seq_latency.png)

#### Batch × Seq → KV Cache

KV-cache footprint as a function of batch size and sequence length. MLA's
compressed cache keeps the surface an order of magnitude lower than MHA at
all (batch, seq) combinations.

![15 3D Batch × Seq → KV Cache](assets/plots/inference/15_3d_batch_seq_kvcache.png)

#### 2D Projections — Batch × Seq → KV Cache

![19 2D Batch × Seq → KV Cache](assets/plots/inference/19_2d_batch_seq_kvcache.png)

#### FLOPs × Latency → Throughput

Analytical FLOPs vs. measured latency coloured by throughput. MLA sits in the
high-FLOPs / low-latency quadrant at small sequences because XLA's JIT fuses
the low-rank projections efficiently; at long sequences the quadratic attention
cost dominates for all variants.

![16 3D FLOPs × Latency → Throughput](assets/plots/inference/16_3d_flops_latency_throughput.png)

#### 2D Projections — FLOPs × Latency → Throughput

![20 2D FLOPs × Latency → Throughput](assets/plots/inference/20_2d_flops_latency_throughput.png)

#### Params × Batch → Throughput

Aggregate throughput as a function of model scale and batch size. The
throughput gap between attention variants narrows as batch size grows because
memory bandwidth increasingly dominates over compute.

![17 3D Params × Batch → Throughput](assets/plots/inference/17_3d_params_batch_throughput.png)

#### 2D Projections — Params × Batch → Throughput

![21 2D Params × Batch → Throughput](assets/plots/inference/21_2d_params_batch_throughput.png)

---

## Core Comparison

High-level trade-offs between attention types on real trained checkpoints:
quality vs. KV-cache Pareto front, VRAM-normalised serving throughput, and the
MLA `down_dim_kv` compression dial.

### Quality vs. KV-Cache Pareto

Scatter of validation loss vs. theoretical KV cache per token. Points on the
lower-left Pareto front dominate in both quality _and_ memory. MLA models
cluster on or near the front at a fraction of the cache cost of equivalent
MHA/GQA models.

![Quality vs KV-Cache Pareto](assets/plots/insight_1_pareto.png)

### VRAM-Normalised Serving Throughput

Aggregate tokens/s as a function of total VRAM budget (500 MB – 80 GB).
Because MLA's smaller KV cache fits more concurrent sequences, it achieves
**3× the throughput of MHA** and ~20% more than GQA at 80 GB.

![VRAM-Normalised Serving Throughput](assets/plots/insight_2_serving.png)

### MLA Compression Dial

Effect of `down_dim_kv` on quality and cache. Left: validation loss vs.
`down_dim_kv` with MHA/GQA reference bands. Right: cache MB vs. `down_dim_kv`
with crossover annotations. A value around 64–96 gives the best
quality/cache trade-off.

![MLA Compression Dial](assets/plots/insight_3_mla_dial.png)

---

## Performance Analysis

Detailed throughput, FLOPs, and latency breakdowns on trained checkpoints.

### KV-Cache Size by Architecture

Absolute KV cache footprint (MB) vs. model params, grouped by depth
(`num_blocks`). MLA achieves a 5–10× cache reduction relative to MHA at the
same parameter count.

![KV-Cache Breakdown](assets/plots/perf_1_cache_breakdown.png)

### Decode Throughput vs. Sequence Length

Tokens per second at context lengths 64 / 128 / 256 / 512. MHA/GQA advantage
at short sequences narrows as context grows.

![Decode Throughput vs Sequence Length](assets/plots/perf_2_seqlen_throughput.png)

### Analytical FLOPs vs. KV-Cache

Decode FLOPs per step vs. theoretical KV cache (Pareto view). MLA sits in the
high-FLOPs / low-cache quadrant because weight–weight products add ~9× extra
compute relative to MHA at bs=1, while shrinking cache 5–10×.

![Analytical FLOPs vs KV-Cache](assets/plots/perf_3_flops_vs_cache.png)

### Batch Throughput Sweep

Measured tokens/s across batch sizes 1 – 64 for representative 256-d and 512-d
models. The crossover point where MLA's smaller cache enables fitting more
sequences only becomes significant at 7B+ scale.

![Batch Throughput Sweep](assets/plots/perf_4_batch_throughput.png)

### Prefill Latency & Cache Extrapolation

Left: prefill latency vs. model parameters. Right: theoretical KV cache size
extrapolated from 512 to 128 k tokens — MLA's linear-but-low slope vs. MHA's
steep growth.

![Prefill Latency and Cache Extrapolation](assets/plots/perf_5_prefill.png)

---

## 3D Cache Surfaces

Three-dimensional views of how KV cache, model quality, and throughput jointly
vary across architecture axes on trained checkpoints.

### KV-Cache vs. Params vs. Sequence Length

Separate surfaces for MHA, GQA, and MLA. Floor contours and VRAM-limit planes
(24 GB / 80 GB) highlight feasibility regions.

![KV-Cache Surface: Params × Sequence Length](assets/plots/3d_1_cache_surface.png)

### Quality — Params — Cache Cube

3D scatter of validation loss × model parameters × KV cache (MB). The
Pareto-optimal cluster is dominated by MLA models.

![Quality–Params–Cache Cube](assets/plots/3d_2_quality_cube.png)

### Efficiency Cube

Three axes all oriented higher-is-better: tokens/s × throughput-per-cache-MB ×
inverse validation loss. MLA models occupy the upper-right-front corner.

![Efficiency Cube](assets/plots/3d_3_efficiency_cube.png)

### VRAM Budget × Seq-Len Serving Surface

Aggregate serving throughput (k-tok/s) as a function of VRAM budget and
sequence length. The MLA surface consistently sits above MHA/GQA.

![Serving Surface: VRAM × Seq-Len](assets/plots/3d_4_serving_surface.png)

---

## down_dim_kv Deep Dive

MLA's key hyperparameter `down_dim_kv` controls the latent KV dimension.

### Cache vs. down_dim_kv vs. Sequence Length

MLA surface: `down_dim_kv` × seq-len → cache (GB). GQA and MHA appear as
horizontal reference planes at their fixed cache levels.

![Cache vs down_dim_kv vs Sequence Length](assets/plots/3d_5_dkv_cache_seqlen.png)

### KV-Dimension Decoupling

MHA/GQA have a steep dim-proportional cache slope; MLA is flat — its cache is
set by `down_dim_kv` independently of model width.

![KV-Dimension Decoupling](assets/plots/3d_6_kv_decoupling.png)

### MLA Quality Surface

Interpolated surface of validation loss over the (`down_dim_kv`, params) plane.
Quality plateaus above `down_dim_kv` ≈ 64.

![MLA Quality Surface](assets/plots/3d_7_mla_quality.png)

### Cache vs. down_dim_kv vs. Depth

Four subplots at fixed sequence lengths (512 / 4 k / 32 k / 128 k tokens).
Deeper models hit the 80 GB wall at shorter contexts.

![Cache vs down_dim_kv vs Depth](assets/plots/3d_8_dkv_num_blocks.png)
