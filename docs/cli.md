---
title: CLI Reference
---

# CLI Reference

The `dantinox` command is installed automatically by `pip install dantinox`.

```bash
dantinox <subcommand> [options]
dantinox --version
```

---

## Commands at a glance

<div class="grid cards" markdown>

-   :material-school: **train**

    Train a model from a YAML config file and a text corpus.

    [Details →](#train)

-   :material-text-box-outline: **generate**

    Generate text from a trained checkpoint. Supports AR, Diffusion, and ELF.

    [Details →](#generate)

-   :material-chart-scatter-plot: **sweep**

    Launch a W&B Bayesian hyperparameter sweep.

    [Details →](#sweep)

-   :material-speedometer: **benchmark**

    Benchmark all (or selected) trained run directories.

    [Details →](#benchmark)

-   :material-chart-bar: **infbench**

    Full 4-stage inference benchmark suite (random-model sweep + trained pipeline).

    [Details →](#infbench)

-   :material-magnify-scan: **find-lr**

    Run the LR range test and suggest an optimal learning rate.

    [Details →](#find-lr)

-   :material-upload: **push**

    Upload a trained checkpoint to the HuggingFace Hub.

    [Details →](#push)

-   :material-download: **pull**

    Download a checkpoint from the HuggingFace Hub.

    [Details →](#pull)

-   :material-image-multiple: **plot**

    Re-generate benchmark plots from an existing results CSV.

    [Details →](#plot)

-   :material-merge: **merge-lora**

    Fold LoRA adapters into base weights and save a standalone checkpoint.

    [Details →](#merge-lora)

-   :material-cpu-64-bit: **profile**

    Print parameter count and FLOPs for a config or checkpoint.

    [Details →](#profile)

-   :material-clipboard-check-outline: **eval**

    Evaluate generation quality (distinct-N, repetition, MAUVE) on a checkpoint.

    [Details →](#eval)

</div>

---

## train

Train a model from a YAML config file. Any `Config` field can be overridden directly on the command line.

```bash
dantinox train --config configs/default_config.yaml --data_path data/wiki.txt
```

### Arguments

| Flag | Default | Description |
|:-----|:-------:|:------------|
| `--config` | `configs/default_config.yaml` | Path to the YAML config file. |
| `--data_path` | — | Training corpus `.txt` file, or HF dataset identifier when using `--dataset_source huggingface`. |
| `--run_dir` | auto-generated | Output directory for checkpoints and logs. Auto-generates a timestamped directory under `runs/` if omitted. |
| `--wandb_project` | — | W&B project name. Enables Weights & Biases logging. |
| `--resume` | `False` | Resume training from the latest checkpoint in `--run_dir`. |
| `--<config_field>` | — | Override any `Config` field (e.g. `--lr 1e-4 --model_type diffusion`). |

### Output structure

```
runs/
└── <run_name>/
    ├── config.yaml               ← saved config (reproducible)
    ├── best_model_weights.msgpack
    ├── training_log.csv
    └── model_summary.json
```

### Examples

=== "Basic AR training"

    ```bash
    dantinox train \
        --config configs/default_config.yaml \
        --data_path wiki.txt
    ```

=== "Diffusion model"

    ```bash
    dantinox train \
        --config configs/diffusion_base.yaml \
        --data_path wiki.txt \
        --model_type diffusion \
        --noise_schedule cosine \
        --lr 3e-4
    ```

=== "Multi-GPU + bfloat16"

    ```bash
    dantinox train \
        --config configs/large.yaml \
        --data_path wiki.txt \
        --n_devices 4 \
        --grad_accum 8 \
        --batch_size 32 \
        --use_bf16 true
    ```

=== "Resume from checkpoint"

    ```bash
    dantinox train \
        --config configs/default_config.yaml \
        --data_path wiki.txt \
        --run_dir runs/ar_mha_512d_12b \
        --resume
    ```

!!! tip "Effective batch size"
    Effective batch size = `batch_size × grad_accum × n_devices`. Tune `grad_accum` to reach your target without exceeding VRAM.

---

## generate

Generate text from a trained checkpoint.

```bash
dantinox generate --run_dir runs/ar_mha_512d_12b --prompt "Once upon a time"
```

The subcommand auto-detects the model type from `config.yaml` and routes to the correct generation function — no manual flag required.

### Arguments

| Flag | Default | Description |
|:-----|:-------:|:------------|
| `--run_dir` | **required** | Run directory containing `config.yaml` and `best_model_weights.msgpack`. |
| `--prompt` | `"Nel mezzo del cammin "` | Input text prompt. |
| `--max_new_tokens` | `150` | Tokens to generate beyond the prompt. |
| `--greedy` | `False` | Greedy decoding (argmax). Overrides sampling parameters. |
| `--top_k` | `None` | Keep only the `k` highest-probability tokens. |
| `--top_p` | `None` | Nucleus sampling — keep the smallest set with cumulative probability ≥ `p`. |
| `--temperature` | `1.0` | Sampling temperature. Lower = more focused, higher = more random. |
| `--no_cache` | `False` | Disable KV cache (slower; useful for debugging). AR only. |
| `--stream` | `False` | Stream tokens to stdout as they are produced. AR only. |
| `--seed` | `42` | Random seed for sampling. |
| `--n_steps` | `50` | Denoising steps. Diffusion and ELF models only. |
| `--block_size` | `32` | Token block size for Fast-dLLM DualCache. Diffusion only. |
| `--use_dual_cache` | `True` | Enable Fast-dLLM DualCache for ~1.8× speedup. Diffusion only. |
| `--confidence_threshold` | `0.9` | Confidence threshold for early token commitment. Diffusion only. |
| `--cfg_scale` | `1.5` | Classifier-free guidance scale. ELF only. |

### Examples

=== "Nucleus sampling"

    ```bash
    dantinox generate \
        --run_dir runs/ar_mha_512d \
        --prompt "In the beginning" \
        --top_p 0.9 --temperature 0.8 \
        --max_new_tokens 300
    ```

=== "Streaming output"

    ```bash
    dantinox generate \
        --run_dir runs/ar_mha_512d \
        --prompt "Chapter 1:" \
        --stream --top_p 0.95
    ```

=== "Greedy / deterministic"

    ```bash
    dantinox generate \
        --run_dir runs/ar_mha_512d \
        --prompt "The capital of France is" \
        --greedy
    ```

---

## sweep

Launch a W&B hyperparameter sweep over your training config.

```bash
dantinox sweep \
    --sweep_config configs/sweep.yaml \
    --data_path wiki.txt \
    --wandb_project DantinoX
```

### Arguments

| Flag | Default | Description |
|:-----|:-------:|:------------|
| `--sweep_config` | `configs/sweep.yaml` | W&B sweep YAML (defines method, metric, parameter grid). |
| `--config` | `configs/default_config.yaml` | Base model config — overridden by sweep parameters. |
| `--data_path` | **required** | Training corpus. |
| `--wandb_project` | `"DantinoX"` | W&B project name. |
| `--count` | `None` | Maximum number of sweep runs (default: unlimited). |

??? example "Sweep YAML example"
    ```yaml title="configs/sweep.yaml"
    method: bayes
    metric:
      name: val_loss
      goal: minimize
    parameters:
      lr:
        distribution: log_uniform_values
        min: 1e-5
        max: 1e-2
      batch_size:
        values: [32, 64, 128]
      num_blocks:
        values: [6, 12, 18]
    ```

---

## benchmark

Benchmark all (or selected) trained run directories and write a results CSV.

```bash
dantinox benchmark --runs_dir runs --out_csv results/benchmark.csv
```

### Arguments

| Flag | Default | Description |
|:-----|:-------:|:------------|
| `--runs_dir` | `runs` | Directory containing run sub-directories. |
| `--runs` | all | Specific run names to benchmark (space-separated list). |
| `--out_csv` | `None` | Write results to this CSV file. |

---

## infbench

Full 4-stage inference benchmark suite. Runs random-model sweeps, plots, and optionally trained-model analysis.

```bash
dantinox infbench               # random-model sweep only
dantinox infbench --trained     # add trained-model analysis
dantinox infbench --eval        # add quality evaluation (PPL, gen quality)
```

### Pipeline stages

| Stage | Script | Output | Requires |
|:------|:-------|:-------|:---------|
| 1 | `benchmarks/inference_sweep.py` | `inference_sweep.csv` | always |
| 2 | `benchmarks/plot_inference.py` | 21 PNG plots | always |
| 3 | `benchmarks/trained_analysis.py` | `benchmark_results.csv` | `--trained` |
| 4 | `benchmarks/trained_batch_sweep.py` | `batch_sweep_results.csv` | `--trained` |

### Arguments

| Flag | Default | Description |
|:-----|:-------:|:------------|
| `--out-csv` | `results/inference_sweep.csv` | Sweep output CSV. |
| `--out-dir` | `results/plots/` | Plot output directory. |
| `--groups` | all 13 | Restrict sweep to specific groups (e.g. `attention_type scale`). |
| `--n-warmup` | `3` | Warmup repetitions per experiment. |
| `--n-trials` | `10` | Measured repetitions per experiment. |
| `--device` | env | CUDA device index (`CUDA_VISIBLE_DEVICES`). |
| `--sweep-only` | `False` | Run sweep only, skip plotting. |
| `--plot-only` | `False` | Skip sweep, re-plot existing `--out-csv`. |
| `--verbose` | `False` | Print per-experiment metrics. |
| `--trained` | `False` | Run trained-model analysis (stages 3–4). |
| `--diff-ar` | `False` | Run the AR vs. Diffusion sweep. |
| `--eval` | `False` | Run quality evaluation (PPL + generation quality). Implies `--trained`. |
| `--inference-off` | `False` | Skip inference pipeline. Requires at least one of `--trained`/`--diff-ar`/`--eval`. |
| `--no-mla` | `False` | Skip MLA experiments. |
| `--pdf` | `False` | Save figures as PDF in addition to PNG. |
| `--runs-dir` | `runs` | Trained run sub-directories. |
| `--batch-sizes` | `1 2 4 8 16 32 64` | Batch sizes for the batch sweep. |
| `--batch-seq-len` | `512` | Sequence length for the batch sweep. |

### Examples

=== "Quick sweep"

    ```bash
    dantinox infbench --groups attention_type --n-trials 30 --verbose
    ```

=== "Full pipeline"

    ```bash
    dantinox infbench --trained --diff-ar --eval
    ```

=== "Re-plot from existing CSV"

    ```bash
    dantinox infbench --plot-only --out-csv results/inference_sweep.csv
    ```

---

## find-lr

Run the LR range test to find a good learning rate before full training.

```bash
dantinox find-lr \
    --config configs/default_config.yaml \
    --data_path wiki.txt \
    --plot
```

### Arguments

| Flag | Default | Description |
|:-----|:-------:|:------------|
| `--config` | `configs/default_config.yaml` | YAML config file. |
| `--data_path` | **required** | Training corpus. |
| `--min_lr` | `1e-7` | Starting learning rate. |
| `--max_lr` | `1.0` | Maximum learning rate. |
| `--num_steps` | `100` | Steps in the exponential sweep. |
| `--plot` | `False` | Save a loss-vs-LR PNG. |
| `--plot_out` | `lr_finder.png` | Output PNG path. |
| `--<config_field>` | — | Override any `Config` field. |

!!! tip
    Run `find-lr` before any new architecture — the optimal LR can vary 10–100× across model sizes and optimizers.

---

## push

Upload a trained checkpoint to HuggingFace Hub.

```bash
dantinox push --run_dir runs/ar_mha_512d --repo my-org/my-model
```

### Arguments

| Flag | Default | Description |
|:-----|:-------:|:------------|
| `--run_dir` | **required** | Local run directory to upload. |
| `--repo` | **required** | Hub repository ID (e.g. `my-org/my-model`). |
| `--private` | `False` | Create a private repository. |
| `--token` | `None` | HuggingFace access token. Falls back to `HF_TOKEN` env var. |
| `--message` | `None` | Commit message for the Hub upload. |

---

## pull

Download a checkpoint from HuggingFace Hub.

```bash
dantinox pull --repo my-org/my-model --local_dir runs/my-model
```

### Arguments

| Flag | Default | Description |
|:-----|:-------:|:------------|
| `--repo` | **required** | Hub repository ID. |
| `--local_dir` | `None` | Where to save files. Defaults to `runs/<repo-name>`. |
| `--token` | `None` | HuggingFace access token. |
| `--revision` | `None` | Branch, tag, or commit SHA to download. |

---

## plot

Generate benchmark plots from an existing results CSV.

```bash
dantinox plot --in_csv results/benchmark.csv --out_dir results/plots/
```

### Arguments

| Flag | Default | Description |
|:-----|:-------:|:------------|
| `--in_csv` | `benchmark_results.csv` | Input CSV produced by `dantinox benchmark` or `dantinox infbench`. |
| `--out_dir` | `plots/` | Output directory for PNG files. |
| `--batch_csv` | `None` | Optional batch sweep CSV for the throughput-vs-batch-size figure. |
| `--groups` | all | Plot groups: `perf` · `insights` · `3d` · `3d_dkv`. |

| Group | Charts |
|:------|:-------|
| `perf` | Throughput, throughput-vs-batch-size, prefill latency |
| `insights` | Pareto frontier (quality vs. speed) |
| `3d` | 3D parameter / quality / throughput surface |
| `3d_dkv` | 3D KV-cache / throughput surface |

---

## merge-lora

Fold trained LoRA adapters back into the base model weights to produce a standalone checkpoint with no LoRA overhead.

```bash
dantinox merge-lora \
    --run_dir runs/lora_finetune \
    --out_dir runs/lora_merged
```

### Arguments

| Flag | Default | Description |
|:-----|:-------:|:------------|
| `--run_dir` | **required** | Run directory containing a LoRA checkpoint and `config.yaml`. |
| `--out_dir` | **required** | Output directory. Receives `best_model_weights.msgpack` + `config.yaml`. |
| `--overwrite` | `False` | Overwrite `--out_dir` if it already exists. |

### Examples

=== "Merge and verify"

    ```bash
    dantinox merge-lora \
        --run_dir runs/ar_lora_rank8 \
        --out_dir runs/ar_lora_merged

    # Verify merged model generates correctly
    dantinox generate \
        --run_dir runs/ar_lora_merged \
        --prompt "In the beginning"
    ```

=== "Overwrite existing output"

    ```bash
    dantinox merge-lora \
        --run_dir runs/ar_lora_rank8 \
        --out_dir runs/ar_lora_merged \
        --overwrite
    ```

!!! tip
    After merging, `config.yaml` in `--out_dir` has `use_lora: false`. The merged weights are identical in size to the base model and can be pushed to the Hub directly.

---

## profile

Print parameter count and estimated FLOPs for a model config or checkpoint. Works from a saved run directory or a raw YAML config file.

```bash
dantinox profile --run_dir runs/ar_mha_512d
```

### Arguments

| Flag | Default | Description |
|:-----|:-------:|:------------|
| `--run_dir` | — | Run directory (reads `config.yaml`). Mutually exclusive with `--config`. |
| `--config` | — | Path to a YAML config file. Mutually exclusive with `--run_dir`. |
| `--seq_len` | `512` | Sequence length for FLOPs estimation. |
| `--batch_size` | `1` | Batch size for FLOPs estimation. |

### Examples

=== "From a run directory"

    ```bash
    dantinox profile --run_dir runs/diffusion_512d_28k
    ```

=== "From a config file"

    ```bash
    dantinox profile \
        --config configs/large.yaml \
        --seq_len 1024 \
        --batch_size 4
    ```

### Output example

```
Model: DiffusionTransformer (diffusion · MHA)
──────────────────────────────────────────────
Parameters:   48,234,496  (48.2 M)
Embedding:     8,192,000
Backbone:     39,845,376
LM head:         197,120

FLOPs per forward pass (seq=512, batch=1)
  Total:    12.3 GFLOPs
  Attention: 4.1 GFLOPs   (33.3 %)
  FFN:       8.2 GFLOPs   (66.7 %)
──────────────────────────────────────────────
```

---

## eval

Evaluate generation quality for a checkpoint by generating samples and computing diversity and repetition metrics (distinct-1, distinct-2, rep-4, and optionally MAUVE).

```bash
dantinox eval --run_dir runs/diffusion_512d_28k
```

### Arguments

| Flag | Default | Description |
|:-----|:-------:|:------------|
| `--run_dir` | **required** | Run directory with checkpoint and `config.yaml`. |
| `--n_samples` | `50` | Number of samples to generate for evaluation. |
| `--gen_len` | `128` | Generation length in tokens per sample. |
| `--seed` | `42` | Random seed. |
| `--out_csv` | `None` | Save metrics row to this CSV file (appends if exists). |

### Examples

=== "Quick quality check"

    ```bash
    dantinox eval \
        --run_dir runs/ar_mha_512d \
        --n_samples 50 \
        --gen_len 128
    ```

=== "Compare two checkpoints"

    ```bash
    dantinox eval --run_dir runs/ar_mha_512d   --out_csv quality.csv
    dantinox eval --run_dir runs/diff_mha_512d  --out_csv quality.csv
    ```

### Metrics

| Metric | Range | Meaning |
|:-------|:-----:|:--------|
| `distinct_1` | 0–1 | Fraction of unique unigrams across all samples (higher = more diverse) |
| `distinct_2` | 0–1 | Fraction of unique bigrams (higher = more diverse) |
| `rep_4` | 0–1 | Fraction of 4-grams repeated within the same sample (lower = less repetitive) |

---

## See also

- [Configuration Reference](configuration.md) — all `Config` fields and valid values
- [Cookbook](cookbook.md) — end-to-end CLI recipes
- [Experiments & Results](paper.md) — running the full benchmark pipeline
