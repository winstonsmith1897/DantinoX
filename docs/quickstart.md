# Quickstart

Get from zero to a running language model in under two minutes.

---

## Installation

```bash
git clone https://github.com/winstonsmith1897/DantinoX.git
cd DantinoX
pip install -U "jax[cuda12]" jaxlib          # GPU install; use jax[cpu] for CPU-only
pip install -e ".[all]"
```

!!! tip "Conda environment"
    ```bash
    conda create -n dantinox python=3.10 -y && conda activate dantinox
    pip install -U "jax[cuda12]" && pip install -e ".[all]"
    ```

!!! note "Python requirement"
    DantinoX requires **Python ≥ 3.10**. JAX ≥ 0.4.25 is recommended for Flash Attention support.

---

## Train and generate in 8 lines

```python
import dantinox as dx

run_dir = dx.fit(
    "ar",                               # paradigm: "ar" | "discrete" | "continuous"
    "data/wiki.txt",                    # path to training corpus
    dim=512, n_heads=8, head_size=64,
    num_blocks=12, vocab_size=32_000,
    lr=3e-4, epochs=5,
)

print(dx.quick_generate(run_dir, "Once upon a time"))
```

That's it. `dx.fit` builds the model, trains it, saves the best checkpoint to `runs/<timestamp>/`, and returns the directory path. `dx.quick_generate` loads the checkpoint and generates text.

---

## Switching paradigms

Change `"ar"` to `"discrete"` or `"continuous"` — the trainer, optimizer, and checkpoint logic are identical:

=== "Autoregressive"

    ```python
    run_dir = dx.fit("ar", "data/wiki.txt",
                     dim=512, n_heads=8, head_size=64, num_blocks=12,
                     vocab_size=32_000, causal=True,
                     lr=3e-4, epochs=5)
    ```

=== "Discrete Diffusion (LLaDA)"

    ```python
    run_dir = dx.fit("discrete", "data/wiki.txt",
                     dim=512, n_heads=8, head_size=64, num_blocks=12,
                     vocab_size=32_000, causal=False,
                     noise_schedule="cosine", mask_token_id=4,
                     lr=3e-4, epochs=5)
    ```

=== "ELF Continuous Flow"

    ```python
    run_dir = dx.fit("continuous", "data/wiki.txt",
                     embed_dim=768, model_dim=512,
                     n_heads=8, head_size=64, num_blocks=12,
                     vocab_size=32_128, lr=1e-4, epochs=10)
    ```

---

## Benchmark and profile

```python
import dantinox as dx
from dantinox.benchmarking import BenchmarkSuite

paradigm = dx.ARParadigm(dx.ModelConfig(
    dim=512, n_heads=8, head_size=64, num_blocks=12, vocab_size=32_000
))
from flax import nnx
model = paradigm.build_model(nnx.Rngs(0))

report = BenchmarkSuite.default().run(paradigm, model, save_csv="results.csv")
print(report.summary())
```

FLOPs estimate (no model needed):

```python
flops = dx.profile(dx.ModelConfig(dim=512, n_heads=8, head_size=64,
                                   num_blocks=12, vocab_size=32_000),
                   seq_len=512, batch_size=4)
print(flops)
```

---

## Use the CLI

Every operation is accessible from the command line:

```bash
# Train
dantinox train --config configs/default_config.yaml --data_path data/wiki.txt

# Generate
dantinox generate --run_dir runs/20260101_120000 --prompt "In the beginning"

# Benchmark
dantinox benchmark --runs_dir runs --out_csv results/benchmark.csv

# Plot results
dantinox plot --in_csv results/benchmark.csv --out_dir plots/
```

---

## Next steps

| Goal | Where to go |
| :--- | :--- |
| Understand the three-layer architecture | [Architecture](architecture.md) |
| Deep-dive on discrete diffusion | [LLaDA Paradigm](paradigms/diffusion.md) |
| Add Muon, LoRA, or multi-GPU training | [Training Guide](training/index.md) |
| Write a custom benchmark task | [Developer Guide — Custom Task](guides/new-benchmark.md) |
| Full API docs for every class | [API Reference](api/index.md) |
