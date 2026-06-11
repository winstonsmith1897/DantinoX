---
hide:
  - toc
---

<div class="dnx-hero" markdown>

# DantinoX

**Una libreria JAX/Flax NNX per la ricerca su modelli linguistici.**
Tre paradigmi di generazione — Autoregressive, Masked Diffusion e ELF — sulla stessa architettura transformer, con un unico trainer e zero boilerplate.

<div class="hero-badges" markdown>
[![JAX](https://img.shields.io/badge/JAX-000000?style=flat-square&logo=google&logoColor=white)](https://github.com/google/jax)
[![Flax NNX](https://img.shields.io/badge/Flax_NNX-5E17EB?style=flat-square&logoColor=white)](https://github.com/google/flax)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License MIT](https://img.shields.io/badge/License-MIT-16a34a?style=flat-square)](https://opensource.org/licenses/MIT)
[![Docs](https://readthedocs.org/projects/dantinox/badge/?version=latest&style=flat-square)](https://dantinox.readthedocs.io)
</div>

[Inizia subito →](quickstart.md){ .md-button .md-button--primary }
[API Reference](api/index.md){ .md-button }
[Cookbook](cookbook.md){ .md-button }
[GitHub](https://github.com/winstonsmith1897/DantinoX){ .md-button }

</div>

---

## Cos'è DantinoX?

DantinoX è una libreria di ricerca scritta in puro JAX per costruire e addestrare transformer per la generazione di linguaggio naturale. Nasce per rispondere a una domanda semplice: **come si comportano diversi paradigmi di generazione — autoregressive, masked diffusion, e flow-matching — sulla stessa architettura e con lo stesso codice di training?**

La libreria è progettata per tre tipi di utenti:

- **Ricercatori** che vogliono confrontare AR vs. Diffusion vs. ELF in modo riproducibile
- **Studenti** che vogliono capire i dettagli interni di un transformer moderno
- **Ingegneri** che vogliono sperimentare con varianti architetturali (GQA, MLA, MoE, LoRA) senza riscrivere il trainer da zero

---

## I tre paradigmi in sintesi

<div class="grid cards" markdown>

-   :material-arrow-right-circle: **Autoregressive (AR)**

    Il paradigma classico: genera un token alla volta, da sinistra a destra. Ogni token prodotto viene aggiunto al contesto e usato per predire il successivo.

    **Pro:** Semplice, veloce con KV-cache, ottimo come baseline.

    **Contro:** Non può "correggere" token già generati.

    [Approfondisci →](paradigms/autoregressive.md)

-   :material-blur: **Masked Diffusion (LLaDA)**

    Genera tutti i token in parallelo, partendo da una sequenza completamente mascherata e rimuovendo i `[MASK]` in modo iterativo. L'attention è bidirezionale — vede l'intera sequenza.

    **Pro:** Output più diversificati, coerenti su sequenze lunghe.

    **Contro:** Richiede più step in inferenza (accelerabile con Fast-dLLM).

    [Approfondisci →](paradigms/diffusion.md)

-   :material-wave: **ELF — Continuous Flow**

    Opera nello spazio degli embedding continui anziché su token discreti. Trasforma rumore gaussiano in embedding puliti tramite un ODE di Euler.

    **Pro:** Paradigma sperimentale, ottimo per ricerca.

    **Contro:** Più complesso da addestrare, richiede più dati.

    [Approfondisci →](paradigms/elf.md)

</div>

---

## Cosa include la libreria

<div class="grid cards" markdown>

-   :material-layers: **Layer neurali completi**

    MHA, GQA, MLA (Multi-Latent Attention), Flash Attention, Sliding Window Attention, SwiGLU, GELU, Sparse MoE, RMSNorm, LayerNorm, RoPE, NTK-aware RoPE, Sinusoidale, Learned PE.

-   :material-school: **Trainer unificato**

    Un unico `Trainer` funziona per tutti e tre i paradigmi. Supporta: gradient accumulation, bfloat16, multi-GPU con JAX SPMD, checkpointing automatico, logging su W&B, LR range test.

-   :material-speedometer: **Ottimizzatori e schedule**

    AdamW, Lion, Muon, Adafactor. Schedule: cosine, lineare, WSD (warmup-stable-decay). Warmup configurabile.

-   :material-lightning-bolt: **Inferenza ottimizzata**

    KV-cache statico pre-allocato (AR). Fast-dLLM DualCache per Diffusion (speedup 1.4–2.1×). Streaming per AR.

-   :material-tune: **Fine-tuning con LoRA**

    LoRA integrato (`use_lora=True`). I pesi base vengono congelati automaticamente. Supporto per merge dei pesi (`merge_lora()`).

-   :material-chart-bar: **Benchmarking sistematico**

    `BenchmarkSuite` con task plug-in. Throughput, latenza, perplexity. Export CSV. 21 grafici automatici.

-   :material-cloud-sync: **Integrazione ecosistema**

    HuggingFace Hub (push/pull). W&B sweeps. CLI completa con 12 sottocomandi. Notebook Colab.

-   :material-wrench: **Strumenti di analisi**

    `count_flops()` per FLOPs teorici. `LatencyTracker` per misurazioni reali. `Visualizer` per grafici.

</div>

---

## I tre livelli di API

La libreria è pensata per essere usata a diversi livelli di astrazione, dal più semplice al più dettagliato.

=== "Livello 1 — Una riga"

    Ideale per prototipazione rapida. `dx.fit` fa tutto: costruisce il modello, lo addestra, salva il checkpoint.

    ```python
    import dantinox as dx

    run_dir = dx.fit("ar", "data/wiki.txt",
                     dim=512, n_heads=8, head_size=64,
                     num_blocks=12, vocab_size=32_000)

    print(dx.quick_generate(run_dir, "Nel mezzo del cammin"))
    ```

=== "Livello 2 — API esplicita"

    Separa configurazione dell'architettura, del training, e del paradigma. Permette di customizzare ogni componente.

    ```python
    import dantinox as dx
    from flax import nnx

    model_cfg = dx.ModelConfig(
        dim=512, n_heads=8, head_size=64, num_blocks=12,
        vocab_size=32_000, attention_type="gqa", kv_heads=2,
    )
    train_cfg = dx.TrainingConfig(lr=3e-4, epochs=5, grad_accum=4)

    paradigm = dx.ARParadigm(model_cfg)
    run_dir  = dx.Trainer(paradigm, train_cfg).fit("data/wiki.txt")
    model    = dx.load(run_dir, paradigm=paradigm)
    tokens   = paradigm.generate(model, prompt_ids, rng=nnx.Rngs(0))
    ```

=== "Livello 3 — Controllo totale"

    Accedi direttamente a tutti i componenti interni. Ideale per modificare il loop di training o aggiungere componenti custom.

    ```python
    from core.config import ModelConfig
    from core.model import Transformer
    from dantinox.paradigms.ar import ARParadigm
    from dantinox.training.trainer import Trainer
    from dantinox.training.optimizer import build_optimizer, build_schedule
    from dantinox.profiling import LatencyTracker, count_flops
    from flax import nnx
    import jax, optax

    cfg      = ModelConfig(dim=512, n_heads=8, head_size=64,
                           num_blocks=12, vocab_size=32_000)
    paradigm = ARParadigm(cfg)
    model    = paradigm.build_model(nnx.Rngs(42))

    tx      = build_optimizer(cfg)
    schedule = build_schedule(cfg)
    # ... loop di training custom ...
    ```

---

## Struttura del progetto

```text
DantinoX/
│
├── core/                        ← Primitivi neurali (Attention, FFN, MoE, LoRA, …)
│   ├── config.py                   ModelConfig · TrainingConfig · Config · ELFConfig
│   ├── model.py                    Transformer · DiffusionTransformer
│   ├── elf.py                      ELFTransformer
│   ├── attention.py                MHA / GQA / MLA + RoPE + KV-cache
│   ├── block.py                    TransformerBlock (Attention + FFN + Norm)
│   ├── mlp.py                      Dense MLP (SwiGLU / GELU)
│   ├── moe.py                      Sparse MoE con load-balancing
│   ├── diffusion.py                NoiseSchedule · make_noise_schedule
│   ├── lora.py                     LoRAParam · merge_lora
│   └── generation.py               generate · diffusion_generate · elf_generate · fast_dllm_generate
│
├── dantinox/                    ← Pacchetto installabile
│   ├── cli.py                      12 sottocomandi CLI
│   ├── generator.py                Classe Generator (AR, carica checkpoint)
│   ├── paradigms/
│   │   ├── ar.py                   ARParadigm
│   │   └── diffusion/
│   │       ├── discrete.py         DiscreteParadigm (LLaDA)
│   │       └── continuous.py       ContinuousParadigm (ELF)
│   ├── training/
│   │   ├── trainer.py              Trainer — JIT loop, checkpointing, multi-GPU
│   │   └── optimizer.py            build_optimizer · build_schedule
│   ├── benchmarking/               BenchmarkSuite · task plug-in
│   ├── profiling/                  LatencyTracker · count_flops
│   ├── visualization/              Visualizer · registro grafici
│   └── hub.py                      push · pull verso HuggingFace Hub
│
├── utils/
│   ├── tokenizer.py                CharTokenizer · BPETokenizer
│   └── helpers.py                  Funzioni di loss, campionamento batch
│
├── benchmarks/                  ← Script benchmark stand-alone
├── configs/                     ← Template YAML (default, diffusion, sweep)
├── docs/                        ← Documentazione MkDocs Material
└── tests/                       ← Suite pytest (CPU-only)
```

---

## Citazione

```bibtex
@software{dantinox2026,
  author  = {Simoni, Marco},
  title   = {DantinoX: A Unified {JAX}/Flax Framework for {AR},
             Masked Diffusion, and Flow-Matching Language Models},
  year    = {2026},
  url     = {https://github.com/winstonsmith1897/DantinoX},
}
```
