# Quickstart

Questa guida ti porta da zero a un modello linguistico funzionante in pochi minuti.
Sono illustrati tutti i passaggi: installazione, addestramento, generazione, e i tre paradigmi disponibili.

---

## 1. Installazione

### Prerequisiti

| Requisito | Versione minima | Note |
|:----------|:---------------:|:-----|
| Python | 3.10 | Richiesto per le type annotations |
| JAX | 0.4.25 | Serve XLA e JIT compilation |
| Flax NNX | 0.8 | API a stato mutabile (diversa da Linen) |
| CUDA | 12.x | Solo per GPU NVIDIA |

### Da sorgente (consigliato per la ricerca)

```bash title="Terminale"
# 1. Clona il repository
git clone https://github.com/winstonsmith1897/DantinoX.git
cd DantinoX

# 2. (Opzionale ma consigliato) Crea un ambiente virtuale dedicato
conda create -n dantinox python=3.10 -y
conda activate dantinox

# 3. Installa JAX con supporto GPU CUDA 12
pip install -U "jax[cuda12]" jaxlib

# 4. Installa DantinoX in modalità editabile con tutte le dipendenze
pip install -e ".[all]"
```

!!! note "Solo CPU"
    Se non hai una GPU o vuoi girare su CPU, sostituisci `jax[cuda12]` con `jax[cpu]`. Il codice funziona identicamente, solo più lento.

!!! tip "Verifica l'installazione"
    Dopo l'installazione, verifica che JAX veda la tua GPU:
    ```python
    import jax
    print(jax.devices())   # dovrebbe stampare [CudaDevice(id=0), ...]
    ```

### Come pacchetto PyPI

```bash
pip install dantinox                   # solo il core
pip install "dantinox[data]"          # + HuggingFace datasets
pip install "dantinox[benchmark]"     # + pandas, matplotlib, scipy
pip install "dantinox[all]"           # tutto inclusi dev e doc tools
```

---

## 2. Il tuo primo modello in 10 righe

DantinoX offre tre livelli di astrazione. La funzione `dx.fit` è il livello più alto: fa tutto in automatico.

```python title="primo_modello.py"
import dantinox as dx

# dx.fit costruisce il modello, lo addestra e salva il checkpoint
run_dir = dx.fit(
    "ar",                               # paradigma: "ar" | "discrete" | "continuous"
    "data/wiki.txt",                    # file di testo per l'addestramento
    dim=512,                            # dimensione degli embedding (latent space)
    n_heads=8,                          # numero di teste dell'attention
    head_size=64,                       # dim per testa — DEVE valere: dim = n_heads × head_size
    num_blocks=12,                      # numero di layer Transformer
    vocab_size=32_000,                  # dimensione del vocabolario
    lr=3e-4,                            # learning rate iniziale (Adam)
    epochs=5,                           # numero di epoche di addestramento
)

# run_dir è la cartella salvata, es. "runs/20260611_142301"
print(dx.quick_generate(run_dir, "Once upon a time"))
```

**Cosa succede internamente:**

1. `dx.fit` costruisce il `Transformer` con la configurazione specificata
2. Istanzia un `CharTokenizer` (o BPE se `tokenizer_type="bpe"`)
3. Crea un `Trainer` con `AdamW` e schedule cosine
4. Addestra per `epochs` epoche, salvando il checkpoint migliore in `runs/<timestamp>/best_model_weights.msgpack`
5. Restituisce il path della cartella

!!! warning "Vincolo fondamentale"
    `dim` deve essere esattamente uguale a `n_heads × head_size`.
    Con `n_heads=8` e `head_size=64`, devi usare `dim=512`.
    Se i valori non corrispondono, il costruttore lancia un `ValueError`.

---

## 3. I tre paradigmi

DantinoX supporta tre modi diversi di generare testo, tutti con la stessa architettura transformer di base. Cambia solo come il modello viene addestrato e come genera.

### Paradigma 1 — Autoregressive (AR)

Il modello classico: genera un token alla volta, da sinistra a destra.
Ogni token generato dipende da tutti i token precedenti.

```python
run_dir = dx.fit(
    "ar",
    "data/wiki.txt",
    dim=512, n_heads=8, head_size=64, num_blocks=12,
    vocab_size=32_000,
    causal=True,          # applica una maschera causale (lower triangular)
    lr=3e-4,
    epochs=5,
)
```

**Quando usarlo:** È il paradigma più semplice da addestrare e più veloce in inferenza con KV-cache. Buono come baseline.

### Paradigma 2 — Masked Diffusion (LLaDA / Discrete)

Il modello è addestrato a "de-noisare": durante il training, parte dei token viene sostituita con un token `[MASK]`, e il modello impara a predirli tutti simultaneamente.
In generazione, parte da una sequenza completamente mascherata e rimuove i mask iterativamente.

```python
run_dir = dx.fit(
    "discrete",
    "data/wiki.txt",
    dim=512, n_heads=8, head_size=64, num_blocks=12,
    vocab_size=32_000,
    causal=False,             # l'attention è bidirezionale (vede tutta la sequenza)
    noise_schedule="cosine",  # schedule per decidere quanti token mascherare
    mask_token_id=4,          # ID del token [MASK] nel vocabolario
    lr=3e-4,
    epochs=20,                # richiede più epoche dell'AR
)
```

**Quando usarlo:** Genera output più coerenti e diversificati rispetto all'AR su certi task. È più lento in inferenza perché richiede più passaggi, ma può essere accelerato con Fast-dLLM (vedi sotto).

### Paradigma 3 — ELF (Continuous Flow-Matching)

Il modello opera nello spazio degli embedding continui, non su token discreti.
Parte da rumore gaussiano e, con un ODE di Euler, lo trasforma negli embedding dei token.

```python
run_dir = dx.fit(
    "continuous",
    "data/wiki.txt",
    embed_dim=768,     # dimensione dello spazio di embedding continuo
    model_dim=512,     # dimensione interna del transformer
    n_heads=8, head_size=64, num_blocks=12,
    vocab_size=32_128,
    elf_cfg_scale=1.5, # scala del Classifier-Free Guidance (0 = no guidance)
    lr=1e-4,
    epochs=30,
)
```

**Quando usarlo:** Paradigma sperimentale per ricerca su flow-matching discreto. Richiede più dati e più epoche.

---

## 4. API esplicita (livello 2)

Se hai bisogno di più controllo — ad esempio per customizzare l'ottimizzatore o accedere al modello direttamente — usa l'API a paradigma esplicito.

```python title="training_esplicito.py"
import dantinox as dx
from flax import nnx

# Definisci separatamente architettura e training
model_cfg    = dx.ModelConfig(
    dim=512, n_heads=8, head_size=64,
    num_blocks=12, vocab_size=32_000,
    attention_type="gqa",   # usa Grouped-Query Attention invece di MHA
    kv_heads=2,             # 2 KV heads condivisi da 8 query heads
)

training_cfg = dx.TrainingConfig(
    lr=3e-4,
    batch_size=64,
    grad_accum=4,           # effective batch = 64 × 4 = 256
    optimizer="adamw",
    lr_schedule="cosine",
    warmup_steps=400,
    epochs=5,
)

# Crea il paradigma e costruisci il modello
paradigm = dx.ARParadigm(model_cfg)
model    = paradigm.build_model(nnx.Rngs(params=42))

# Addestra
run_dir = dx.Trainer(paradigm, training_cfg).fit("data/wiki.txt")

# Carica e genera
model   = dx.load(run_dir, paradigm=paradigm)
tokens  = paradigm.generate(model, prompt_ids, rng=nnx.Rngs(0))
```

---

## 5. Generazione

### AR — generazione autoregressive

```python title="generazione_ar.py"
from dantinox.generator import Generator

gen    = Generator("runs/ar_512d_12b")
output = gen.generate(
    "In the beginning",
    max_new_tokens=200,
    top_p=0.9,          # nucleus sampling: considera solo i token che coprono il 90% di prob
    temperature=0.8,    # abbassa la casualità
    use_cache=True,     # usa il KV-cache per velocità 3-4× maggiore
)
print(output)
```

### Diffusion — generazione con Fast-dLLM

```python title="generazione_diffusion.py"
from core.generation import fast_dllm_generate
from core.diffusion import make_noise_schedule
import yaml, msgpack
from core.config import Config
from core.model import DiffusionTransformer
from flax import nnx

# Carica config e modello
cfg      = Config.from_yaml("runs/diffusion_512d/config.yaml")
schedule = make_noise_schedule(cfg)
model    = DiffusionTransformer(cfg, rngs=nnx.Rngs(0))
# ... carica i pesi ...

tokens = fast_dllm_generate(
    model,
    prefix=prefix_ids,
    gen_len=128,
    schedule=schedule,
    mask_token_id=cfg.mask_token_id,
    block_size=32,              # decodifica 32 token per blocco
    use_dual_cache=True,        # cache duale: ~1.8× più veloce
    confidence_threshold=0.9,  # committa un token quando la confidenza > 90%
)
```

### ELF — generazione con flow-matching

```python title="generazione_elf.py"
from core.generation import elf_generate

tokens = elf_generate(
    model,
    gen_len=128,
    batch_size=4,
    n_steps=64,       # passi dell'ODE di Euler (più passi = più qualità)
    cfg_scale=1.5,    # forza del guidance
    seed=42,
)
```

---

## 6. Interfaccia CLI

Ogni operazione disponibile in Python è accessibile anche da terminale. Utile per script di training e automazione.

```bash title="Terminale"
# Addestra con un file di configurazione YAML
dantinox train \
    --config configs/default_config.yaml \
    --data_path data/wiki.txt

# Sovrascrivi parametri al volo senza toccare il YAML
dantinox train \
    --config configs/default_config.yaml \
    --data_path data/wiki.txt \
    --model_type diffusion \
    --lr 1e-4 \
    --use_bf16 true \
    --n_devices 4

# Genera testo da un checkpoint salvato
dantinox generate \
    --run_dir runs/ar_512d_12b \
    --prompt "In the beginning" \
    --top_p 0.9 \
    --max_new_tokens 300 \
    --stream              # stampa i token man mano che vengono generati

# Trova il learning rate ottimale prima di addestrare
dantinox find-lr \
    --config configs/default_config.yaml \
    --data_path data/wiki.txt \
    --plot

# Mostra parametri e FLOPs per un checkpoint
dantinox profile --run_dir runs/ar_512d_12b

# Valuta la qualità della generazione (distinct-1, distinct-2, rep-4)
dantinox eval \
    --run_dir runs/ar_512d_12b \
    --n_samples 50 \
    --gen_len 128

# Fondi i pesi LoRA nel modello base (per deployment)
dantinox merge-lora \
    --run_dir runs/lora_finetune \
    --out_dir runs/lora_merged
```

Vedi la [CLI Reference](cli.md) per l'elenco completo dei comandi e di tutti i loro argomenti.

---

## 7. Struttura dell'output di training

Quando esegui un training, DantinoX salva tutto in una cartella strutturata così:

```
runs/
└── 20260611_142301/          ← nome generato automaticamente (data + ora)
    ├── config.yaml           ← copia esatta della configurazione usata (riproducibile!)
    ├── best_model_weights.msgpack  ← pesi del checkpoint con validation loss migliore
    ├── training_log.csv      ← log step-by-step di loss, lr, grad_norm, …
    └── model_summary.json    ← riepilogo architettura (n. parametri, FLOPs, …)
```

Il file `config.yaml` ti permette di riprodurre esattamente lo stesso training in futuro, oppure di riprendere da dove si era interrotto con `--resume`.

---

## 8. Prossimi passi

<div class="grid cards" markdown>

-   :material-book-open-variant: **Architettura**

    Capire i layer interni: MHA, GQA, MLA, SwiGLU, MoE, RoPE, LoRA.

    [Architettura →](architecture.md)

-   :material-blur: **Masked Diffusion (LLaDA)**

    Forward process, noise schedule cosine, ELBO loss, unmasking iterativo.

    [Paradigma Diffusion →](paradigms/diffusion.md)

-   :material-tune: **Guida al Training**

    Ottimizzatori (Muon, AdamW, Lion), multi-GPU, gradient accumulation, sweep W&B.

    [Training →](training/index.md)

-   :material-chef-hat: **Cookbook**

    Ricette copia-incolla per ogni scenario: training, generazione, LoRA, Hub, benchmark.

    [Cookbook →](cookbook.md)

-   :material-console: **CLI Reference**

    Tutti i 12 sottocomandi con tabelle complete degli argomenti.

    [CLI →](cli.md)

-   :material-file-cog: **Configurazione**

    Ogni campo di `ModelConfig`, `TrainingConfig`, `Config`, `ELFConfig` spiegato in dettaglio.

    [Configurazione →](configuration.md)

</div>
