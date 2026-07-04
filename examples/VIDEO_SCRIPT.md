# DantinoX — EMNLP 2026 System Demo · Screencast Script

**Hard limit: 2.5 minutes.** Format per CFP: silent screencast, minimal editing.

**No audio narration is needed**: every point below is rendered **on screen** by the
demo itself as bright `▶` caption lines (e.g. *"watch ② Diffusion — starts fully
masked ░ …"*). Your only job while recording is to **press ENTER at each orange
bar** when its `⏱ mm:ss` timestamp matches the recording clock, leaving ~3 seconds
of reading time after each new screen appears. The narration text below is kept
as a reading-pace guide — it mirrors what appears on screen.

## Setup (before recording — not on camera)

```bash
# 1. Train the demo models once (~8 min). Produces runs/_demo_cache/{tiny_ar,tiny_diff}
python examples/demo_video.py --prepare

# 2. Terminal: ≥ 104 columns, dark theme, font ≥ 14 pt. Hide the tab/title bar.
# 3. Start the script — wait for "✓ ready — start recording" (JIT warmup, pre-roll):
python examples/demo_video.py
```

**Optional — pixel-perfect figure in STEP 5**: the benchmark chart is drawn natively
in the terminal (crisp braille lines + real-text labels). If you record inside
VS Code, you can *additionally* show the true PNG at full resolution:

1. VS Code setting → `terminal.integrated.enableImages: true`
2. Run with `DANTINOX_INLINE_IMG=1 python examples/demo_video.py`

(Also works in iTerm2 and WezTerm; silently ignored in other terminals.)

The script pauses at every orange bar (`⏱ mm:ss · STEP n`). The timestamp tells you
where you should be in the video — press ENTER when your narration reaches that point.
The breadcrumb at the top of each screen (① CONFIG ② TRAIN ③ GENERATE ④ PROFILE
⑤ RESULTS) always shows the viewer where they are.

---

## 0:00 – 0:12 · TITLE

*Screen: DantinoX banner + "Why DantinoX?" box.*

> "Comparing autoregressive decoding, masked diffusion, and flow-matching is hard —
> each usually lives in its own codebase, so measured differences reflect
> implementations, not paradigms. DantinoX is a JAX library that puts all three
> on one modular Transformer backbone."

**ENTER** at 0:12.

## 0:12 – 0:40 · STEP 1 — CONFIG (the paradigm switch)

*Screen: three ModelConfigs; the `paradigm="…"` field is highlighted in yellow,
then the configuration-space panel.*

> "Here is the entire paradigm switch: one field. AR, discrete diffusion,
> continuous flow-matching — same backbone, same weights layout, same trainer.
> And every other axis is a flag too: three attention mechanisms, dense or
> Mixture-of-Experts feed-forward, four positional encodings, three tokenizers,
> LoRA, data- and tensor-parallel sharding. Thousands of valid combinations,
> zero code changes."

**ENTER** at 0:40.

## 0:40 – 1:00 · STEP 2 — TRAIN

*Screen: TrainingConfig + two fit() calls; press ENTER on the run bar;
the diffusion model trains live (progress bars, loss dropping, ★ best).*

> "Training is a single fit call — same call for any paradigm. Watch the loss:
> this is a three-million-parameter diffusion model on Tiny Shakespeare,
> two hundred epochs. The checkpoint directory it returns is self-contained:
> weights, config, tokenizer."

**ENTER** at 1:00.

## 1:00 – 1:55 · STEP 3 — GENERATE (the triptych + Fast-dLLM)

*Screen: one `gen.stream()` loop, then five streaming boxes:
cyan = AR, magenta = Diffusion (plain + 2× Fast-dLLM blocks), green = ELF.*

> "One Generator, one stream call — it reads the checkpoint and dispatches the
> right inference loop. Watch the signatures:
> **AR** appends tokens left to right with a KV cache.
> **Discrete diffusion** starts from fully masked text and reveals positions
> over the denoising steps.
> **Fast-dLLM block mode** denoises one 32-token block at a time, left to
> right — with or without the DualCache KV reuse, same flag.
> And **ELF flow-matching**: every position evolves simultaneously through
> ODE steps — here on a 768-dimensional checkpoint trained on WikiText-103.
> Same API from 3 million to 235 million parameters."

**ENTER** at 1:55.

## 1:55 – 2:15 · STEP 4 — PROFILE (CLI)

*Screen: CLI commands + KV-cache table.*

> "Everything is also a CLI: train, generate, profile, and a full inference
> benchmark suite. count_flops profiles any configuration analytically — zero
> GPU execution: same FLOPs, but GQA and MLA shrink the KV cache six to
> twelve-fold. That swap is one flag."

**ENTER** at 2:15.

## 2:15 – 2:30 · STEP 5 — RESULTS + outro

*Screen: BenchmarkSuite API, then two live terminal charts drawn from the real
`results/paradigm_bench_gqa.csv`, then the quality table + decision rule.*

> "All the paper numbers come from the built-in BenchmarkSuite — one call
> sweeps latency, throughput, and energy, and these charts are its actual CSV
> rendered right here: diffusion dominates throughput at low batch, AR wins
> time-to-first-token. Flow-matching wins fluency, diffusion diversity.
> Switching needs zero retraining. DantinoX — pip install dantinox."

**ENTER** twice → outro screen. Stop recording at 2:30.

---

## Timing recovery

If you fall behind: the two places to save time are STEP 1 (skip the
configuration-space narration, just say "and every axis is a flag") and
STEP 4 (skip the table commentary). Never cut STEP 3 — the triptych is the demo.
