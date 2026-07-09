# DantinoX — EMNLP 2026 System Demo · Screencast Script

**Hard limit: 2.5 minutes.** Format per CFP: silent screencast, minimal editing.

**No audio narration is needed**: every point below is rendered **on screen** by the
demo itself as bright `▶` caption lines.

**Recommended: autopilot mode — a zero-keystroke take of exactly 2:20**

```bash
DANTINOX_AUTO=1 python examples/demo_video.py
# wait for "✓ ready — start recording" → start capture → press ENTER once
# (the cold-open gate). Everything else advances by itself; the take ends
# holding on the QR code. Measured duration: 140 s.
```

Manual mode (no `DANTINOX_AUTO`) pauses at every orange bar for ENTER instead —
use it for rehearsal or if you want to control the pace yourself. The `⏱ mm:ss`
labels tell you where you should be. The narration text below is a reading-pace
guide — it mirrors what appears on screen.

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

**Pacing**: all reveal animations scale with `DANTINOX_DEMO_SPEED` (default 1.0;
1.3 = 30% slower, 0.8 = faster). Tune it during rehearsal, not by editing code.

**Recording for maximum text sharpness** (recommended over plain screen capture —
screen-capture compression blurs terminal text):

```bash
# 1. record the terminal session losslessly
asciinema rec demo.cast          # then run the demo inside, Ctrl-D when done

# 2. render to video at exact font/theme (no compression artifacts)
agg --font-size 20 --theme monokai demo.cast demo.gif
ffmpeg -i demo.gif -movflags faststart -pix_fmt yuv420p demo.mp4
```

If you use a normal screen recorder instead: 2× display scaling / large font
(≥ 16 pt), disable terminal transparency, record at native resolution.
The demo hides the terminal cursor by itself and ends on a scannable QR code
to the GitHub repo — leave it on screen for the final ~3 seconds.

The script pauses at every orange bar (`⏱ mm:ss · STEP n`). The timestamp tells you
where you should be in the video — press ENTER when your narration reaches that point.
The breadcrumb at the top of each screen (① CONFIG ② TRAIN ③ GENERATE ④ PROFILE
⑤ RESULTS) always shows the viewer where they are.

---

## 0:00 – 0:10 · COLD OPEN (trailer)

*After the warmup, the script shows "▶ start recording, then ENTER for the cold
open". Start your capture, press ENTER. The three-paradigm race plays with NO
explanation — AR typing, Diffusion revealing ░, Flow-Matching rewriting — each
row ending in a green `✓ tokens/passes` counter, then the punchline:*
**"one library · one backbone · one config field apart."** *Cut to title.*

## 0:10 – 0:15 · TITLE

*Screen: DantinoX banner + "Why DantinoX?" box.*

> "Comparing autoregressive decoding, masked diffusion, and flow-matching is hard —
> each usually lives in its own codebase, so measured differences reflect
> implementations, not paradigms. DantinoX puts all three on one modular
> Transformer backbone."

**ENTER** at 0:15.

## 0:15 – 0:38 · STEP 1 — CONFIG (the paradigm switch)

*Screen: one config line whose `paradigm=` value FLIPS in place
(ar → discrete → continuous), then the three ModelConfigs with the field
highlighted, then the configuration-space panel (9 axes reveal one by one).*

> "Here is the entire paradigm switch: one field. AR, discrete diffusion,
> continuous flow-matching — same backbone, same weights layout, same trainer.
> And every other axis is a flag too: three attention mechanisms, dense or
> Mixture-of-Experts feed-forward, four positional encodings, three tokenizers,
> LoRA, data- and tensor-parallel sharding. Thousands of valid combinations,
> zero code changes."

**ENTER** at 0:38.

## 0:38 – 0:55 · STEP 2 — TRAIN

*Screen: TrainingConfig + one fit() call. On ENTER: the Trainer prints its own
run panel (paradigm · model · data · training), then trains live — progress
bars, loss dropping, ★ best, val-loss sparkline.*

> "Training is a single fit call — same call for any paradigm. Watch the
> 21.5M-parameter diffusion model train on Tiny Shakespeare, 250 epochs.
> The run_dir it returns is self-contained: weights, config.yaml, tokenizer."

**ENTER** at 0:55.

## 0:55 – 1:50 · STEP 3 — GENERATE (the signatures + THE RACE)

*Screen: one `gen.stream()` loop, then four streaming boxes (cyan = AR,
magenta = Diffusion + Fast-dLLM blocks, green = Continuous), and the finale:
THE RACE — all three paradigms generating simultaneously, each row ending in
a green `✓ tokens/passes` counter with the run conditions stated below.*

> "One Generator, one stream call — it reads the checkpoint and dispatches the
> right inference loop. Watch the signatures:
> **AR** appends tokens left to right with a KV cache.
> **Discrete diffusion** starts from fully masked text and reveals positions
> over the denoising steps.
> **Fast-dLLM block mode** denoises one 32-token block at a time, left to
> right — DualCache KV reuse is the same flag.
> **Continuous flow-matching**: every position evolves through ODE steps —
> a 235M-parameter checkpoint trained on WikiText-103.
> Then all three at once: AR needs one pass per token; diffusion and flow
> finish in far fewer passes — that is where their throughput comes from."

**ENTER** at 1:50.

## 1:50 – 2:05 · STEP 4 — PROFILE

*Screen: profiling API code (count_flops · LatencyMetric · EnergyMetric ·
FLOPsMetric) + KV-cache table + measured per-paradigm metrics table.*

> "The profiling API measures FLOPs, latency, energy and MFU for any config.
> count_flops is analytical — zero GPU execution: same FLOPs, but GQA and MLA
> shrink the KV cache six to twelve-fold. That swap is one flag."

**ENTER** at 2:05.

## 2:05 – 2:30 · STEP 5 — RESULTS + outro

*Screen: BenchmarkSuite API, two live terminal charts from the real CSV, the
batch-sweep braille chart, quality table, decision rule, and the external-
validation note (dllm & xLM reproduce the training curves).*

> "All the paper numbers come from the built-in BenchmarkSuite — one call
> sweeps latency, throughput, and energy, and these charts are its actual CSV
> rendered right here: diffusion dominates throughput at low batch, AR wins
> time-to-first-token. Flow-matching wins fluency, diffusion diversity.
> Switching needs zero retraining. DantinoX — pip install dantinox."

**ENTER** twice → outro: claim, integrity line ("everything ran live on one
GPU"), `pip install dantinox`, and the scannable QR code to the repo.
Hold ~3 s on the QR. Stop recording at 2:30.

---

## Timing recovery

If you fall behind: the two places to save time are STEP 1 (skip the
configuration-space narration, just say "and every axis is a flag") and
STEP 4 (skip the table commentary). Never cut STEP 3 — the triptych is the demo.
