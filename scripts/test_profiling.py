"""
Comprehensive test for dantinox.profiling redesign.

Sections
--------
1. discover() — 12 filter combinations with expected counts
2. Standalone metric APIs (LatencyMetric, ThroughputMetric, FLOPsMetric,
   PerplexityMetric, EntropyMetric) without loading any model
3. RunsProfiler.run() on a real small AR model (128d)  — metrics: latency, throughput, flops
4. RunsProfiler.run() on two AR models at once (dim=128, model_type=autoregressive)
5. MultiRunReport: summary(), to_dataframe(), save_csv()

Usage
-----
    python scripts/test_profiling.py
"""
import math
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
import numpy as np

from dantinox.profiling import (
    LatencyMetric, ThroughputMetric, FLOPsMetric, PerplexityMetric, EntropyMetric,
    EnergyMetric,
    LatencyResult, ThroughputResult, FLOPsResult, PerplexityResult, EntropyResult,
    RunsProfiler, MultiRunReport, RunProfile,
    # backward compat
    LatencyTracker, ProfilingResult,
)

RUNS_DIR = "runs"
OK  = "\033[32mOK\033[0m"
ERR = "\033[31mFAIL\033[0m"
HDR = "\033[1m"
RST = "\033[0m"

_failures: list[str] = []


def check(condition: bool, label: str, detail: str = "") -> None:
    if condition:
        print(f"  {OK}  {label}")
    else:
        msg = f"{label}  {detail}"
        print(f"  {ERR}  {msg}")
        _failures.append(msg)


def section(title: str) -> None:
    print(f"\n{HDR}{'─'*60}{RST}")
    print(f"{HDR}{title}{RST}")
    print(f"{HDR}{'─'*60}{RST}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. discover() — filter combinations
# ══════════════════════════════════════════════════════════════════════════════

section("1. RunsProfiler.discover() — filter combinations")

FILTER_TESTS = [
    # (filter_dict, expected_count_min, expected_count_max, label)
    ({},                                                    180, 210,  "no filter  → all runs"),
    ({"model_type": "autoregressive"},                       65,  75,  "AR only"),
    ({"model_type": "diffusion"},                            28,  38,  "diffusion only"),
    ({"model_type": "elf"},                                   4,  10,  "ELF only"),
    ({"dim": 128},                                            2,   8,  "dim=128"),
    ({"dim": 256},                                           75,  90,  "dim=256"),
    ({"dim": 512},                                           50,  65,  "dim=512"),
    ({"dim": 768},                                           30,  45,  "dim=768"),
    ({"optimizer": "muon"},                                  30,  40,  "optimizer=muon"),
    ({"use_moe": True},                                      40,  55,  "use_moe=True"),
    ({"use_swiglu": False},                                  40,  55,  "use_swiglu=False (GELU)"),
    ({"norm_type": "rmsnorm"},                                3,  10,  "norm_type=rmsnorm"),
    # multi-key filters
    ({"model_type": "autoregressive", "dim": 256},           35,  45,  "AR + dim=256"),
    ({"model_type": "autoregressive", "dim": 128},            2,   5,  "AR + dim=128"),
    ({"model_type": "autoregressive", "dim": 768},            2,   6,  "AR + dim=768"),
    ({"model_type": "diffusion",      "optimizer": "muon"},   25,  35,  "diffusion + muon"),
    ({"model_type": "diffusion",      "optimizer": "adamw"},   1,   4,  "diffusion + adamw"),
    ({"model_type": "elf",            "dim": 512},             2,   5,  "ELF + dim=512"),
    ({"model_type": "elf",            "dim": 768},             2,   5,  "ELF + dim=768"),
    ({"model_type": "autoregressive", "use_moe": True},       10,  25,  "AR + MoE"),
    ({"model_type": "autoregressive", "norm_type": "rmsnorm"},  1,   5,  "AR + RMSNorm"),
    # no-match cases
    ({"model_type": "transformer"},                            0,   0,  "unknown model_type → 0"),
    ({"dim": 9999},                                            0,   0,  "dim=9999 → 0"),
    ({"model_type": "elf", "optimizer": "lion"},               0,   0,  "ELF + lion → 0 (ELF uses muon)"),
]

for filt, lo, hi, label in FILTER_TESTS:
    p = RunsProfiler(RUNS_DIR, filter_config=filt)
    found = p.discover()
    n = len(found)
    ok = lo <= n <= hi
    check(ok, f"{label:50s}  found={n}  expected=[{lo},{hi}]",
          detail="" if ok else f"got {n}, expected [{lo},{hi}]")

# Extra: check that filtering is reproducible
p1 = RunsProfiler(RUNS_DIR, filter_config={"model_type": "autoregressive", "dim": 256})
p2 = RunsProfiler(RUNS_DIR, filter_config={"model_type": "autoregressive", "dim": 256})
check(p1.discover() == p2.discover(), "discover() is deterministic")

# Extra: explicit list of run dirs
explicit = [
    os.path.join(RUNS_DIR, "ar_gqa_128d_12b_Dense"),
    os.path.join(RUNS_DIR, "ar_gqa_256d_12b_Dense"),
]
p3 = RunsProfiler(explicit, filter_config={})
check(len(p3.discover()) == 2, "explicit list of run dirs → 2 discovered")

p4 = RunsProfiler(explicit, filter_config={"dim": 256})
check(len(p4.discover()) == 1, "explicit list + dim=256 filter → 1 discovered")

p5 = RunsProfiler(explicit, filter_config={"dim": 999})
check(len(p5.discover()) == 0, "explicit list + dim=999 → 0 discovered")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Standalone metric APIs (no model loading)
# ══════════════════════════════════════════════════════════════════════════════

section("2. Standalone metric APIs")

# ── LatencyMetric ─────────────────────────────────────────────────────────────
print("\n  LatencyMetric")
lat = LatencyMetric(n_warmup=2, n_measure=20)

# constant-time fn
result = lat.measure(lambda: time.sleep(0.002), n_tokens=256)
check(isinstance(result, LatencyResult), "returns LatencyResult")
check(1.5 <= result.mean_ms <= 5.0,     f"mean_ms≈2ms  got={result.mean_ms:.2f}")
check(result.p50_ms <= result.p99_ms,   "p50 ≤ p99")
check(result.p50_ms <= result.p95_ms,   "p50 ≤ p95")
check(result.n_samples == 20,           "n_samples=20")
check(result.total_tokens == 20 * 256,  "total_tokens correct")
d = result.to_dict()
check(all(k in d for k in ("lat_mean_ms","lat_p50_ms","lat_p95_ms","lat_p99_ms")),
      "to_dict() has all keys")
check(str(result).startswith("Latency"), "str(result) works")

# variable-time fn (ensures ordering)
import random
result2 = lat.measure(lambda: time.sleep(random.uniform(0.001, 0.005)), n_tokens=64)
check(result2.p50_ms <= result2.p99_ms, "variable latency: p50 ≤ p99")

# ── ThroughputMetric ──────────────────────────────────────────────────────────
print("\n  ThroughputMetric")
thr = ThroughputMetric(n_warmup=1, n_measure=10,
                       batch_sizes=[1, 2, 4, 8],
                       seq_lens=[64, 128, 256])

dummy_model = lambda x: jax.block_until_ready(x.sum())
make_batch  = lambda bs, sl: jnp.ones((bs, sl), jnp.int32)
result = thr.measure(make_batch, dummy_model, seq_len=128)

check(isinstance(result, ThroughputResult), "returns ThroughputResult")
check(result.peak_tps > 0,                 f"peak_tps > 0  got={result.peak_tps:,.0f}")
check(set(result.by_batch.keys()) == {1,2,4,8}, f"by_batch has all sizes: {set(result.by_batch.keys())}")
check(set(result.by_seq.keys())   == {64,128,256}, f"by_seq has all lens: {set(result.by_seq.keys())}")
# throughput should increase with batch size
bs_vals = [result.by_batch[b] for b in [1,2,4,8]]
check(bs_vals[0] < bs_vals[-1],  f"tps grows with batch: bs1={bs_vals[0]:,.0f} bs8={bs_vals[-1]:,.0f}")
d = result.to_dict()
check("peak_tps"   in d,  "to_dict() has peak_tps")
check("tps_bs1"    in d,  "to_dict() has tps_bs1")
check("tps_sl64"   in d,  "to_dict() has tps_sl64")

# OOM-safe: batch size that triggers exception
call_count = [0]
def fragile_model(x):
    call_count[0] += 1
    if x.shape[0] >= 4:
        raise RuntimeError("OOM")
    return x.sum()

thr_oom = ThroughputMetric(n_warmup=0, n_measure=3, batch_sizes=[1, 2, 4, 8])
result_oom = thr_oom.measure(make_batch, fragile_model, seq_len=32)
check(4 not in result_oom.by_batch, "OOM at bs=4 stops batch sweep correctly")
check(8 not in result_oom.by_batch, "bs=8 also absent after OOM stop")

# ── FLOPsMetric ───────────────────────────────────────────────────────────────
print("\n  FLOPsMetric")
from dantinox.core.config import Config

cfg_ar = Config.from_dict({
    "dim": 256, "n_heads": 4, "num_blocks": 6, "expansion": 4,
    "vocab_size": 32128, "use_swiglu": True, "head_size": 64,
    "kv_heads": 4, "max_context": 512,
})

fm = FLOPsMetric(gpu_peak_tflops=312.0)

# without elapsed_s → efficiency NaN
r = fm.measure(cfg_ar, seq_len=256, batch_size=1)
check(isinstance(r, FLOPsResult),      "returns FLOPsResult")
check(r.total_gflops > 0,             f"total_gflops > 0  got={r.total_gflops:.2f}")
check(math.isnan(r.efficiency_pct),   "efficiency_pct is NaN without elapsed_s")
check(r.attention_gflops > 0,         "attention_gflops > 0")
check(r.ffn_gflops > 0,               "ffn_gflops > 0")
check(r.total_gflops > r.attention_gflops, "total > attention")

# with elapsed_s → efficiency computed
r2 = fm.measure(cfg_ar, seq_len=256, batch_size=1, elapsed_s=0.01)
check(not math.isnan(r2.efficiency_pct), "efficiency_pct computed with elapsed_s")
check(0 < r2.efficiency_pct < 100,       f"efficiency in (0,100): {r2.efficiency_pct:.3f}%")

# batch_size doubles → flops double
r_bs1 = fm.measure(cfg_ar, seq_len=256, batch_size=1)
r_bs2 = fm.measure(cfg_ar, seq_len=256, batch_size=2)
ratio = r_bs2.total_gflops / r_bs1.total_gflops
check(1.9 < ratio < 2.1, f"batch×2 → flops×2  ratio={ratio:.3f}")

d = r.to_dict()
check(all(k in d for k in ("flops_total_g","flops_attn_g","flops_ffn_g","flops_efficiency_pct")),
      "to_dict() has all keys")

# ── PerplexityMetric ──────────────────────────────────────────────────────────
print("\n  PerplexityMetric")

# synthetic token data
rng = jax.random.PRNGKey(42)
vocab = 100
fake_data = jax.random.randint(rng, (5000,), 0, vocab).tolist()

# fake loss_fn: returns constant loss = 3.0
def const_loss_fn(batch, rng):
    return 3.0, None

ppl = PerplexityMetric(data=fake_data, seq_len=32, batch_size=2, n_batches=10)
result = ppl.measure(const_loss_fn, rng)
check(isinstance(result, PerplexityResult), "returns PerplexityResult")
check(abs(result.eval_loss - 3.0) < 1e-6,  f"eval_loss=3.0  got={result.eval_loss:.6f}")
check(abs(result.perplexity - math.exp(3.0)) < 0.01,
      f"perplexity=e^3={math.exp(3.0):.3f}  got={result.perplexity:.3f}")
check(abs(result.bpb - 3.0/math.log(2)) < 0.01,
      f"bpb=3/ln2={3.0/math.log(2):.4f}  got={result.bpb:.4f}")
check(result.n_batches == 10,              "n_batches=10")

# varying loss
import itertools
losses_iter = iter([1.0, 2.0, 3.0, 4.0])
def var_loss_fn(batch, rng):
    v = next(losses_iter, 2.5)
    return v, None

ppl2 = PerplexityMetric(data=fake_data, seq_len=32, batch_size=2, n_batches=4)
result2 = ppl2.measure(var_loss_fn, rng)
expected_mean = (1.0 + 2.0 + 3.0 + 4.0) / 4
check(abs(result2.eval_loss - expected_mean) < 1e-6,
      f"mean loss correct: {result2.eval_loss:.4f} vs {expected_mean:.4f}")

d = result.to_dict()
check(all(k in d for k in ("perplexity","bpb","eval_loss")), "to_dict() keys OK")

# ── EntropyMetric ─────────────────────────────────────────────────────────────
print("\n  EntropyMetric")

vocab_size = 50

# uniform logits → max entropy = log(V)
def uniform_logits(x):
    return jnp.zeros((*x.shape, vocab_size))

ent_metric = EntropyMetric(data=fake_data, seq_len=32, batch_size=2, n_batches=5)
result = ent_metric.measure(uniform_logits, rng)
check(isinstance(result, EntropyResult),  "returns EntropyResult")
expected_H = math.log(vocab_size)
check(abs(result.mean_entropy - expected_H) < 0.05,
      f"uniform → H=ln({vocab_size})={expected_H:.3f}  got={result.mean_entropy:.3f}")
check(abs(result.mean_top1_prob - 1/vocab_size) < 0.01,
      f"uniform → top1_prob=1/{vocab_size}={1/vocab_size:.4f}  got={result.mean_top1_prob:.4f}")

# peaked logits → low entropy, high top1_prob
def peaked_logits(x):
    logits = jnp.full((*x.shape, vocab_size), -100.0)
    return logits.at[..., 0].set(100.0)  # always predict token 0

result2 = ent_metric.measure(peaked_logits, rng)
check(result2.mean_entropy < 0.01,    f"peaked → near-zero entropy: {result2.mean_entropy:.4f}")
check(result2.mean_top1_prob > 0.99,  f"peaked → top1_prob≈1: {result2.mean_top1_prob:.4f}")
check(result2.std_entropy >= 0,       "std_entropy ≥ 0")

# to_dict
d = result.to_dict()
check(all(k in d for k in ("entropy_mean_nats","entropy_std","top1_prob_mean")),
      "to_dict() keys OK")

# ── Backward compat ───────────────────────────────────────────────────────────
print("\n  Backward compat: LatencyTracker / ProfilingResult")
tracker = LatencyTracker()
with tracker.measure(n_tokens=256):
    time.sleep(0.001)
r = tracker.result()
check(isinstance(r, ProfilingResult), "LatencyTracker.result() → ProfilingResult")
check(r.latency_mean_ms > 0,          "ProfilingResult.latency_mean_ms > 0")
check(r.n_samples == 1,               "ProfilingResult.n_samples == 1")


# ══════════════════════════════════════════════════════════════════════════════
# 3. RunsProfiler.run() — real AR model (128d, 12b, Dense)  light metrics
# ══════════════════════════════════════════════════════════════════════════════

section("3. RunsProfiler.run() — AR 128d model (latency + throughput + flops)")

profiler = RunsProfiler(
    runs=RUNS_DIR,
    filter_config={"model_type": "autoregressive", "dim": 128},
    metrics=["latency", "throughput", "flops"],
    batch_sizes=[1, 4, 16],
    seq_len=64,
    n_warmup=2,
    n_measure=10,
)

found = profiler.discover()
print(f"  Discovered {len(found)} run(s): {[os.path.basename(d) for d in found]}")
check(len(found) >= 1, f"at least 1 AR 128d run found")

report = profiler.run()

check(isinstance(report, MultiRunReport),           "returns MultiRunReport")
check(len(report.profiles) == len(found),           f"profile count matches discovered ({len(found)})")
check(report.total_time_s > 0,                      "total_time_s > 0")
check(report.metrics == ["latency","throughput","flops"], "metrics list stored correctly")
check(report.filter_used == {"model_type": "autoregressive", "dim": 128}, "filter_used stored")

for p in report.profiles:
    check(p.error is None,          f"[{p.run_name}] no error")
    # latency / throughput may be None if the model's forward pass fails
    # (e.g. corrupted config — ar_gqa_128d has attention_type=mla + mla=False)
    if p.latency is not None:
        check(p.latency.mean_ms > 0, f"[{p.run_name}] latency.mean_ms > 0")
        check(p.latency.p99_ms >= p.latency.p50_ms, f"[{p.run_name}] p99 ≥ p50")
    check(p.throughput is not None,  f"[{p.run_name}] throughput measured")
    check(p.flops is not None,       f"[{p.run_name}] flops computed")
    if p.throughput and not math.isnan(p.throughput.peak_tps):
        check(p.throughput.peak_tps > 0,  f"[{p.run_name}] throughput > 0")
        check(1 in p.throughput.by_batch, f"[{p.run_name}] bs=1 in by_batch")
    if p.flops:
        check(p.flops.total_gflops > 0,   f"[{p.run_name}] flops > 0")
    check(p.perplexity is None, f"[{p.run_name}] perplexity=None (not requested)")
    check(p.energy is None,     f"[{p.run_name}] energy=None (not requested)")
    check("dim" in p.config,    f"[{p.run_name}] config has 'dim'")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Multi-filter run: AR dim=128 + AR dim=192 via explicit list
# ══════════════════════════════════════════════════════════════════════════════

section("4. RunsProfiler.run() — AR 192d model (throughput + flops only)")

profiler2 = RunsProfiler(
    runs=RUNS_DIR,
    filter_config={"model_type": "autoregressive", "dim": 192},
    metrics=["throughput", "flops"],
    batch_sizes=[1, 4],
    seq_len=32,
    n_warmup=1,
    n_measure=5,
)

found2 = profiler2.discover()
print(f"  Discovered {len(found2)} run(s): {[os.path.basename(d) for d in found2]}")
report2 = profiler2.run()

check(len(report2.profiles) == len(found2), "profile count matches")
for p in report2.profiles:
    check(p.latency is None,    f"[{p.run_name}] latency=None (not requested)")
    check(p.throughput is not None, f"[{p.run_name}] throughput measured")
    check(p.flops is not None,  f"[{p.run_name}] flops measured")


# ══════════════════════════════════════════════════════════════════════════════
# 5. MultiRunReport — summary(), to_dataframe(), save_csv()
# ══════════════════════════════════════════════════════════════════════════════

section("5. MultiRunReport output")

summary = report.summary()
check("MultiRunReport" in summary,   "summary() contains 'MultiRunReport'")
check("latency" in summary,          "summary() mentions 'latency'")
print("  summary preview:")
for line in summary.split("\n"):
    print(f"    {line}")

df = report.to_dataframe()
check(len(df) == len(report.profiles),  f"DataFrame rows = {len(df)}")
check("run_name" in df.columns,         "DataFrame has 'run_name' column")
check("lat_mean_ms" in df.columns,      "DataFrame has 'lat_mean_ms' column")
check("peak_tps" in df.columns,         "DataFrame has 'peak_tps' column")
check("flops_total_g" in df.columns,    "DataFrame has 'flops_total_g' column")
print(f"  DataFrame columns: {list(df.columns)}")
print(f"  DataFrame:\n{df.to_string(index=False)}")

# save_csv
with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
    tmp_path = f.name
report.save_csv(tmp_path)
check(os.path.exists(tmp_path), f"CSV saved to {tmp_path}")
with open(tmp_path) as f:
    csv_content = f.read()
check("run_name" in csv_content, "CSV has 'run_name' header")
check(len(csv_content.strip().split("\n")) == len(report.profiles) + 1,
      f"CSV has header + {len(report.profiles)} data row(s)")
os.unlink(tmp_path)

# to_dict per profile
for p in report.profiles:
    d = p.to_dict()
    check("run_name" in d, f"[{p.run_name}] RunProfile.to_dict() has run_name")
    if p.latency is not None:
        check("lat_mean_ms" in d, f"[{p.run_name}] RunProfile.to_dict() has lat_mean_ms")

# Combined report from two runs
combined = MultiRunReport(
    profiles=report.profiles + report2.profiles,
    total_time_s=report.total_time_s + report2.total_time_s,
    metrics=["latency","throughput","flops"],
    filter_used={},
)
df2 = combined.to_dataframe()
check(len(df2) == len(report.profiles) + len(report2.profiles),
      f"Combined DataFrame has {len(df2)} rows")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Sweep lists + 3D plots
# ══════════════════════════════════════════════════════════════════════════════

section("6. Sweep lists (seq_lens / batch_sizes as list) + 3D plots")

# ── PerplexityMetric with lists ───────────────────────────────────────────────
print("\n  PerplexityMetric — sweep grid")

rng = jax.random.PRNGKey(99)
fake_loss_fn = lambda batch, rng: (2.5, None)

ppl_sweep = PerplexityMetric(
    data=fake_data,
    seq_lens=[32, 64],
    batch_sizes=[2, 4],
    n_batches=5,
)
r_ppl = ppl_sweep.measure(fake_loss_fn, rng)
check(isinstance(r_ppl, PerplexityResult),  "returns PerplexityResult")
check(len(r_ppl.sweep) == 4,               f"sweep has 2×2=4 entries  got={len(r_ppl.sweep)}")
check(all("seq_len"    in e for e in r_ppl.sweep), "every sweep entry has seq_len")
check(all("batch_size" in e for e in r_ppl.sweep), "every sweep entry has batch_size")
check(all("perplexity" in e for e in r_ppl.sweep), "every sweep entry has perplexity")
check(all("bpb"        in e for e in r_ppl.sweep), "every sweep entry has bpb")
seq_lens_in_sweep  = sorted(set(e["seq_len"]    for e in r_ppl.sweep))
batch_sizes_in_sweep = sorted(set(e["batch_size"] for e in r_ppl.sweep))
check(seq_lens_in_sweep  == [32, 64], f"sweep seq_lens correct: {seq_lens_in_sweep}")
check(batch_sizes_in_sweep == [2, 4], f"sweep batch_sizes correct: {batch_sizes_in_sweep}")

# backward compat: single point → sweep is empty
ppl_single = PerplexityMetric(data=fake_data, seq_len=32, batch_size=2, n_batches=3)
r_single = ppl_single.measure(fake_loss_fn, rng)
check(len(r_single.sweep) == 0, "single (seq_len, batch_size) → sweep is empty")

# ── EntropyMetric with lists ──────────────────────────────────────────────────
print("\n  EntropyMetric — sweep grid")

ent_sweep = EntropyMetric(
    data=fake_data,
    seq_lens=[16, 32],
    batch_sizes=[2, 4],
    n_batches=3,
)
r_ent = ent_sweep.measure(uniform_logits, rng)
check(isinstance(r_ent, EntropyResult),    "returns EntropyResult")
check(len(r_ent.sweep) == 4,              f"sweep has 2×2=4 entries  got={len(r_ent.sweep)}")
check(all("mean_entropy"   in e for e in r_ent.sweep), "every sweep entry has mean_entropy")
check(all("mean_top1_prob" in e for e in r_ent.sweep), "every sweep entry has mean_top1_prob")
check(all("std_entropy"    in e for e in r_ent.sweep), "every sweep entry has std_entropy")
check(r_ent.mean_entropy > 0,             "aggregate mean_entropy > 0")

# backward compat: single point → sweep is empty
ent_single = EntropyMetric(data=fake_data, seq_len=32, batch_size=2, n_batches=2)
r_ent_s = ent_single.measure(uniform_logits, rng)
check(len(r_ent_s.sweep) == 0, "single (seq_len, batch_size) → sweep is empty")

# ── LatencyMetric.measure_sweep() ────────────────────────────────────────────
print("\n  LatencyMetric.measure_sweep()")

lat_sweep = LatencyMetric(n_warmup=1, n_measure=5)
r_lat = lat_sweep.measure_sweep(
    get_batch_fn=lambda bs, sl: jnp.ones((bs, sl), jnp.int32),
    model_fn=lambda x: jax.block_until_ready(x.sum()),
    batch_sizes=[1, 2, 4],
    seq_lens=[64, 128],
)
check(isinstance(r_lat, LatencyResult),  "returns LatencyResult")
check(len(r_lat.grid) == 6,             f"grid has 3×2=6 entries  got={len(r_lat.grid)}")
check(all("batch_size" in e for e in r_lat.grid), "every grid entry has batch_size")
check(all("seq_len"    in e for e in r_lat.grid), "every grid entry has seq_len")
check(all("mean_ms"    in e for e in r_lat.grid), "every grid entry has mean_ms")
check(all("p50_ms"     in e for e in r_lat.grid), "every grid entry has p50_ms")
check(all("p99_ms"     in e for e in r_lat.grid), "every grid entry has p99_ms")
bs_in_grid = sorted(set(e["batch_size"] for e in r_lat.grid))
sl_in_grid = sorted(set(e["seq_len"]    for e in r_lat.grid))
check(bs_in_grid == [1, 2, 4],   f"grid batch_sizes: {bs_in_grid}")
check(sl_in_grid == [64, 128],   f"grid seq_lens: {sl_in_grid}")
check("grid:" in str(r_lat) and "6" in str(r_lat), "str() reflects grid count")

# ── ThroughputMetric 2-D grid ──────────────────────────────────────────────
print("\n  ThroughputMetric — 2-D grid")

thr_grid = ThroughputMetric(
    n_warmup=1, n_measure=5,
    batch_sizes=[1, 2, 4],
    seq_lens=[32, 64],
)
r_thr = thr_grid.measure(
    get_batch_fn=lambda bs, sl: jnp.ones((bs, sl), jnp.int32),
    model_fn=lambda x: jax.block_until_ready(x.sum()),
)
check(isinstance(r_thr, ThroughputResult),  "returns ThroughputResult")
check(len(r_thr.grid) == 6,               f"grid has 3×2=6 entries  got={len(r_thr.grid)}")
check(all("tps" in e for e in r_thr.grid), "every grid entry has tps")
check(r_thr.peak_tps > 0,                 f"peak_tps > 0  got={r_thr.peak_tps:,.0f}")

# ── 3D plots (no browser, write to /tmp HTML) ─────────────────────────────────
print("\n  3D plots (write HTML only, no browser)")

import tempfile, os as _os

from dantinox.profiling.plots import (
    plot_3d_surface, plot_3d_compare, plot_bar_compare,
)

# Build a synthetic RunProfile with throughput grid for plotting
from dantinox.profiling import RunProfile, MultiRunReport, ThroughputResult

def _make_profile(name, bs_list, sl_list, tps_fn):
    grid = [
        {"batch_size": bs, "seq_len": sl, "tps": tps_fn(bs, sl)}
        for bs in bs_list for sl in sl_list
    ]
    thr  = ThroughputResult(
        peak_tps=max(e["tps"] for e in grid),
        by_batch={bs: tps_fn(bs, sl_list[0]) for bs in bs_list},
        by_seq={sl: tps_fn(1, sl) for sl in sl_list},
        seq_len=sl_list[0],
        grid=grid,
    )
    return RunProfile(
        run_name=name, run_dir=f"/tmp/{name}",
        config={"dim": 128},
        throughput=thr,
    )

p_a = _make_profile("model_A", [1,4,16,64], [64,128,256,512],
                    lambda bs, sl: bs * 50_000 / math.log(sl + 1))
p_b = _make_profile("model_B", [1,4,16,64], [64,128,256,512],
                    lambda bs, sl: bs * 80_000 / math.log(sl + 1))

synth_report = MultiRunReport(
    profiles=[p_a, p_b],
    total_time_s=1.0,
    metrics=["throughput"],
    filter_used={},
)

with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
    html_surf = f.name
with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
    html_cmp = f.name
with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
    html_bar = f.name

try:
    fig1 = plot_3d_surface(p_a, metric="tps", show=False, output=html_surf)
    check(_os.path.exists(html_surf) and _os.path.getsize(html_surf) > 1000,
          f"plot_3d_surface writes HTML  ({_os.path.getsize(html_surf):,} bytes)")
    import plotly.graph_objects as go
    check(isinstance(fig1, go.Figure), "plot_3d_surface returns Figure")
    check(len(fig1.data) == 1, "single-run surface has 1 trace")
    check(fig1.data[0].type in ("surface", "scatter3d"), "trace type is surface or scatter3d")
except Exception as exc:
    check(False, f"plot_3d_surface raised: {exc}")

try:
    fig2 = plot_3d_compare(synth_report, metric="tps", mode="surface",
                           show=False, output=html_cmp)
    check(_os.path.exists(html_cmp) and _os.path.getsize(html_cmp) > 1000,
          f"plot_3d_compare writes HTML  ({_os.path.getsize(html_cmp):,} bytes)")
    check(isinstance(fig2, go.Figure), "plot_3d_compare returns Figure")
    check(len(fig2.data) == 2, "two runs → 2 traces")
except Exception as exc:
    check(False, f"plot_3d_compare raised: {exc}")

try:
    fig3 = plot_3d_compare(synth_report, metric="tps", mode="scatter",
                           show=False)
    check(isinstance(fig3, go.Figure), "scatter mode returns Figure")
    check(all(t.type == "scatter3d" for t in fig3.data), "scatter mode → all scatter3d traces")
except Exception as exc:
    check(False, f"scatter mode raised: {exc}")

# bar chart on scalar metric — use report from section 3 (has lat_mean_ms)
try:
    if len(report.profiles) > 0 and report.profiles[0].latency is not None:
        fig4 = plot_bar_compare(report, metric="lat_mean_ms",
                                show=False, output=html_bar)
        check(isinstance(fig4, go.Figure), "plot_bar_compare returns Figure")
        check(_os.path.exists(html_bar) and _os.path.getsize(html_bar) > 500,
              "plot_bar_compare writes HTML")
    else:
        print("  SKIP  plot_bar_compare (no real run latency available)")
except Exception as exc:
    check(False, f"plot_bar_compare raised: {exc}")

# MultiRunReport.plot() convenience method
try:
    fig5 = synth_report.plot("tps", kind="3d", show=False)
    check(isinstance(fig5, go.Figure), "MultiRunReport.plot('tps') returns Figure")
except Exception as exc:
    check(False, f"MultiRunReport.plot() raised: {exc}")

# ValueError for missing metric
try:
    plot_3d_surface(p_a, metric="perplexity", show=False)
    check(False, "should raise ValueError for missing sweep data")
except ValueError:
    check(True, "plot_3d_surface raises ValueError for missing sweep data")

# cleanup
for f in (html_surf, html_cmp, html_bar):
    try: _os.unlink(f)
    except Exception: pass


# ══════════════════════════════════════════════════════════════════════════════
# Final summary
# ══════════════════════════════════════════════════════════════════════════════

section("Results")
total = 0  # counted by check()
if _failures:
    print(f"\n  {ERR}  {len(_failures)} failure(s):")
    for f in _failures:
        print(f"    • {f}")
else:
    print(f"\n  {OK}  All checks passed.")

print()
