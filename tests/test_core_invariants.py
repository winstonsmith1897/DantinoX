"""Invariant tests protecting the paper's claims and cross-path consistency.

These lock down properties that individual unit tests miss:

* the split (ModelConfig/FlowMatchingConfig) ↔ legacy (Config) bridge is lossless,
* legacy serialized keys (``elf_*``) still load,
* KV-cached and uncached AR generation agree,
* MHA and GQA-with-all-heads are the same model,
* the absorbed MLA inference path equals the explicit training path,
* eager and streaming Fast-dLLM generation agree,
* the vectorised factor decoder matches the reference algorithm,
* checkpoints round-trip through every loader (flat and train-state formats).
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml
from flax import nnx

from dantinox.core.checkpoint import load_model
from dantinox.core.config import Config, FlowMatchingConfig, ModelConfig, TrainingConfig
from dantinox.core.diffusion import confidence_unmask_factor, make_noise_schedule
from dantinox.core.generation import fast_dllm_generate, generate, stream_fast_dllm_generate
from dantinox.core.model import Transformer

# ── Config bridge invariants ──────────────────────────────────────────────────


def _assert_fields_equal(a, b, exclude=()):
    diffs = [
        (f.name, getattr(a, f.name), getattr(b, f.name))
        for f in dataclasses.fields(type(a))
        if f.name not in exclude and getattr(a, f.name) != getattr(b, f.name)
    ]
    assert not diffs, f"fields changed across round-trip: {diffs}"


def test_model_config_roundtrip_is_lossless():
    """ModelConfig → Config → ModelConfig must be the identity on every field."""
    m = ModelConfig(
        paradigm="discrete", noise_schedule="sqrt", attention="gqa", kv_heads=4,
        no_sink=True, differential=True, lambda_init=0.5, ffn="moe", n_experts=8,
        top_k=3, moe_latent=True, moe_latent_dim=32, vocab_size=1000,
        pos_encoding="learned", norm="layernorm", dropout=0.2, use_lora=True,
        lora_targets="ffn", sliding_window=True, context_window=7, mask_token_id=9,
        flow_n_steps=17, sde_gamma=0.3, rope_scale=2.0, use_flash=True,
    )
    c = Config.from_parts(m, TrainingConfig(lr=1e-3))
    assert c.noise_schedule == "sqrt"      # the field silently dropped pre-refactor
    assert c.model_type == "diffusion"
    assert c.lr == 1e-3
    _assert_fields_equal(m, c.to_model_config(), exclude=("embed_pooling", "embed_temperature"))


def test_flow_config_roundtrip_is_lossless():
    """FlowMatchingConfig → Config → FlowMatchingConfig must be the identity."""
    f = FlowMatchingConfig(
        embed_dim=512, model_dim=256, n_heads=4, num_blocks=2, vocab_size=32100,
        attention="gqa", kv_heads=2, ffn="moe", n_experts=6, flow_n_steps=9,
        sde_gamma=0.2, pos_encoding="absolute", norm="layernorm", dropout=0.1,
    )
    c = Config.from_parts(f)
    assert c.model_type == "elf" and c.use_moe and c.attention_type == "gqa"
    assert c.tokenizer_type == "t5"        # frozen T5 encoder needs T5 token IDs
    _assert_fields_equal(f, c.to_flow_config())


def test_legacy_yaml_keys_still_load(tmp_path):
    """config.yaml files written with the old elf_* field names keep loading."""
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump({"dim": 64, "n_heads": 4, "vocab_size": 128,
                               "elf_n_steps": 7, "elf_cfg_scale": 2.5}))
    cfg = ModelConfig.from_yaml(str(path))
    assert cfg.flow_n_steps == 7
    assert cfg.flow_cfg_scale == 2.5
    # deprecated accessors still answer
    assert cfg.elf_n_steps == 7 and cfg.elf_cfg_scale == 2.5


def test_default_configs_agree():
    """Bare Config() and bare ModelConfig() must describe the same architecture."""
    # gradient_checkpointing is intentionally excluded: it lives on TrainingConfig
    # now, not ModelConfig, so it no longer has a shared-architecture meaning here.
    c, m = Config(), ModelConfig()
    assert (c.norm_type, c.no_sink, c.noise_schedule, c.num_blocks, c.kv_heads,
            c.attention_type, c.dropout_rate) == (
        m.norm, m.no_sink, m.noise_schedule, m.num_blocks, m.kv_heads,
        m.attention, m.dropout,
    )


# ── Tiny models ───────────────────────────────────────────────────────────────


def _tiny(paradigm="ar", **kw) -> ModelConfig:
    base = dict(dim=64, n_heads=4, num_blocks=2, vocab_size=97, max_context=48)
    base.update(kw)
    return ModelConfig(paradigm=paradigm, **base)


# ── Generation invariants ─────────────────────────────────────────────────────


def test_cached_and_uncached_ar_generation_agree():
    """KV-cached decoding must produce the same tokens as full recomputation."""
    model = Transformer(_tiny(), rngs=nnx.Rngs(0))
    x = jnp.array([[5, 17, 3, 42, 8]], dtype=jnp.int32)
    out_cached   = generate(model, x, 16, greedy=True, use_cache=True)
    out_uncached = generate(model, x, 16, greedy=True, use_cache=False)
    np.testing.assert_array_equal(np.asarray(out_cached), np.asarray(out_uncached))


def test_gqa_with_all_kv_heads_equals_mha():
    """GQA with kv_heads == n_heads is MHA: same params, same logits."""
    m1 = Transformer(_tiny(attention="mha"), rngs=nnx.Rngs(0))
    m2 = Transformer(_tiny(attention="gqa", kv_heads=4), rngs=nnx.Rngs(0))
    x = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    l1 = m1(x, deterministic=True).logits
    l2 = m2(x, deterministic=True).logits
    np.testing.assert_allclose(np.asarray(l1), np.asarray(l2), rtol=0, atol=0)


def test_mla_absorbed_path_equals_explicit_path():
    """MLA absorbed inference (latent cache) must equal the training path."""
    cfg = _tiny(attention="mla", down_dim_q=16, down_dim_kv=16, rope_dim=8)
    model = Transformer(cfg, rngs=nnx.Rngs(0))
    x = jnp.array([[7, 1, 30, 4, 11, 2]], dtype=jnp.int32)

    explicit = model(x, deterministic=True).logits
    for block in model.blocks:
        block.attention.inference = True
    absorbed = model(x, deterministic=True).logits

    np.testing.assert_allclose(np.asarray(explicit), np.asarray(absorbed),
                               rtol=1e-4, atol=1e-4)


def test_fast_dllm_eager_equals_stream():
    """The eager Fast-dLLM API must return exactly the streaming final state."""
    model = Transformer(_tiny(paradigm="discrete", mask_token_id=4), rngs=nnx.Rngs(0))
    schedule = make_noise_schedule("linear", n_steps=8)
    prefix = jnp.array([[9, 12, 5]], dtype=jnp.int32)

    eager = fast_dllm_generate(model, prefix, 16, schedule, mask_token_id=4,
                               block_size=8, steps_per_block=4)
    last = None
    for _, _, last in stream_fast_dllm_generate(model, prefix, 16, schedule,
                                                mask_token_id=4, block_size=8,
                                                steps_per_block=4):
        pass
    np.testing.assert_array_equal(np.asarray(eager), np.asarray(last))
    assert not (np.asarray(eager) == 4).any(), "mask tokens must never survive"


def test_confidence_unmask_factor_matches_reference():
    """Vectorised factor decoding must equal the sequential reference algorithm."""

    def reference(logits, x_t, mask_id, factor):
        logits, x_t = np.asarray(logits), np.asarray(x_t).copy()
        e = np.exp(logits - logits.max(-1, keepdims=True))
        probs = e / e.sum(-1, keepdims=True)
        conf, x0 = probs.max(-1), logits.argmax(-1)
        for b in range(x_t.shape[0]):
            pos = np.where(x_t[b] == mask_id)[0]
            if pos.size == 0:
                continue
            order = np.argsort(-conf[b][pos])
            sc, sp = conf[b][pos][order], pos[order]
            n_unmask = 1
            for n in range(1, len(sc)):
                if (n + 1) * (1.0 - float(sc[n])) < factor:
                    n_unmask = n + 1
                else:
                    break
            for i in range(n_unmask):
                x_t[b, sp[i]] = x0[b, sp[i]]
        return x_t

    rng = np.random.default_rng(0)
    for _ in range(20):
        B, T, V = int(rng.integers(1, 4)), int(rng.integers(2, 16)), 11
        logits = rng.normal(0, 3, (B, T, V)).astype(np.float32)
        x_t = rng.integers(0, V, (B, T)).astype(np.int32)
        x_t = np.where(rng.random((B, T)) < rng.random(), 4,
                       np.where(x_t == 4, 0, x_t)).astype(np.int32)
        factor = float(rng.uniform(0.1, 3.0))
        got = confidence_unmask_factor(jnp.asarray(logits), jnp.asarray(x_t), 4, factor)
        np.testing.assert_array_equal(np.asarray(got), reference(logits, x_t, 4, factor))


# ── Checkpoint round-trips ────────────────────────────────────────────────────


def _save_run_dir(tmp_path, name, cfg, model, wrap_train_state=False):
    import flax.serialization

    run = tmp_path / name
    run.mkdir()
    cfg.save_yaml(str(run / "config.yaml"))
    pure = nnx.state(model, nnx.Not(nnx.RngState)).to_pure_dict()
    payload = {"model": pure, "opt": {}} if wrap_train_state else pure
    (run / "checkpoint_best.msgpack").write_bytes(
        flax.serialization.msgpack_serialize(payload)
    )
    return str(run)


def _first_kernel(model):
    return np.asarray(model.blocks[0].attention.qkv.kernel[...])


@pytest.mark.parametrize("wrap_train_state", [False, True],
                         ids=["flat-weights", "train-state"])
def test_checkpoint_roundtrip_all_loaders(tmp_path, wrap_train_state):
    """Both loaders must restore both checkpoint formats identically."""
    cfg = _tiny()
    src = Transformer(cfg, rngs=nnx.Rngs(7))
    run_dir = _save_run_dir(tmp_path, "run", cfg, src, wrap_train_state)

    via_from_pretrained = Transformer.from_pretrained(run_dir, rngs=nnx.Rngs(1))
    np.testing.assert_allclose(_first_kernel(via_from_pretrained), _first_kernel(src))

    via_load_model, loaded_cfg, _ = load_model(run_dir, seed=2)
    assert isinstance(loaded_cfg, ModelConfig)
    np.testing.assert_allclose(_first_kernel(via_load_model), _first_kernel(src))

    # And the restored model must behave like the source model.
    x = jnp.array([[3, 1, 4, 1, 5]], dtype=jnp.int32)
    np.testing.assert_allclose(
        np.asarray(src(x, deterministic=True).logits),
        np.asarray(via_from_pretrained(x, deterministic=True).logits),
        rtol=1e-6, atol=1e-6,
    )


def test_checkpoint_loader_detects_legacy_config(tmp_path):
    """A run dir with a legacy monolithic config.yaml loads as a diffusion model."""
    legacy = Config(dim=64, n_heads=4, head_size=16, num_blocks=2, vocab_size=97,
                    max_context=48, model_type="diffusion")
    model = Transformer(legacy, rngs=nnx.Rngs(0))
    run_dir = _save_run_dir(tmp_path, "legacy_run", legacy, model)

    restored, cfg, _ = load_model(run_dir)
    assert isinstance(cfg, Config)
    assert cfg.model_type == "diffusion"
    assert restored.causal is False  # bidirectional — not silently causal
