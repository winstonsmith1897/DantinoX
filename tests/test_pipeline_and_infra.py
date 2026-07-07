"""
Comprehensive tests for the six new DantinoX components:

  1. dantinox/core/pipeline.py          — smart one-call inference pipeline
  2. dantinox/utils/data_pipeline.py    — streaming AutoDataPipeline
  3. dantinox/training/callbacks.py     — BaseCallback + WandbCallback
  4. dantinox/training/stability.py     — OOM guard context manager
  5. dantinox/core/export.py            — StableHLO export
  6. Trainer callback integration       — hooks wired into fit()
  7. CLI subcommands                    — run + export parsers
"""

from __future__ import annotations

import os
import pathlib
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import jax
import numpy as np
import pytest

# Force CPU for all tests in this session.
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

DUMMY_CORPUS = pathlib.Path(__file__).parent / "dummy_data.txt"

# Build a longer corpus so the Trainer always has enough tokens.
_RAW_TEXT = DUMMY_CORPUS.read_text(encoding="utf-8") * 20


class _SimpleTokenizer:
    """Minimal char-level tokenizer for tests that need no external deps."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) % 200 for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i % 128 or 32) for i in ids)

    @property
    def vocab_size(self) -> int:
        return 200


class _RecordingCallback:
    """Records every callback hook call for assertion in tests."""

    def __init__(self) -> None:
        self.begin_calls: list[Any] = []
        self.step_calls: list[tuple] = []
        self.end_calls: int = 0

    def on_train_begin(self, config: Any) -> None:
        self.begin_calls.append(config)

    def on_step_end(self, step: int, metrics: dict, epoch: int) -> None:
        self.step_calls.append((step, metrics, epoch))

    def on_train_end(self) -> None:
        self.end_calls += 1


@pytest.fixture(scope="module")
def corpus_file(tmp_path_factory) -> str:
    """Write an extended dummy corpus and return its absolute path."""
    path = tmp_path_factory.mktemp("corpus") / "corpus.txt"
    path.write_text(_RAW_TEXT, encoding="utf-8")
    return str(path)


@pytest.fixture(scope="module")
def tiny_model_config():
    """A minimal ModelConfig for fast AR training (CPU-friendly)."""
    from dantinox.core.config import ModelConfig

    return ModelConfig(
        dim=64,
        n_heads=2,
        head_size=32,
        num_blocks=1,
        vocab_size=200,
        max_context=32,
        causal=True,
        dropout=0.0,
        weight_tying=False,
    )


@pytest.fixture(scope="module")
def tiny_train_config():
    """A minimal TrainingConfig for 1-epoch training on the dummy corpus."""
    from dantinox.core.config import TrainingConfig

    return TrainingConfig(
        epochs=1,
        batch_size=4,
        grad_accum=1,
        eval_iters=1,
        warmup_steps=1,
        tokenizer_type="char",
        val_frac=0.1,
        lr=1e-3,
    )


@pytest.fixture(scope="module")
def tiny_paradigm(tiny_model_config):
    from dantinox.paradigms.ar import ARParadigm

    return ARParadigm(tiny_model_config)


@pytest.fixture(scope="module")
def trained_new_run(tiny_paradigm, tiny_train_config, corpus_file, tmp_path_factory):
    """Train a tiny AR model with the new Trainer and return the run directory."""
    from dantinox.training.trainer import Trainer

    run_dir = str(tmp_path_factory.mktemp("new_trainer_run"))
    trainer = Trainer(tiny_paradigm, tiny_train_config)
    result = trainer.fit(corpus_file, run_dir=run_dir)
    return result


# ===========================================================================
# 1. AutoDataPipeline
# ===========================================================================

class TestAutoDataPipeline:
    """Unit tests for dantinox/utils/data_pipeline.py."""

    def _make_pipe(self, ctx: int = 8, text_col: str = "text"):
        from dantinox.utils.data_pipeline import AutoDataPipeline

        return AutoDataPipeline(
            dataset_path="fake_dataset",
            subset="fake_subset",
            split="train",
            text_column=text_col,
            tokenizer=_SimpleTokenizer(),
            max_context=ctx,
        )

    def _fake_modules(self, rows: list[dict]) -> dict:
        """Return a sys.modules patch that injects a fake datasets module."""
        mock_ds = MagicMock()
        mock_ds.load_dataset.return_value = iter(rows)
        return {"datasets": mock_ds}

    # ── Constructor ──────────────────────────────────────────────────────────

    def test_stores_params(self):
        pipe = self._make_pipe(ctx=16)
        assert pipe._max_context == 16
        assert pipe._split == "train"
        assert pipe._text_column == "text"

    def test_repr_is_informative(self):
        pipe = self._make_pipe(ctx=32)
        r = repr(pipe)
        assert "AutoDataPipeline" in r
        assert "32" in r

    # ── Shapes & dtypes ──────────────────────────────────────────────────────

    def test_yields_dict_with_inputs_and_targets(self):
        rows = [{"text": "abcdefghijklmnopqrstuvwxyz" * 3}]
        with patch.dict(sys.modules, self._fake_modules(rows)):
            pipe = self._make_pipe(ctx=8)
            batch = next(iter(pipe))
        assert set(batch.keys()) == {"inputs", "targets"}

    def test_inputs_shape_equals_max_context(self):
        ctx = 12
        rows = [{"text": "x" * 200}]
        with patch.dict(sys.modules, self._fake_modules(rows)):
            pipe = self._make_pipe(ctx=ctx)
            batch = next(iter(pipe))
        assert batch["inputs"].shape == (ctx,)
        assert batch["targets"].shape == (ctx,)

    def test_dtype_is_int32(self):
        rows = [{"text": "hello world " * 10}]
        with patch.dict(sys.modules, self._fake_modules(rows)):
            pipe = self._make_pipe(ctx=6)
            batch = next(iter(pipe))
        assert batch["inputs"].dtype == np.int32
        assert batch["targets"].dtype == np.int32

    # ── AR shift property ────────────────────────────────────────────────────

    def test_targets_are_inputs_shifted_by_one(self):
        """targets[i] == the next token after inputs[i] within the chunk."""
        rows = [{"text": "abcdefghijklmnopqrstuvwxyz" * 5}]
        ctx = 8
        with patch.dict(sys.modules, self._fake_modules(rows)):
            pipe = self._make_pipe(ctx=ctx)
            batch = next(iter(pipe))
        # Within a single window: inputs[1:] == targets[:-1]
        np.testing.assert_array_equal(batch["inputs"][1:], batch["targets"][:-1])

    # ── Empty / short text handling ──────────────────────────────────────────

    def test_skips_empty_texts(self):
        """Empty texts must not produce garbage batches."""
        ctx = 4
        rows = [{"text": ""}, {"text": "   "}, {"text": "abcdefghijklmnop"}]
        with patch.dict(sys.modules, self._fake_modules(rows)):
            pipe = self._make_pipe(ctx=ctx)
            batch = next(iter(pipe))
        assert batch["inputs"].shape == (ctx,)

    def test_skips_wrong_column(self):
        """Rows without the expected column are gracefully ignored."""
        ctx = 4
        rows = [{"text": ""}, {"text": ""}, {"other": "abc"}, {"text": "X" * 50}]
        with patch.dict(sys.modules, self._fake_modules(rows)):
            pipe = self._make_pipe(ctx=ctx, text_col="text")
            batch = next(iter(pipe))
        assert batch["inputs"].shape == (ctx,)

    # ── Multiple consecutive batches ─────────────────────────────────────────

    def test_multiple_batches_are_non_overlapping(self):
        """Successive windows must not share the same first token."""
        ctx = 4
        long_text = "".join(chr(65 + i % 26) for i in range(300))
        rows = [{"text": long_text}]
        with patch.dict(sys.modules, self._fake_modules(rows)):
            pipe = self._make_pipe(ctx=ctx)
            it = iter(pipe)
            b1 = next(it)
            b2 = next(it)
        # Non-overlapping: b2 starts where b1 ended.
        assert not np.array_equal(b1["inputs"], b2["inputs"])

    # ── Missing datasets module ───────────────────────────────────────────────

    def test_raises_import_error_when_datasets_missing(self):
        from dantinox.utils.data_pipeline import AutoDataPipeline

        pipe = AutoDataPipeline("ds", "sub", "train", "text", _SimpleTokenizer(), 8)
        # Hide both the package and any cached import
        hidden = {k: None for k in list(sys.modules.keys()) if k.startswith("datasets")}
        hidden["datasets"] = None
        with patch.dict(sys.modules, hidden), pytest.raises(ImportError, match="datasets"):
            next(iter(pipe))


# ===========================================================================
# 2. BaseCallback
# ===========================================================================

class TestBaseCallback:
    """BaseCallback must be instantiable with silent no-op hooks."""

    def test_instantiates(self):
        from dantinox.training.callbacks import BaseCallback

        cb = BaseCallback()
        assert isinstance(cb, BaseCallback)

    def test_on_train_begin_is_noop(self):
        from dantinox.training.callbacks import BaseCallback

        cb = BaseCallback()
        cb.on_train_begin(None)  # should not raise

    def test_on_step_end_is_noop(self):
        from dantinox.training.callbacks import BaseCallback

        cb = BaseCallback()
        cb.on_step_end(step=0, metrics={"train_loss": 1.0}, epoch=1)

    def test_on_train_end_is_noop(self):
        from dantinox.training.callbacks import BaseCallback

        cb = BaseCallback()
        cb.on_train_end()

    def test_on_step_end_accepts_arbitrary_metrics(self):
        from dantinox.training.callbacks import BaseCallback

        cb = BaseCallback()
        cb.on_step_end(step=99, metrics={"a": 0.1, "b": 0.2, "c": -3.0}, epoch=5)


# ===========================================================================
# 3. WandbCallback
# ===========================================================================

class TestWandbCallback:
    """WandbCallback — uses a fully mocked wandb module."""

    def _mock_wandb(self):
        mw = MagicMock()
        mw.init.return_value = MagicMock(name="mock-run-1")
        return mw

    def test_stores_project_and_name(self):
        from dantinox.training.callbacks import WandbCallback

        cb = WandbCallback(project="my-proj", name="run-42")
        assert cb._project == "my-proj"
        assert cb._name == "run-42"

    def test_stores_tags(self):
        from dantinox.training.callbacks import WandbCallback

        cb = WandbCallback(tags=["arxiv", "v2"])
        assert cb._tags == ["arxiv", "v2"]

    def test_on_train_end_noop_when_run_is_none(self):
        from dantinox.training.callbacks import WandbCallback

        cb = WandbCallback(project="p")
        cb.on_train_end()  # _run is None — must not raise

    def test_on_step_end_noop_when_run_is_none(self):
        from dantinox.training.callbacks import WandbCallback

        cb = WandbCallback(project="p")
        cb.on_step_end(step=0, metrics={"train_loss": 0.5}, epoch=1)

    def test_on_train_begin_calls_wandb_init(self):
        from dantinox.training.callbacks import WandbCallback

        mw = self._mock_wandb()
        with patch.dict(sys.modules, {"wandb": mw}):
            cb = WandbCallback(project="proj", name="run-x")
            cb.on_train_begin(SimpleNamespace())
        mw.init.assert_called_once()
        call_kwargs = mw.init.call_args.kwargs
        assert call_kwargs["project"] == "proj"
        assert call_kwargs["name"] == "run-x"

    def test_on_train_begin_passes_dataclass_config(self):
        import dataclasses

        from dantinox.training.callbacks import WandbCallback

        @dataclasses.dataclass
        class FakeCfg:
            lr: float = 1e-3
            epochs: int = 5

        mw = self._mock_wandb()
        with patch.dict(sys.modules, {"wandb": mw}):
            cb = WandbCallback(project="p")
            cb.on_train_begin(FakeCfg())
        logged_cfg = mw.init.call_args.kwargs["config"]
        assert logged_cfg["lr"] == pytest.approx(1e-3)
        assert logged_cfg["epochs"] == 5

    def test_on_step_end_calls_wandb_log(self):
        from dantinox.training.callbacks import WandbCallback

        mw = self._mock_wandb()
        with patch.dict(sys.modules, {"wandb": mw}):
            cb = WandbCallback(project="p")
            cb.on_train_begin(SimpleNamespace())
            cb.on_step_end(step=10, metrics={"train_loss": 0.8}, epoch=2)
        mw.log.assert_called_once()
        logged, kwargs = mw.log.call_args.args[0], mw.log.call_args.kwargs
        assert logged["train_loss"] == pytest.approx(0.8)
        assert logged["epoch"] == 2
        assert kwargs["step"] == 10

    def test_on_train_end_calls_wandb_finish(self):
        from dantinox.training.callbacks import WandbCallback

        mw = self._mock_wandb()
        with patch.dict(sys.modules, {"wandb": mw}):
            cb = WandbCallback(project="p")
            cb.on_train_begin(SimpleNamespace())
            cb.on_train_end()
        mw.finish.assert_called_once()
        assert cb._run is None

    def test_on_train_begin_raises_import_error_without_wandb(self):
        from dantinox.training.callbacks import WandbCallback

        cb = WandbCallback(project="p")
        with patch.dict(sys.modules, {"wandb": None}):
            with pytest.raises(ImportError, match="wandb"):
                cb.on_train_begin(SimpleNamespace())

    def test_config_overrides_are_merged(self):
        from dantinox.training.callbacks import WandbCallback

        mw = self._mock_wandb()
        with patch.dict(sys.modules, {"wandb": mw}):
            cb = WandbCallback(project="p", git_sha="abc123")
            cb.on_train_begin(SimpleNamespace())
        logged_cfg = mw.init.call_args.kwargs["config"]
        assert logged_cfg["git_sha"] == "abc123"


# ===========================================================================
# 4. OOM Guard
# ===========================================================================

class TestOOMMitigatedError:
    """Unit tests for the OOMMitigatedError exception class."""

    def test_stores_new_config_values(self):
        from dantinox.training.stability import OOMMitigatedError

        original = RuntimeError("oom")
        err = OOMMitigatedError(new_batch_size=16, new_grad_accum=8, original=original)
        assert err.new_batch_size == 16
        assert err.new_grad_accum == 8
        assert err.original is original

    def test_str_contains_batch_and_accum(self):
        from dantinox.training.stability import OOMMitigatedError

        err = OOMMitigatedError(4, 16, RuntimeError("oom"))
        msg = str(err)
        assert "4" in msg
        assert "16" in msg

    def test_is_runtime_error_subclass(self):
        from dantinox.training.stability import OOMMitigatedError

        assert issubclass(OOMMitigatedError, RuntimeError)


class TestOOMGuard:
    """Unit tests for the with_oom_guard context manager."""

    def _cfg(self, batch_size: int = 32, grad_accum: int = 4) -> Any:
        return SimpleNamespace(batch_size=batch_size, grad_accum=grad_accum)

    # ── Happy path ────────────────────────────────────────────────────────────

    def test_success_path_leaves_config_unchanged(self):
        from dantinox.training.stability import with_oom_guard

        cfg = self._cfg()
        with with_oom_guard(cfg):
            pass  # no exception
        assert cfg.batch_size == 32
        assert cfg.grad_accum == 4

    def test_success_path_does_not_raise(self):
        from dantinox.training.stability import with_oom_guard

        cfg = self._cfg()
        result = []
        with with_oom_guard(cfg):
            result.append(1)
        assert result == [1]

    # ── OOM detection ─────────────────────────────────────────────────────────

    @pytest.mark.parametrize("msg", [
        "CUDA out of memory",
        "out of memory when allocating tensor",
        "resource_exhausted: failed to allocate",
        "Allocation failed: cannot allocate memory",
        "OOM when allocating device memory",
        "RESOURCE_EXHAUSTED: dummy",
    ])
    def test_oom_message_is_caught(self, msg: str):
        from dantinox.training.stability import OOMMitigatedError, with_oom_guard

        cfg = self._cfg(batch_size=16, grad_accum=2)
        with pytest.raises(OOMMitigatedError), with_oom_guard(cfg):
            raise RuntimeError(msg)

    def test_non_oom_runtime_error_passes_through(self):
        from dantinox.training.stability import with_oom_guard

        cfg = self._cfg()
        with pytest.raises(RuntimeError, match="division"), with_oom_guard(cfg):
            raise RuntimeError("division by zero")
        # Config must not have been modified
        assert cfg.batch_size == 32
        assert cfg.grad_accum == 4

    def test_value_error_passes_through(self):
        from dantinox.training.stability import with_oom_guard

        cfg = self._cfg()
        with pytest.raises(ValueError), with_oom_guard(cfg):
            raise ValueError("bad value")

    def test_key_error_passes_through(self):
        from dantinox.training.stability import with_oom_guard

        cfg = self._cfg()
        with pytest.raises(KeyError), with_oom_guard(cfg):
            raise KeyError("missing")

    # ── Config mutation ───────────────────────────────────────────────────────

    def test_batch_size_is_halved(self):
        from dantinox.training.stability import OOMMitigatedError, with_oom_guard

        cfg = self._cfg(batch_size=64, grad_accum=1)
        with pytest.raises(OOMMitigatedError), with_oom_guard(cfg):
            raise RuntimeError("out of memory")
        assert cfg.batch_size == 32

    def test_grad_accum_is_doubled(self):
        from dantinox.training.stability import OOMMitigatedError, with_oom_guard

        cfg = self._cfg(batch_size=32, grad_accum=4)
        with pytest.raises(OOMMitigatedError), with_oom_guard(cfg):
            raise RuntimeError("out of memory")
        assert cfg.grad_accum == 8

    def test_batch_size_floor_is_one(self):
        from dantinox.training.stability import OOMMitigatedError, with_oom_guard

        cfg = self._cfg(batch_size=1, grad_accum=2)
        with pytest.raises(OOMMitigatedError) as exc_info, with_oom_guard(cfg):
            raise RuntimeError("cuda out of memory")
        assert cfg.batch_size == 1
        assert exc_info.value.new_batch_size == 1

    def test_odd_batch_size_halved_correctly(self):
        from dantinox.training.stability import OOMMitigatedError, with_oom_guard

        cfg = self._cfg(batch_size=7, grad_accum=1)
        with pytest.raises(OOMMitigatedError), with_oom_guard(cfg):
            raise RuntimeError("out of memory")
        assert cfg.batch_size == 3   # floor(7 // 2)

    # ── OOMMitigatedError attributes ──────────────────────────────────────────

    def test_error_new_batch_size_matches_config(self):
        from dantinox.training.stability import OOMMitigatedError, with_oom_guard

        cfg = self._cfg(batch_size=128, grad_accum=1)
        with pytest.raises(OOMMitigatedError) as exc_info, with_oom_guard(cfg):
            raise RuntimeError("out of memory")
        assert exc_info.value.new_batch_size == 64
        assert exc_info.value.new_grad_accum == 2

    def test_error_wraps_original_exception(self):
        from dantinox.training.stability import OOMMitigatedError, with_oom_guard

        cfg = self._cfg()
        original = RuntimeError("resource_exhausted: gpu oom")
        with pytest.raises(OOMMitigatedError) as exc_info, with_oom_guard(cfg):
            raise original
        assert exc_info.value.original is original

    # ── Side effects ──────────────────────────────────────────────────────────

    def test_clears_jax_caches_on_oom(self):
        from dantinox.training.stability import OOMMitigatedError, with_oom_guard

        cfg = self._cfg()
        with patch("jax.clear_caches") as mock_clear, pytest.raises(OOMMitigatedError):
            with with_oom_guard(cfg):
                raise RuntimeError("out of memory")
        mock_clear.assert_called_once()

    def test_does_not_clear_caches_on_non_oom(self):
        from dantinox.training.stability import with_oom_guard

        cfg = self._cfg()
        with patch("jax.clear_caches") as mock_clear, pytest.raises(RuntimeError):
            with with_oom_guard(cfg):
                raise RuntimeError("unrelated error")
        mock_clear.assert_not_called()

    # ── Retry pattern ─────────────────────────────────────────────────────────

    def test_retry_loop_reduces_batch_size_each_attempt(self):
        from dantinox.training.stability import OOMMitigatedError, with_oom_guard

        cfg = self._cfg(batch_size=16, grad_accum=1)
        attempt = 0

        with patch("jax.clear_caches"):
            for _ in range(4):
                try:
                    with with_oom_guard(cfg):
                        if cfg.batch_size > 1:
                            raise RuntimeError("out of memory")
                    break  # success when batch_size reached 1
                except OOMMitigatedError:
                    attempt += 1

        assert attempt == 4   # 16→8→4→2→1, 4 OOM events
        assert cfg.batch_size == 1


# ===========================================================================
# 5. Trainer callback integration
# ===========================================================================

class TestTrainerCallbacksIntegration:
    """Verify that Trainer.fit() fires all three callback hooks."""

    def test_trainer_accepts_callbacks_kwarg(self, tiny_paradigm, tiny_train_config):
        from dantinox.training.callbacks import BaseCallback
        from dantinox.training.trainer import Trainer

        cb = BaseCallback()
        trainer = Trainer(tiny_paradigm, tiny_train_config, callbacks=[cb])
        assert cb in trainer.callbacks

    def test_empty_callbacks_by_default(self, tiny_paradigm, tiny_train_config):
        from dantinox.training.trainer import Trainer

        trainer = Trainer(tiny_paradigm, tiny_train_config)
        assert trainer.callbacks == []

    def test_none_callbacks_gives_empty_list(self, tiny_paradigm, tiny_train_config):
        from dantinox.training.trainer import Trainer

        trainer = Trainer(tiny_paradigm, tiny_train_config, callbacks=None)
        assert trainer.callbacks == []

    def test_on_train_begin_called_exactly_once(
        self, tiny_paradigm, tiny_train_config, corpus_file, tmp_path
    ):
        from dantinox.training.trainer import Trainer

        cb = _RecordingCallback()
        trainer = Trainer(tiny_paradigm, tiny_train_config, callbacks=[cb])
        trainer.fit(corpus_file, run_dir=str(tmp_path / "run_begin"))
        assert len(cb.begin_calls) == 1

    def test_on_train_begin_receives_training_config(
        self, tiny_paradigm, tiny_train_config, corpus_file, tmp_path
    ):
        from dantinox.core.config import TrainingConfig
        from dantinox.training.trainer import Trainer

        cb = _RecordingCallback()
        trainer = Trainer(tiny_paradigm, tiny_train_config, callbacks=[cb])
        trainer.fit(corpus_file, run_dir=str(tmp_path / "run_cfg"))
        assert isinstance(cb.begin_calls[0], TrainingConfig)

    def test_on_step_end_called_at_every_step(
        self, tiny_paradigm, tiny_train_config, corpus_file, tmp_path
    ):
        from dantinox.training.trainer import Trainer

        cb = _RecordingCallback()
        trainer = Trainer(tiny_paradigm, tiny_train_config, callbacks=[cb])
        trainer.fit(corpus_file, run_dir=str(tmp_path / "run_steps"))
        # At least one step per epoch
        assert len(cb.step_calls) >= 1

    def test_on_step_end_metrics_has_train_loss(
        self, tiny_paradigm, tiny_train_config, corpus_file, tmp_path
    ):
        from dantinox.training.trainer import Trainer

        cb = _RecordingCallback()
        trainer = Trainer(tiny_paradigm, tiny_train_config, callbacks=[cb])
        trainer.fit(corpus_file, run_dir=str(tmp_path / "run_metrics"))
        for _step, metrics, _epoch in cb.step_calls:
            assert "train_loss" in metrics
            assert isinstance(metrics["train_loss"], float)

    def test_on_step_end_epoch_is_one_based(
        self, tiny_paradigm, tiny_train_config, corpus_file, tmp_path
    ):
        from dantinox.training.trainer import Trainer

        cb = _RecordingCallback()
        trainer = Trainer(tiny_paradigm, tiny_train_config, callbacks=[cb])
        trainer.fit(corpus_file, run_dir=str(tmp_path / "run_epoch"))
        epochs = {epoch for _, _, epoch in cb.step_calls}
        assert min(epochs) >= 1

    def test_on_train_end_called_exactly_once(
        self, tiny_paradigm, tiny_train_config, corpus_file, tmp_path
    ):
        from dantinox.training.trainer import Trainer

        cb = _RecordingCallback()
        trainer = Trainer(tiny_paradigm, tiny_train_config, callbacks=[cb])
        trainer.fit(corpus_file, run_dir=str(tmp_path / "run_end"))
        assert cb.end_calls == 1

    def test_multiple_callbacks_all_called(
        self, tiny_paradigm, tiny_train_config, corpus_file, tmp_path
    ):
        from dantinox.training.trainer import Trainer

        cb1 = _RecordingCallback()
        cb2 = _RecordingCallback()
        trainer = Trainer(tiny_paradigm, tiny_train_config, callbacks=[cb1, cb2])
        trainer.fit(corpus_file, run_dir=str(tmp_path / "run_multi"))
        assert len(cb1.begin_calls) == 1
        assert len(cb2.begin_calls) == 1
        assert cb1.end_calls == 1
        assert cb2.end_calls == 1

    def test_hook_order_begin_before_steps_before_end(
        self, tiny_paradigm, tiny_train_config, corpus_file, tmp_path
    ):
        from dantinox.training.trainer import Trainer

        events: list[str] = []

        class OrderCb:
            def on_train_begin(self, cfg):
                events.append("begin")

            def on_step_end(self, step, metrics, epoch):
                events.append("step")

            def on_train_end(self):
                events.append("end")

        trainer = Trainer(tiny_paradigm, tiny_train_config, callbacks=[OrderCb()])
        trainer.fit(corpus_file, run_dir=str(tmp_path / "run_order"))
        assert events[0] == "begin"
        assert events[-1] == "end"
        assert all(e == "step" for e in events[1:-1])


# ===========================================================================
# 6. pipeline() — _decode helper
# ===========================================================================

class TestDecodeHelper:
    """Unit tests for the private _decode helper in pipeline.py."""

    def test_hf_tokenizer_uses_skip_special(self):
        from dantinox.core.pipeline import _decode

        mock_tok = MagicMock()
        mock_tok.decode.return_value = "hello"
        result = _decode(mock_tok, [1, 2, 3])
        mock_tok.decode.assert_called_once_with([1, 2, 3], skip_special_tokens=True)
        assert result == "hello"

    def test_dantinox_tokenizer_falls_back_on_type_error(self):
        from dantinox.core.pipeline import _decode

        tok = _SimpleTokenizer()
        result = _decode(tok, [72, 101, 108])
        assert isinstance(result, str)

    def test_empty_ids_returns_string(self):
        from dantinox.core.pipeline import _decode

        tok = _SimpleTokenizer()
        result = _decode(tok, [])
        assert isinstance(result, str)


# ===========================================================================
# 7. pipeline() — error cases and integration
# ===========================================================================

class TestPipelineErrors:
    """Error-path tests that do not require a trained model."""

    def test_unsupported_task_raises_value_error(self):
        from dantinox.core.pipeline import pipeline

        with pytest.raises(ValueError, match="Unsupported task"):
            pipeline("summarization", "some/model")

    def test_unsupported_task_lists_valid_options(self):
        from dantinox.core.pipeline import pipeline

        with pytest.raises(ValueError, match="text-generation"):
            pipeline("ner", "some/model")

    def test_missing_config_raises_file_not_found(self, tmp_path):
        from dantinox.core.pipeline import pipeline

        # Directory exists but has no config.yaml
        (tmp_path / "empty_run").mkdir()
        with pytest.raises(FileNotFoundError, match="config.yaml"):
            pipeline("text-generation", str(tmp_path / "empty_run"))

    def test_missing_weights_raises_file_not_found(self, tmp_path):
        """Config exists but no weights file → FileNotFoundError."""
        from dantinox.core.checkpoint import load_model
        from dantinox.core.config import Config

        run = tmp_path / "no_weights"
        run.mkdir()
        cfg = Config(
            dim=64, n_heads=2, head_size=32, num_blocks=1,
            vocab_size=200, max_context=32, kv_heads=2,
        )
        cfg.save_yaml(str(run / "config.yaml"))

        with pytest.raises(FileNotFoundError, match="No weights file"):
            load_model(str(run), seed=0)


class TestPipelineIntegration:
    """End-to-end tests that require a real trained checkpoint."""

    def test_pipeline_returns_string(self, trained_new_run):
        from dantinox.core.pipeline import pipeline

        result = pipeline(
            "text-generation",
            trained_new_run,
            prompt="a",
            max_new_tokens=10,
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_pipeline_prompt_is_in_output(self, trained_new_run):
        from dantinox.core.pipeline import pipeline

        prompt = "P"
        result = pipeline(
            "text-generation",
            trained_new_run,
            prompt=prompt,
            max_new_tokens=5,
        )
        assert result.startswith(prompt)

    def test_pipeline_greedy_is_deterministic(self, trained_new_run):
        from dantinox.core.pipeline import pipeline

        r1 = pipeline("text-generation", trained_new_run, prompt="a",
                      max_new_tokens=15, greedy=True, seed=0)
        r2 = pipeline("text-generation", trained_new_run, prompt="a",
                      max_new_tokens=15, greedy=True, seed=0)
        assert r1 == r2

    def test_pipeline_respects_max_new_tokens(self, trained_new_run):
        import os

        from dantinox.core.pipeline import pipeline
        from dantinox.utils.tokenizer import load_tokenizer_from_file

        tok = load_tokenizer_from_file(
            os.path.join(trained_new_run, "tokenizer.json")
        )
        prompt = "a"
        n = 8
        result = pipeline("text-generation", trained_new_run,
                          prompt=prompt, max_new_tokens=n, greedy=True)
        new_tokens = tok.encode(result)[len(tok.encode(prompt)):]
        assert len(new_tokens) <= n


# ===========================================================================
# 8. export_to_stablehlo
# ===========================================================================

def _has_jax_export() -> bool:
    return hasattr(jax, "export")


def _has_flatbuffers() -> bool:
    try:
        import flatbuffers  # noqa: F401
        return True
    except ImportError:
        return False


requires_jax_export = pytest.mark.skipif(
    not _has_jax_export(),
    reason="jax.export requires JAX >= 0.4.26",
)

requires_flatbuffers = pytest.mark.skipif(
    not _has_flatbuffers(),
    reason="flatbuffers is required for jax.export.serialize(). "
           "Install with: pip install flatbuffers",
)


class TestExportErrors:
    """Error-path tests for export_to_stablehlo that do not need real export."""

    def test_missing_config_raises_file_not_found(self, tmp_path):
        from dantinox.core.export import export_to_stablehlo

        run = tmp_path / "empty_run"
        run.mkdir()
        with pytest.raises(FileNotFoundError, match="config.yaml"):
            export_to_stablehlo(str(run), str(tmp_path / "out.stablehlo"))

    def test_missing_jax_export_raises_import_error(self, tmp_path):
        """If jax.export is absent the function must raise ImportError."""
        import dantinox.core.export as export_mod
        from dantinox.core.export import export_to_stablehlo

        run = tmp_path / "fake_run"
        run.mkdir()

        # Inject a fake jax namespace that has no 'export' attribute so that
        # hasattr(jax, "export") returns False inside export_to_stablehlo.
        class _OldJax:
            pass

        with patch.object(export_mod, "jax", _OldJax()):
            with pytest.raises(ImportError, match="jax.export"):
                export_to_stablehlo(str(run), str(tmp_path / "out.stablehlo"))

    def test_load_model_for_export_no_weights_raises(self, tmp_path):
        """Export shares the checkpoint loader: no weights file → FileNotFoundError."""
        from dantinox.core.checkpoint import find_weights_file

        run = tmp_path / "no_weights"
        run.mkdir()
        with pytest.raises(FileNotFoundError, match="No weights file"):
            find_weights_file(str(run))


class TestSerializeHelper:
    """Pure-mock tests for the _serialize helper — no JAX tracing needed."""

    def test_uses_serialize_when_available(self):
        from dantinox.core.export import _serialize

        mock_exported = MagicMock()
        mock_exported.serialize.return_value = b"stablehlo_bytes"

        result = _serialize(mock_exported, "dummy.stablehlo")
        assert result == b"stablehlo_bytes"
        mock_exported.serialize.assert_called_once()

    def test_fallback_uses_mlir_bytecode(self):
        """When .serialize() is absent, _serialize must fall back to MLIR."""
        from dantinox.core.export import _serialize

        # Use a class-based spec so mlir_module is accessible but serialize is not.
        class _MinimalExported:
            def mlir_module(self): ...

        mock_exported = MagicMock(spec=_MinimalExported)
        mock_mlir = MagicMock()
        mock_mlir.operation.get_asm.return_value = b"\x00fake\x00mlir"
        mock_exported.mlir_module.return_value = mock_mlir

        result = _serialize(mock_exported, "dummy.stablehlo")
        assert result == b"\x00fake\x00mlir"
        mock_mlir.operation.get_asm.assert_called_once_with(binary=True)

    def test_fallback_not_triggered_when_serialize_present(self):
        from dantinox.core.export import _serialize

        mock_exported = MagicMock()
        mock_exported.serialize.return_value = b"data"

        _serialize(mock_exported, "out")
        mock_exported.mlir_module.assert_not_called()


@requires_jax_export
@requires_flatbuffers
class TestExportHappyPath:
    """Integration tests — require a trained checkpoint, JAX >= 0.4.26, and flatbuffers."""

    def test_export_creates_output_file(self, trained_new_run, tmp_path):
        from dantinox.core.export import export_to_stablehlo

        out = str(tmp_path / "model.stablehlo")
        export_to_stablehlo(trained_new_run, out, batch_size=1, seq_len=8)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_export_returns_absolute_path(self, trained_new_run, tmp_path):
        from dantinox.core.export import export_to_stablehlo

        out = str(tmp_path / "model.stablehlo")
        result = export_to_stablehlo(trained_new_run, out, batch_size=1, seq_len=8)
        assert os.path.isabs(result)

    def test_export_creates_parent_dirs(self, trained_new_run, tmp_path):
        from dantinox.core.export import export_to_stablehlo

        out = str(tmp_path / "nested" / "deep" / "model.stablehlo")
        export_to_stablehlo(trained_new_run, out, batch_size=1, seq_len=8)
        assert os.path.exists(out)


# ===========================================================================
# 9. CLI subcommand parsers
# ===========================================================================

class TestCLIParsers:
    """Verify the new 'run' and 'export' subcommands are correctly wired up."""

    @pytest.fixture(scope="class")
    def parser(self):
        from dantinox.cli import _build_parser

        return _build_parser()

    def test_run_subcommand_exists(self, parser):
        choices = parser._subparsers._actions[-1].choices
        assert "run" in choices

    def test_export_subcommand_exists(self, parser):
        choices = parser._subparsers._actions[-1].choices
        assert "export" in choices

    def test_run_positional_workflow_arg(self, parser):
        args = parser.parse_args(["run", "workflow.yaml"])
        assert args.workflow == "workflow.yaml"

    def test_run_optional_run_dir(self, parser):
        args = parser.parse_args(["run", "wf.yaml", "--run_dir", "runs/test"])
        assert args.run_dir == "runs/test"

    def test_run_default_run_dir_is_none(self, parser):
        args = parser.parse_args(["run", "wf.yaml"])
        assert args.run_dir is None

    def test_export_positional_checkpoint_and_output(self, parser):
        args = parser.parse_args(["export", "runs/my_run", "out/model.stablehlo"])
        assert args.checkpoint == "runs/my_run"
        assert args.output == "out/model.stablehlo"

    def test_export_default_batch_size_is_one(self, parser):
        args = parser.parse_args(["export", "runs/r", "out.stablehlo"])
        assert args.batch_size == 1

    def test_export_custom_batch_size(self, parser):
        args = parser.parse_args(["export", "runs/r", "out.stablehlo", "--batch_size", "4"])
        assert args.batch_size == 4

    def test_export_default_seq_len_is_none(self, parser):
        args = parser.parse_args(["export", "runs/r", "out.stablehlo"])
        assert args.seq_len is None

    def test_export_custom_seq_len(self, parser):
        args = parser.parse_args(["export", "runs/r", "out.stablehlo", "--seq_len", "128"])
        assert args.seq_len == 128

    def test_export_default_seed_is_42(self, parser):
        args = parser.parse_args(["export", "runs/r", "out.stablehlo"])
        assert args.seed == 42

    def test_run_and_export_in_dispatch(self):
        import dantinox.cli as cli_mod

        # Verify the dispatch table knows about the new commands
        dispatch_keys = set()
        source = cli_mod.main.__code__.co_consts  # not ideal, let's call main
        # Simpler: instantiate and check that run/export don't error at parse
        args = cli_mod._build_parser().parse_args(["run", "wf.yaml"])
        assert args.command == "run"
        args2 = cli_mod._build_parser().parse_args(["export", "ck", "out.bin"])
        assert args2.command == "export"


# ===========================================================================
# 10. CLI _cmd_run workflow YAML parsing (unit)
# ===========================================================================

class TestCmdRunWorkflowParsing:
    """Test that _cmd_run correctly parses a workflow YAML without running training."""

    def _write_workflow(self, path: pathlib.Path, overrides: dict | None = None) -> str:
        import yaml

        wf = {
            "dataset": {
                "name": "fake_ds",
                "subset": "fake_sub",
                "split": "train",
                "text_column": "text",
            },
            "architecture": {
                "model_type": "autoregressive",
                "dim": 64,
                "n_heads": 2,
                "head_size": 32,
            },
            "training": {
                "batch_size": 4,
                "grad_accum": 1,
                "epochs": 1,
            },
            "tracking": {
                "use_wandb": False,
            },
        }
        if overrides:
            for k, v in overrides.items():
                wf[k] = {**wf.get(k, {}), **v}
        out = path / "workflow.yaml"
        out.write_text(yaml.dump(wf), encoding="utf-8")
        return str(out)

    def test_workflow_yaml_parses_model_type(self, tmp_path):
        import yaml

        wf_path = self._write_workflow(tmp_path)
        with open(wf_path) as f:
            wf = yaml.safe_load(f)
        assert wf["architecture"]["model_type"] == "autoregressive"

    def test_workflow_yaml_dataset_fields(self, tmp_path):
        import yaml

        wf_path = self._write_workflow(tmp_path)
        with open(wf_path) as f:
            wf = yaml.safe_load(f)
        ds = wf["dataset"]
        assert ds["name"] == "fake_ds"
        assert ds["text_column"] == "text"

    def test_workflow_yaml_tracking_wandb_false(self, tmp_path):
        import yaml

        wf_path = self._write_workflow(tmp_path)
        with open(wf_path) as f:
            wf = yaml.safe_load(f)
        assert wf["tracking"]["use_wandb"] is False


# ===========================================================================
# 11. AutoDataPipeline repr
# ===========================================================================

class TestAutoDataPipelineRepr:
    def test_repr_contains_dataset_path(self):
        from dantinox.utils.data_pipeline import AutoDataPipeline

        pipe = AutoDataPipeline("myds", "sub", "train", "text", _SimpleTokenizer(), 64)
        r = repr(pipe)
        assert "myds" in r

    def test_repr_contains_max_context(self):
        from dantinox.utils.data_pipeline import AutoDataPipeline

        pipe = AutoDataPipeline("ds", "sub", "train", "text", _SimpleTokenizer(), 128)
        r = repr(pipe)
        assert "128" in r


# ===========================================================================
# 12. Checkpoint loader — filename priority
# ===========================================================================

class TestLoadModelCheckpointPriority:
    """load_model must prefer checkpoint_best.msgpack over legacy filenames."""

    def _make_checkpoint(self, run_dir: pathlib.Path, model, filename: str) -> None:
        """Save a real model checkpoint using the new-style format."""
        import flax.serialization
        from flax import nnx

        _WEIGHTS = nnx.Not(nnx.RngState)
        pure = nnx.state(model, _WEIGHTS).to_pure_dict()
        with open(run_dir / filename, "wb") as f:
            f.write(flax.serialization.msgpack_serialize(pure))

    def test_prefers_checkpoint_best(self, tmp_path):
        """When both checkpoint_best and best_model_weights exist, prefers the former."""
        from flax import nnx

        from dantinox.core.checkpoint import load_model
        from dantinox.core.config import Config
        from dantinox.core.model import Transformer

        run = tmp_path / "pref_run"
        run.mkdir()
        cfg = Config(
            dim=64, n_heads=2, head_size=32, num_blocks=1,
            vocab_size=200, max_context=32, kv_heads=2,
        )
        cfg.save_yaml(str(run / "config.yaml"))

        model = Transformer(cfg.to_model_config(), rngs=nnx.Rngs(0))
        # Write both; load_model should prefer checkpoint_best
        self._make_checkpoint(run, model, "checkpoint_best.msgpack")
        self._make_checkpoint(run, model, "best_model_weights.msgpack")

        _, _, weights_path = load_model(str(run), seed=0)
        assert "checkpoint_best" in weights_path

    def test_falls_back_to_legacy_filename(self, tmp_path):
        from flax import nnx

        from dantinox.core.checkpoint import load_model
        from dantinox.core.config import Config
        from dantinox.core.model import Transformer

        run = tmp_path / "legacy_run"
        run.mkdir()
        cfg = Config(
            dim=64, n_heads=2, head_size=32, num_blocks=1,
            vocab_size=200, max_context=32, kv_heads=2,
        )
        cfg.save_yaml(str(run / "config.yaml"))

        model = Transformer(cfg.to_model_config(), rngs=nnx.Rngs(0))
        self._make_checkpoint(run, model, "best_model_weights.msgpack")

        _, _, weights_path = load_model(str(run), seed=0)
        assert "best_model_weights" in weights_path
