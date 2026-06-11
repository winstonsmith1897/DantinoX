"""
Integration tests for Trainer, Generator, and tokenizer persistence.

These tests use a small dummy corpus and a tiny model config so they run
quickly on CPU without a GPU.
"""

from __future__ import annotations

import json
import os
import pathlib

import pytest

from dantinox.core.config import Config
from dantinox.generator import Generator
from dantinox.trainer import Trainer
from dantinox.utils.tokenizer import BPETokenizer, CharTokenizer, load_tokenizer_from_file

DUMMY_CORPUS = pathlib.Path(__file__).parent / "dummy_data.txt"


@pytest.fixture(scope="module")
def train_config(tmp_path_factory) -> Config:
    tmp = tmp_path_factory.mktemp("corpus")
    corpus = tmp / "corpus.txt"
    corpus.write_text(DUMMY_CORPUS.read_text(encoding="utf-8")[:5000], encoding="utf-8")
    cfg = Config(
        dim=64,
        n_heads=2,
        head_size=32,
        num_blocks=1,
        vocab_size=128,
        max_context=32,
        kv_heads=1,
        gradient_checkpointing=False,
        dropout_rate=0.0,
        epochs=1,
        batch_size=4,
        grad_accum=1,
        eval_iters=1,
        warmup_steps=1,
        dataset_name=str(corpus),
        tokenizer_type="char",
        grad_clip=1.0,
        patience=0,
    )
    return cfg


@pytest.fixture(scope="module")
def trained_run(train_config, tmp_path_factory):
    run_dir = str(tmp_path_factory.mktemp("run"))
    trainer = Trainer(train_config)
    result = trainer.fit(run_dir=run_dir)
    return result


class TestTokenizerPersistence:
    def test_char_save_load_roundtrip(self, tmp_path):
        tok = CharTokenizer()
        tok.train_from_text("abcdef abcdef")
        path = str(tmp_path / "tok.json")
        tok.save(path)

        loaded = load_tokenizer_from_file(path)
        assert loaded.vocab_size == tok.vocab_size
        sample = "abc"
        assert loaded.encode(sample) == tok.encode(sample)
        assert loaded.decode(tok.encode(sample)) == sample

    def test_char_save_file_format(self, tmp_path):
        tok = CharTokenizer()
        tok.train_from_text("hello")
        path = str(tmp_path / "tok.json")
        tok.save(path)
        with open(path) as f:
            payload = json.load(f)
        assert payload["type"] == "char"
        assert "vocab" in payload

    @pytest.mark.slow
    def test_bpe_save_load_roundtrip(self, tmp_path):
        tok = BPETokenizer()
        tok.train_from_text("hello world hello world", vocab_size=50)
        path = str(tmp_path / "tok.json")
        tok.save(path)

        loaded = load_tokenizer_from_file(path)
        assert loaded.vocab_size == tok.vocab_size
        ids = tok.encode("hello")
        assert loaded.encode("hello") == ids


class TestTrainerFit:
    def test_creates_run_dir(self, trained_run):
        assert os.path.isdir(trained_run)

    def test_saves_config_yaml(self, trained_run):
        assert os.path.exists(os.path.join(trained_run, "config.yaml"))

    def test_saves_model_weights(self, trained_run):
        # The best checkpoint is always written; the periodic resume
        # checkpoint (model_weights.msgpack) only appears for runs longer
        # than config.checkpoint_every steps.
        assert os.path.exists(
            os.path.join(trained_run, "best_model_weights.msgpack"))

    def test_saves_tokenizer_json(self, trained_run):
        assert os.path.exists(os.path.join(trained_run, "tokenizer.json"))

    def test_removes_training_cursor_on_completion(self, trained_run):
        # A finished run must not look like an interrupted one.
        assert not os.path.exists(
            os.path.join(trained_run, "training_cursor.json"))

    def test_saves_model_summary(self, trained_run):
        summary_path = os.path.join(trained_run, "model_summary.json")
        assert os.path.exists(summary_path)
        with open(summary_path) as f:
            summary = json.load(f)
        assert "total_params_M" in summary

    def test_saves_best_weights(self, trained_run):
        assert os.path.exists(os.path.join(trained_run, "best_model_weights.msgpack"))

    def test_tokenizer_json_is_valid(self, trained_run):
        tok = load_tokenizer_from_file(os.path.join(trained_run, "tokenizer.json"))
        assert tok.vocab_size > 0
        ids = tok.encode("a")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)


class TestGeneratorLoads:
    def test_generator_loads_without_corpus(self, trained_run):
        gen = Generator(trained_run, seed=0)
        assert gen.config is not None
        assert gen.model is not None
        assert gen.tokenizer is not None

    def test_generator_produces_string(self, trained_run):
        gen = Generator(trained_run, seed=0)
        # Use a single character guaranteed to be in the char vocabulary.
        prompt = gen.tokenizer.decode([0])
        result = gen.generate(prompt, max_new_tokens=10)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generator_repr(self, trained_run):
        gen = Generator(trained_run, seed=0)
        assert "Generator" in repr(gen)
        assert trained_run in repr(gen)


class TestCheckpointResume:
    def test_resume_continues_from_saved_step(self, train_config, tmp_path_factory):
        run_dir = str(tmp_path_factory.mktemp("resume_run"))
        trainer = Trainer(train_config)
        trainer.fit(run_dir=run_dir)

        # The cursor is removed once a run completes; resume from a finished
        # run should not raise and should return the same run_dir.
        result = trainer.fit(run_dir=run_dir, resume=True)
        assert result == run_dir


class TestEarlyStopping:
    def test_patience_zero_does_not_stop_early(self, train_config, tmp_path_factory):
        run_dir = str(tmp_path_factory.mktemp("es_run"))
        cfg = Config(
            dim=64,
            n_heads=2,
            head_size=32,
            num_blocks=1,
            vocab_size=128,
            max_context=32,
            kv_heads=1,
            gradient_checkpointing=False,
            dropout_rate=0.0,
            epochs=1,
            batch_size=4,
            grad_accum=1,
            eval_iters=1,
            warmup_steps=1,
            dataset_name=train_config.dataset_name,
            tokenizer_type="char",
            patience=0,
        )
        trainer = Trainer(cfg)
        result = trainer.fit(run_dir=run_dir)
        assert os.path.isdir(result)

    def test_patience_one_saves_best_weights(self, train_config, tmp_path_factory):
        run_dir = str(tmp_path_factory.mktemp("es_patience_run"))
        cfg = Config(
            dim=64,
            n_heads=2,
            head_size=32,
            num_blocks=1,
            vocab_size=128,
            max_context=32,
            kv_heads=1,
            gradient_checkpointing=False,
            dropout_rate=0.0,
            epochs=1,
            batch_size=4,
            grad_accum=1,
            eval_iters=1,
            warmup_steps=1,
            dataset_name=train_config.dataset_name,
            tokenizer_type="char",
            patience=1,
        )
        trainer = Trainer(cfg)
        trainer.fit(run_dir=run_dir)
        assert os.path.exists(os.path.join(run_dir, "best_model_weights.msgpack"))
