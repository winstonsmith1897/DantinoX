"""OOM guard utilities for stable JAX/GPU/TPU training.

Provides :func:`with_oom_guard`, a context manager that catches GPU/TPU
out-of-memory errors, halves ``cfg.batch_size``, doubles ``cfg.grad_accum``
to maintain the effective batch size, clears compiled JAX artefacts, and
raises :exc:`OOMMitigatedError` so the caller can rebuild the Trainer.

Typical usage in a CLI command::

    from dantinox.training.stability import OOMMitigatedError, with_oom_guard

    for attempt in range(5):
        try:
            with with_oom_guard(train_cfg):
                trainer = Trainer(paradigm, train_cfg, callbacks=callbacks)
                trainer.fit(run_dir=run_dir)
            break                           # clean exit
        except OOMMitigatedError as exc:
            log.warning("OOM attempt %d — %s", attempt + 1, exc)
            # train_cfg.batch_size and .grad_accum already updated in-place
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Generator

import jax

log = logging.getLogger(__name__)

# XlaRuntimeError was added in JAX 0.4.14; guard the import so the module
# works on older installs where only RuntimeError is available.
try:
    from jax.errors import XlaRuntimeError as _XlaRuntimeError
    _CATCHABLE: tuple[type[BaseException], ...] = (RuntimeError, _XlaRuntimeError)
except ImportError:
    _CATCHABLE = (RuntimeError,)

# Lower-cased substrings that identify GPU/TPU memory-exhaustion messages
# across JAX, XLA, CUDA, and TPU backends.
_OOM_SIGNATURES: frozenset[str] = frozenset({
    "out of memory",
    "resource_exhausted",
    "oom when allocating",
    "cuda out of memory",
    "cannot allocate memory",
    "allocation failed",
    "failed to allocate",
})


class OOMMitigatedError(RuntimeError):
    """Raised after :func:`with_oom_guard` successfully mitigates an OOM.

    The config passed to :func:`with_oom_guard` has already been mutated
    in-place before this exception is raised.  Callers should rebuild the
    Trainer with the same config object and resume training.

    Attributes:
        new_batch_size:  ``cfg.batch_size`` after the reduction.
        new_grad_accum:  ``cfg.grad_accum`` after the increase.
        original:        The underlying ``RuntimeError`` or
                         ``XlaRuntimeError``.
    """

    def __init__(
        self,
        new_batch_size: int,
        new_grad_accum: int,
        original: BaseException,
    ) -> None:
        self.new_batch_size = new_batch_size
        self.new_grad_accum = new_grad_accum
        self.original = original
        super().__init__(
            f"OOM mitigated — batch_size reduced to {new_batch_size}, "
            f"grad_accum increased to {new_grad_accum}. "
            f"Rebuild the Trainer with the updated config and resume. "
            f"(Original error: {original})"
        )


def _is_oom(exc: BaseException) -> bool:
    """Return ``True`` when *exc* looks like a memory-exhaustion error."""
    msg = str(exc).lower()
    return any(sig in msg for sig in _OOM_SIGNATURES)


@contextlib.contextmanager
def with_oom_guard(cfg: Any) -> Generator[None, None, None]:
    """Context manager that catches OOM errors and adjusts training config.

    On catching a :class:`RuntimeError` or
    :class:`jax.errors.XlaRuntimeError` whose message matches a known
    out-of-memory pattern, the manager:

    1. Sets ``cfg.batch_size = max(1, cfg.batch_size // 2)``.
    2. Sets ``cfg.grad_accum *= 2`` (preserves effective batch size).
    3. Calls :func:`jax.clear_caches` to free compiled XLA kernels.
    4. Raises :exc:`OOMMitigatedError`.

    Any non-OOM exception propagates unchanged.

    Args:
        cfg: A config object with ``batch_size: int`` and ``grad_accum: int``
             attributes — either
             :class:`~dantinox.core.config.TrainingConfig` or the legacy
             :class:`~dantinox.core.config.Config`.

    Raises:
        OOMMitigatedError: When OOM is detected and the config is patched.
        Exception:         Any non-OOM exception passes through unmodified.

    Example::

        while True:
            try:
                with with_oom_guard(train_cfg):
                    Trainer(paradigm, train_cfg).fit("data.txt")
                break           # success
            except OOMMitigatedError:
                pass            # config already patched; retry
    """
    try:
        yield
    except _CATCHABLE as exc:
        if not _is_oom(exc):
            raise

        old_bs = cfg.batch_size
        old_ga = cfg.grad_accum
        cfg.batch_size = max(1, cfg.batch_size // 2)
        cfg.grad_accum = cfg.grad_accum * 2

        log.warning(
            "OOM detected — batch_size %d → %d, grad_accum %d → %d. "
            "Clearing JAX caches and raising OOMMitigatedError.",
            old_bs, cfg.batch_size, old_ga, cfg.grad_accum,
        )
        jax.clear_caches()
        raise OOMMitigatedError(cfg.batch_size, cfg.grad_accum, exc) from exc
