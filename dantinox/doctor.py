"""Environment health check — ``dx.doctor()`` / ``dantinox doctor``.

Diagnoses the version-skew and GPU-visibility problems that most often break
DantinoX on managed environments (Colab in particular):

* **jax / jaxlib / CUDA-plugin mismatch** — three packages that must move in
  lockstep; a partial upgrade leaves the GPU unusable with cryptic PJRT
  errors (``PJRT_FFI_UserData_Add_Args size mismatch`` and friends).
* **flax too old** — ``nnx.remat`` + ``Param[...]`` crashes on flax < 0.12
  (``tuple indices must be integers or slices, not ellipsis``).
* **optax too old** — ``optax.contrib.muon`` is an unstable contrib API;
  versions before 0.2.8 crash.
* **No GPU visible** — jax silently falls back to CPU.
"""

from __future__ import annotations

import importlib.metadata as _md
from typing import Any

_OK   = "\033[92m✓\033[0m"
_BAD  = "\033[91m✗\033[0m"
_WARN = "\033[93m⚠\033[0m"


def _ver(pkg: str) -> str | None:
    """Installed version of *pkg*, or ``None`` when absent."""
    try:
        return _md.version(pkg)
    except _md.PackageNotFoundError:
        return None


def _parse(v: str) -> tuple[int, ...]:
    """Best-effort numeric parse of a version string ('0.12.6' → (0, 12, 6))."""
    parts: list[int] = []
    for tok in v.split("."):
        digits = "".join(ch for ch in tok if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def doctor(verbose: bool = True) -> dict[str, Any]:
    """Check the environment for the known DantinoX failure modes.

    Prints a ✓/⚠/✗ report (unless ``verbose=False``) and returns a dict with
    the findings: ``versions`` (installed packages), ``problems`` (blocking),
    ``warnings`` (suspicious), ``gpu`` (devices seen by jax), ``ok`` (bool).

    Quick-start::

        import dantinox as dx
        dx.doctor()
    """
    problems: list[str] = []
    warnings_: list[str] = []
    lines: list[str] = []

    versions = {
        pkg: _ver(pkg)
        for pkg in ("dantinox", "jax", "jaxlib", "flax", "optax",
                    "jax-cuda12-plugin", "jax-cuda12-pjrt", "transformers", "datasets")
    }

    # ── jax / jaxlib / CUDA plugin alignment ─────────────────────────────────
    jax_v, jaxlib_v = versions["jax"], versions["jaxlib"]
    plugin_v = versions["jax-cuda12-plugin"] or versions["jax-cuda12-pjrt"]
    if jax_v and jaxlib_v and _parse(jax_v)[:2] != _parse(jaxlib_v)[:2]:
        problems.append(
            f"jax {jax_v} vs jaxlib {jaxlib_v} — versions out of sync; "
            f'fix: pip install -U "jax[cuda12]"'
        )
    if plugin_v and jaxlib_v and _parse(plugin_v)[:2] != _parse(jaxlib_v)[:2]:
        problems.append(
            f"jax-cuda12-plugin {plugin_v} vs jaxlib {jaxlib_v} — the CUDA "
            f"plugin must match jaxlib exactly (PJRT errors otherwise); "
            f'fix: pip install -U "jax[cuda12]"'
        )

    # ── flax (nnx.remat bug) ─────────────────────────────────────────────────
    flax_v = versions["flax"]
    if flax_v and _parse(flax_v) < (0, 12):
        problems.append(
            f"flax {flax_v} < 0.12 — nnx.remat/Param indexing crashes "
            f"(gradient_checkpointing, muon); fix: pip install -U 'flax>=0.12,<0.13'"
        )

    # ── optax (muon contrib API) ─────────────────────────────────────────────
    optax_v = versions["optax"]
    if optax_v and _parse(optax_v) < (0, 2, 8):
        warnings_.append(
            f"optax {optax_v} < 0.2.8 — optimizer='muon' may crash; "
            f"fix: pip install -U 'optax>=0.2.8'"
        )

    # ── optional extras ──────────────────────────────────────────────────────
    if versions["transformers"] is None:
        warnings_.append("transformers not installed — continuous/flow-matching "
                         "paradigm unavailable (pip install dantinox[elf])")
    if versions["datasets"] is None:
        warnings_.append("datasets not installed — HuggingFace corpora "
                         "unavailable (pip install dantinox[data])")

    # ── device visibility + smoke test ───────────────────────────────────────
    gpu_info: list[str] = []
    try:
        import jax as _jax
        devs = _jax.devices()
        gpu_info = [str(d) for d in devs]
        if not any("cuda" in s.lower() or "gpu" in s.lower() or "tpu" in s.lower()
                   for s in gpu_info):
            warnings_.append(
                "no GPU/TPU visible to jax — running on CPU "
                "(check CUDA_VISIBLE_DEVICES / runtime type on Colab)"
            )
        # tiny smoke test: catches PJRT/plugin breakage that import alone misses
        import jax.numpy as _jnp
        _ = (_jnp.ones((8, 8)) @ _jnp.ones((8, 8))).block_until_ready()
    except Exception as exc:  # noqa: BLE001 — any device failure is a finding
        problems.append(f"device smoke test FAILED: {type(exc).__name__}: {exc}")

    # ── report ───────────────────────────────────────────────────────────────
    if verbose:
        lines.append("DantinoX doctor")
        for pkg, v in versions.items():
            if v is not None:
                lines.append(f"  {pkg:20s} {v}")
        if gpu_info:
            lines.append(f"  {'devices':20s} {', '.join(gpu_info)}")
        for p in problems:
            lines.append(f"  {_BAD} {p}")
        for w in warnings_:
            lines.append(f"  {_WARN} {w}")
        if not problems and not warnings_:
            lines.append(f"  {_OK} environment looks healthy")
        elif not problems:
            lines.append(f"  {_OK} no blocking problems")
        print("\n".join(lines), flush=True)

    return {
        "versions": versions,
        "problems": problems,
        "warnings": warnings_,
        "gpu": gpu_info,
        "ok": not problems,
    }
