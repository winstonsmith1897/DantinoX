"""
HuggingFace Hub integration for DantinoX.

Push a trained checkpoint to the Hub and pull it back on any machine.

Examples
--------
CLI:
    dantinox push --run_dir runs/my_run --repo my-org/dantinox-dante
    dantinox pull --repo my-org/dantinox-dante --local_dir runs/pulled

Python API:
    from dantinox.hub import push, pull

    url = push("runs/my_run", "my-org/dantinox-dante", private=True)
    run_dir = pull("my-org/dantinox-dante")
    gen = Generator(run_dir)
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

_UPLOAD_IGNORE = ["*.log", "__pycache__/", "*.pyc"]


def resolve_checkpoint(
    path_or_repo: str,
    *,
    token: str | None = None,
    revision: str | None = None,
) -> str:
    """Return a local directory path for *path_or_repo*.

    If *path_or_repo* is an existing local directory it is returned unchanged.
    Otherwise it is treated as a HuggingFace Hub repo ID (e.g.
    ``"my-org/dantinox-dante"``) and the checkpoint is downloaded via
    :func:`pull` before returning the local cache path.

    Parameters
    ----------
    path_or_repo:
        Local run directory **or** HuggingFace Hub repo ID.
    token:
        HuggingFace access token for private repositories.
    revision:
        Branch, tag, or commit SHA to download.

    Returns
    -------
    str
        Absolute path to a local directory suitable for passing to
        ``Generator()``, ``Transformer.from_pretrained()``, etc.
    """
    if os.path.isdir(path_or_repo):
        return path_or_repo
    return pull(path_or_repo, token=token, revision=revision)


def push(
    run_dir: str,
    repo_id: str,
    *,
    private: bool = False,
    token: str | None = None,
    commit_message: str | None = None,
) -> str:
    """
    Upload a run directory to a HuggingFace Hub model repository.

    Creates the repository if it does not exist.  Only the core checkpoint
    files are uploaded (``config.yaml``, ``tokenizer.json``,
    ``model_weights.msgpack``, ``best_model_weights.msgpack``,
    ``model_summary.json``).  Log files are excluded.

    Parameters
    ----------
    run_dir : str
        Local path to a DantinoX run directory.
    repo_id : str
        Hub repository in the form ``"owner/repo-name"``.
    private : bool
        Create the repository as private (default False).
    token : str, optional
        HuggingFace access token.  Falls back to the ``HF_TOKEN``
        environment variable or the cached login token.
    commit_message : str, optional
        Commit message for the upload (auto-generated if omitted).

    Returns
    -------
    str
        URL of the Hub repository after the upload.

    Raises
    ------
    ImportError
        If ``huggingface_hub`` is not installed.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for Hub integration: "
            "pip install huggingface-hub"
        ) from exc

    import os
    msg = commit_message or f"Upload DantinoX checkpoint from {os.path.basename(run_dir)}"

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)

    url = api.upload_folder(
        folder_path=run_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message=msg,
        ignore_patterns=_UPLOAD_IGNORE,
    )

    log.info("Pushed %s → %s", run_dir, url)
    return str(url)


def pull(
    repo_id: str,
    *,
    local_dir: str | None = None,
    token: str | None = None,
    revision: str | None = None,
) -> str:
    """
    Download a DantinoX checkpoint from HuggingFace Hub.

    Parameters
    ----------
    repo_id : str
        Hub repository in the form ``"owner/repo-name"``.
    local_dir : str, optional
        Where to store the downloaded files.  Defaults to the HuggingFace
        cache directory (``~/.cache/huggingface/hub/...``).
    token : str, optional
        HuggingFace access token for private repositories.
    revision : str, optional
        Git revision (branch, tag, or commit SHA) to download.

    Returns
    -------
    str
        Path to the local directory containing the checkpoint.  Pass this
        directly to ``Generator(run_dir)``.

    Raises
    ------
    ImportError
        If ``huggingface_hub`` is not installed.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for Hub integration: "
            "pip install huggingface-hub"
        ) from exc

    run_dir: str = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=local_dir,
        token=token,
        revision=revision,
    )

    log.info("Pulled %s → %s", repo_id, run_dir)
    return run_dir
