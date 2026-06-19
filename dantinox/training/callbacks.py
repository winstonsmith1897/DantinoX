"""Training callback system for DantinoX.

Callbacks let you attach arbitrary side-effects to the training loop without
modifying the Trainer.  Subclass :class:`BaseCallback` and override only the
hooks you need.

Trainer integration
-------------------
Add ``callbacks`` to ``Trainer.__init__`` and wire up **three lines** inside
``Trainer.fit()`` (``dantinox/training/trainer.py``):

.. code-block:: python

    # ① just before the epoch for-loop (~line 386, after PRNGKey is made):
    for cb in self.callbacks: cb.on_train_begin(cfg)

    # ② inside the inner step loop, right after loss is appended (~line 411):
    for cb in self.callbacks: cb.on_step_end(step, {"train_loss": float(loss)}, epoch)

    # ③ at the very end of fit(), just before `return run_dir` (~line 444):
    for cb in self.callbacks: cb.on_train_end()

You also need to add one line to ``__init__``:

.. code-block:: python

    self.callbacks: list[BaseCallback] = list(callbacks or [])

Example::

    from dantinox.training.callbacks import WandbCallback
    from dantinox.training.trainer import Trainer

    cb      = WandbCallback(project="my-project", name="run-001")
    trainer = Trainer(paradigm, train_cfg, callbacks=[cb])
    trainer.fit("data/corpus.txt")
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


class BaseCallback:
    """Minimal callback interface with no-op default implementations.

    Override only the hooks you need in a subclass.
    """

    def on_train_begin(self, config: Any) -> None:
        """Called once before the first training step.

        Args:
            config: The active :class:`~dantinox.core.config.TrainingConfig`.
        """

    def on_step_end(
        self,
        step: int,
        metrics: dict[str, float],
        epoch: int,
    ) -> None:
        """Called after every gradient update step.

        Args:
            step:    Step index within the current epoch (0-based).
            metrics: Scalar metrics for this step, e.g.
                     ``{"train_loss": 0.42}``.
            epoch:   Current epoch index (1-based).
        """

    def on_train_end(self) -> None:
        """Called once after the final epoch or early-stopping trigger."""


class WandbCallback(BaseCallback):
    """Weights & Biases experiment-tracking callback.

    Initialises a W&B run at :meth:`on_train_begin`, logs per-step metrics
    at :meth:`on_step_end`, and finishes the run at :meth:`on_train_end`.

    Args:
        project: W&B project name (default ``"dantinox"``).
        name: Optional run name.  W&B auto-generates one when omitted.
        tags: Optional list of string tags attached to the run.
        **config_overrides: Additional key-value pairs merged into the W&B
            run config dict (useful for adding git SHA, hostname, etc.).

    Example::

        cb = WandbCallback(project="dantinox", name="ar-512", tags=["arxiv"])
        Trainer(paradigm, train_cfg, callbacks=[cb]).fit("data/corpus.txt")
    """

    def __init__(
        self,
        project: str = "dantinox",
        name: str | None = None,
        tags: list[str] | None = None,
        **config_overrides: Any,
    ) -> None:
        self._project = project
        self._name = name
        self._tags = list(tags or [])
        self._config_overrides = config_overrides
        self._run: Any = None

    # ── Hooks ─────────────────────────────────────────────────────────────────

    def on_train_begin(self, config: Any) -> None:
        """Initialise W&B and log the training configuration."""
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "wandb is required for WandbCallback. "
                "Install it with:  pip install wandb"
            ) from exc

        import dataclasses

        cfg_dict: dict[str, Any] = {}
        if dataclasses.is_dataclass(config):
            cfg_dict = dataclasses.asdict(config)
        elif hasattr(config, "to_dict"):
            cfg_dict = config.to_dict()
        cfg_dict.update(self._config_overrides)

        self._run = wandb.init(
            project=self._project,
            name=self._name,
            tags=self._tags,
            config=cfg_dict,
            reinit=True,
        )
        log.info(
            "WandbCallback: run initialised — project=%s  name=%s",
            self._project,
            self._run.name,
        )

    def on_step_end(
        self,
        step: int,
        metrics: dict[str, float],
        epoch: int,
    ) -> None:
        """Log *metrics* to W&B at the current step."""
        if self._run is None:
            return
        import wandb
        wandb.log({"epoch": epoch, **metrics}, step=step)

    def on_train_end(self) -> None:
        """Flush and close the W&B run."""
        if self._run is not None:
            import wandb
            wandb.finish()
            log.info("WandbCallback: run finished")
            self._run = None
