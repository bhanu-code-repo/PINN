"""Weights & Biases integration for PINNs.

Provides a trainer callback that logs per-epoch losses to W&B, plus
helpers for initialising and finishing runs. ``wandb`` is imported
lazily so the core library works without it installed.

Usage::

    from pinn import wandb_callback, wandb_init, wandb_finish

    run = wandb_init(project="pinn", config=config, name="burgers-rar")
    cb = wandb_callback()  # logs every epoch by default
    trainer.train(..., callbacks=[cb])
    wandb_finish(run_dir)  # saves artifacts and closes the run

Copyright 2026 Bhanu Thakur. All rights reserved.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from loguru import logger


def _import_wandb():
    """Lazy import with a clear error message."""
    try:
        import wandb
        return wandb
    except ImportError:
        raise ImportError(
            "wandb is required for W&B integration. "
            "Install it with: uv add wandb  or  pip install wandb"
        ) from None


def wandb_init(
    project: str = "pinn",
    config: dict | None = None,
    name: str | None = None,
    tags: list[str] | None = None,
    group: str | None = None,
    **kwargs,
):
    """Initialise a W&B run.

    Args:
        project: W&B project name.
        config: Hyperparameters / run config to log.
        name: Run name (defaults to W&B auto-generated name).
        tags: Optional tags for filtering runs.
        group: Optional group name (e.g. experiment name).
        **kwargs: Extra args forwarded to ``wandb.init()``.

    Returns:
        The ``wandb.Run`` object.
    """
    wandb = _import_wandb()
    run = wandb.init(
        project=project,
        config=config,
        name=name,
        tags=tags,
        group=group,
        **kwargs,
    )
    logger.info("W&B run initialised: {} ({})", run.name, run.url)
    return run


def wandb_callback(
    log_every: int = 1,
    prefix: str = "",
) -> Callable[[int, dict[str, float]], None]:
    """Create an epoch callback that logs losses to W&B.

    Args:
        log_every: Log every N epochs (1 = every epoch).
        prefix: Optional prefix for metric names (e.g. ``"train/"``).

    Returns:
        A callback compatible with ``PINNTrainer.train(callbacks=[...])``.
    """
    wandb = _import_wandb()

    def callback(epoch: int, epoch_losses: dict[str, float]) -> None:
        if epoch % log_every != 0:
            return
        metrics = {f"{prefix}{k}": v for k, v in epoch_losses.items()}
        metrics["epoch"] = epoch
        wandb.log(metrics, step=epoch)

    return callback


def wandb_finish(
    run_dir: str | Path | None = None,
    artifact_name: str | None = None,
) -> None:
    """Finish the current W&B run, optionally saving artifacts.

    Args:
        run_dir: If provided, saves ``checkpoint.pt``, ``metrics.json``,
            and all PNG plots as a W&B artifact.
        artifact_name: Name for the artifact (defaults to ``"run-artifacts"``).
    """
    wandb = _import_wandb()

    if run_dir is not None:
        run_dir = Path(run_dir)
        name = artifact_name or "run-artifacts"
        artifact = wandb.Artifact(name, type="model")

        for pattern in ["checkpoint.pt", "best_model.pt", "metrics.json", "*.png"]:
            for path in run_dir.glob(pattern):
                artifact.add_file(str(path), name=path.name)
                logger.debug("W&B artifact: added {}", path.name)

        wandb.log_artifact(artifact)
        logger.info("W&B artifact '{}' saved with run files", name)

    wandb.finish()
    logger.info("W&B run finished")
