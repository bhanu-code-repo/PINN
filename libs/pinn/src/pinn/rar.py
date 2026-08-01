"""Residual-based Adaptive Refinement (RAR) for PINNs.

RAR improves convergence in regions where the PDE residual is large
(e.g. shocks, boundary layers) by iteratively adding collocation points
where the model struggles most.

Algorithm (Lu et al. 2021, "DeepXDE"):
    1. Train for E epochs with the current point set.
    2. Evaluate residuals on a dense candidate set.
    3. Select the K candidates with the highest residual magnitude.
    4. Append them to the physics collocation set.
    5. Rebuild loss closures with the enlarged point set.
    6. Repeat for P phases.

Public API:

- :func:`select_rar_points` — one-shot point selection from residuals.
- :func:`adaptive_train` — full multi-phase RAR training loop.

Copyright 2026 Bhanu Thakur. All rights reserved.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
from loguru import logger

from .trainer.trainer import PINNTrainer


def select_rar_points(
    model: nn.Module,
    candidates: torch.Tensor,
    residual_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    n_select: int,
) -> torch.Tensor:
    """Select collocation points where the PDE residual is largest.

    Args:
        model: Trained (or partially trained) PINN model.
        candidates: Dense candidate point set, shape ``(N, d)``.
        residual_fn: ``fn(model, points) -> residuals`` where residuals has
            shape ``(N,)`` or ``(N, c)`` (per-point residual magnitudes).
            Multi-component residuals are L2-normed across components.
        n_select: Number of high-residual points to return.

    Returns:
        Tensor of shape ``(n_select, d)`` — the selected points, detached.
    """
    model.eval()
    # No torch.no_grad() here — residual_fn may need autograd
    # (e.g. computing PDE residuals via derivatives of model output).
    residuals = residual_fn(model, candidates)

    # Handle multi-component residuals: norm across last dim
    if residuals.ndim > 1:
        residuals = torch.norm(residuals, dim=-1)

    # Select top-k by residual magnitude
    k = min(n_select, len(candidates))
    _, indices = torch.topk(residuals.abs(), k)
    selected = candidates[indices].detach().clone()

    logger.debug(
        "RAR: selected {} points | residual range [{:.4e}, {:.4e}]",
        k,
        residuals.abs().min().item(),
        residuals.abs().max().item(),
    )
    return selected


def adaptive_train(
    trainer: PINNTrainer,
    build_losses: Callable[[torch.Tensor], dict[str, Callable[[nn.Module], torch.Tensor]]],
    residual_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    candidate_sampler: Callable[[int], torch.Tensor],
    initial_points: torch.Tensor,
    optimizer_fn: Callable[[nn.Module], torch.optim.Optimizer],
    n_phases: int = 5,
    epochs_per_phase: int = 5000,
    n_candidates: int = 10_000,
    n_select: int = 500,
    weights: dict[str, float] | None = None,
    **train_kwargs,
) -> dict:
    """Run multi-phase RAR training.

    Each phase trains the model, then selects high-residual points from a
    fresh candidate set and appends them to the physics collocation points.
    Loss closures are rebuilt with the enlarged point set before the next phase.

    Args:
        trainer: A ``PINNTrainer`` instance (model already set).
        build_losses: ``fn(physics_points) -> loss_dict``. Called at the start
            of each phase to rebuild loss closures with the current point set.
            The returned dict must include all loss terms (IC, BC, physics, etc.).
        residual_fn: ``fn(model, points) -> per_point_residuals``. Used to
            score candidate points. Should return shape ``(N,)`` or ``(N, c)``.
            Must support autograd (points may need ``requires_grad``).
        candidate_sampler: ``fn(n) -> points`` tensor of shape ``(n, d)``.
            Generates random candidate points in the physics domain.
        initial_points: Starting physics collocation points, shape ``(M, d)``.
        optimizer_fn: ``fn(model) -> optimizer``. Called once per phase so the
            optimizer state is fresh (avoids stale momentum from old point sets).
        n_phases: Number of RAR phases.
        epochs_per_phase: Training epochs per phase.
        n_candidates: Number of candidate points sampled for residual evaluation.
        n_select: Points added per phase (top-K by residual).
        weights: Loss term weights passed to ``trainer.train()``.
        **train_kwargs: Extra kwargs forwarded to ``trainer.train()``
            (e.g. ``log_every``, ``grad_clip``, ``save_best``).

    Returns:
        Dict with ``"points"`` (final collocation set), ``"points_per_phase"``
        (list of point counts), and ``"loss_history"`` (full history across
        all phases).
    """
    physics_points = initial_points.clone()
    points_per_phase = []

    for phase in range(n_phases):
        n_pts = len(physics_points)
        points_per_phase.append(n_pts)
        logger.info(
            "RAR phase {}/{} — {} collocation points, {} epochs",
            phase + 1, n_phases, n_pts, epochs_per_phase,
        )

        # Rebuild losses with current point set
        loss_functions = build_losses(physics_points)
        optimizer = optimizer_fn(trainer.model)

        # Train this phase
        trainer.train(
            n_epochs=epochs_per_phase,
            optimizer=optimizer,
            loss_functions=loss_functions,
            weights=weights,
            **train_kwargs,
        )

        # Skip refinement after the last phase
        if phase < n_phases - 1:
            candidates = candidate_sampler(n_candidates)
            new_points = select_rar_points(
                trainer.model, candidates, residual_fn, n_select,
            )
            physics_points = torch.cat([physics_points, new_points], dim=0)
            logger.info(
                "RAR: added {} points → {} total",
                len(new_points), len(physics_points),
            )

        trainer.model.train()  # Ensure model is back in train mode

    return {
        "points": physics_points,
        "points_per_phase": points_per_phase,
        "loss_history": trainer.loss_history,
    }
