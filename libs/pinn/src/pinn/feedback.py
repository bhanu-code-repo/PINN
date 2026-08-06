"""Training feedback: health monitoring, adaptive loss weighting, quality evaluation.

Inspired by the Feedback Agent concept from Lang-PINN (He et al. 2025),
adapted to work as pure callbacks within the existing PINNTrainer.

Three components:

- :class:`TrainingHealthMonitor` — epoch callback that tracks loss smoothness,
  gradient health, and convergence. Useful for diagnostics and post-training
  quality evaluation.
- :class:`AdaptiveLossWeighter` — epoch callback that dynamically rebalances
  loss term weights when one term dominates, preventing gradient starvation.
- :func:`evaluate_quality` — post-training quality scoring across effectiveness,
  efficiency, and robustness dimensions.

Copyright 2026 Bhanu Thakur. All rights reserved.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch.nn as nn
from loguru import logger


class TrainingHealthMonitor:
    """Epoch callback that tracks training health metrics.

    Monitors three aspects of training quality in real time:

    1. **Loss smoothness** — how stable the loss trajectory is (low variance
       in epoch-to-epoch changes = smooth).
    2. **Gradient health** — whether gradient norms stay in a healthy range
       (not vanishing, not exploding).
    3. **Convergence** — whether the loss has reached a target threshold.

    Use as a callback::

        monitor = TrainingHealthMonitor(model)
        trainer.train(..., callbacks=[monitor])
        print(monitor.report())

    Args:
        model: The model being trained (for gradient norm computation).
        window: Rolling window size for smoothness/gradient statistics.
        grad_eps: Lower bound for healthy gradient norm.
        grad_kappa: Upper bound for healthy gradient norm.
        log_every: Log health summary every N epochs (0 = no logging).
    """

    def __init__(
        self,
        model: nn.Module,
        window: int = 100,
        grad_eps: float = 1e-6,
        grad_kappa: float = 1e2,
        log_every: int = 0,
    ):
        self.model = model
        self.window = window
        self.grad_eps = grad_eps
        self.grad_kappa = grad_kappa
        self.log_every = log_every

        self.total_losses: list[float] = []
        self.grad_norms: list[float] = []
        self.per_term_losses: dict[str, list[float]] = defaultdict(list)

    def __call__(self, epoch: int, epoch_losses: dict[str, float]) -> None:
        self.total_losses.append(epoch_losses["total"])
        for name, val in epoch_losses.items():
            self.per_term_losses[name].append(val)

        # Compute gradient norm (gradients are available at callback time)
        grad_norm = self._grad_norm()
        self.grad_norms.append(grad_norm)

        if self.log_every > 0 and epoch > 0 and epoch % self.log_every == 0:
            logger.info(
                "Health [epoch {}] | smoothness={:.4f} | grad_health={} | "
                "grad_norm={:.4e}",
                epoch, self.loss_smoothness, self.gradient_healthy, grad_norm,
            )

    def _grad_norm(self) -> float:
        total = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total += p.grad.norm().item() ** 2
        return total**0.5

    @property
    def loss_smoothness(self) -> float:
        """Loss smoothness in [0, 1]. Higher = smoother training.

        Computed as ``1 - Std(delta_L) / Mean(L)`` over the recent window.
        """
        losses = self.total_losses[-self.window:]
        if len(losses) < 2:
            return 0.5
        deltas = np.diff(losses)
        std_delta = float(np.std(deltas))
        mean_loss = float(np.mean(losses))
        if mean_loss < 1e-12:
            return 1.0
        smoothness = 1.0 - std_delta / mean_loss
        return float(np.clip(smoothness, 0.0, 1.0))

    @property
    def gradient_healthy(self) -> bool:
        """Whether recent gradient norms are in the healthy range [eps, kappa]."""
        if not self.grad_norms:
            return False
        recent = self.grad_norms[-self.window:]
        avg = float(np.mean(recent))
        return self.grad_eps <= avg <= self.grad_kappa

    @property
    def convergence_epoch(self) -> int | None:
        """First epoch where total loss dropped below 1e-4, or None."""
        for i, loss in enumerate(self.total_losses):
            if loss < 1e-4:
                return i
        return None

    def report(self) -> dict[str, float | bool | int | None]:
        """Return a summary dict of all health metrics."""
        return {
            "loss_smoothness": self.loss_smoothness,
            "gradient_healthy": self.gradient_healthy,
            "mean_grad_norm": float(np.mean(self.grad_norms)) if self.grad_norms else 0.0,
            "convergence_epoch": self.convergence_epoch,
            "final_loss": self.total_losses[-1] if self.total_losses else None,
            "epochs_tracked": len(self.total_losses),
        }


class AdaptiveLossWeighter:
    """Epoch callback that dynamically rebalances loss term weights.

    When one loss term dominates (its weighted contribution is much larger
    than others), gradient updates are almost entirely driven by that term
    and the other terms stagnate. This callback detects imbalance and
    adjusts weights to keep all terms contributing.

    The ``weights`` dict is mutated in-place — since the trainer references
    the same dict object, changes take effect on the next epoch automatically.

    Algorithm (each ``rebalance_every`` epochs):
        1. Compute the mean loss for each term over the recent window.
        2. If ``max_term / min_term > ratio_threshold``, rebalance:
           - Scale each weight inversely proportional to its mean loss.
           - Normalize so total weight magnitude is preserved.

    Usage::

        weights = {"ic": 1.0, "bc": 1.0, "physics": 1.0}
        weighter = AdaptiveLossWeighter(weights)
        trainer.train(..., weights=weights, callbacks=[weighter])

    Args:
        weights: The mutable weights dict (same object passed to ``trainer.train``).
        rebalance_every: Check and rebalance every N epochs.
        ratio_threshold: Trigger rebalancing when max/min loss ratio exceeds this.
        window: Number of recent epochs to average over for ratio computation.
        max_weight: Cap individual weights to prevent extreme values.
        min_weight: Floor individual weights to prevent zeroing out.
    """

    def __init__(
        self,
        weights: dict[str, float],
        rebalance_every: int = 500,
        ratio_threshold: float = 5.0,
        window: int = 100,
        max_weight: float = 100.0,
        min_weight: float = 0.01,
    ):
        self.weights = weights
        self.rebalance_every = rebalance_every
        self.ratio_threshold = ratio_threshold
        self.window = window
        self.max_weight = max_weight
        self.min_weight = min_weight

        self.history: dict[str, list[float]] = defaultdict(list)
        self.rebalance_count = 0

    def __call__(self, epoch: int, epoch_losses: dict[str, float]) -> None:
        # Track per-term losses (exclude "total")
        for name, val in epoch_losses.items():
            if name != "total":
                self.history[name].append(val)

        if epoch > 0 and epoch % self.rebalance_every == 0:
            self._maybe_rebalance(epoch)

    def _maybe_rebalance(self, epoch: int) -> None:
        # Compute recent mean for each tracked term
        term_means = {}
        for name, vals in self.history.items():
            recent = vals[-self.window:]
            mean = float(np.mean(recent)) if recent else 0.0
            term_means[name] = max(mean, 1e-12)  # avoid division by zero

        if not term_means:
            return

        max_loss = max(term_means.values())
        min_loss = min(term_means.values())
        ratio = max_loss / min_loss

        if ratio <= self.ratio_threshold:
            return  # Losses are balanced, no action needed

        # Rebalance: weight inversely proportional to mean loss
        # This boosts underfitting terms and dampens dominating ones
        total_weight = sum(self.weights.get(name, 1.0) for name in term_means)
        new_weights = {}
        for name, mean_loss in term_means.items():
            raw = 1.0 / mean_loss
            new_weights[name] = raw

        # Normalize to preserve total weight magnitude
        raw_sum = sum(new_weights.values())
        scale = total_weight / raw_sum if raw_sum > 0 else 1.0

        for name, raw in new_weights.items():
            clamped = float(np.clip(raw * scale, self.min_weight, self.max_weight))
            self.weights[name] = clamped

        self.rebalance_count += 1
        logger.info(
            "Adaptive weights [epoch {}]: rebalanced (ratio {:.1f}x) → {}",
            epoch, ratio,
            {k: f"{v:.3f}" for k, v in self.weights.items() if k in term_means},
        )


def evaluate_quality(
    loss_history: list[dict[str, float]],
    convergence_threshold: float = 1e-4,
) -> dict[str, float]:
    """Post-training quality evaluation across multiple dimensions.

    Scores a completed training run on effectiveness, efficiency, and
    robustness — the three quality dimensions from the Lang-PINN framework.

    Args:
        loss_history: The ``trainer.loss_history`` list of per-epoch loss dicts.
        convergence_threshold: Loss value that counts as "converged".

    Returns:
        Dict with individual dimension scores (0-1) and an overall
        ``quality_score`` (weighted combination).
    """
    if not loss_history:
        return {
            "effectiveness": 0.0,
            "efficiency": 0.0,
            "robustness": 0.0,
            "quality_score": 0.0,
        }

    total_losses = [epoch["total"] for epoch in loss_history]
    n_epochs = len(total_losses)

    # 1. Effectiveness: how low did the final loss get?
    # Normalized via log scale: lower loss → higher score
    final_loss = total_losses[-1]
    effectiveness = 1.0 / (1.0 + np.log1p(final_loss))

    # 2. Efficiency: how fast did it converge?
    conv_epoch = n_epochs  # default: never converged
    for i, loss in enumerate(total_losses):
        if loss < convergence_threshold:
            conv_epoch = i
            break
    efficiency = 1.0 - (conv_epoch / n_epochs)

    # 3. Robustness: loss smoothness
    if n_epochs < 2:
        robustness = 0.5
    else:
        deltas = np.diff(total_losses)
        std_delta = float(np.std(deltas))
        mean_loss = float(np.mean(total_losses))
        smoothness = 1.0 - std_delta / max(mean_loss, 1e-12)
        robustness = float(np.clip(smoothness, 0.0, 1.0))

    # Overall: weighted combination (paper uses 0.4, 0.3, 0.3)
    quality_score = 0.4 * effectiveness + 0.3 * efficiency + 0.3 * robustness
    quality_score = float(np.clip(quality_score, 0.0, 1.0))

    return {
        "effectiveness": round(float(effectiveness), 4),
        "efficiency": round(float(efficiency), 4),
        "robustness": round(float(robustness), 4),
        "quality_score": round(quality_score, 4),
        "final_loss": final_loss,
        "convergence_epoch": conv_epoch if conv_epoch < n_epochs else None,
    }
