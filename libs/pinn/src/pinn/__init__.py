"""pinn — core library for building and training Physics-Informed Neural Networks.

Copyright 2026 Bhanu Thakur. All rights reserved.

Public API:

- :class:`pinn.PINN` — fully-connected MLP backbone
- :class:`pinn.PINNTrainer` — generic multi-loss training loop
- :func:`pinn.set_seed` — reproducibility helper (random / numpy / torch)
- :func:`pinn.setup_logging` — loguru configuration with optional file sink
- plotting helpers: :func:`plot_contour`, :func:`plot_comparison_1d`,
  :func:`plot_loss_comparison`
- :func:`pinn.select_rar_points` — residual-based adaptive point selection
- :func:`pinn.adaptive_train` — multi-phase RAR training loop
"""

from .core.network import PINN
from .rar import adaptive_train, select_rar_points
from .trainer.trainer import PINNTrainer
from .utils.logging import setup_logging
from .utils.plotting import plot_comparison_1d, plot_contour, plot_loss_comparison
from .utils.seed import set_seed

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "PINN",
    "PINNTrainer",
    "adaptive_train",
    "select_rar_points",
    "set_seed",
    "setup_logging",
    "plot_contour",
    "plot_comparison_1d",
    "plot_loss_comparison",
]
