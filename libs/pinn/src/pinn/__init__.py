"""pinn — core library for building and training Physics-Informed Neural Networks.

Public API:

- :class:`pinn.PINN` — fully-connected MLP backbone
- :class:`pinn.PINNTrainer` — generic multi-loss training loop
- :func:`pinn.set_seed` — reproducibility helper (random / numpy / torch)
- :func:`pinn.setup_logging` — loguru configuration with optional file sink
- plotting helpers: :func:`plot_contour`, :func:`plot_comparison_1d`,
  :func:`plot_loss_comparison`
"""

from .core.network import PINN
from .trainer.trainer import PINNTrainer
from .utils.logging import setup_logging
from .utils.plotting import plot_comparison_1d, plot_contour, plot_loss_comparison
from .utils.seed import set_seed

__all__ = [
    "PINN",
    "PINNTrainer",
    "set_seed",
    "setup_logging",
    "plot_contour",
    "plot_comparison_1d",
    "plot_loss_comparison",
]
