import matplotlib.pyplot as plt
import numpy as np
from loguru import logger


def _finish(save_path: str | None, show: bool) -> None:
    """Shared save/show/close epilogue for all plotting helpers."""
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Plot saved to {}", save_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_contour(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    title: str = "Contour Plot",
    xlabel: str = "x",
    ylabel: str = "y",
    clabel: str = "z",
    save_path: str | None = None,
    show: bool = True,
):
    """Plot a standard filled contour plot.

    Args:
        X: Meshgrid coordinates for the horizontal axis.
        Y: Meshgrid coordinates for the vertical axis.
        Z: Values to contour; same shape as ``X`` and ``Y``.
        title: Plot title.
        xlabel: Horizontal axis label.
        ylabel: Vertical axis label.
        clabel: Colorbar label.
        save_path: If provided, saves the plot (300 dpi).
        show: Call ``plt.show()``. Set ``False`` for headless runs.
    """
    plt.figure(figsize=(10, 6))
    contour = plt.contourf(X, Y, Z, 20, cmap="viridis")
    plt.colorbar(contour, label=clabel)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    _finish(save_path, show)


def plot_comparison_1d(
    x: np.ndarray,
    y_exact: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Comparison Plot",
    xlabel: str = "x",
    ylabel: str = "y",
    exact_label: str = "Exact",
    pred_label: str = "Prediction",
    save_path: str | None = None,
    show: bool = True,
):
    """Plot a 1D comparison between exact and predicted values.

    Args:
        x: X-axis coordinates.
        y_exact: Exact values (black solid line).
        y_pred: Predicted values (red dashed line).
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        exact_label: Legend label for the exact curve.
        pred_label: Legend label for the predicted curve.
        save_path: If provided, saves the plot (300 dpi).
        show: Call ``plt.show()``. Set ``False`` for headless runs.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_exact, "k-", label=exact_label, linewidth=2, alpha=0.9)
    plt.plot(x, y_pred, "r--", label=pred_label, linewidth=2, alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    _finish(save_path, show)


def plot_loss_comparison(
    loss_history: dict,
    title: str = "Loss Comparison",
    save_path: str | None = None,
    show: bool = True,
):
    """Overlay multiple loss histories on one log-scale plot.

    Useful for comparing experiments or hyperparameter settings.

    Args:
        loss_history: Mapping of experiment name to a list of loss values.
        title: Plot title.
        save_path: If provided, saves the plot (300 dpi).
        show: Call ``plt.show()``. Set ``False`` for headless runs.
    """
    plt.figure(figsize=(10, 6))

    for name, losses in loss_history.items():
        plt.plot(losses, label=name, linewidth=1.5)

    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    _finish(save_path, show)
