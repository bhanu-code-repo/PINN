import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

def plot_contour(
    X: np.ndarray, 
    Y: np.ndarray, 
    Z: np.ndarray, 
    title: str = "Contour Plot",
    xlabel: str = "x",
    ylabel: str = "y",
    clabel: str = "z",
    save_path: Optional[str] = None
):
    """
    Plots a standard contour plot.
    
    Args:
        X, Y: Meshgrid coordinates.
        Z: Values to contour.
        title: Plot title.
        xlabel, ylabel: Axis labels.
        clabel: Colorbar label.
        save_path: If provided, saves the plot.
    """
    plt.figure(figsize=(10, 6))
    contour = plt.contourf(Y, X, Z, 20, cmap='viridis')
    plt.colorbar(contour, label=clabel)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_comparison_1d(
    x: np.ndarray,
    y_exact: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Comparison Plot",
    xlabel: str = "x",
    ylabel: str = "y",
    exact_label: str = "Exact",
    pred_label: str = "Prediction",
    save_path: Optional[str] = None
):
    """
    Plots a 1D comparison between exact and predicted values.
    
    Args:
        x: X-axis coordinates.
        y_exact: Exact values.
        y_pred: Predicted values.
        title, xlabel, ylabel: Plot labels.
        exact_label, pred_label: Legend labels.
        save_path: If provided, saves the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_exact, 'k-', label=exact_label, linewidth=2, alpha=0.9)
    plt.plot(x, y_pred, 'r--', label=pred_label, linewidth=2, alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_loss_comparison(
    loss_history: dict,
    title: str = "Loss Comparison",
    save_path: Optional[str] = None
):
    """
    Plots multiple loss histories on the same plot (useful for comparing experiments).
    
    Args:
        loss_history: Dictionary where keys are experiment names and values are lists of losses.
        title: Plot title.
        save_path: If provided, saves the plot.
    """
    plt.figure(figsize=(10, 6))
    
    for name, losses in loss_history.items():
        plt.plot(losses, label=name, linewidth=1.5)
        
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()