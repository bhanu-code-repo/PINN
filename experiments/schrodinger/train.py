#!/usr/bin/env python3
"""Schrödinger Equation PINN training CLI.

Trains a Physics-Informed Neural Network to solve the 1D focusing nonlinear
Schrödinger equation ``i*h_t + 0.5*h_xx + |h|^2*h = 0`` with the
``2*sech(x)`` soliton initial condition and periodic boundary conditions.
The complex field is represented as two real output channels
``h = u + i*v``.

Every run writes a self-contained artifact directory (checkpoint, metrics,
plots, logs). See the README in this directory for the full methodology.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import typer
from loguru import logger
from pinn import PINN, PINNTrainer, plot_contour

from experiments.common import (
    compare_runs,
    find_latest_run,
    get_device,
    init_run,
    load_model,
    print_summary,
    save_metrics,
    show_banner,
)

app = typer.Typer(help="Train a PINN for the Schrödinger Equation.")

EXPERIMENT = "schrodinger"
X_DOMAIN = (-5.0, 5.0)
T_DOMAIN = (0.0, np.pi / 2)


class ComplexPINN(nn.Module):
    """Two-output PINN returning ``(Re h, Im h)`` for a complex field ``h``."""

    def __init__(self, input_dim: int, hidden_layers: int, hidden_neurons: int):
        super().__init__()
        self.network = PINN(input_dim, hidden_layers, hidden_neurons, output_dim=2)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.network(torch.cat([x, t], dim=1))
        return out[:, 0:1], out[:, 1:2]


def build_model(config: dict) -> nn.Module:
    """Reconstruct the model architecture from a run config.

    Used by both training and prediction so that checkpoints are
    self-describing: ``load_model(run_dir, build_model)`` needs no manually
    remembered hyperparameters.
    """
    return ComplexPINN(
        input_dim=2,
        hidden_layers=config["hidden_layers"],
        hidden_neurons=config["hidden_neurons"],
    )


def build_losses(device: torch.device) -> dict:
    """Create the named loss functions (closures own their collocation points)."""
    x_ic = torch.linspace(*X_DOMAIN, 100).view(-1, 1).to(device).requires_grad_(True)
    t_bc = torch.linspace(*T_DOMAIN, 50).view(-1, 1).to(device)

    x_physics = torch.rand(5000, 1) * (X_DOMAIN[1] - X_DOMAIN[0]) + X_DOMAIN[0]
    t_physics = torch.rand(5000, 1) * (T_DOMAIN[1] - T_DOMAIN[0]) + T_DOMAIN[0]
    x_physics = x_physics.to(device).requires_grad_(True)
    t_physics = t_physics.to(device).requires_grad_(True)

    def pde_residual(model, x, t):
        u, v = model(x, t)
        h = u + 1j * v
        h_conj = u - 1j * v

        u_t = autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        v_t = autograd.grad(v, t, torch.ones_like(v), create_graph=True)[0]
        u_x = autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        v_x = autograd.grad(v, x, torch.ones_like(v), create_graph=True)[0]
        u_xx = autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
        v_xx = autograd.grad(v_x, x, torch.ones_like(v_x), create_graph=True)[0]

        h_t = u_t + 1j * v_t
        h_xx = u_xx + 1j * v_xx

        f = 1j * h_t + 0.5 * h_xx + (h * h_conj) * h
        return torch.mean(torch.abs(f) ** 2)

    def ic_loss(model):
        u, v = model(x_ic, torch.zeros_like(x_ic))
        h_exact = 2 / torch.cosh(x_ic)  # h(0,x) = 2*sech(x), purely real
        return torch.mean((u - h_exact) ** 2 + v**2)

    def bc_loss(model):
        # Periodic BC on value and x-derivative: h(t,-5) = h(t,5), h_x(t,-5) = h_x(t,5)
        x_l = X_DOMAIN[0] * torch.ones_like(t_bc).requires_grad_(True)
        x_r = X_DOMAIN[1] * torch.ones_like(t_bc).requires_grad_(True)

        u_l, v_l = model(x_l, t_bc)
        u_r, v_r = model(x_r, t_bc)

        u_l_x = autograd.grad(u_l, x_l, torch.ones_like(u_l), create_graph=True)[0]
        v_l_x = autograd.grad(v_l, x_l, torch.ones_like(v_l), create_graph=True)[0]
        u_r_x = autograd.grad(u_r, x_r, torch.ones_like(u_r), create_graph=True)[0]
        v_r_x = autograd.grad(v_r, x_r, torch.ones_like(v_r), create_graph=True)[0]

        loss_value = torch.mean((u_l - u_r) ** 2 + (v_l - v_r) ** 2)
        loss_deriv = torch.mean((u_l_x - u_r_x) ** 2 + (v_l_x - v_r_x) ** 2)
        return loss_value + loss_deriv

    def physics_loss(model):
        return pde_residual(model, x_physics, t_physics)

    return {"ic": ic_loss, "bc": bc_loss, "physics": physics_loss}


def evaluate(model: nn.Module, device: torch.device) -> tuple[dict, dict]:
    """Evaluate |h| on a test grid, at t=0 (vs exact IC), and at the t=pi/4 peak.

    Returns:
        ``(metrics, arrays)`` where ``arrays`` holds everything needed for plots.
    """
    n_x, n_t = 200, 100
    x_test = torch.linspace(*X_DOMAIN, n_x).view(-1, 1).to(device)
    t_test = torch.linspace(*T_DOMAIN, n_t).view(-1, 1).to(device)
    X, T = torch.meshgrid(x_test.squeeze(), t_test.squeeze(), indexing="ij")

    with torch.no_grad():
        u_pred, v_pred = model(X.flatten().unsqueeze(1), T.flatten().unsqueeze(1))
        h_mag = torch.sqrt(u_pred**2 + v_pred**2).cpu().numpy().reshape(n_x, n_t)

        u0, v0 = model(x_test, torch.zeros_like(x_test))
        h_mag_0 = torch.sqrt(u0**2 + v0**2).cpu().numpy()

        t_peak = (np.pi / 4) * torch.ones_like(x_test)
        up, vp = model(x_test, t_peak)
        h_mag_peak = torch.sqrt(up**2 + vp**2).cpu().numpy()

    x_np = x_test.cpu().numpy()
    h_exact_0 = 2 / np.cosh(x_np)
    rel_l2_ic = float(np.linalg.norm(h_mag_0 - h_exact_0) / np.linalg.norm(h_exact_0))

    metrics = {
        "relative_l2_error_t0": rel_l2_ic,
        "peak_magnitude_t_pi4": float(h_mag_peak.max()),
    }
    arrays = {
        "X": X.cpu().numpy(), "T": T.cpu().numpy(), "h_mag": h_mag,
        "x": x_np, "h_mag_0": h_mag_0, "h_exact_0": h_exact_0, "h_mag_peak": h_mag_peak,
    }
    return metrics, arrays


def make_snapshot_plot(arrays: dict, save_path: str, show: bool) -> None:
    """Side-by-side t=0 (vs exact IC) and t=pi/4 (breathing peak) snapshots."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(arrays["x"], arrays["h_exact_0"], "k-", label="Exact (t=0)", linewidth=2)
    axes[0].plot(arrays["x"], arrays["h_mag_0"], "r--", label="PINN (t=0)", linewidth=2)
    axes[0].set(title="Initial condition check", xlabel="x", ylabel="|h(0,x)|")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(arrays["x"], arrays["h_mag_peak"], "b-", label="PINN (t=pi/4)", linewidth=2)
    axes[1].set(title="Breathing peak near t = pi/4", xlabel="x", ylabel="|h(pi/4,x)|")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    logger.info("Plot saved to {}", save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


def solve_schrodinger_equation(
    epochs: int = 25000,
    lr: float = 5e-4,
    hidden_neurons: int = 100,
    hidden_layers: int = 4,
    seed: int = 42,
    output_dir: str | None = None,
    show: bool = True,
) -> dict:
    """Train, evaluate, and persist a nonlinear Schrödinger PINN run.

    Artifacts written to the run directory: ``checkpoint.pt``,
    ``metrics.json``, ``loss_history.png``, ``solution_contour.png``,
    ``snapshots.png``, ``logs/``.

    Returns:
        The metrics dict (also saved as ``metrics.json``).
    """
    run_dir, device = init_run(EXPERIMENT, output_dir, seed)

    config = {
        "epochs": epochs, "lr": lr, "hidden_neurons": hidden_neurons,
        "hidden_layers": hidden_layers, "seed": seed,
    }
    logger.info("Config: {}", config)

    # 1. Model, losses, trainer
    model = build_model(config)
    loss_functions = build_losses(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = PINNTrainer(model, device=device)

    # 2. Train
    trainer.train(n_epochs=epochs, optimizer=optimizer, loss_functions=loss_functions, save_best=run_dir / "best_model.pt")
    trainer.save_checkpoint(run_dir / "checkpoint.pt", optimizer=optimizer, metadata=config)
    trainer.plot_loss_history(show_total=True, save_path=run_dir / "loss_history.png", show=show)

    # 3. Evaluate
    metrics, arrays = evaluate(model, device)
    final = trainer.loss_history[-1]
    metrics.update({
        "final_total_loss": final["total"],
        "final_ic_loss": final["ic"],
        "final_bc_loss": final["bc"],
        "final_physics_loss": final["physics"],
        "epochs_run": len(trainer.loss_history),
    })
    save_metrics({"config": config, "metrics": metrics}, run_dir)

    # 4. Plots and summary
    plot_contour(
        arrays["T"], arrays["X"], arrays["h_mag"],
        title="PINN Solution Magnitude for Schrödinger Equation",
        xlabel="t", ylabel="x", clabel="|h(t,x)|",
        save_path=str(run_dir / "solution_contour.png"), show=show,
    )
    make_snapshot_plot(arrays, str(run_dir / "snapshots.png"), show)
    print_summary("Training Summary", {
        "Final Total Loss": f"{metrics['final_total_loss']:.4e}",
        "Final IC Loss": f"{metrics['final_ic_loss']:.4e}",
        "Final BC Loss": f"{metrics['final_bc_loss']:.4e}",
        "Final Physics Loss": f"{metrics['final_physics_loss']:.4e}",
        "Relative L2 Error (t=0)": f"{metrics['relative_l2_error_t0']:.4e}",
        "Peak |h| at t=pi/4": f"{metrics['peak_magnitude_t_pi4']:.3f}",
        "Epochs Run": str(metrics["epochs_run"]),
        "Artifacts": str(run_dir),
    })
    return metrics


@app.command()
def train(
    epochs: int = typer.Option(25000, "--epochs", "-e", help="Number of training epochs."),
    lr: float = typer.Option(5e-4, "--lr", help="Learning rate."),
    neurons: int = typer.Option(100, "--neurons", "-n", help="Neurons per hidden layer."),
    layers: int = typer.Option(4, "--layers", "-l", help="Number of hidden layers."),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility."),
    output_dir: str | None = typer.Option(
        None, "--output-dir", "-o",
        help="Artifact directory (default: outputs/schrodinger/<timestamp>).",
    ),
    show: bool = typer.Option(True, "--show/--no-show", help="Display plots interactively."),
):
    """Train a PINN to solve the 1D nonlinear Schrödinger equation."""
    show_banner("SCHRODINGER", "1D Nonlinear Schrödinger Equation PINN Solver")
    solve_schrodinger_equation(
        epochs=epochs,
        lr=lr,
        hidden_neurons=neurons,
        hidden_layers=layers,
        seed=seed,
        output_dir=output_dir,
        show=show,
    )


@app.command()
def predict(
    run: str | None = typer.Option(
        None, "--run", "-r",
        help="Run directory containing checkpoint.pt (default: latest run).",
    ),
    show: bool = typer.Option(True, "--show/--no-show", help="Display plots interactively."),
):
    """Load a trained model and evaluate |h| on the space-time grid.

    Writes predictions.npz, prediction_contour.png, and prediction_snapshots.png
    into the run directory.
    """
    from pinn import setup_logging

    setup_logging()
    run_dir = Path(run) if run else find_latest_run(EXPERIMENT)
    device = get_device()
    model, _config = load_model(run_dir, build_model, device)

    metrics, arrays = evaluate(model, device)
    np.savez(
        run_dir / "predictions.npz",
        X=arrays["X"], T=arrays["T"], h_mag=arrays["h_mag"],
        x=arrays["x"], h_mag_0=arrays["h_mag_0"], h_exact_0=arrays["h_exact_0"],
        h_mag_peak=arrays["h_mag_peak"],
    )
    logger.info("Predictions saved to {}", run_dir / "predictions.npz")

    plot_contour(
        arrays["T"], arrays["X"], arrays["h_mag"],
        title=f"Prediction from {run_dir.name} — Schr\u00f6dinger Equation",
        xlabel="t", ylabel="x", clabel="|h(t,x)|",
        save_path=str(run_dir / "prediction_contour.png"), show=show,
    )
    make_snapshot_plot(arrays, str(run_dir / "prediction_snapshots.png"), show)
    print_summary("Prediction Summary", {
        "Run": str(run_dir),
        "Relative L2 Error (t=0)": f"{metrics['relative_l2_error_t0']:.4e}",
        "Peak |h| at t=pi/4": f"{metrics['peak_magnitude_t_pi4']:.3f}",
    })


@app.command()
def compare():
    """Rank all runs of this experiment by their recorded metrics."""
    compare_runs(EXPERIMENT)


if __name__ == "__main__":
    app()
