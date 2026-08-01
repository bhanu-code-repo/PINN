#!/usr/bin/env python3
"""Harmonic Oscillator PINN training CLI.

Trains a Physics-Informed Neural Network to solve the damped harmonic
oscillator ODE ``u'' + mu*u' + k*u = 0`` with ``u(0) = 1``, ``u'(0) = 0``,
using a learnable sinusoidal Ansatz to handle high-frequency oscillations.

Every run writes a self-contained artifact directory (checkpoint, metrics,
plots, logs). See the README in this directory for the full methodology.
"""

from pathlib import Path

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import typer
from loguru import logger
from pinn import PINN, PINNTrainer, plot_comparison_1d

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

app = typer.Typer(help="Train a PINN for the Damped Harmonic Oscillator.")

EXPERIMENT = "harmonic_oscillator"


class Ansatz(nn.Module):
    """u(t) = NN(t) * sin(a*t + b) with trainable frequency ``a`` and phase ``b``.

    The backbone only has to learn the slowly-varying envelope; the sinusoid
    carries the high-frequency content, defeating spectral bias.
    """

    def __init__(self, backbone: nn.Module, freq_init: float = 70.0, phase_init: float = 1.0):
        super().__init__()
        self.backbone = backbone
        self.a = nn.Parameter(torch.tensor(freq_init))
        self.b = nn.Parameter(torch.tensor(phase_init))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.backbone(t) * torch.sin(self.a * t + self.b)


def build_model(config: dict) -> nn.Module:
    """Reconstruct the model architecture from a run config.

    Used by both training and prediction so that checkpoints are
    self-describing: ``load_model(run_dir, build_model)`` needs no manually
    remembered hyperparameters.
    """
    backbone = PINN(
        input_dim=1,
        hidden_layers=config["hidden_layers"],
        hidden_neurons=config["hidden_neurons"],
    )
    return Ansatz(backbone)


def exact_solution(d: float, w0: float, t: np.ndarray) -> np.ndarray:
    """Closed-form under-damped solution (d < w0), used for validation only."""
    w = np.sqrt(w0**2 - d**2)
    phi = np.arctan(-d / w)
    A = 1 / (2 * np.cos(phi))
    return np.exp(-d * t) * 2 * A * np.cos(phi + w * t)


def build_losses(mu: float, k: float, t_domain: tuple[float, float],
                 n_collocation: int, device: torch.device) -> dict:
    """Create the named loss functions (closures own their collocation points)."""
    t_ic = torch.tensor([[0.0]], dtype=torch.float32, device=device, requires_grad=True)
    t_physics = (
        torch.linspace(*t_domain, n_collocation).view(-1, 1).to(device).requires_grad_(True)
    )

    def pde_residual(model, t):
        u = model(t)
        u_t = autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        u_tt = autograd.grad(u_t, t, torch.ones_like(u_t), create_graph=True)[0]
        return u_tt + mu * u_t + k * u

    def ic_loss(model):
        u = model(t_ic)
        u_t = autograd.grad(u, t_ic, torch.ones_like(u), create_graph=True)[0]
        return ((u - 1.0) ** 2 + (u_t - 0.0) ** 2).squeeze()

    def physics_loss(model):
        return torch.mean(pde_residual(model, t_physics) ** 2)

    return {"ic": ic_loss, "physics": physics_loss}


def evaluate(model: nn.Module, d: float, w0: float, t_domain: tuple[float, float],
             device: torch.device) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Compare the trained model against the closed-form solution.

    Returns:
        ``(metrics, t, u_pinn, u_exact)``.
    """
    t_test = torch.linspace(*t_domain, 300).view(-1, 1).to(device)
    with torch.no_grad():
        u_pinn = model(t_test).cpu().numpy()
    t_np = t_test.cpu().numpy()
    u_exact = exact_solution(d, w0, t_np)

    rel_l2 = float(np.linalg.norm(u_pinn - u_exact) / np.linalg.norm(u_exact))
    metrics = {
        "relative_l2_error": rel_l2,
        "learned_frequency_a": float(model.a.item()),
        "learned_phase_b": float(model.b.item()),
        "true_damped_frequency": float(np.sqrt(w0**2 - d**2)),
    }
    return metrics, t_np, u_pinn, u_exact


def solve_harmonic_oscillator(
    epochs: int = 15000,
    lr: float = 1e-3,
    hidden_neurons: int = 32,
    hidden_layers: int = 3,
    w0: float = 80.0,
    d: float = 2.0,
    seed: int = 42,
    output_dir: str | None = None,
    show: bool = True,
) -> dict:
    """Train, evaluate, and persist a harmonic-oscillator PINN run.

    Artifacts written to the run directory: ``checkpoint.pt``,
    ``metrics.json``, ``loss_history.png``, ``solution.png``, ``logs/``.

    Returns:
        The metrics dict (also saved as ``metrics.json``).
    """
    run_dir, device = init_run(EXPERIMENT, output_dir, seed)

    # 1. Problem setup
    mu, k = 2 * d, w0**2
    t_domain = (0.0, 1.0)
    weights = {"ic": 0.1, "physics": 1e-4}
    config = {
        "epochs": epochs, "lr": lr, "hidden_neurons": hidden_neurons,
        "hidden_layers": hidden_layers, "w0": w0, "d": d, "seed": seed,
        "loss_weights": weights,
    }
    logger.info("Config: {}", config)

    # 2. Model, losses, trainer
    model = build_model(config)
    loss_functions = build_losses(mu, k, t_domain, n_collocation=100, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = PINNTrainer(model, device=device)

    # 3. Train
    trainer.train(
        n_epochs=epochs,
        optimizer=optimizer,
        loss_functions=loss_functions,
        weights=weights,
        save_best=run_dir / "best_model.pt",
    )
    trainer.save_checkpoint(run_dir / "checkpoint.pt", optimizer=optimizer, metadata=config)
    trainer.plot_loss_history(show_total=True, save_path=run_dir / "loss_history.png", show=show)

    # 4. Evaluate against the exact solution
    metrics, t_np, u_pinn, u_exact = evaluate(model, d, w0, t_domain, device)
    metrics["final_total_loss"] = trainer.loss_history[-1]["total"]
    metrics["epochs_run"] = len(trainer.loss_history)
    save_metrics({"config": config, "metrics": metrics}, run_dir)

    # 5. Plots and summary
    plot_comparison_1d(
        t_np, u_exact, u_pinn,
        title=f"High-Frequency Damped Harmonic Oscillator (w0={w0})",
        xlabel="t", ylabel="u(t)",
        exact_label="Exact Solution", pred_label="PINN Solution",
        save_path=str(run_dir / "solution.png"), show=show,
    )
    print_summary("Training Summary", {
        "Final Loss": f"{metrics['final_total_loss']:.4e}",
        "Relative L2 Error": f"{metrics['relative_l2_error']:.4e}",
        "Learned 'a' (Frequency)": f"{metrics['learned_frequency_a']:.4f}",
        "True Damped Frequency": f"{metrics['true_damped_frequency']:.4f}",
        "Learned 'b' (Phase)": f"{metrics['learned_phase_b']:.4f}",
        "Epochs Run": str(metrics["epochs_run"]),
        "Artifacts": str(run_dir),
    })
    return metrics


@app.command()
def train(
    epochs: int = typer.Option(15000, "--epochs", "-e", help="Number of training epochs."),
    lr: float = typer.Option(1e-3, "--lr", help="Learning rate."),
    neurons: int = typer.Option(32, "--neurons", "-n", help="Neurons per hidden layer."),
    layers: int = typer.Option(3, "--layers", "-l", help="Number of hidden layers."),
    w0: float = typer.Option(80.0, "--w0", help="Natural frequency of the oscillator."),
    damping: float = typer.Option(2.0, "--damping", "-d", help="Damping coefficient."),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility."),
    output_dir: str | None = typer.Option(
        None, "--output-dir", "-o",
        help="Artifact directory (default: outputs/harmonic_oscillator/<timestamp>).",
    ),
    show: bool = typer.Option(True, "--show/--no-show", help="Display plots interactively."),
):
    """Train a PINN to solve the damped harmonic oscillator."""
    show_banner("PINN", "Damped Harmonic Oscillator PINN Solver")
    solve_harmonic_oscillator(
        epochs=epochs,
        lr=lr,
        hidden_neurons=neurons,
        hidden_layers=layers,
        w0=w0,
        d=damping,
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
    n_points: int = typer.Option(300, "--n-points", help="Number of evaluation points."),
    show: bool = typer.Option(True, "--show/--no-show", help="Display plots interactively."),
):
    """Load a trained model and evaluate it against the exact solution.

    Writes predictions.npz and prediction.png into the run directory.
    """
    from pinn import setup_logging

    setup_logging()
    run_dir = Path(run) if run else find_latest_run(EXPERIMENT)
    device = get_device()
    model, config = load_model(run_dir, build_model, device)

    d, w0 = config["d"], config["w0"]
    t_domain = (0.0, 1.0)
    t_test = torch.linspace(*t_domain, n_points).view(-1, 1).to(device)
    with torch.no_grad():
        u_pinn = model(t_test).cpu().numpy()
    t_np = t_test.cpu().numpy()
    u_exact = exact_solution(d, w0, t_np)
    rel_l2 = float(np.linalg.norm(u_pinn - u_exact) / np.linalg.norm(u_exact))

    np.savez(run_dir / "predictions.npz", t=t_np, u_pinn=u_pinn, u_exact=u_exact)
    logger.info("Predictions saved to {}", run_dir / "predictions.npz")

    plot_comparison_1d(
        t_np, u_exact, u_pinn,
        title=f"Prediction from {run_dir.name} (w0={w0})",
        xlabel="t", ylabel="u(t)",
        exact_label="Exact Solution", pred_label="PINN Solution",
        save_path=str(run_dir / "prediction.png"), show=show,
    )
    print_summary("Prediction Summary", {
        "Run": str(run_dir),
        "Relative L2 Error": f"{rel_l2:.4e}",
        "Evaluation Points": str(n_points),
    })


@app.command()
def compare():
    """Rank all runs of this experiment by their recorded metrics."""
    compare_runs(EXPERIMENT)


if __name__ == "__main__":
    app()
