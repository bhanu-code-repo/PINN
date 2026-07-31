#!/usr/bin/env python3
"""Lid-Driven Cavity PINN training CLI.

Trains a Physics-Informed Neural Network to solve the 2D steady
incompressible Navier-Stokes equations for the lid-driven cavity flow —
the standard benchmark for 2D NS solvers.

The network outputs ``(u, v, p)`` from inputs ``(x, y)`` (no time — steady
state). Boundary conditions: u = 1, v = 0 on the top wall (y = 1);
u = v = 0 on the other three walls. Validated against the tabulated
centreline velocities of Ghia, Ghia & Shin (1982) at Re = 100.

Domain: [0, 1]² with Dirichlet BCs on all walls.

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
from pinn import PINN, PINNTrainer

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

app = typer.Typer(help="Train a PINN for the 2D lid-driven cavity (steady Navier-Stokes).")

EXPERIMENT = "lid_driven_cavity"
DEFAULT_RE = 100.0  # Reynolds number

# Ghia, Ghia & Shin (1982) benchmark data at Re = 100
# u-velocity along the vertical centreline (x = 0.5)
GHIA_Y = np.array([
    0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813,
    0.4531, 0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609,
    0.9688, 0.9766, 1.0000,
])
GHIA_U = np.array([
    0.0000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662,
    -0.21090, -0.20581, -0.13641, 0.00332, 0.23151, 0.68717, 0.73722,
    0.78871, 0.84123, 1.00000,
])
# v-velocity along the horizontal centreline (y = 0.5)
GHIA_X = np.array([
    0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266,
    0.2344, 0.5000, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531,
    0.9609, 0.9688, 1.0000,
])
GHIA_V = np.array([
    0.0000, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077, 0.17507,
    0.17527, 0.05454, -0.24533, -0.22445, -0.16914, -0.10313, -0.08864,
    -0.07391, -0.05906, 0.00000,
])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SteadyNavierStokesPINN(nn.Module):
    """Three-output PINN returning ``(u, v, p)`` for 2D steady NS.

    Hard-encodes the no-slip BC on the three lower/side walls by
    construction: ``u_raw`` and ``v_raw`` are multiplied by a mask
    ``x(1-x)y`` that vanishes on x=0, x=1, y=0. The lid BC (u=1 at y=1)
    is added analytically.
    """

    def __init__(self, hidden_layers: int, hidden_neurons: int):
        super().__init__()
        self.network = PINN(
            input_dim=2, hidden_layers=hidden_layers,
            hidden_neurons=hidden_neurons, output_dim=3,
        )

    def forward(
        self, x: torch.Tensor, y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.network(torch.cat([x, y], dim=1))
        u_raw, v_raw, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]

        # Hard BC mask: vanishes on x=0, x=1, y=0; only y=1 is open
        mask = x * (1 - x) * y
        # u: mask * NN + y^10 * 1 (smooth ramp to lid velocity at y=1)
        # Using y^10 so the ramp is nearly zero except very close to y=1
        u = mask * u_raw + y**10
        # v: mask * (1-y) * NN — also vanishes at y=1 (v=0 on lid)
        v = mask * (1 - y) * v_raw
        return u, v, p


def build_model(config: dict) -> nn.Module:
    """Reconstruct the model architecture from a run config (self-describing checkpoints)."""
    return SteadyNavierStokesPINN(
        hidden_layers=config["hidden_layers"],
        hidden_neurons=config["hidden_neurons"],
    )


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def build_losses(n_physics: int, device: torch.device, re: float = DEFAULT_RE) -> dict:
    """Create the named loss functions.

    With hard BCs encoded in the model, only two loss terms remain:
        - ``physics``: NS momentum (x + y) + continuity residuals.
        - ``bc_lid``: soft reinforcement of the lid BC (u=1, v=0 at y=1)
          as a safety net — the hard BC handles most of it, but the soft
          term helps during early training when the network is far from
          the solution.
    """
    nu = 1.0 / re

    # Interior collocation points
    x_p = torch.rand(n_physics, 1, device=device, requires_grad=True)
    y_p = torch.rand(n_physics, 1, device=device, requires_grad=True)

    # Lid BC reinforcement points
    n_lid = 200
    x_lid = torch.rand(n_lid, 1, device=device)
    y_lid = torch.ones(n_lid, 1, device=device)

    def physics_loss(model):
        u, v, p = model(x_p, y_p)
        ones = torch.ones_like(u)

        u_x = autograd.grad(u, x_p, ones, create_graph=True)[0]
        u_y = autograd.grad(u, y_p, ones, create_graph=True)[0]
        u_xx = autograd.grad(u_x, x_p, ones, create_graph=True)[0]
        u_yy = autograd.grad(u_y, y_p, ones, create_graph=True)[0]

        v_x = autograd.grad(v, x_p, ones, create_graph=True)[0]
        v_y = autograd.grad(v, y_p, ones, create_graph=True)[0]
        v_xx = autograd.grad(v_x, x_p, ones, create_graph=True)[0]
        v_yy = autograd.grad(v_y, y_p, ones, create_graph=True)[0]

        p_x = autograd.grad(p, x_p, ones, create_graph=True)[0]
        p_y = autograd.grad(p, y_p, ones, create_graph=True)[0]

        # Steady NS: u·u_x + v·u_y + p_x - (1/Re)(u_xx + u_yy) = 0
        mom_x = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
        mom_y = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
        cont = u_x + v_y

        return torch.mean(mom_x**2 + mom_y**2 + cont**2)

    def bc_lid_loss(model):
        u, v, _p = model(x_lid, y_lid)
        return torch.mean((u - 1.0) ** 2 + v**2)

    return {"physics": physics_loss, "bc_lid": bc_lid_loss}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module, device: torch.device,
) -> tuple[dict, dict]:
    """Evaluate against Ghia benchmark on centrelines.

    Returns:
        ``(metrics, arrays)`` for downstream plotting and persistence.
    """
    n = 100

    # u along vertical centreline (x = 0.5)
    y_line = torch.linspace(0, 1, n).view(-1, 1).to(device)
    x_half = 0.5 * torch.ones(n, 1, device=device)
    with torch.no_grad():
        u_centre, _, _ = model(x_half, y_line)
    u_centre = u_centre.cpu().numpy().flatten()

    # v along horizontal centreline (y = 0.5)
    x_line = torch.linspace(0, 1, n).view(-1, 1).to(device)
    y_half = 0.5 * torch.ones(n, 1, device=device)
    with torch.no_grad():
        _, v_centre, _ = model(x_line, y_half)
    v_centre = v_centre.cpu().numpy().flatten()

    # Interpolate PINN at Ghia locations for error
    u_at_ghia = np.interp(GHIA_Y, np.linspace(0, 1, n), u_centre)
    v_at_ghia = np.interp(GHIA_X, np.linspace(0, 1, n), v_centre)
    u_err = float(np.linalg.norm(u_at_ghia - GHIA_U) / np.linalg.norm(GHIA_U))
    v_err = float(np.linalg.norm(v_at_ghia - GHIA_V) / np.linalg.norm(GHIA_V))

    # Full 2D field for contour plot
    xx = torch.linspace(0, 1, n).view(-1, 1).to(device)
    yy = torch.linspace(0, 1, n).view(-1, 1).to(device)
    X, Y = torch.meshgrid(xx.squeeze(), yy.squeeze(), indexing="ij")
    x_flat = X.flatten().unsqueeze(1)
    y_flat = Y.flatten().unsqueeze(1)
    with torch.no_grad():
        u_2d, v_2d, p_2d = model(x_flat, y_flat)
    u_2d = u_2d.cpu().numpy().reshape(n, n)
    v_2d = v_2d.cpu().numpy().reshape(n, n)
    p_2d = p_2d.cpu().numpy().reshape(n, n)

    metrics = {"rel_l2_u_ghia": u_err, "rel_l2_v_ghia": v_err}
    arrays = {
        "y_line": np.linspace(0, 1, n), "u_centre": u_centre,
        "x_line": np.linspace(0, 1, n), "v_centre": v_centre,
        "X": X.cpu().numpy(), "Y": Y.cpu().numpy(),
        "u_2d": u_2d, "v_2d": v_2d, "p_2d": p_2d,
    }
    return metrics, arrays


def make_plots(arrays: dict, save_path: str, show: bool) -> None:
    """Centreline profiles vs Ghia + 2D velocity/pressure contours."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Top-left: u along vertical centreline
    axes[0, 0].plot(arrays["u_centre"], arrays["y_line"], "b-", linewidth=2, label="PINN")
    axes[0, 0].plot(GHIA_U, GHIA_Y, "ro", markersize=6, label="Ghia et al. (1982)")
    axes[0, 0].set(xlabel="u", ylabel="y", title="u-velocity at x = 0.5")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Top-right: v along horizontal centreline
    axes[0, 1].plot(arrays["x_line"], arrays["v_centre"], "b-", linewidth=2, label="PINN")
    axes[0, 1].plot(GHIA_X, GHIA_V, "ro", markersize=6, label="Ghia et al. (1982)")
    axes[0, 1].set(xlabel="x", ylabel="v", title="v-velocity at y = 0.5")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Bottom-left: velocity magnitude contour
    X, Y = arrays["X"], arrays["Y"]
    speed = np.sqrt(arrays["u_2d"] ** 2 + arrays["v_2d"] ** 2)
    im = axes[1, 0].contourf(X, Y, speed, levels=30, cmap="viridis")
    axes[1, 0].set(xlabel="x", ylabel="y", title="Velocity magnitude")
    axes[1, 0].set_aspect("equal")
    plt.colorbar(im, ax=axes[1, 0])

    # Bottom-right: pressure contour
    p_ms = arrays["p_2d"] - arrays["p_2d"].mean()
    im2 = axes[1, 1].contourf(X, Y, p_ms, levels=30, cmap="RdBu_r")
    axes[1, 1].set(xlabel="x", ylabel="y", title="Pressure (mean-subtracted)")
    axes[1, 1].set_aspect("equal")
    plt.colorbar(im2, ax=axes[1, 1])

    plt.suptitle("Lid-Driven Cavity at Re = 100", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    logger.info("Plot saved to {}", save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve_cavity(
    epochs: int = 30000,
    lr: float = 1e-3,
    hidden_neurons: int = 64,
    hidden_layers: int = 5,
    n_physics: int = 10000,
    re: float = DEFAULT_RE,
    seed: int = 42,
    output_dir: str | None = None,
    show: bool = True,
) -> dict:
    """Train, evaluate, and persist a lid-driven cavity PINN run."""
    run_dir, device = init_run(EXPERIMENT, output_dir, seed)

    config = {
        "epochs": epochs, "lr": lr, "hidden_neurons": hidden_neurons,
        "hidden_layers": hidden_layers, "n_physics": n_physics,
        "re": re, "seed": seed,
    }
    logger.info("Config: {}", config)

    model = build_model(config)
    loss_functions = build_losses(n_physics, device, re)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = PINNTrainer(model, device=device)

    trainer.train(n_epochs=epochs, optimizer=optimizer, loss_functions=loss_functions)
    trainer.save_checkpoint(run_dir / "checkpoint.pt", optimizer=optimizer, metadata=config)
    trainer.plot_loss_history(show_total=True, save_path=run_dir / "loss_history.png", show=show)

    metrics, arrays = evaluate(model, device)
    final = trainer.loss_history[-1]
    metrics.update({
        "final_total_loss": final["total"],
        "final_physics_loss": final["physics"],
        "final_bc_lid_loss": final["bc_lid"],
        "epochs_run": len(trainer.loss_history),
    })
    save_metrics({"config": config, "metrics": metrics}, run_dir)

    make_plots(arrays, str(run_dir / "cavity_results.png"), show)
    print_summary("Training Summary", {
        "Final Total Loss": f"{metrics['final_total_loss']:.4e}",
        "Rel-L2 u vs Ghia": f"{metrics['rel_l2_u_ghia']:.4e}",
        "Rel-L2 v vs Ghia": f"{metrics['rel_l2_v_ghia']:.4e}",
        "Epochs Run": str(metrics["epochs_run"]),
        "Artifacts": str(run_dir),
    })
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def train(
    epochs: int = typer.Option(30000, "--epochs", "-e", help="Number of training epochs."),
    lr: float = typer.Option(1e-3, "--lr", help="Learning rate."),
    neurons: int = typer.Option(64, "--neurons", "-n", help="Neurons per hidden layer."),
    layers: int = typer.Option(5, "--layers", "-l", help="Number of hidden layers."),
    n_physics: int = typer.Option(10000, "--n-physics", help="Number of collocation points."),
    re: float = typer.Option(DEFAULT_RE, "--re", help="Reynolds number."),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility."),
    output_dir: str | None = typer.Option(
        None, "--output-dir", "-o",
        help="Artifact directory (default: outputs/lid_driven_cavity/<timestamp>).",
    ),
    show: bool = typer.Option(True, "--show/--no-show", help="Display plots interactively."),
):
    """Train a PINN to solve the 2D lid-driven cavity (steady NS at Re = 100)."""
    show_banner("CAVITY", "2D Lid-Driven Cavity — Steady Navier-Stokes PINN")
    solve_cavity(
        epochs=epochs, lr=lr, hidden_neurons=neurons, hidden_layers=layers,
        n_physics=n_physics, re=re, seed=seed, output_dir=output_dir, show=show,
    )


@app.command()
def predict(
    run: str | None = typer.Option(
        None, "--run", "-r",
        help="Run directory containing checkpoint.pt (default: latest run).",
    ),
    show: bool = typer.Option(True, "--show/--no-show", help="Display plots interactively."),
):
    """Load a trained cavity model and evaluate against the Ghia benchmark.

    Writes predictions.npz and prediction_cavity.png into the run directory.
    """
    from pinn import setup_logging

    setup_logging()
    run_dir = Path(run) if run else find_latest_run(EXPERIMENT)
    device = get_device()
    model, _config = load_model(run_dir, build_model, device)

    metrics, arrays = evaluate(model, device)
    np.savez(
        run_dir / "predictions.npz",
        u_2d=arrays["u_2d"], v_2d=arrays["v_2d"], p_2d=arrays["p_2d"],
        u_centre=arrays["u_centre"], v_centre=arrays["v_centre"],
        ghia_y=GHIA_Y, ghia_u=GHIA_U, ghia_x=GHIA_X, ghia_v=GHIA_V,
    )
    logger.info("Predictions saved to {}", run_dir / "predictions.npz")

    make_plots(arrays, str(run_dir / "prediction_cavity.png"), show)
    print_summary("Prediction Summary", {
        "Run": str(run_dir),
        "Rel-L2 u vs Ghia": f"{metrics['rel_l2_u_ghia']:.4e}",
        "Rel-L2 v vs Ghia": f"{metrics['rel_l2_v_ghia']:.4e}",
    })


@app.command()
def compare():
    """Rank all runs of this experiment by their recorded metrics."""
    compare_runs(EXPERIMENT)


if __name__ == "__main__":
    app()
