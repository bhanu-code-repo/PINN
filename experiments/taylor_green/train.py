#!/usr/bin/env python3
"""Taylor-Green Vortex PINN training CLI.

Trains a Physics-Informed Neural Network to solve the 2D incompressible
Navier-Stokes equations for the Taylor-Green vortex — the simplest
unsteady NS problem with a closed-form solution valid for all time,
making it ideal for rigorous PINN validation.

The network outputs ``(u, v, p)`` from inputs ``(x, y, t)`` and is
trained by minimising the NS momentum + continuity residuals, the
initial-condition mismatch, and periodic boundary conditions.

Exact solution (valid for any ν, all t):

    u =  -cos(x) sin(y) exp(-2νt)
    v =   sin(x) cos(y) exp(-2νt)
    p = -¼(cos 2x + cos 2y) exp(-4νt)

Domain: [0, 2π]² × [0, T] with periodic BCs.

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

app = typer.Typer(help="Train a PINN for the 2D Taylor-Green vortex (Navier-Stokes).")

EXPERIMENT = "taylor_green"
XY_DOMAIN = (0.0, 2 * np.pi)
T_DOMAIN = (0.0, 1.0)
DEFAULT_NU = 0.01  # kinematic viscosity


# ---------------------------------------------------------------------------
# Exact solution
# ---------------------------------------------------------------------------

def exact_taylor_green(x, y, t, nu=DEFAULT_NU):
    """Exact Taylor-Green vortex solution (numpy arrays)."""
    u = -np.cos(x) * np.sin(y) * np.exp(-2 * nu * t)
    v = np.sin(x) * np.cos(y) * np.exp(-2 * nu * t)
    p = -0.25 * (np.cos(2 * x) + np.cos(2 * y)) * np.exp(-4 * nu * t)
    return u, v, p


def exact_taylor_green_torch(x, y, t, nu=DEFAULT_NU):
    """Exact Taylor-Green vortex solution (torch tensors, no grad)."""
    u = -torch.cos(x) * torch.sin(y) * torch.exp(-2 * nu * t)
    v = torch.sin(x) * torch.cos(y) * torch.exp(-2 * nu * t)
    p = -0.25 * (torch.cos(2 * x) + torch.cos(2 * y)) * torch.exp(-4 * nu * t)
    return u, v, p


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class NavierStokesPINN(nn.Module):
    """Three-output PINN returning ``(u, v, p)`` for 2D incompressible NS."""

    def __init__(self, hidden_layers: int, hidden_neurons: int):
        super().__init__()
        self.network = PINN(
            input_dim=3, hidden_layers=hidden_layers,
            hidden_neurons=hidden_neurons, output_dim=3,
        )

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.network(torch.cat([x, y, t], dim=1))
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]


def build_model(config: dict) -> nn.Module:
    """Reconstruct the model architecture from a run config (self-describing checkpoints)."""
    return NavierStokesPINN(
        hidden_layers=config["hidden_layers"],
        hidden_neurons=config["hidden_neurons"],
    )


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def build_losses(n_physics: int, device: torch.device, nu: float = DEFAULT_NU) -> dict:
    """Create the named loss functions (closures own their collocation points).

    Losses:
        - ``physics``: NS momentum (x + y) + continuity residuals at random
          interior points.
        - ``ic``: mismatch with the exact solution at t = 0.
        - ``bc``: periodic boundary conditions at x = 0/2π and y = 0/2π.
    """
    L = XY_DOMAIN[1] - XY_DOMAIN[0]

    # Physics collocation points (interior)
    x_p = (torch.rand(n_physics, 1) * L + XY_DOMAIN[0]).to(device).requires_grad_(True)
    y_p = (torch.rand(n_physics, 1) * L + XY_DOMAIN[0]).to(device).requires_grad_(True)
    t_p = (torch.rand(n_physics, 1) * (T_DOMAIN[1] - T_DOMAIN[0]) + T_DOMAIN[0]).to(device).requires_grad_(True)

    # IC points (t = 0)
    n_ic = 500
    x_ic = (torch.rand(n_ic, 1) * L + XY_DOMAIN[0]).to(device)
    y_ic = (torch.rand(n_ic, 1) * L + XY_DOMAIN[0]).to(device)
    t_ic = torch.zeros(n_ic, 1, device=device)
    u_ic, v_ic, p_ic = exact_taylor_green_torch(x_ic, y_ic, t_ic, nu)

    # BC points (periodic boundaries — shared time coordinates)
    n_bc = 200
    t_bc = (torch.rand(n_bc, 1) * (T_DOMAIN[1] - T_DOMAIN[0]) + T_DOMAIN[0]).to(device)
    # For x-periodicity: random y, x at left/right boundary
    y_bc_x = (torch.rand(n_bc, 1) * L + XY_DOMAIN[0]).to(device)
    # For y-periodicity: random x, y at bottom/top boundary
    x_bc_y = (torch.rand(n_bc, 1) * L + XY_DOMAIN[0]).to(device)

    def ns_residual(model, x, y, t):
        """Compute NS momentum and continuity residuals."""
        u, v, p = model(x, y, t)
        ones = torch.ones_like(u)

        # Velocity gradients
        u_t = autograd.grad(u, t, ones, create_graph=True)[0]
        u_x = autograd.grad(u, x, ones, create_graph=True)[0]
        u_y = autograd.grad(u, y, ones, create_graph=True)[0]
        u_xx = autograd.grad(u_x, x, ones, create_graph=True)[0]
        u_yy = autograd.grad(u_y, y, ones, create_graph=True)[0]

        v_t = autograd.grad(v, t, ones, create_graph=True)[0]
        v_x = autograd.grad(v, x, ones, create_graph=True)[0]
        v_y = autograd.grad(v, y, ones, create_graph=True)[0]
        v_xx = autograd.grad(v_x, x, ones, create_graph=True)[0]
        v_yy = autograd.grad(v_y, y, ones, create_graph=True)[0]

        p_x = autograd.grad(p, x, ones, create_graph=True)[0]
        p_y = autograd.grad(p, y, ones, create_graph=True)[0]

        # Momentum residuals: u_t + u·u_x + v·u_y + p_x - ν(u_xx + u_yy) = 0
        mom_x = u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
        mom_y = v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
        # Continuity: u_x + v_y = 0
        cont = u_x + v_y

        return mom_x, mom_y, cont

    def physics_loss(model):
        mom_x, mom_y, cont = ns_residual(model, x_p, y_p, t_p)
        return torch.mean(mom_x**2 + mom_y**2 + cont**2)

    def ic_loss(model):
        u, v, p = model(x_ic, y_ic, t_ic)
        return torch.mean((u - u_ic) ** 2 + (v - v_ic) ** 2 + (p - p_ic) ** 2)

    def bc_loss(model):
        """Periodic BCs: solution must match at opposite boundaries."""
        x_left = torch.full((n_bc, 1), XY_DOMAIN[0], device=device)
        x_right = torch.full((n_bc, 1), XY_DOMAIN[1], device=device)
        u_l, v_l, p_l = model(x_left, y_bc_x, t_bc)
        u_r, v_r, p_r = model(x_right, y_bc_x, t_bc)
        loss_x = torch.mean((u_l - u_r) ** 2 + (v_l - v_r) ** 2 + (p_l - p_r) ** 2)

        y_bottom = torch.full((n_bc, 1), XY_DOMAIN[0], device=device)
        y_top = torch.full((n_bc, 1), XY_DOMAIN[1], device=device)
        u_b, v_b, p_b = model(x_bc_y, y_bottom, t_bc)
        u_t_, v_t_, p_t_ = model(x_bc_y, y_top, t_bc)
        loss_y = torch.mean((u_b - u_t_) ** 2 + (v_b - v_t_) ** 2 + (p_b - p_t_) ** 2)

        return loss_x + loss_y

    return {"ic": ic_loss, "bc": bc_loss, "physics": physics_loss}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model: nn.Module, device: torch.device, nu: float = DEFAULT_NU) -> tuple[dict, dict]:
    """Evaluate the trained model against the exact solution on a test grid.

    Returns:
        ``(metrics, arrays)`` for downstream plotting and persistence.
    """
    n_xy, n_t = 50, 50
    xy = torch.linspace(*XY_DOMAIN, n_xy)
    ts = torch.linspace(*T_DOMAIN, n_t)
    X, Y, T = torch.meshgrid(xy, xy, ts, indexing="ij")
    x_flat = X.flatten().unsqueeze(1).to(device)
    y_flat = Y.flatten().unsqueeze(1).to(device)
    t_flat = T.flatten().unsqueeze(1).to(device)

    with torch.no_grad():
        u_pred, v_pred, p_pred = model(x_flat, y_flat, t_flat)
    u_pred = u_pred.cpu().numpy().reshape(n_xy, n_xy, n_t)
    v_pred = v_pred.cpu().numpy().reshape(n_xy, n_xy, n_t)
    p_pred = p_pred.cpu().numpy().reshape(n_xy, n_xy, n_t)

    x_np, y_np, t_np = X.numpy(), Y.numpy(), T.numpy()
    u_exact, v_exact, p_exact = exact_taylor_green(x_np, y_np, t_np, nu)

    # Velocity rel-L2
    vel_err = np.sqrt(np.sum((u_pred - u_exact) ** 2 + (v_pred - v_exact) ** 2))
    vel_ref = np.sqrt(np.sum(u_exact**2 + v_exact**2))
    rel_l2_vel = float(vel_err / vel_ref) if vel_ref > 0 else float(vel_err)

    # Pressure rel-L2 (mean-subtracted — pressure is defined up to a constant)
    p_pred_ms = p_pred - p_pred.mean()
    p_exact_ms = p_exact - p_exact.mean()
    p_err = float(np.linalg.norm(p_pred_ms - p_exact_ms) / np.linalg.norm(p_exact_ms))

    metrics = {"rel_l2_velocity": rel_l2_vel, "rel_l2_pressure": p_err}

    # Snapshot at t = T/2 for plotting
    t_mid = n_t // 2
    arrays = {
        "xy": xy.numpy(),
        "u_pred_mid": u_pred[:, :, t_mid],
        "v_pred_mid": v_pred[:, :, t_mid],
        "p_pred_mid": p_pred[:, :, t_mid],
        "u_exact_mid": u_exact[:, :, t_mid],
        "v_exact_mid": v_exact[:, :, t_mid],
        "p_exact_mid": p_exact[:, :, t_mid],
        "u_pred": u_pred, "v_pred": v_pred, "p_pred": p_pred,
        "u_exact": u_exact, "v_exact": v_exact, "p_exact": p_exact,
    }
    return metrics, arrays


def make_comparison_plot(arrays: dict, save_path: str, show: bool) -> None:
    """Side-by-side PINN vs exact velocity magnitude and pressure at t = T/2."""
    xy = arrays["xy"]
    X, Y = np.meshgrid(xy, xy, indexing="ij")

    u_p, v_p = arrays["u_pred_mid"], arrays["v_pred_mid"]
    u_e, v_e = arrays["u_exact_mid"], arrays["v_exact_mid"]
    speed_pred = np.sqrt(u_p**2 + v_p**2)
    speed_exact = np.sqrt(u_e**2 + v_e**2)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: velocity magnitude
    vmax = max(speed_exact.max(), speed_pred.max())
    im0 = axes[0, 0].contourf(X, Y, speed_exact, levels=30, vmin=0, vmax=vmax)
    axes[0, 0].set_title("Exact |vel|")
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].contourf(X, Y, speed_pred, levels=30, vmin=0, vmax=vmax)
    axes[0, 1].set_title("PINN |vel|")
    plt.colorbar(im1, ax=axes[0, 1])

    im2 = axes[0, 2].contourf(X, Y, np.abs(speed_pred - speed_exact), levels=30)
    axes[0, 2].set_title("|vel| error")
    plt.colorbar(im2, ax=axes[0, 2])

    # Row 2: pressure (mean-subtracted)
    p_p = arrays["p_pred_mid"] - arrays["p_pred_mid"].mean()
    p_e = arrays["p_exact_mid"] - arrays["p_exact_mid"].mean()
    pmax = max(abs(p_e).max(), abs(p_p).max())

    im3 = axes[1, 0].contourf(X, Y, p_e, levels=30, vmin=-pmax, vmax=pmax, cmap="RdBu_r")
    axes[1, 0].set_title("Exact pressure")
    plt.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].contourf(X, Y, p_p, levels=30, vmin=-pmax, vmax=pmax, cmap="RdBu_r")
    axes[1, 1].set_title("PINN pressure")
    plt.colorbar(im4, ax=axes[1, 1])

    im5 = axes[1, 2].contourf(X, Y, np.abs(p_p - p_e), levels=30)
    axes[1, 2].set_title("Pressure error")
    plt.colorbar(im5, ax=axes[1, 2])

    for ax in axes.flat:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")

    plt.suptitle("Taylor-Green Vortex: PINN vs Exact at t = T/2", fontsize=14)
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

def solve_taylor_green(
    epochs: int = 30000,
    lr: float = 1e-3,
    hidden_neurons: int = 64,
    hidden_layers: int = 5,
    n_physics: int = 10000,
    nu: float = DEFAULT_NU,
    seed: int = 42,
    output_dir: str | None = None,
    show: bool = True,
) -> dict:
    """Train, evaluate, and persist a Taylor-Green vortex PINN run."""
    run_dir, device = init_run(EXPERIMENT, output_dir, seed)

    config = {
        "epochs": epochs, "lr": lr, "hidden_neurons": hidden_neurons,
        "hidden_layers": hidden_layers, "n_physics": n_physics,
        "nu": nu, "seed": seed,
    }
    logger.info("Config: {}", config)

    model = build_model(config)
    loss_functions = build_losses(n_physics, device, nu)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = PINNTrainer(model, device=device)

    trainer.train(n_epochs=epochs, optimizer=optimizer, loss_functions=loss_functions, save_best=run_dir / "best_model.pt")
    trainer.save_checkpoint(run_dir / "checkpoint.pt", optimizer=optimizer, metadata=config)
    trainer.plot_loss_history(show_total=True, save_path=run_dir / "loss_history.png", show=show)

    metrics, arrays = evaluate(model, device, nu)
    final = trainer.loss_history[-1]
    metrics.update({
        "final_total_loss": final["total"],
        "final_ic_loss": final["ic"],
        "final_bc_loss": final["bc"],
        "final_physics_loss": final["physics"],
        "epochs_run": len(trainer.loss_history),
    })
    save_metrics({"config": config, "metrics": metrics}, run_dir)

    make_comparison_plot(arrays, str(run_dir / "comparison.png"), show)
    print_summary("Training Summary", {
        "Final Total Loss": f"{metrics['final_total_loss']:.4e}",
        "Rel-L2 Velocity": f"{metrics['rel_l2_velocity']:.4e}",
        "Rel-L2 Pressure": f"{metrics['rel_l2_pressure']:.4e}",
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
    nu: float = typer.Option(DEFAULT_NU, "--nu", help="Kinematic viscosity."),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility."),
    output_dir: str | None = typer.Option(
        None, "--output-dir", "-o",
        help="Artifact directory (default: outputs/taylor_green/<timestamp>).",
    ),
    show: bool = typer.Option(True, "--show/--no-show", help="Display plots interactively."),
):
    """Train a PINN to solve the 2D Taylor-Green vortex (Navier-Stokes)."""
    show_banner("TAYLOR GREEN", "2D Taylor-Green Vortex — Navier-Stokes PINN")
    solve_taylor_green(
        epochs=epochs, lr=lr, hidden_neurons=neurons, hidden_layers=layers,
        n_physics=n_physics, nu=nu, seed=seed, output_dir=output_dir, show=show,
    )


@app.command()
def predict(
    run: str | None = typer.Option(
        None, "--run", "-r",
        help="Run directory containing checkpoint.pt (default: latest run).",
    ),
    show: bool = typer.Option(True, "--show/--no-show", help="Display plots interactively."),
):
    """Load a trained model and evaluate against the exact Taylor-Green solution.

    Writes predictions.npz and prediction_comparison.png into the run directory.
    """
    from pinn import setup_logging

    setup_logging()
    run_dir = Path(run) if run else find_latest_run(EXPERIMENT)
    device = get_device()
    model, config = load_model(run_dir, build_model, device)
    nu = config.get("nu", DEFAULT_NU)

    metrics, arrays = evaluate(model, device, nu)
    np.savez(
        run_dir / "predictions.npz",
        u_pred=arrays["u_pred"], v_pred=arrays["v_pred"], p_pred=arrays["p_pred"],
        u_exact=arrays["u_exact"], v_exact=arrays["v_exact"], p_exact=arrays["p_exact"],
    )
    logger.info("Predictions saved to {}", run_dir / "predictions.npz")

    make_comparison_plot(arrays, str(run_dir / "prediction_comparison.png"), show)
    print_summary("Prediction Summary", {
        "Run": str(run_dir),
        "Viscosity (ν)": f"{nu}",
        "Rel-L2 Velocity": f"{metrics['rel_l2_velocity']:.4e}",
        "Rel-L2 Pressure": f"{metrics['rel_l2_pressure']:.4e}",
    })


@app.command()
def compare():
    """Rank all runs of this experiment by their recorded metrics."""
    compare_runs(EXPERIMENT)


if __name__ == "__main__":
    app()
