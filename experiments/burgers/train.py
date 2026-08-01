#!/usr/bin/env python3
"""Burgers' Equation PINN training CLI.

Trains a Physics-Informed Neural Network to solve the 1D viscous Burgers'
equation ``u_t + u*u_x - nu*u_xx = 0`` with ``u(0,x) = -sin(pi*x)`` and
homogeneous Dirichlet boundaries — the classic shock-formation benchmark
from Raissi et al. (2019).

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
from pinn import PINN, PINNTrainer, adaptive_train, plot_contour

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

app = typer.Typer(help="Train a PINN for Burgers' Equation.")

EXPERIMENT = "burgers"
X_DOMAIN = (-1.0, 1.0)
T_DOMAIN = (0.0, 1.0)


def build_model(config: dict) -> nn.Module:
    """Reconstruct the model architecture from a run config.

    Used by both training and prediction so that checkpoints are
    self-describing: ``load_model(run_dir, build_model)`` needs no manually
    remembered hyperparameters.
    """
    return PINN(
        input_dim=2,
        hidden_layers=config["hidden_layers"],
        hidden_neurons=config["hidden_neurons"],
    )


def _pde_residual(model, x, t, nu):
    """Per-point Burgers PDE residual: u_t + u*u_x - nu*u_xx."""
    u = model(torch.cat([x, t], dim=1))
    u_t = autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    return u_t + u * u_x - nu * u_xx


def build_losses(nu: float, device: torch.device) -> dict:
    """Create the named loss functions (closures own their collocation points)."""
    x_physics = torch.rand(5000, 1) * (X_DOMAIN[1] - X_DOMAIN[0]) + X_DOMAIN[0]
    t_physics = torch.rand(5000, 1) * (T_DOMAIN[1] - T_DOMAIN[0]) + T_DOMAIN[0]
    physics_points = torch.cat([x_physics, t_physics], dim=1).to(device)
    return build_losses_with_points(nu, device, physics_points)


def build_losses_with_points(
    nu: float, device: torch.device, physics_points: torch.Tensor,
) -> dict:
    """Create loss functions using a specific set of physics collocation points."""
    x_ic = torch.linspace(*X_DOMAIN, 100).view(-1, 1).to(device)
    t_bc = torch.linspace(*T_DOMAIN, 50).view(-1, 1).to(device)

    x_phys = physics_points[:, 0:1].clone().requires_grad_(True)
    t_phys = physics_points[:, 1:2].clone().requires_grad_(True)

    def ic_loss(model):
        u = model(torch.cat([x_ic, torch.zeros_like(x_ic)], dim=1))
        u_exact = -torch.sin(np.pi * x_ic)
        return torch.mean((u - u_exact) ** 2)

    def bc_loss(model):
        u_left = model(torch.cat([-torch.ones_like(t_bc), t_bc], dim=1))
        u_right = model(torch.cat([torch.ones_like(t_bc), t_bc], dim=1))
        return torch.mean(u_left**2 + u_right**2)

    def physics_loss(model):
        return torch.mean(_pde_residual(model, x_phys, t_phys, nu) ** 2)

    return {"ic": ic_loss, "bc": bc_loss, "physics": physics_loss}


def _burgers_residual_fn(nu: float):
    """Return a residual function for RAR point selection."""
    def residual_fn(model, points):
        x = points[:, 0:1].clone().requires_grad_(True)
        t = points[:, 1:2].clone().requires_grad_(True)
        return _pde_residual(model, x, t, nu).squeeze(-1)
    return residual_fn


def _burgers_candidate_sampler(device: torch.device):
    """Return a sampler that generates random (x, t) candidates in the domain."""
    def sampler(n):
        x = torch.rand(n, 1) * (X_DOMAIN[1] - X_DOMAIN[0]) + X_DOMAIN[0]
        t = torch.rand(n, 1) * (T_DOMAIN[1] - T_DOMAIN[0]) + T_DOMAIN[0]
        return torch.cat([x, t], dim=1).to(device)
    return sampler


def evaluate(model: nn.Module, device: torch.device) -> tuple[dict, dict]:
    """Evaluate on a test grid and at the t=0 / t=1 snapshots.

    Returns:
        ``(metrics, arrays)`` where ``arrays`` holds everything needed for plots.
    """
    n_grid = 200
    x_test = torch.linspace(*X_DOMAIN, n_grid).view(-1, 1).to(device)
    t_test = torch.linspace(*T_DOMAIN, n_grid).view(-1, 1).to(device)
    X, T = torch.meshgrid(x_test.squeeze(), t_test.squeeze(), indexing="ij")

    with torch.no_grad():
        xt_test = torch.stack([X.flatten(), T.flatten()], dim=1)
        u_grid = model(xt_test).cpu().numpy().reshape(n_grid, n_grid)
        u_pinn_0 = model(torch.cat([x_test, torch.zeros_like(x_test)], dim=1)).cpu().numpy()
        u_pinn_1 = model(torch.cat([x_test, torch.ones_like(x_test)], dim=1)).cpu().numpy()

    x_np = x_test.cpu().numpy()
    u_exact_0 = -np.sin(np.pi * x_np)
    rel_l2_ic = float(np.linalg.norm(u_pinn_0 - u_exact_0) / np.linalg.norm(u_exact_0))

    metrics = {"relative_l2_error_t0": rel_l2_ic}
    arrays = {
        "X": X.cpu().numpy(), "T": T.cpu().numpy(), "u_grid": u_grid,
        "x": x_np, "u_pinn_0": u_pinn_0, "u_pinn_1": u_pinn_1, "u_exact_0": u_exact_0,
    }
    return metrics, arrays


def make_snapshot_plot(arrays: dict, save_path: str, show: bool) -> None:
    """Side-by-side t=0 (vs exact IC) and t=1 (shock) snapshots."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(arrays["x"], arrays["u_exact_0"], "k-", label="Exact (t=0)", linewidth=2)
    axes[0].plot(arrays["x"], arrays["u_pinn_0"], "r--", label="PINN (t=0)", linewidth=2)
    axes[0].set(title="Snapshot at t = 0", xlabel="x", ylabel="u(0,x)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(arrays["x"], arrays["u_pinn_1"], "r-", label="PINN (t=1)", linewidth=2)
    axes[1].set(title="Snapshot at t = 1 (Steep Shock Formed)", xlabel="x", ylabel="u(1,x)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    logger.info("Plot saved to {}", save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


def solve_burgers_equation(
    epochs: int = 30000,
    lr: float = 1e-3,
    hidden_neurons: int = 50,
    hidden_layers: int = 5,
    nu: float = 0.01 / np.pi,
    seed: int = 42,
    output_dir: str | None = None,
    show: bool = True,
    rar: bool = False,
    rar_phases: int = 5,
    rar_points: int = 500,
    use_wandb: bool = False,
    wandb_project: str = "pinn",
) -> dict:
    """Train, evaluate, and persist a Burgers' equation PINN run.

    Artifacts written to the run directory: ``checkpoint.pt``,
    ``metrics.json``, ``loss_history.png``, ``solution_contour.png``,
    ``snapshots.png``, ``logs/``.

    Returns:
        The metrics dict (also saved as ``metrics.json``).
    """
    run_dir, device = init_run(EXPERIMENT, output_dir, seed)

    config = {
        "epochs": epochs, "lr": lr, "hidden_neurons": hidden_neurons,
        "hidden_layers": hidden_layers, "nu": nu, "seed": seed,
        "rar": rar, "rar_phases": rar_phases, "rar_points": rar_points,
    }
    logger.info("Config: {}", config)

    # W&B integration (optional)
    callbacks = []
    if use_wandb:
        from pinn import wandb_callback, wandb_init
        wandb_init(project=wandb_project, config=config, group=EXPERIMENT)
        callbacks.append(wandb_callback(prefix="train/"))

    # 1. Model + trainer
    model = build_model(config)
    trainer = PINNTrainer(model, device=device)

    # 2. Train (standard or RAR)
    if rar:
        initial_points = _burgers_candidate_sampler(device)(5000)
        rar_result = adaptive_train(
            trainer=trainer,
            build_losses=lambda pts: build_losses_with_points(nu, device, pts),
            residual_fn=_burgers_residual_fn(nu),
            candidate_sampler=_burgers_candidate_sampler(device),
            initial_points=initial_points,
            optimizer_fn=lambda m: torch.optim.Adam(m.parameters(), lr=lr),
            n_phases=rar_phases,
            epochs_per_phase=epochs // rar_phases,
            n_candidates=10_000,
            n_select=rar_points,
            save_best=run_dir / "best_model.pt",
            callbacks=callbacks or None,
        )
        config["rar_points_per_phase"] = rar_result["points_per_phase"]
    else:
        loss_functions = build_losses(nu, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        trainer.train(
            n_epochs=epochs, optimizer=optimizer,
            loss_functions=loss_functions, save_best=run_dir / "best_model.pt",
            callbacks=callbacks or None,
        )
    trainer.save_checkpoint(run_dir / "checkpoint.pt", metadata=config)
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
        arrays["T"], arrays["X"], arrays["u_grid"],
        title="PINN Solution for Burgers' Equation",
        xlabel="t", ylabel="x", clabel="u(t,x)",
        save_path=str(run_dir / "solution_contour.png"), show=show,
    )
    make_snapshot_plot(arrays, str(run_dir / "snapshots.png"), show)
    print_summary("Training Summary", {
        "Final Total Loss": f"{metrics['final_total_loss']:.4e}",
        "Final IC Loss": f"{metrics['final_ic_loss']:.4e}",
        "Final BC Loss": f"{metrics['final_bc_loss']:.4e}",
        "Final Physics Loss": f"{metrics['final_physics_loss']:.4e}",
        "Relative L2 Error (t=0)": f"{metrics['relative_l2_error_t0']:.4e}",
        "Epochs Run": str(metrics["epochs_run"]),
        "Artifacts": str(run_dir),
    })

    if use_wandb:
        from pinn import wandb_finish
        wandb_finish(run_dir)

    return metrics


@app.command()
def train(
    epochs: int = typer.Option(30000, "--epochs", "-e", help="Number of training epochs."),
    lr: float = typer.Option(1e-3, "--lr", help="Learning rate."),
    neurons: int = typer.Option(50, "--neurons", "-n", help="Neurons per hidden layer."),
    layers: int = typer.Option(5, "--layers", "-l", help="Number of hidden layers."),
    nu: float = typer.Option(0.01 / np.pi, "--nu", help="Viscosity coefficient."),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility."),
    output_dir: str | None = typer.Option(
        None, "--output-dir", "-o",
        help="Artifact directory (default: outputs/burgers/<timestamp>).",
    ),
    show: bool = typer.Option(True, "--show/--no-show", help="Display plots interactively."),
    rar: bool = typer.Option(False, "--rar", help="Enable Residual-based Adaptive Refinement."),
    rar_phases: int = typer.Option(5, "--rar-phases", help="Number of RAR refinement phases."),
    rar_points: int = typer.Option(500, "--rar-points", help="New collocation points per phase."),
    use_wandb: bool = typer.Option(False, "--wandb", help="Log to Weights & Biases."),
    wandb_project: str = typer.Option("pinn", "--wandb-project", help="W&B project name."),
):
    """Train a PINN to solve the 1D Burgers' equation."""
    show_banner("BURGERS", "1D Burgers' Equation PINN Solver")
    solve_burgers_equation(
        epochs=epochs,
        lr=lr,
        hidden_neurons=neurons,
        hidden_layers=layers,
        nu=nu,
        seed=seed,
        output_dir=output_dir,
        show=show,
        rar=rar,
        rar_phases=rar_phases,
        rar_points=rar_points,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
    )


@app.command()
def predict(
    run: str | None = typer.Option(
        None, "--run", "-r",
        help="Run directory containing checkpoint.pt (default: latest run).",
    ),
    show: bool = typer.Option(True, "--show/--no-show", help="Display plots interactively."),
):
    """Load a trained model and evaluate it on the space-time grid.

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
        X=arrays["X"], T=arrays["T"], u_grid=arrays["u_grid"],
        x=arrays["x"], u_pinn_0=arrays["u_pinn_0"], u_pinn_1=arrays["u_pinn_1"],
        u_exact_0=arrays["u_exact_0"],
    )
    logger.info("Predictions saved to {}", run_dir / "predictions.npz")

    plot_contour(
        arrays["T"], arrays["X"], arrays["u_grid"],
        title=f"Prediction from {run_dir.name} — Burgers' Equation",
        xlabel="t", ylabel="x", clabel="u(t,x)",
        save_path=str(run_dir / "prediction_contour.png"), show=show,
    )
    make_snapshot_plot(arrays, str(run_dir / "prediction_snapshots.png"), show)
    print_summary("Prediction Summary", {
        "Run": str(run_dir),
        "Relative L2 Error (t=0)": f"{metrics['relative_l2_error_t0']:.4e}",
    })


@app.command()
def compare():
    """Rank all runs of this experiment by their recorded metrics."""
    compare_runs(EXPERIMENT)


if __name__ == "__main__":
    app()
