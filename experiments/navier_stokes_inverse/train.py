#!/usr/bin/env python3
"""Navier-Stokes Inverse Problem PINN training CLI.

Demonstrates the **inverse problem** methodology: given scattered noisy
velocity observations, infer the unknown Reynolds number (or equivalently
the viscosity ν) while simultaneously learning the flow field. This is
the same approach used by Raissi et al. (2019) for the cylinder wake,
but with the **Kovasznay flow** — an exact steady NS solution — so the
experiment is fully self-contained (no external CFD data required) and
rigorously validatable.

Kovasznay flow (exact solution for any Re):

    λ  = Re/2 - √(Re²/4 + 4π²)
    u  = 1 - exp(λx) cos(2πy)
    v  = (λ / 2π) exp(λx) sin(2πy)
    p  = (1 - exp(2λx)) / 2

Domain: [-0.5, 1.0] × [-0.5, 1.5].

The network outputs ``(u, v, p)`` and holds a **learnable scalar**
``log_Re`` (optimised jointly with the weights). The physics loss uses the
*current estimate* of Re inside the NS residual, so the network is
simultaneously fitting the data and inferring the parameter that makes
the physics consistent.

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

app = typer.Typer(help="Inverse NS PINN: infer Re from scattered velocity data (Kovasznay flow).")

EXPERIMENT = "navier_stokes_inverse"
X_DOMAIN = (-0.5, 1.0)
Y_DOMAIN = (-0.5, 1.5)
DEFAULT_RE_TRUE = 20.0  # ground-truth Re used to generate synthetic data
DEFAULT_NOISE = 0.01  # observation noise (fraction of signal amplitude)


# ---------------------------------------------------------------------------
# Exact Kovasznay solution
# ---------------------------------------------------------------------------

def kovasznay_lambda(re: float) -> float:
    """Kovasznay eigenvalue λ = Re/2 - √(Re²/4 + 4π²)."""
    return re / 2 - np.sqrt(re**2 / 4 + 4 * np.pi**2)


def exact_kovasznay(x, y, re):
    """Exact Kovasznay flow (numpy arrays)."""
    lam = kovasznay_lambda(re)
    u = 1.0 - np.exp(lam * x) * np.cos(2 * np.pi * y)
    v = (lam / (2 * np.pi)) * np.exp(lam * x) * np.sin(2 * np.pi * y)
    p = 0.5 * (1.0 - np.exp(2 * lam * x))
    return u, v, p


def generate_observations(
    n_obs: int, re_true: float, noise: float, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate scattered noisy (u, v) observations from the exact solution.

    Returns:
        ``(x_obs, y_obs, u_obs, v_obs)`` — each shape ``(n_obs,)``.
    """
    rng = np.random.default_rng(seed)
    x_obs = rng.uniform(*X_DOMAIN, n_obs)
    y_obs = rng.uniform(*Y_DOMAIN, n_obs)
    u_exact, v_exact, _ = exact_kovasznay(x_obs, y_obs, re_true)
    u_obs = u_exact + noise * rng.standard_normal(n_obs) * np.std(u_exact)
    v_obs = v_exact + noise * rng.standard_normal(n_obs) * np.std(v_exact)
    return x_obs, y_obs, u_obs, v_obs


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class InverseNavierStokesPINN(nn.Module):
    """PINN with a learnable Reynolds number for inverse NS problems.

    The network outputs ``(u, v, p)`` from ``(x, y)`` and exposes a
    learnable ``log_Re`` parameter (optimised in log-space for positivity
    and better conditioning).
    """

    def __init__(self, hidden_layers: int, hidden_neurons: int, log_re_init: float):
        super().__init__()
        self.network = PINN(
            input_dim=2, hidden_layers=hidden_layers,
            hidden_neurons=hidden_neurons, output_dim=3,
        )
        self.log_re = nn.Parameter(torch.tensor(log_re_init))

    @property
    def re(self) -> torch.Tensor:
        return torch.exp(self.log_re)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.network(torch.cat([x, y], dim=1))
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]


def build_model(config: dict) -> nn.Module:
    """Reconstruct the model architecture from a run config (self-describing checkpoints)."""
    return InverseNavierStokesPINN(
        hidden_layers=config["hidden_layers"],
        hidden_neurons=config["hidden_neurons"],
        log_re_init=config.get("log_re_init", np.log(10.0)),
    )


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def build_losses(
    n_physics: int,
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    u_obs: np.ndarray,
    v_obs: np.ndarray,
    device: torch.device,
) -> dict:
    """Create physics + data loss functions.

    The physics loss uses ``model.re`` (the learnable Re) inside the NS
    residual, so gradients flow through to ``log_re`` and the parameter
    is inferred jointly with the flow field.
    """
    # Physics collocation points
    Lx = X_DOMAIN[1] - X_DOMAIN[0]
    Ly = Y_DOMAIN[1] - Y_DOMAIN[0]
    x_p = (torch.rand(n_physics, 1) * Lx + X_DOMAIN[0]).to(device).requires_grad_(True)
    y_p = (torch.rand(n_physics, 1) * Ly + Y_DOMAIN[0]).to(device).requires_grad_(True)

    # Observation tensors
    x_d = torch.tensor(x_obs, dtype=torch.float32).view(-1, 1).to(device)
    y_d = torch.tensor(y_obs, dtype=torch.float32).view(-1, 1).to(device)
    u_d = torch.tensor(u_obs, dtype=torch.float32).view(-1, 1).to(device)
    v_d = torch.tensor(v_obs, dtype=torch.float32).view(-1, 1).to(device)

    def physics_loss(model):
        nu = 1.0 / model.re  # learnable!
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

        # Steady NS residuals with learnable ν = 1/Re
        mom_x = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
        mom_y = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
        cont = u_x + v_y

        return torch.mean(mom_x**2 + mom_y**2 + cont**2)

    def data_loss(model):
        u_pred, v_pred, _ = model(x_d, y_d)
        return torch.mean((u_pred - u_d) ** 2 + (v_pred - v_d) ** 2)

    return {"data": data_loss, "physics": physics_loss}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module, device: torch.device, re_true: float,
) -> tuple[dict, dict]:
    """Evaluate the inferred Re and field accuracy."""
    re_pred = float(model.re.detach().cpu())
    re_error = abs(re_pred - re_true) / re_true

    n = 80
    x_test = np.linspace(*X_DOMAIN, n)
    y_test = np.linspace(*Y_DOMAIN, n)
    X, Y = np.meshgrid(x_test, y_test, indexing="ij")

    x_t = torch.tensor(X.flatten(), dtype=torch.float32).view(-1, 1).to(device)
    y_t = torch.tensor(Y.flatten(), dtype=torch.float32).view(-1, 1).to(device)

    with torch.no_grad():
        u_pred, v_pred, p_pred = model(x_t, y_t)
    u_pred = u_pred.cpu().numpy().reshape(n, n)
    v_pred = v_pred.cpu().numpy().reshape(n, n)
    p_pred = p_pred.cpu().numpy().reshape(n, n)

    u_exact, v_exact, p_exact = exact_kovasznay(X, Y, re_true)

    vel_err = np.sqrt(np.sum((u_pred - u_exact) ** 2 + (v_pred - v_exact) ** 2))
    vel_ref = np.sqrt(np.sum(u_exact**2 + v_exact**2))
    rel_l2_vel = float(vel_err / vel_ref) if vel_ref > 0 else float(vel_err)

    metrics = {
        "re_true": re_true,
        "re_inferred": re_pred,
        "re_relative_error": re_error,
        "rel_l2_velocity": rel_l2_vel,
    }
    arrays = {
        "X": X, "Y": Y,
        "u_pred": u_pred, "v_pred": v_pred, "p_pred": p_pred,
        "u_exact": u_exact, "v_exact": v_exact, "p_exact": p_exact,
    }
    return metrics, arrays


def make_plots(
    arrays: dict, re_history: list[float], re_true: float,
    save_path: str, show: bool,
) -> None:
    """Field comparison + Re convergence history."""
    X, Y = arrays["X"], arrays["Y"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Row 1: velocity magnitude comparison
    speed_exact = np.sqrt(arrays["u_exact"] ** 2 + arrays["v_exact"] ** 2)
    speed_pred = np.sqrt(arrays["u_pred"] ** 2 + arrays["v_pred"] ** 2)
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

    for ax in axes[0]:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    # Row 2: Re convergence + pressure
    axes[1, 0].plot(re_history, "b-", linewidth=1.5)
    axes[1, 0].axhline(re_true, color="r", linestyle="--", linewidth=2, label=f"True Re = {re_true}")
    axes[1, 0].set(xlabel="Epoch", ylabel="Inferred Re", title="Re convergence")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    p_exact_ms = arrays["p_exact"] - arrays["p_exact"].mean()
    p_pred_ms = arrays["p_pred"] - arrays["p_pred"].mean()
    pmax = max(abs(p_exact_ms).max(), abs(p_pred_ms).max())

    im3 = axes[1, 1].contourf(X, Y, p_exact_ms, levels=30, vmin=-pmax, vmax=pmax, cmap="RdBu_r")
    axes[1, 1].set_title("Exact pressure")
    plt.colorbar(im3, ax=axes[1, 1])

    im4 = axes[1, 2].contourf(X, Y, p_pred_ms, levels=30, vmin=-pmax, vmax=pmax, cmap="RdBu_r")
    axes[1, 2].set_title("PINN pressure")
    plt.colorbar(im4, ax=axes[1, 2])

    for ax in axes[1, 1:]:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    plt.suptitle(
        f"Inverse NS (Kovasznay): Re_true = {re_true}, Re_inferred = {re_history[-1]:.2f}",
        fontsize=14,
    )
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

def solve_inverse_ns(
    epochs: int = 30000,
    lr: float = 1e-3,
    hidden_neurons: int = 64,
    hidden_layers: int = 5,
    n_physics: int = 5000,
    n_obs: int = 200,
    re_true: float = DEFAULT_RE_TRUE,
    noise: float = DEFAULT_NOISE,
    re_init: float = 10.0,
    seed: int = 42,
    output_dir: str | None = None,
    show: bool = True,
) -> dict:
    """Train, evaluate, and persist an inverse NS PINN run."""
    run_dir, device = init_run(EXPERIMENT, output_dir, seed)

    config = {
        "epochs": epochs, "lr": lr, "hidden_neurons": hidden_neurons,
        "hidden_layers": hidden_layers, "n_physics": n_physics,
        "n_obs": n_obs, "re_true": re_true, "noise": noise,
        "log_re_init": float(np.log(re_init)), "seed": seed,
    }
    logger.info("Config: {}", config)

    # Generate synthetic observations
    x_obs, y_obs, u_obs, v_obs = generate_observations(n_obs, re_true, noise, seed)
    logger.info("Generated {} noisy observations (noise = {}%)", n_obs, noise * 100)

    model = build_model(config)
    loss_functions = build_losses(n_physics, x_obs, y_obs, u_obs, v_obs, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = PINNTrainer(model, device=device)

    # Track Re convergence via callback
    re_history: list[float] = []

    def record_re(epoch, losses):
        re_history.append(float(model.re.detach().cpu()))
        if epoch % 1000 == 0:
            logger.info("Epoch {}: Re = {:.4f} (true: {})", epoch, re_history[-1], re_true)

    trainer.train(
        n_epochs=epochs, optimizer=optimizer, loss_functions=loss_functions,
        callbacks=[record_re], save_best=run_dir / "best_model.pt",
    )
    trainer.save_checkpoint(run_dir / "checkpoint.pt", optimizer=optimizer, metadata=config)
    trainer.plot_loss_history(show_total=True, save_path=run_dir / "loss_history.png", show=show)

    metrics, arrays = evaluate(model, device, re_true)
    final = trainer.loss_history[-1]
    metrics.update({
        "final_total_loss": final["total"],
        "final_data_loss": final["data"],
        "final_physics_loss": final["physics"],
        "epochs_run": len(trainer.loss_history),
    })
    save_metrics({"config": config, "metrics": metrics}, run_dir)

    # Save observations for reproducibility
    np.savez(
        run_dir / "observations.npz",
        x_obs=x_obs, y_obs=y_obs, u_obs=u_obs, v_obs=v_obs,
    )

    make_plots(arrays, re_history, re_true, str(run_dir / "inverse_results.png"), show)
    print_summary("Training Summary", {
        "Final Total Loss": f"{metrics['final_total_loss']:.4e}",
        "True Re": f"{re_true}",
        "Inferred Re": f"{metrics['re_inferred']:.4f}",
        "Re Relative Error": f"{metrics['re_relative_error']:.4e}",
        "Rel-L2 Velocity": f"{metrics['rel_l2_velocity']:.4e}",
        "Observations": f"{n_obs} (noise = {noise * 100:.1f}%)",
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
    n_physics: int = typer.Option(5000, "--n-physics", help="Number of collocation points."),
    n_obs: int = typer.Option(200, "--n-obs", help="Number of observation points."),
    re_true: float = typer.Option(DEFAULT_RE_TRUE, "--re-true", help="Ground-truth Reynolds number."),
    noise: float = typer.Option(DEFAULT_NOISE, "--noise", help="Observation noise level (fraction)."),
    re_init: float = typer.Option(10.0, "--re-init", help="Initial Re guess (deliberately wrong)."),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility."),
    output_dir: str | None = typer.Option(
        None, "--output-dir", "-o",
        help="Artifact directory (default: outputs/navier_stokes_inverse/<timestamp>).",
    ),
    show: bool = typer.Option(True, "--show/--no-show", help="Display plots interactively."),
):
    """Infer Re from noisy velocity data — the PINN inverse problem."""
    show_banner("NS INVERSE", "Inverse Navier-Stokes — Infer Re from Data (Kovasznay)")
    solve_inverse_ns(
        epochs=epochs, lr=lr, hidden_neurons=neurons, hidden_layers=layers,
        n_physics=n_physics, n_obs=n_obs, re_true=re_true, noise=noise,
        re_init=re_init, seed=seed, output_dir=output_dir, show=show,
    )


@app.command()
def predict(
    run: str | None = typer.Option(
        None, "--run", "-r",
        help="Run directory containing checkpoint.pt (default: latest run).",
    ),
    show: bool = typer.Option(True, "--show/--no-show", help="Display plots interactively."),
):
    """Load a trained inverse model and report the inferred Re and field accuracy.

    Writes predictions.npz and prediction_inverse.png into the run directory.
    """
    from pinn import setup_logging

    setup_logging()
    run_dir = Path(run) if run else find_latest_run(EXPERIMENT)
    device = get_device()
    model, config = load_model(run_dir, build_model, device)
    re_true = config.get("re_true", DEFAULT_RE_TRUE)

    metrics, arrays = evaluate(model, device, re_true)
    np.savez(
        run_dir / "predictions.npz",
        u_pred=arrays["u_pred"], v_pred=arrays["v_pred"], p_pred=arrays["p_pred"],
        u_exact=arrays["u_exact"], v_exact=arrays["v_exact"], p_exact=arrays["p_exact"],
        re_inferred=metrics["re_inferred"],
    )
    logger.info("Predictions saved to {}", run_dir / "predictions.npz")

    # Re-plot without full history (only final state)
    make_plots(
        arrays, [metrics["re_inferred"]], re_true,
        str(run_dir / "prediction_inverse.png"), show,
    )
    print_summary("Prediction Summary", {
        "Run": str(run_dir),
        "True Re": f"{re_true}",
        "Inferred Re": f"{metrics['re_inferred']:.4f}",
        "Re Relative Error": f"{metrics['re_relative_error']:.4e}",
        "Rel-L2 Velocity": f"{metrics['rel_l2_velocity']:.4e}",
    })


@app.command()
def compare():
    """Rank all runs of this experiment by their recorded metrics."""
    compare_runs(EXPERIMENT)


if __name__ == "__main__":
    app()
