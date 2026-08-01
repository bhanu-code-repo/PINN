#!/usr/bin/env python3
"""Parametric Taylor-Green Vortex PINN training CLI.

Extends the single-instance Taylor-Green experiment to a **parametric PINN**:
the network takes ``(x, y, t, nu)`` as input and learns the solution family

    u(x, y, t; nu),  v(x, y, t; nu),  p(x, y, t; nu)

over the viscosity range ``nu in [0.001, 0.1]`` (log-sampled — the decay rate
scales linearly with nu, so log-space is the natural coordinate). After one
training run, ``predict --nu 0.05`` solves a never-trained viscosity in
milliseconds.

The Taylor-Green vortex has an **exact closed-form solution for any nu**:

    u =  -cos(x) sin(y) exp(-2 nu t)
    v =   sin(x) cos(y) exp(-2 nu t)
    p = -1/4 (cos 2x + cos 2y) exp(-4 nu t)

This makes it the ideal testbed for parametric NS: unlike parametric Burgers
(no closed form), we can compute exact rel-L2 errors at every held-out nu.

Domain: [0, 2pi]^2 x [0, 1], periodic BCs.

Optionally trains a deep ensemble (``--ensemble N``) for epistemic uncertainty.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import typer
from loguru import logger
from pinn import PINN, PINNTrainer, set_seed

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

app = typer.Typer(help="Train a parametric PINN for the Taylor-Green vortex family (Reynolds sweep).")

EXPERIMENT = "parametric_taylor_green"
XY_DOMAIN = (0.0, 2 * np.pi)
T_DOMAIN = (0.0, 1.0)
NU_RANGE = (0.001, 0.1)  # log-sampled during training

# Held-out viscosities for validation — never sampled explicitly during training
EVAL_NUS = [0.002, 0.015, 0.07]

_LOG_NU_LO = float(np.log10(NU_RANGE[0]))
_LOG_NU_HI = float(np.log10(NU_RANGE[1]))


# ---------------------------------------------------------------------------
# Exact solution
# ---------------------------------------------------------------------------

def exact_taylor_green(x, y, t, nu):
    """Exact Taylor-Green vortex solution (numpy arrays)."""
    u = -np.cos(x) * np.sin(y) * np.exp(-2 * nu * t)
    v = np.sin(x) * np.cos(y) * np.exp(-2 * nu * t)
    p = -0.25 * (np.cos(2 * x) + np.cos(2 * y)) * np.exp(-4 * nu * t)
    return u, v, p


def exact_taylor_green_torch(x, y, t, nu):
    """Exact Taylor-Green vortex solution (torch tensors, no grad)."""
    u = -torch.cos(x) * torch.sin(y) * torch.exp(-2 * nu * t)
    v = torch.sin(x) * torch.cos(y) * torch.exp(-2 * nu * t)
    p = -0.25 * (torch.cos(2 * x) + torch.cos(2 * y)) * torch.exp(-4 * nu * t)
    return u, v, p


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ParametricTaylorGreenPINN(nn.Module):
    """(u, v, p)(x, y, t; nu) from a plain MLP with normalised inputs.

    Viscosity enters through its **log**, normalised to [-1, 1] — nu spans
    a factor of 100 and its physical effect (decay rate) is multiplicative,
    so log-scaling is the right coordinate for the network.
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    @staticmethod
    def _normalise_nu(nu):
        log_nu = torch.log10(nu)
        return 2 * (log_nu - _LOG_NU_LO) / (_LOG_NU_HI - _LOG_NU_LO) - 1

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, nu: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nu_n = self._normalise_nu(nu)
        out = self.backbone(torch.cat([x, y, t, nu_n], dim=1))
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]


def build_model(config: dict) -> nn.Module:
    """Reconstruct the model architecture from a run config (self-describing checkpoints)."""
    backbone = PINN(
        input_dim=4,  # (x, y, t, nu_normalised)
        hidden_layers=config["hidden_layers"],
        hidden_neurons=config["hidden_neurons"],
        output_dim=3,  # (u, v, p)
    )
    return ParametricTaylorGreenPINN(backbone)


def _sample_nu(n: int, device: torch.device) -> torch.Tensor:
    """Log-uniform viscosity samples over NU_RANGE."""
    log_nu = torch.rand(n, 1) * (_LOG_NU_HI - _LOG_NU_LO) + _LOG_NU_LO
    return (10.0**log_nu).to(device)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def build_losses(n_physics: int, n_ic: int, n_bc: int, device: torch.device) -> dict:
    """Create the named loss functions (closures own their collocation points).

    All collocation points sample nu log-uniformly — each point in the
    (x, y, t, nu) box sees a different viscosity, teaching the network
    the full family simultaneously.
    """
    L = XY_DOMAIN[1] - XY_DOMAIN[0]

    # Physics collocation points (interior)
    x_p = (torch.rand(n_physics, 1) * L + XY_DOMAIN[0]).to(device).requires_grad_(True)
    y_p = (torch.rand(n_physics, 1) * L + XY_DOMAIN[0]).to(device).requires_grad_(True)
    t_p = (torch.rand(n_physics, 1) * (T_DOMAIN[1] - T_DOMAIN[0]) + T_DOMAIN[0]).to(device).requires_grad_(True)
    nu_p = _sample_nu(n_physics, device)

    # IC points (t = 0, random x, y, nu)
    x_ic = (torch.rand(n_ic, 1) * L + XY_DOMAIN[0]).to(device)
    y_ic = (torch.rand(n_ic, 1) * L + XY_DOMAIN[0]).to(device)
    t_ic = torch.zeros(n_ic, 1, device=device)
    nu_ic = _sample_nu(n_ic, device)
    # IC is the same for all nu: u = -cos(x)sin(y), v = sin(x)cos(y), p = -1/4(cos2x+cos2y)
    u_ic = -torch.cos(x_ic) * torch.sin(y_ic)
    v_ic = torch.sin(x_ic) * torch.cos(y_ic)
    p_ic = -0.25 * (torch.cos(2 * x_ic) + torch.cos(2 * y_ic))

    # BC points (periodic boundaries, random t and nu)
    t_bc = (torch.rand(n_bc, 1) * (T_DOMAIN[1] - T_DOMAIN[0]) + T_DOMAIN[0]).to(device)
    nu_bc = _sample_nu(n_bc, device)
    y_bc_x = (torch.rand(n_bc, 1) * L + XY_DOMAIN[0]).to(device)
    x_bc_y = (torch.rand(n_bc, 1) * L + XY_DOMAIN[0]).to(device)

    def physics_loss(model):
        u, v, p = model(x_p, y_p, t_p, nu_p)
        ones = torch.ones_like(u)

        u_t = autograd.grad(u, t_p, ones, create_graph=True)[0]
        u_x = autograd.grad(u, x_p, ones, create_graph=True)[0]
        u_y = autograd.grad(u, y_p, ones, create_graph=True)[0]
        u_xx = autograd.grad(u_x, x_p, ones, create_graph=True)[0]
        u_yy = autograd.grad(u_y, y_p, ones, create_graph=True)[0]

        v_t = autograd.grad(v, t_p, ones, create_graph=True)[0]
        v_x = autograd.grad(v, x_p, ones, create_graph=True)[0]
        v_y = autograd.grad(v, y_p, ones, create_graph=True)[0]
        v_xx = autograd.grad(v_x, x_p, ones, create_graph=True)[0]
        v_yy = autograd.grad(v_y, y_p, ones, create_graph=True)[0]

        p_x = autograd.grad(p, x_p, ones, create_graph=True)[0]
        p_y = autograd.grad(p, y_p, ones, create_graph=True)[0]

        mom_x = u_t + u * u_x + v * u_y + p_x - nu_p * (u_xx + u_yy)
        mom_y = v_t + u * v_x + v * v_y + p_y - nu_p * (v_xx + v_yy)
        cont = u_x + v_y

        return torch.mean(mom_x**2 + mom_y**2 + cont**2)

    def ic_loss(model):
        u, v, p = model(x_ic, y_ic, t_ic, nu_ic)
        return torch.mean((u - u_ic) ** 2 + (v - v_ic) ** 2 + (p - p_ic) ** 2)

    def bc_loss(model):
        x_left = torch.full((n_bc, 1), XY_DOMAIN[0], device=device)
        x_right = torch.full((n_bc, 1), XY_DOMAIN[1], device=device)
        u_l, v_l, p_l = model(x_left, y_bc_x, t_bc, nu_bc)
        u_r, v_r, p_r = model(x_right, y_bc_x, t_bc, nu_bc)
        loss_x = torch.mean((u_l - u_r) ** 2 + (v_l - v_r) ** 2 + (p_l - p_r) ** 2)

        y_bottom = torch.full((n_bc, 1), XY_DOMAIN[0], device=device)
        y_top = torch.full((n_bc, 1), XY_DOMAIN[1], device=device)
        u_b, v_b, p_b = model(x_bc_y, y_bottom, t_bc, nu_bc)
        u_t_, v_t_, p_t_ = model(x_bc_y, y_top, t_bc, nu_bc)
        loss_y = torch.mean((u_b - u_t_) ** 2 + (v_b - v_t_) ** 2 + (p_b - p_t_) ** 2)

        return loss_x + loss_y

    return {"ic": ic_loss, "bc": bc_loss, "physics": physics_loss}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _member_checkpoints(run_dir: Path) -> list[Path]:
    """All ensemble member checkpoints in a run directory, member 0 first."""
    extra = sorted(run_dir.glob("checkpoint_[0-9]*.pt"))
    return [run_dir / "checkpoint.pt", *extra]


def ensemble_predict_grid(
    models: list[nn.Module], nu: float, n_xy: int, n_t: int, device: torch.device,
) -> dict:
    """Ensemble mean/std of (u, v, p)(x, y, t; nu) on a grid."""
    xy = torch.linspace(*XY_DOMAIN, n_xy)
    ts = torch.linspace(*T_DOMAIN, n_t)
    X, Y, T = torch.meshgrid(xy, xy, ts, indexing="ij")
    x_flat = X.flatten().unsqueeze(1).to(device)
    y_flat = Y.flatten().unsqueeze(1).to(device)
    t_flat = T.flatten().unsqueeze(1).to(device)
    nu_flat = torch.full_like(x_flat, nu)

    u_stack, v_stack, p_stack = [], [], []
    with torch.no_grad():
        for m in models:
            u, v, p = m(x_flat, y_flat, t_flat, nu_flat)
            u_stack.append(u.cpu())
            v_stack.append(v.cpu())
            p_stack.append(p.cpu())

    shape = (n_xy, n_xy, n_t)
    u_all = torch.stack(u_stack).numpy().reshape(len(models), *shape)
    v_all = torch.stack(v_stack).numpy().reshape(len(models), *shape)
    p_all = torch.stack(p_stack).numpy().reshape(len(models), *shape)

    return {
        "xy": xy.numpy(), "ts": ts.numpy(),
        "X": X.numpy(), "Y": Y.numpy(), "T": T.numpy(),
        "u_mean": u_all.mean(axis=0), "u_std": u_all.std(axis=0),
        "v_mean": v_all.mean(axis=0), "v_std": v_all.std(axis=0),
        "p_mean": p_all.mean(axis=0), "p_std": p_all.std(axis=0),
    }


def evaluate(models: list[nn.Module], device: torch.device) -> dict:
    """Validate at held-out viscosities against the exact solution."""
    metrics: dict = {}
    vel_errors = []
    p_errors = []

    for nu in EVAL_NUS:
        arrays = ensemble_predict_grid(models, nu, n_xy=30, n_t=30, device=device)
        x_np, y_np, t_np = arrays["X"], arrays["Y"], arrays["T"]
        u_exact, v_exact, p_exact = exact_taylor_green(x_np, y_np, t_np, nu)

        # Velocity rel-L2
        vel_err = np.sqrt(np.sum((arrays["u_mean"] - u_exact) ** 2 + (arrays["v_mean"] - v_exact) ** 2))
        vel_ref = np.sqrt(np.sum(u_exact**2 + v_exact**2))
        rel_l2_vel = float(vel_err / vel_ref) if vel_ref > 0 else float(vel_err)

        # Pressure rel-L2 (mean-subtracted)
        p_pred_ms = arrays["p_mean"] - arrays["p_mean"].mean()
        p_exact_ms = p_exact - p_exact.mean()
        p_ref = np.linalg.norm(p_exact_ms)
        rel_l2_p = float(np.linalg.norm(p_pred_ms - p_exact_ms) / p_ref) if p_ref > 0 else 0.0

        metrics[f"rel_l2_velocity_nu={nu:g}"] = rel_l2_vel
        metrics[f"rel_l2_pressure_nu={nu:g}"] = rel_l2_p
        vel_errors.append(rel_l2_vel)
        p_errors.append(rel_l2_p)

    metrics["rel_l2_velocity_mean_heldout"] = float(np.mean(vel_errors))
    metrics["rel_l2_pressure_mean_heldout"] = float(np.mean(p_errors))
    return metrics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_comparison_plot(
    models: list[nn.Module], nu: float, device: torch.device,
    save_path: str, show: bool,
) -> None:
    """Side-by-side PINN vs exact at t = T/2 for a specific nu."""
    arrays = ensemble_predict_grid(models, nu, n_xy=50, n_t=50, device=device)
    xy = arrays["xy"]
    X, Y = np.meshgrid(xy, xy, indexing="ij")
    t_mid = len(arrays["ts"]) // 2

    u_exact, v_exact, p_exact = exact_taylor_green(
        X, Y, np.full_like(X, arrays["ts"][t_mid]), nu,
    )
    speed_pred = np.sqrt(arrays["u_mean"][:, :, t_mid] ** 2 + arrays["v_mean"][:, :, t_mid] ** 2)
    speed_exact = np.sqrt(u_exact**2 + v_exact**2)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

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

    p_p = arrays["p_mean"][:, :, t_mid] - arrays["p_mean"][:, :, t_mid].mean()
    p_e = p_exact - p_exact.mean()
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

    re = 1.0 / nu
    plt.suptitle(f"Parametric Taylor-Green: nu={nu:g} (Re={re:.0f}) at t = T/2", fontsize=14)
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

def solve_parametric_taylor_green(
    epochs: int = 40000,
    lr: float = 1e-3,
    hidden_neurons: int = 64,
    hidden_layers: int = 5,
    n_physics: int = 10000,
    ensemble: int = 1,
    seed: int = 42,
    output_dir: str | None = None,
    show: bool = True,
) -> dict:
    """Train, evaluate, and persist a parametric Taylor-Green PINN run.

    With ``ensemble > 1``, trains N independent members (seeds ``seed + i``)
    saved as ``checkpoint.pt``, ``checkpoint_1.pt``, ...

    Returns:
        The metrics dict (also saved as ``metrics.json``).
    """
    run_dir, device = init_run(EXPERIMENT, output_dir, seed)

    config = {
        "epochs": epochs, "lr": lr, "hidden_neurons": hidden_neurons,
        "hidden_layers": hidden_layers, "n_physics": n_physics,
        "ensemble": ensemble, "seed": seed, "nu_range": list(NU_RANGE),
    }
    logger.info("Config: {}", config)

    models: list[nn.Module] = []
    metrics: dict = {}
    for member in range(ensemble):
        member_seed = seed + member
        set_seed(member_seed)
        logger.info("Training ensemble member {}/{} (seed {})", member + 1, ensemble, member_seed)

        model = build_model(config)
        loss_functions = build_losses(n_physics=n_physics, n_ic=500, n_bc=200, device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        trainer = PINNTrainer(model, device=device)
        trainer.train(
            n_epochs=epochs, optimizer=optimizer, loss_functions=loss_functions,
            save_best=run_dir / ("best_model.pt" if member == 0 else f"best_model_{member}.pt"),
        )

        name = "checkpoint.pt" if member == 0 else f"checkpoint_{member}.pt"
        trainer.save_checkpoint(
            run_dir / name, optimizer=optimizer,
            metadata={**config, "member": member, "member_seed": member_seed},
        )
        metrics[f"final_total_loss_member_{member}"] = trainer.loss_history[-1]["total"]
        if member == 0:
            trainer.plot_loss_history(
                show_total=True, save_path=run_dir / "loss_history.png", show=show,
            )
            metrics["final_total_loss"] = trainer.loss_history[-1]["total"]
            metrics["epochs_run"] = len(trainer.loss_history)
        models.append(model.eval())

    metrics.update(evaluate(models, device))
    save_metrics({"config": config, "metrics": metrics}, run_dir)

    # Comparison plot at a held-out nu
    make_comparison_plot(models, EVAL_NUS[1], device, str(run_dir / "comparison.png"), show)

    summary = {
        "Final Loss (member 0)": f"{metrics['final_total_loss']:.4e}",
        "Ensemble Members": str(ensemble),
        "Mean Velocity Rel-L2 (held-out)": f"{metrics['rel_l2_velocity_mean_heldout']:.4e}",
        "Mean Pressure Rel-L2 (held-out)": f"{metrics['rel_l2_pressure_mean_heldout']:.4e}",
        "Epochs Run": str(metrics["epochs_run"]),
        "Artifacts": str(run_dir),
    }
    for nu in EVAL_NUS:
        re = 1.0 / nu
        summary[f"Vel err @ nu={nu:g} (Re={re:.0f})"] = f"{metrics[f'rel_l2_velocity_nu={nu:g}']:.4e}"
    print_summary("Training Summary", summary)
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def train(
    epochs: int = typer.Option(40000, "--epochs", "-e", help="Epochs per ensemble member."),
    lr: float = typer.Option(1e-3, "--lr", help="Learning rate."),
    neurons: int = typer.Option(64, "--neurons", "-n", help="Neurons per hidden layer."),
    layers: int = typer.Option(5, "--layers", "-l", help="Number of hidden layers."),
    n_physics: int = typer.Option(10000, "--n-physics", help="Collocation points in the (x,y,t,nu) box."),
    ensemble: int = typer.Option(1, "--ensemble", help="Number of ensemble members (>1 enables uncertainty)."),
    seed: int = typer.Option(42, "--seed", help="Base random seed (member i uses seed+i)."),
    output_dir: str | None = typer.Option(
        None, "--output-dir", "-o",
        help="Artifact directory (default: outputs/parametric_taylor_green/<timestamp>).",
    ),
    show: bool = typer.Option(True, "--show/--no-show", help="Display plots interactively."),
):
    """Train a parametric PINN over the viscosity family — optionally as a deep ensemble."""
    show_banner("TAYLOR GREEN", "Parametric Taylor-Green Vortex (Reynolds sweep)")
    solve_parametric_taylor_green(
        epochs=epochs, lr=lr, hidden_neurons=neurons, hidden_layers=layers,
        n_physics=n_physics, ensemble=ensemble, seed=seed,
        output_dir=output_dir, show=show,
    )


@app.command()
def predict(
    nu: float = typer.Option(0.01, "--nu", help="Viscosity of the instance to solve."),
    run: str | None = typer.Option(
        None, "--run", "-r",
        help="Run directory containing checkpoint(s) (default: latest run).",
    ),
    show: bool = typer.Option(True, "--show/--no-show", help="Display plots interactively."),
):
    """Solve a NEW viscosity instance in milliseconds — no retraining.

    Writes predictions.npz and comparison plot into the run directory.
    """
    from pinn import setup_logging

    setup_logging()
    run_dir = Path(run) if run else find_latest_run(EXPERIMENT)
    device = get_device()

    if not (NU_RANGE[0] <= nu <= NU_RANGE[1]):
        logger.warning(
            "nu={} lies OUTSIDE the trained box nu in [{}, {}] — "
            "this is parameter-space extrapolation; the result is unreliable.",
            nu, *NU_RANGE,
        )

    models = []
    for path in _member_checkpoints(run_dir):
        model, _config = load_model(run_dir, build_model, device, checkpoint_name=path.name)
        models.append(model)
    logger.info("Loaded {} ensemble member(s) from {}", len(models), run_dir)

    arrays = ensemble_predict_grid(models, nu, n_xy=50, n_t=50, device=device)
    x_np, y_np, t_np = arrays["X"], arrays["Y"], arrays["T"]
    u_exact, v_exact, p_exact = exact_taylor_green(x_np, y_np, t_np, nu)

    vel_err = np.sqrt(np.sum((arrays["u_mean"] - u_exact) ** 2 + (arrays["v_mean"] - v_exact) ** 2))
    vel_ref = np.sqrt(np.sum(u_exact**2 + v_exact**2))
    rel_l2_vel = float(vel_err / vel_ref) if vel_ref > 0 else float(vel_err)

    p_pred_ms = arrays["p_mean"] - arrays["p_mean"].mean()
    p_exact_ms = p_exact - p_exact.mean()
    p_ref = np.linalg.norm(p_exact_ms)
    rel_l2_p = float(np.linalg.norm(p_pred_ms - p_exact_ms) / p_ref) if p_ref > 0 else 0.0

    np.savez(
        run_dir / "predictions.npz", nu=nu,
        u_pred=arrays["u_mean"], v_pred=arrays["v_mean"], p_pred=arrays["p_mean"],
        u_exact=u_exact, v_exact=v_exact, p_exact=p_exact,
    )
    logger.info("Predictions saved to {}", run_dir / "predictions.npz")

    make_comparison_plot(models, nu, device, str(run_dir / "prediction_comparison.png"), show)

    re = 1.0 / nu
    summary = {
        "Run": str(run_dir),
        "Instance": f"nu={nu:g} (Re={re:.0f})",
        "Ensemble Members": str(len(models)),
        "Rel-L2 Velocity": f"{rel_l2_vel:.4e}",
        "Rel-L2 Pressure": f"{rel_l2_p:.4e}",
    }
    print_summary("Prediction Summary", summary)


@app.command()
def compare():
    """Rank all runs of this experiment by their recorded metrics."""
    compare_runs(EXPERIMENT)


if __name__ == "__main__":
    app()
