#!/usr/bin/env python3
"""Parametric Burgers' Equation PINN training CLI.

Lifts "one model = one problem instance" for a PDE: the network takes
``(x, t, nu)`` as input and learns the Burgers' solution family

    u_t + u*u_x - nu*u_xx = 0,   u(0,x) = -sin(pi*x),   u(t,+-1) = 0

over the viscosity range ``nu in [0.01/pi, 0.1]`` (log-sampled — shock
sharpness scales with 1/nu). After one training run, ``predict --nu 0.05``
solves a never-trained viscosity in milliseconds, sweeping from sharp-shock
to diffusive regimes with a single checkpoint.

No closed-form solution exists, so validation uses the honest tools from
docs/prediction.md: the initial-condition error and the PDE residual.

Optionally trains a deep ensemble (``--ensemble N``) for +/-2 sigma
epistemic-uncertainty bands at prediction time.
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

app = typer.Typer(help="Train a parametric PINN for the Burgers' equation family.")

EXPERIMENT = "parametric_burgers"
X_DOMAIN = (-1.0, 1.0)
T_DOMAIN = (0.0, 1.0)
NU_RANGE = (0.01 / np.pi, 0.1)  # log-sampled during training
# Held-out viscosities for validation — never sampled explicitly during training
EVAL_NUS = [0.005, 0.02, 0.08]

_LOG_NU_LO = float(np.log10(NU_RANGE[0]))
_LOG_NU_HI = float(np.log10(NU_RANGE[1]))


class ParametricBurgersPINN(nn.Module):
    """u(x, t; nu) from a plain MLP with normalised ``(x, t, log10(nu))`` inputs.

    Viscosity enters through its **log**, normalised to ``[-1, 1]`` — nu spans
    a factor of ~30 and its physical effect (shock width) is multiplicative,
    so log-scaling is the right coordinate for the network.
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    @staticmethod
    def _normalise(x, t, nu):
        log_nu = torch.log10(nu)
        nu_n = 2 * (log_nu - _LOG_NU_LO) / (_LOG_NU_HI - _LOG_NU_LO) - 1
        t_n = 2 * (t - T_DOMAIN[0]) / (T_DOMAIN[1] - T_DOMAIN[0]) - 1
        return torch.cat([x, t_n, nu_n], dim=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
        return self.backbone(self._normalise(x, t, nu))


def build_model(config: dict) -> nn.Module:
    """Reconstruct the model architecture from a run config (self-describing checkpoints)."""
    backbone = PINN(
        input_dim=3,
        hidden_layers=config["hidden_layers"],
        hidden_neurons=config["hidden_neurons"],
    )
    return ParametricBurgersPINN(backbone)


def _sample_nu(n: int, device: torch.device) -> torch.Tensor:
    """Log-uniform viscosity samples over NU_RANGE."""
    log_nu = torch.rand(n, 1) * (_LOG_NU_HI - _LOG_NU_LO) + _LOG_NU_LO
    return (10.0**log_nu).to(device)


def build_losses(n_physics: int, n_ic: int, n_bc: int, device: torch.device) -> dict:
    """Create the named loss functions (closures own their collocation points)."""
    # Interior: uniform in (x, t), log-uniform in nu
    x_p = (torch.rand(n_physics, 1) * (X_DOMAIN[1] - X_DOMAIN[0]) + X_DOMAIN[0]).to(device)
    t_p = (torch.rand(n_physics, 1) * (T_DOMAIN[1] - T_DOMAIN[0]) + T_DOMAIN[0]).to(device)
    nu_p = _sample_nu(n_physics, device)
    x_p.requires_grad_(True)
    t_p.requires_grad_(True)

    # IC: t = 0, random (x, nu)
    x_ic = (torch.rand(n_ic, 1) * (X_DOMAIN[1] - X_DOMAIN[0]) + X_DOMAIN[0]).to(device)
    t_ic = torch.zeros(n_ic, 1, device=device)
    nu_ic = _sample_nu(n_ic, device)

    # BC: x = +-1, random (t, nu)
    t_bc = (torch.rand(n_bc, 1) * (T_DOMAIN[1] - T_DOMAIN[0]) + T_DOMAIN[0]).to(device)
    nu_bc = _sample_nu(n_bc, device)
    x_left = torch.full_like(t_bc, X_DOMAIN[0])
    x_right = torch.full_like(t_bc, X_DOMAIN[1])

    def pde_residual(model, x, t, nu):
        u = model(x, t, nu)
        u_t = autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        u_x = autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_xx = autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
        return u_t + u * u_x - nu * u_xx

    def physics_loss(model):
        return torch.mean(pde_residual(model, x_p, t_p, nu_p) ** 2)

    def ic_loss(model):
        u = model(x_ic, t_ic, nu_ic)
        return torch.mean((u + torch.sin(np.pi * x_ic)) ** 2)

    def bc_loss(model):
        u_l = model(x_left, t_bc, nu_bc)
        u_r = model(x_right, t_bc, nu_bc)
        return torch.mean(u_l**2 + u_r**2)

    return {"ic": ic_loss, "bc": bc_loss, "physics": physics_loss}


def _member_checkpoints(run_dir: Path) -> list[Path]:
    """All ensemble member checkpoints in a run directory, member 0 first."""
    extra = sorted(run_dir.glob("checkpoint_[0-9]*.pt"))
    return [run_dir / "checkpoint.pt", *extra]


def ensemble_predict_grid(
    models: list[nn.Module], nu: float, n_x: int, n_t: int, device: torch.device,
) -> dict:
    """Ensemble mean/std of u(x,t;nu) on a grid, plus t=0 and t=1 profiles."""
    x = torch.linspace(*X_DOMAIN, n_x).view(-1, 1).to(device)
    t = torch.linspace(*T_DOMAIN, n_t).view(-1, 1).to(device)
    X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing="ij")
    x_flat = X.flatten().unsqueeze(1)
    t_flat = T.flatten().unsqueeze(1)
    nu_flat = torch.full_like(x_flat, nu)

    with torch.no_grad():
        stack = torch.stack([m(x_flat, t_flat, nu_flat) for m in models])  # (K, N, 1)
    mean = stack.mean(dim=0).cpu().numpy().reshape(n_x, n_t)
    std = stack.std(dim=0).cpu().numpy().reshape(n_x, n_t)

    return {
        "X": X.cpu().numpy(), "T": T.cpu().numpy(),
        "u_mean": mean, "u_std": std,
        "x": x.cpu().numpy(),
        "u_mean_0": mean[:, 0:1], "u_std_0": std[:, 0:1],
        "u_mean_1": mean[:, -1:], "u_std_1": std[:, -1:],
    }


def mean_abs_residual(model: nn.Module, nu: float, n: int, device: torch.device) -> float:
    """Mean |PDE residual| of one model on random interior points at fixed nu."""
    x = (torch.rand(n, 1) * (X_DOMAIN[1] - X_DOMAIN[0]) + X_DOMAIN[0]).to(device)
    t = (torch.rand(n, 1) * (T_DOMAIN[1] - T_DOMAIN[0]) + T_DOMAIN[0]).to(device)
    x.requires_grad_(True)
    t.requires_grad_(True)
    nu_t = torch.full_like(x, nu)

    u = model(x, t, nu_t)
    u_t = autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    residual = u_t + u * u_x - nu_t * u_xx
    return float(residual.abs().mean().item())


def evaluate(models: list[nn.Module], device: torch.device) -> dict:
    """Validation without a closed form: IC error + residual at held-out nus."""
    metrics: dict = {}
    ic_errors = []
    for nu in EVAL_NUS:
        arrays = ensemble_predict_grid(models, nu, n_x=200, n_t=50, device=device)
        u_exact_0 = -np.sin(np.pi * arrays["x"])
        rel_l2_ic = float(
            np.linalg.norm(arrays["u_mean_0"] - u_exact_0) / np.linalg.norm(u_exact_0)
        )
        residual = mean_abs_residual(models[0], nu, n=2000, device=device)
        metrics[f"rel_l2_ic_nu={nu:g}"] = rel_l2_ic
        metrics[f"mean_abs_residual_nu={nu:g}"] = residual
        ic_errors.append(rel_l2_ic)
    metrics["rel_l2_ic_mean_heldout"] = float(np.mean(ic_errors))
    return metrics


def solve_parametric_burgers(
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
    """Train, evaluate, and persist a parametric Burgers PINN run.

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
        loss_functions = build_losses(n_physics=n_physics, n_ic=200, n_bc=200, device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        trainer = PINNTrainer(model, device=device)
        trainer.train(n_epochs=epochs, optimizer=optimizer, loss_functions=loss_functions)

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

    summary = {
        "Final Loss (member 0)": f"{metrics['final_total_loss']:.4e}",
        "Ensemble Members": str(ensemble),
        "Mean IC Rel-L2 (held-out nus)": f"{metrics['rel_l2_ic_mean_heldout']:.4e}",
        "Epochs Run": str(metrics["epochs_run"]),
        "Artifacts": str(run_dir),
    }
    for nu in EVAL_NUS:
        summary[f"Residual @ nu={nu:g}"] = f"{metrics[f'mean_abs_residual_nu={nu:g}']:.4e}"
    print_summary("Training Summary", summary)
    return metrics


@app.command()
def train(
    epochs: int = typer.Option(40000, "--epochs", "-e", help="Epochs per ensemble member."),
    lr: float = typer.Option(1e-3, "--lr", help="Learning rate."),
    neurons: int = typer.Option(64, "--neurons", "-n", help="Neurons per hidden layer."),
    layers: int = typer.Option(5, "--layers", "-l", help="Number of hidden layers."),
    n_physics: int = typer.Option(10000, "--n-physics", help="Collocation points in the (x,t,nu) box."),
    ensemble: int = typer.Option(1, "--ensemble", help="Number of ensemble members (>1 enables uncertainty bands)."),
    seed: int = typer.Option(42, "--seed", help="Base random seed (member i uses seed+i)."),
    output_dir: str | None = typer.Option(
        None, "--output-dir", "-o",
        help="Artifact directory (default: outputs/parametric_burgers/<timestamp>).",
    ),
    show: bool = typer.Option(True, "--show/--no-show", help="Display plots interactively."),
):
    """Train a parametric PINN over the viscosity family — optionally as a deep ensemble."""
    show_banner("BURGERS", "Parametric Burgers' Equation (viscosity family)")
    solve_parametric_burgers(
        epochs=epochs,
        lr=lr,
        hidden_neurons=neurons,
        hidden_layers=layers,
        n_physics=n_physics,
        ensemble=ensemble,
        seed=seed,
        output_dir=output_dir,
        show=show,
    )


@app.command()
def predict(
    nu: float = typer.Option(0.05, "--nu", help="Viscosity of the instance to solve."),
    run: str | None = typer.Option(
        None, "--run", "-r",
        help="Run directory containing checkpoint(s) (default: latest run).",
    ),
    show: bool = typer.Option(True, "--show/--no-show", help="Display plots interactively."),
):
    """Solve a NEW viscosity instance in milliseconds — no retraining.

    With an ensemble run, snapshots include a +/-2 sigma uncertainty band.
    Writes predictions.npz, prediction_contour.png, prediction_snapshots.png.
    """
    from pinn import setup_logging

    setup_logging()
    run_dir = Path(run) if run else find_latest_run(EXPERIMENT)
    device = get_device()

    if not (NU_RANGE[0] <= nu <= NU_RANGE[1]):
        logger.warning(
            "nu={} lies OUTSIDE the trained box nu in [{:.4f}, {:.4f}] — "
            "this is parameter-space extrapolation; the result is unreliable.",
            nu, *NU_RANGE,
        )

    models = []
    for path in _member_checkpoints(run_dir):
        model, _config = load_model(run_dir, build_model, device, checkpoint_name=path.name)
        models.append(model)
    logger.info("Loaded {} ensemble member(s) from {}", len(models), run_dir)

    arrays = ensemble_predict_grid(models, nu, n_x=200, n_t=200, device=device)
    u_exact_0 = -np.sin(np.pi * arrays["x"])
    rel_l2_ic = float(np.linalg.norm(arrays["u_mean_0"] - u_exact_0) / np.linalg.norm(u_exact_0))
    residual = mean_abs_residual(models[0], nu, n=2000, device=device)

    np.savez(run_dir / "predictions.npz", nu=nu, **arrays)
    logger.info("Predictions saved to {}", run_dir / "predictions.npz")

    # Contour of the (mean) solution
    plt.figure(figsize=(10, 6))
    contour = plt.contourf(arrays["T"], arrays["X"], arrays["u_mean"], 20, cmap="viridis")
    plt.colorbar(contour, label="u(t,x)")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title(f"Never-trained instance: Burgers' at nu={nu:g}")
    plt.savefig(run_dir / "prediction_contour.png", dpi=300, bbox_inches="tight")
    logger.info("Plot saved to {}", run_dir / "prediction_contour.png")
    plt.show() if show else plt.close()

    # Snapshots with uncertainty bands
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, key_m, key_s, title, exact in [
        (axes[0], "u_mean_0", "u_std_0", "t = 0 (vs exact IC)", u_exact_0),
        (axes[1], "u_mean_1", "u_std_1", "t = 1 (shock profile)", None),
    ]:
        mean, std = arrays[key_m][:, 0], arrays[key_s][:, 0]
        x = arrays["x"][:, 0]
        if exact is not None:
            ax.plot(x, exact, "k-", linewidth=2, alpha=0.9, label="Exact IC")
        ax.plot(x, mean, "r--", linewidth=2, label="PINN (ensemble mean)")
        if len(models) > 1:
            ax.fill_between(x, mean - 2 * std, mean + 2 * std,
                            alpha=0.3, color="orange", label="±2σ (ensemble)")
        ax.set(title=title, xlabel="x", ylabel="u")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_dir / "prediction_snapshots.png", dpi=300, bbox_inches="tight")
    logger.info("Plot saved to {}", run_dir / "prediction_snapshots.png")
    plt.show() if show else plt.close(fig)

    summary = {
        "Run": str(run_dir),
        "Instance": f"nu={nu:g}",
        "Ensemble Members": str(len(models)),
        "IC Relative L2 Error": f"{rel_l2_ic:.4e}",
        "Mean |Residual|": f"{residual:.4e}",
    }
    if len(models) > 1:
        summary["Max ±2σ Band Width"] = f"{float(4 * arrays['u_std'].max()):.4e}"
    print_summary("Prediction Summary", summary)


@app.command()
def compare():
    """Rank all runs of this experiment by their recorded metrics."""
    compare_runs(EXPERIMENT)


if __name__ == "__main__":
    app()
