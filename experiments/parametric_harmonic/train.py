#!/usr/bin/env python3
"""Parametric Harmonic Oscillator PINN training CLI.

Lifts the "one model = one problem instance" PINN limitation: the network
takes ``(t, w0, d)`` as input and learns the *whole solution family* of

    u'' + 2*d*u' + w0^2*u = 0,   u(0) = 1,  u'(0) = 0

over the parameter box ``w0 in [20, 100]``, ``d in [0.1, 4]``. After one
training run, ``predict --w0 40 -d 1.5`` solves a never-trained problem
instance in milliseconds.

Optionally trains a deep ensemble (``--ensemble N``) whose member
disagreement provides an epistemic-uncertainty band at prediction time —
mitigating the "no uncertainty awareness" limitation.

Every run writes a self-contained artifact directory (checkpoints, metrics,
plots, logs). See the README in this directory and docs/prediction.md.
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

app = typer.Typer(help="Train a parametric PINN for the damped harmonic oscillator family.")

EXPERIMENT = "parametric_harmonic"
T_DOMAIN = (0.0, 1.0)
W0_RANGE = (20.0, 100.0)
D_RANGE = (0.1, 4.0)
# Held-out (w0, d) combos for validation — never sampled during training on purpose
EVAL_COMBOS = [(40.0, 1.5), (90.0, 3.0), (25.0, 0.5), (60.0, 2.5)]


class ParametricAnsatz(nn.Module):
    """u(t; w0, d) = A(t,w0,d)*cos(w*t) + B(t,w0,d)*sin(w*t), w = sqrt(w0^2 - d^2).

    The known damped frequency is embedded analytically (computed from the
    inputs), so the backbone only learns the two slowly-varying envelopes
    ``A`` and ``B`` — the same spectral-bias defeat as the single-instance
    Ansatz, generalised across the whole parameter family.

    Parameter inputs are normalised to ``[-1, 1]`` before the backbone, which
    matters for ``tanh`` conditioning given ``w0`` spans [20, 100].
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    @staticmethod
    def _normalise(t, w0, d):
        w0n = (w0 - (W0_RANGE[0] + W0_RANGE[1]) / 2) / ((W0_RANGE[1] - W0_RANGE[0]) / 2)
        dn = (d - (D_RANGE[0] + D_RANGE[1]) / 2) / ((D_RANGE[1] - D_RANGE[0]) / 2)
        return torch.cat([t, w0n, dn], dim=1)

    def forward(self, t: torch.Tensor, w0: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        w = torch.sqrt(w0**2 - d**2)
        envelopes = self.backbone(self._normalise(t, w0, d))
        return envelopes[:, 0:1] * torch.cos(w * t) + envelopes[:, 1:2] * torch.sin(w * t)


def build_model(config: dict) -> nn.Module:
    """Reconstruct the model architecture from a run config (self-describing checkpoints)."""
    backbone = PINN(
        input_dim=3,
        hidden_layers=config["hidden_layers"],
        hidden_neurons=config["hidden_neurons"],
        output_dim=2,
    )
    return ParametricAnsatz(backbone)


def exact_solution(d: float, w0: float, t: np.ndarray) -> np.ndarray:
    """Closed-form under-damped solution (d < w0), used for validation only."""
    w = np.sqrt(w0**2 - d**2)
    phi = np.arctan(-d / w)
    A = 1 / (2 * np.cos(phi))
    return np.exp(-d * t) * 2 * A * np.cos(phi + w * t)


def _sample_box(n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Uniform-random collocation points over the (t, w0, d) training box."""
    t = torch.rand(n, 1) * (T_DOMAIN[1] - T_DOMAIN[0]) + T_DOMAIN[0]
    w0 = torch.rand(n, 1) * (W0_RANGE[1] - W0_RANGE[0]) + W0_RANGE[0]
    d = torch.rand(n, 1) * (D_RANGE[1] - D_RANGE[0]) + D_RANGE[0]
    return t.to(device), w0.to(device), d.to(device)


def build_losses(n_physics: int, n_ic: int, device: torch.device) -> dict:
    """Create the named loss functions (closures own their collocation points).

    The physics residual is **normalised by k = w0^2** (i.e. the ODE is divided
    through by its stiffness term), so its magnitude is O(1) across the whole
    parameter box — without this, high-w0 samples would dominate the loss and
    hand-tuned weights would be needed.
    """
    t_p, w0_p, d_p = _sample_box(n_physics, device)
    t_p.requires_grad_(True)

    _, w0_ic, d_ic = _sample_box(n_ic, device)
    t_ic = torch.zeros(n_ic, 1, device=device, requires_grad=True)

    def pde_residual(model, t, w0, d):
        u = model(t, w0, d)
        u_t = autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        u_tt = autograd.grad(u_t, t, torch.ones_like(u_t), create_graph=True)[0]
        k = w0**2
        return u_tt / k + 2 * d * u_t / k + u  # residual / k  ->  O(1) everywhere

    def physics_loss(model):
        return torch.mean(pde_residual(model, t_p, w0_p, d_p) ** 2)

    def ic_loss(model):
        u = model(t_ic, w0_ic, d_ic)
        u_t = autograd.grad(u, t_ic, torch.ones_like(u), create_graph=True)[0]
        return torch.mean((u - 1.0) ** 2 + u_t**2)

    return {"ic": ic_loss, "physics": physics_loss}


def _member_checkpoints(run_dir: Path) -> list[Path]:
    """All ensemble member checkpoints in a run directory, member 0 first."""
    extra = sorted(run_dir.glob("checkpoint_[0-9]*.pt"))
    return [run_dir / "checkpoint.pt", *extra]


def ensemble_predict(
    models: list[nn.Module], t: torch.Tensor, w0: float, d: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std of member predictions at fixed (w0, d) over time points t."""
    w0_t = torch.full_like(t, w0)
    d_t = torch.full_like(t, d)
    with torch.no_grad():
        stack = torch.stack([m(t, w0_t, d_t) for m in models])  # (K, N, 1)
    return stack.mean(dim=0).cpu().numpy(), stack.std(dim=0).cpu().numpy()


def evaluate(models: list[nn.Module], device: torch.device) -> dict:
    """Rel-L2 error of the (ensemble-mean) prediction at held-out (w0, d) combos."""
    t_test = torch.linspace(*T_DOMAIN, 300).view(-1, 1).to(device)
    t_np = t_test.cpu().numpy()

    metrics: dict = {}
    errors = []
    for w0, d in EVAL_COMBOS:
        mean, _std = ensemble_predict(models, t_test, w0, d)
        u_exact = exact_solution(d, w0, t_np)
        rel_l2 = float(np.linalg.norm(mean - u_exact) / np.linalg.norm(u_exact))
        metrics[f"rel_l2_w0={w0:g}_d={d:g}"] = rel_l2
        errors.append(rel_l2)
    metrics["rel_l2_mean_heldout"] = float(np.mean(errors))
    return metrics


def solve_parametric_harmonic(
    epochs: int = 40000,
    lr: float = 1e-3,
    hidden_neurons: int = 64,
    hidden_layers: int = 4,
    n_physics: int = 10000,
    ensemble: int = 1,
    seed: int = 42,
    output_dir: str | None = None,
    show: bool = True,
) -> dict:
    """Train, evaluate, and persist a parametric harmonic PINN run.

    With ``ensemble > 1``, trains N independent members (seeds ``seed + i``)
    saved as ``checkpoint.pt``, ``checkpoint_1.pt``, ... Prediction then uses
    the member mean with a +/-2 sigma epistemic-uncertainty band.

    Returns:
        The metrics dict (also saved as ``metrics.json``).
    """
    run_dir, device = init_run(EXPERIMENT, output_dir, seed)

    config = {
        "epochs": epochs, "lr": lr, "hidden_neurons": hidden_neurons,
        "hidden_layers": hidden_layers, "n_physics": n_physics,
        "ensemble": ensemble, "seed": seed,
        "w0_range": list(W0_RANGE), "d_range": list(D_RANGE),
    }
    logger.info("Config: {}", config)

    models: list[nn.Module] = []
    metrics: dict = {}
    for member in range(ensemble):
        member_seed = seed + member
        set_seed(member_seed)
        logger.info("Training ensemble member {}/{} (seed {})", member + 1, ensemble, member_seed)

        model = build_model(config)
        loss_functions = build_losses(n_physics=n_physics, n_ic=200, device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        trainer = PINNTrainer(model, device=device)
        trainer.train(n_epochs=epochs, optimizer=optimizer, loss_functions=loss_functions, save_best=run_dir / ("best_model.pt" if member == 0 else f"best_model_{member}.pt"))

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

    # Evaluate the (ensemble-mean) prediction at held-out parameter combos
    metrics.update(evaluate(models, device))
    save_metrics({"config": config, "metrics": metrics}, run_dir)

    summary = {
        "Final Loss (member 0)": f"{metrics['final_total_loss']:.4e}",
        "Ensemble Members": str(ensemble),
        "Mean Rel-L2 (held-out combos)": f"{metrics['rel_l2_mean_heldout']:.4e}",
        "Epochs Run": str(metrics["epochs_run"]),
        "Artifacts": str(run_dir),
    }
    for w0, d in EVAL_COMBOS:
        summary[f"Rel-L2 @ w0={w0:g}, d={d:g}"] = f"{metrics[f'rel_l2_w0={w0:g}_d={d:g}']:.4e}"
    print_summary("Training Summary", summary)
    return metrics


@app.command()
def train(
    epochs: int = typer.Option(40000, "--epochs", "-e", help="Epochs per ensemble member."),
    lr: float = typer.Option(1e-3, "--lr", help="Learning rate."),
    neurons: int = typer.Option(64, "--neurons", "-n", help="Neurons per hidden layer."),
    layers: int = typer.Option(4, "--layers", "-l", help="Number of hidden layers."),
    n_physics: int = typer.Option(10000, "--n-physics", help="Collocation points in the (t,w0,d) box."),
    ensemble: int = typer.Option(1, "--ensemble", help="Number of ensemble members (>1 enables uncertainty bands)."),
    seed: int = typer.Option(42, "--seed", help="Base random seed (member i uses seed+i)."),
    output_dir: str | None = typer.Option(
        None, "--output-dir", "-o",
        help="Artifact directory (default: outputs/parametric_harmonic/<timestamp>).",
    ),
    show: bool = typer.Option(True, "--show/--no-show", help="Display plots interactively."),
):
    """Train a parametric PINN over the (w0, d) family — optionally as a deep ensemble."""
    show_banner("PINN", "Parametric Damped Harmonic Oscillator (solution family)")
    solve_parametric_harmonic(
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
    w0: float = typer.Option(55.0, "--w0", help="Natural frequency of the instance to solve."),
    d: float = typer.Option(2.0, "--damping", "-d", help="Damping coefficient of the instance."),
    run: str | None = typer.Option(
        None, "--run", "-r",
        help="Run directory containing checkpoint(s) (default: latest run).",
    ),
    n_points: int = typer.Option(300, "--n-points", help="Number of evaluation points."),
    show: bool = typer.Option(True, "--show/--no-show", help="Display plots interactively."),
):
    """Solve a NEW problem instance (w0, d) in milliseconds — no retraining.

    With an ensemble run, plots the member mean with a +/-2 sigma uncertainty
    band. Writes predictions.npz and prediction.png into the run directory.
    """
    from pinn import setup_logging

    setup_logging()
    run_dir = Path(run) if run else find_latest_run(EXPERIMENT)
    device = get_device()

    if not (W0_RANGE[0] <= w0 <= W0_RANGE[1]) or not (D_RANGE[0] <= d <= D_RANGE[1]):
        logger.warning(
            "(w0={}, d={}) lies OUTSIDE the trained box w0 in {}, d in {} — "
            "this is parameter-space extrapolation; the result is unreliable.",
            w0, d, W0_RANGE, D_RANGE,
        )

    models = []
    for path in _member_checkpoints(run_dir):
        model, _config = load_model(run_dir, build_model, device, checkpoint_name=path.name)
        models.append(model)
    logger.info("Loaded {} ensemble member(s) from {}", len(models), run_dir)

    t_test = torch.linspace(*T_DOMAIN, n_points).view(-1, 1).to(device)
    t_np = t_test.cpu().numpy()
    mean, std = ensemble_predict(models, t_test, w0, d)
    u_exact = exact_solution(d, w0, t_np)
    rel_l2 = float(np.linalg.norm(mean - u_exact) / np.linalg.norm(u_exact))

    np.savez(run_dir / "predictions.npz", t=t_np, u_mean=mean, u_std=std, u_exact=u_exact,
             w0=w0, d=d)
    logger.info("Predictions saved to {}", run_dir / "predictions.npz")

    plt.figure(figsize=(10, 6))
    plt.plot(t_np, u_exact, "k-", linewidth=2, alpha=0.9, label="Exact Solution")
    plt.plot(t_np, mean, "r--", linewidth=2, alpha=0.8, label="PINN (ensemble mean)")
    if len(models) > 1:
        plt.fill_between(
            t_np[:, 0], (mean - 2 * std)[:, 0], (mean + 2 * std)[:, 0],
            alpha=0.3, color="orange", label="±2σ (ensemble)",
        )
    plt.xlabel("t")
    plt.ylabel("u(t)")
    plt.title(f"Never-trained instance: w0={w0:g}, d={d:g}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(run_dir / "prediction.png", dpi=300, bbox_inches="tight")
    logger.info("Plot saved to {}", run_dir / "prediction.png")
    if show:
        plt.show()
    else:
        plt.close()

    summary = {
        "Run": str(run_dir),
        "Instance": f"w0={w0:g}, d={d:g}",
        "Ensemble Members": str(len(models)),
        "Relative L2 Error": f"{rel_l2:.4e}",
    }
    if len(models) > 1:
        summary["Max ±2σ Band Width"] = f"{float(4 * std.max()):.4e}"
    print_summary("Prediction Summary", summary)


@app.command()
def compare():
    """Rank all runs of this experiment by their recorded metrics."""
    compare_runs(EXPERIMENT)


if __name__ == "__main__":
    app()
