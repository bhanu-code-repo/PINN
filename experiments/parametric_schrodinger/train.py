#!/usr/bin/env python3
"""Parametric Schrödinger PINN training CLI — the fundamental soliton family.

Demonstrates how to train a **complex-valued parametric PINN**. The network
takes ``(x, t, A)`` and learns the nonlinear Schrödinger soliton family

    i*h_t + 0.5*h_xx + |h|^2*h = 0,   h(0, x) = A*sech(A*x)

over ``A in [0.75, 2]``. This IC is chosen deliberately: every member has the
**closed-form solution** ``h(t,x) = A*sech(A*x)*exp(i*A^2*t/2)``, so the whole
family is rigorously validatable — unlike the general amplitude family
``A*sech(x)`` (breathers), which remains deferred (see
docs/parametric_pinns.md).

The known phase rotation is embedded analytically in the ansatz:

    h(x, t; A) = W(x, t, A) * exp(i*A^2*t/2)

so the exact target for the backbone is ``W = A*sech(A*x)`` — real and
time-independent. The network only learns a nearly-static envelope.

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

app = typer.Typer(help="Train a parametric PINN for the NLS soliton family.")

EXPERIMENT = "parametric_schrodinger"
X_DOMAIN = (-5.0, 5.0)
T_DOMAIN = (0.0, float(np.pi / 2))
A_RANGE = (0.75, 2.0)
# Held-out amplitudes for validation — never sampled explicitly during training
EVAL_AS = [0.9, 1.3, 1.8]


class SolitonAnsatz(nn.Module):
    """h(x,t;A) = W(x,t,A) * exp(i*A^2*t/2), W complex via two backbone channels.

    Embedding the known phase rotation (Rule 3 of docs/parametric_pinns.md)
    reduces the learning target to the nearly-static envelope W = A*sech(A*x).
    Inputs are normalised to ~[-1, 1] before the backbone.
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    @staticmethod
    def _normalise(x, t, a):
        x_n = x / X_DOMAIN[1]
        t_n = 2 * (t - T_DOMAIN[0]) / (T_DOMAIN[1] - T_DOMAIN[0]) - 1
        a_n = 2 * (a - A_RANGE[0]) / (A_RANGE[1] - A_RANGE[0]) - 1
        return torch.cat([x_n, t_n, a_n], dim=1)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, a: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        w = self.backbone(self._normalise(x, t, a))
        wu, wv = w[:, 0:1], w[:, 1:2]
        theta = 0.5 * a**2 * t
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        u = wu * cos_t - wv * sin_t  # Re h
        v = wu * sin_t + wv * cos_t  # Im h
        return u, v


def build_model(config: dict) -> nn.Module:
    """Reconstruct the model architecture from a run config (self-describing checkpoints)."""
    backbone = PINN(
        input_dim=3,
        hidden_layers=config["hidden_layers"],
        hidden_neurons=config["hidden_neurons"],
        output_dim=2,
    )
    return SolitonAnsatz(backbone)


def exact_solution(a: float, t: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form fundamental soliton: h = A*sech(A*x)*exp(i*A^2*t/2) -> (Re, Im)."""
    envelope = a / np.cosh(a * x)
    theta = 0.5 * a**2 * t
    return envelope * np.cos(theta), envelope * np.sin(theta)


def _sample_a(n: int, device: torch.device) -> torch.Tensor:
    return (torch.rand(n, 1) * (A_RANGE[1] - A_RANGE[0]) + A_RANGE[0]).to(device)


def build_losses(n_physics: int, n_ic: int, n_bc: int, device: torch.device) -> dict:
    """Create the named loss functions (closures own their collocation points)."""
    # Interior: uniform (x, t), uniform A
    x_p = (torch.rand(n_physics, 1) * (X_DOMAIN[1] - X_DOMAIN[0]) + X_DOMAIN[0]).to(device)
    t_p = (torch.rand(n_physics, 1) * (T_DOMAIN[1] - T_DOMAIN[0]) + T_DOMAIN[0]).to(device)
    a_p = _sample_a(n_physics, device)
    x_p.requires_grad_(True)
    t_p.requires_grad_(True)

    # IC: t = 0, random (x, A)
    x_ic = (torch.rand(n_ic, 1) * (X_DOMAIN[1] - X_DOMAIN[0]) + X_DOMAIN[0]).to(device)
    t_ic = torch.zeros(n_ic, 1, device=device)
    a_ic = _sample_a(n_ic, device)

    # Periodic BC: random (t, A) at x = +-5
    t_bc = (torch.rand(n_bc, 1) * (T_DOMAIN[1] - T_DOMAIN[0]) + T_DOMAIN[0]).to(device)
    a_bc = _sample_a(n_bc, device)

    def pde_residual(model, x, t, a):
        u, v = model(x, t, a)
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

    def physics_loss(model):
        return pde_residual(model, x_p, t_p, a_p)

    def ic_loss(model):
        u, v = model(x_ic, t_ic, a_ic)
        h_exact = a_ic / torch.cosh(a_ic * x_ic)  # h(0,x;A) = A*sech(A*x), purely real
        return torch.mean((u - h_exact) ** 2 + v**2)

    def bc_loss(model):
        x_l = torch.full_like(t_bc, X_DOMAIN[0]).requires_grad_(True)
        x_r = torch.full_like(t_bc, X_DOMAIN[1]).requires_grad_(True)

        u_l, v_l = model(x_l, t_bc, a_bc)
        u_r, v_r = model(x_r, t_bc, a_bc)

        u_l_x = autograd.grad(u_l, x_l, torch.ones_like(u_l), create_graph=True)[0]
        v_l_x = autograd.grad(v_l, x_l, torch.ones_like(v_l), create_graph=True)[0]
        u_r_x = autograd.grad(u_r, x_r, torch.ones_like(u_r), create_graph=True)[0]
        v_r_x = autograd.grad(v_r, x_r, torch.ones_like(v_r), create_graph=True)[0]

        loss_value = torch.mean((u_l - u_r) ** 2 + (v_l - v_r) ** 2)
        loss_deriv = torch.mean((u_l_x - u_r_x) ** 2 + (v_l_x - v_r_x) ** 2)
        return loss_value + loss_deriv

    return {"ic": ic_loss, "bc": bc_loss, "physics": physics_loss}


def _member_checkpoints(run_dir: Path) -> list[Path]:
    """All ensemble member checkpoints in a run directory, member 0 first."""
    extra = sorted(run_dir.glob("checkpoint_[0-9]*.pt"))
    return [run_dir / "checkpoint.pt", *extra]


def ensemble_predict_grid(
    models: list[nn.Module], a: float, n_x: int, n_t: int, device: torch.device,
) -> dict:
    """Ensemble mean/std of (Re h, Im h, |h|) on a grid at fixed amplitude A."""
    x = torch.linspace(*X_DOMAIN, n_x).view(-1, 1).to(device)
    t = torch.linspace(*T_DOMAIN, n_t).view(-1, 1).to(device)
    X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing="ij")
    x_flat = X.flatten().unsqueeze(1)
    t_flat = T.flatten().unsqueeze(1)
    a_flat = torch.full_like(x_flat, a)

    with torch.no_grad():
        us, vs = [], []
        for m in models:
            u, v = m(x_flat, t_flat, a_flat)
            us.append(u)
            vs.append(v)
        u_stack = torch.stack(us)  # (K, N, 1)
        v_stack = torch.stack(vs)
        mag_stack = torch.sqrt(u_stack**2 + v_stack**2)

    def stats(stack):
        return (
            stack.mean(dim=0).cpu().numpy().reshape(n_x, n_t),
            stack.std(dim=0).cpu().numpy().reshape(n_x, n_t),
        )

    u_mean, u_std = stats(u_stack)
    v_mean, v_std = stats(v_stack)
    mag_mean, mag_std = stats(mag_stack)
    return {
        "X": X.cpu().numpy(), "T": T.cpu().numpy(),
        "u_mean": u_mean, "u_std": u_std,
        "v_mean": v_mean, "v_std": v_std,
        "h_mag_mean": mag_mean, "h_mag_std": mag_std,
        "x": x.cpu().numpy(), "t": t.cpu().numpy(),
    }


def rel_l2_complex(arrays: dict, a: float) -> float:
    """Rel-L2 of the full complex field (Re and Im stacked) vs the exact soliton."""
    u_exact, v_exact = exact_solution(a, arrays["T"], arrays["X"])
    pred = np.stack([arrays["u_mean"], arrays["v_mean"]])
    exact = np.stack([u_exact, v_exact])
    return float(np.linalg.norm(pred - exact) / np.linalg.norm(exact))


def evaluate(models: list[nn.Module], device: torch.device) -> dict:
    """Full-complex rel-L2 vs the exact soliton at held-out amplitudes."""
    metrics: dict = {}
    errors = []
    for a in EVAL_AS:
        arrays = ensemble_predict_grid(models, a, n_x=200, n_t=50, device=device)
        rel = rel_l2_complex(arrays, a)
        metrics[f"rel_l2_A={a:g}"] = rel
        errors.append(rel)
    metrics["rel_l2_mean_heldout"] = float(np.mean(errors))
    return metrics


def solve_parametric_schrodinger(
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
    """Train, evaluate, and persist a parametric NLS soliton PINN run.

    With ``ensemble > 1``, trains N independent members (seeds ``seed + i``)
    saved as ``checkpoint.pt``, ``checkpoint_1.pt``, ...

    Returns:
        The metrics dict (also saved as ``metrics.json``).
    """
    run_dir, device = init_run(EXPERIMENT, output_dir, seed)

    config = {
        "epochs": epochs, "lr": lr, "hidden_neurons": hidden_neurons,
        "hidden_layers": hidden_layers, "n_physics": n_physics,
        "ensemble": ensemble, "seed": seed, "a_range": list(A_RANGE),
    }
    logger.info("Config: {}", config)

    models: list[nn.Module] = []
    metrics: dict = {}
    for member in range(ensemble):
        member_seed = seed + member
        set_seed(member_seed)
        logger.info("Training ensemble member {}/{} (seed {})", member + 1, ensemble, member_seed)

        model = build_model(config)
        loss_functions = build_losses(n_physics=n_physics, n_ic=300, n_bc=200, device=device)
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

    metrics.update(evaluate(models, device))
    save_metrics({"config": config, "metrics": metrics}, run_dir)

    summary = {
        "Final Loss (member 0)": f"{metrics['final_total_loss']:.4e}",
        "Ensemble Members": str(ensemble),
        "Mean Rel-L2 (held-out A, complex)": f"{metrics['rel_l2_mean_heldout']:.4e}",
        "Epochs Run": str(metrics["epochs_run"]),
        "Artifacts": str(run_dir),
    }
    for a in EVAL_AS:
        summary[f"Rel-L2 @ A={a:g}"] = f"{metrics[f'rel_l2_A={a:g}']:.4e}"
    print_summary("Training Summary", summary)
    return metrics


@app.command()
def train(
    epochs: int = typer.Option(40000, "--epochs", "-e", help="Epochs per ensemble member."),
    lr: float = typer.Option(1e-3, "--lr", help="Learning rate."),
    neurons: int = typer.Option(64, "--neurons", "-n", help="Neurons per hidden layer."),
    layers: int = typer.Option(4, "--layers", "-l", help="Number of hidden layers."),
    n_physics: int = typer.Option(10000, "--n-physics", help="Collocation points in the (x,t,A) box."),
    ensemble: int = typer.Option(1, "--ensemble", help="Number of ensemble members (>1 enables uncertainty bands)."),
    seed: int = typer.Option(42, "--seed", help="Base random seed (member i uses seed+i)."),
    output_dir: str | None = typer.Option(
        None, "--output-dir", "-o",
        help="Artifact directory (default: outputs/parametric_schrodinger/<timestamp>).",
    ),
    show: bool = typer.Option(True, "--show/--no-show", help="Display plots interactively."),
):
    """Train a parametric PINN over the NLS soliton family — optionally as a deep ensemble."""
    show_banner("NLS", "Parametric Schrödinger — Fundamental Soliton Family")
    solve_parametric_schrodinger(
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
    amplitude: float = typer.Option(1.3, "--amplitude", "-a", help="Soliton amplitude A of the instance."),
    run: str | None = typer.Option(
        None, "--run", "-r",
        help="Run directory containing checkpoint(s) (default: latest run).",
    ),
    show: bool = typer.Option(True, "--show/--no-show", help="Display plots interactively."),
):
    """Solve a NEW soliton amplitude in milliseconds — validated against the exact solution.

    With an ensemble run, profiles include a +/-2 sigma uncertainty band.
    Writes predictions.npz, prediction_contour.png, prediction_snapshots.png.
    """
    from pinn import setup_logging

    setup_logging()
    run_dir = Path(run) if run else find_latest_run(EXPERIMENT)
    device = get_device()

    if not (A_RANGE[0] <= amplitude <= A_RANGE[1]):
        logger.warning(
            "A={} lies OUTSIDE the trained box A in [{}, {}] — "
            "this is parameter-space extrapolation; the result is unreliable.",
            amplitude, *A_RANGE,
        )

    models = []
    for path in _member_checkpoints(run_dir):
        model, _config = load_model(run_dir, build_model, device, checkpoint_name=path.name)
        models.append(model)
    logger.info("Loaded {} ensemble member(s) from {}", len(models), run_dir)

    arrays = ensemble_predict_grid(models, amplitude, n_x=200, n_t=100, device=device)
    rel = rel_l2_complex(arrays, amplitude)
    np.savez(run_dir / "predictions.npz", amplitude=amplitude, **arrays)
    logger.info("Predictions saved to {}", run_dir / "predictions.npz")

    # Contour of |h| (mean)
    plt.figure(figsize=(10, 6))
    contour = plt.contourf(arrays["T"], arrays["X"], arrays["h_mag_mean"], 20, cmap="viridis")
    plt.colorbar(contour, label="|h(t,x)|")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title(f"Never-trained soliton: A={amplitude:g}  (exact |h| = A·sech(Ax), static)")
    plt.savefig(run_dir / "prediction_contour.png", dpi=300, bbox_inches="tight")
    logger.info("Plot saved to {}", run_dir / "prediction_contour.png")
    plt.show() if show else plt.close()

    # Snapshots at t=0 and t=pi/4, |h| vs exact envelope, with sigma bands
    x_np = arrays["x"][:, 0]
    envelope_exact = amplitude / np.cosh(amplitude * x_np)
    t_indices = {0: "t = 0", arrays["T"].shape[1] // 2: "t = π/4"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (idx, title) in zip(axes, t_indices.items(), strict=True):
        mean = arrays["h_mag_mean"][:, idx]
        std = arrays["h_mag_std"][:, idx]
        ax.plot(x_np, envelope_exact, "k-", linewidth=2, alpha=0.9, label="Exact |h|")
        ax.plot(x_np, mean, "r--", linewidth=2, label="PINN (ensemble mean)")
        if len(models) > 1:
            ax.fill_between(x_np, mean - 2 * std, mean + 2 * std,
                            alpha=0.3, color="orange", label="±2σ (ensemble)")
        ax.set(title=title, xlabel="x", ylabel="|h|")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_dir / "prediction_snapshots.png", dpi=300, bbox_inches="tight")
    logger.info("Plot saved to {}", run_dir / "prediction_snapshots.png")
    plt.show() if show else plt.close(fig)

    summary = {
        "Run": str(run_dir),
        "Instance": f"A={amplitude:g}",
        "Ensemble Members": str(len(models)),
        "Relative L2 Error (complex)": f"{rel:.4e}",
    }
    if len(models) > 1:
        summary["Max ±2σ Band Width (|h|)"] = f"{float(4 * arrays['h_mag_std'].max()):.4e}"
    print_summary("Prediction Summary", summary)


@app.command()
def compare():
    """Rank all runs of this experiment by their recorded metrics."""
    compare_runs(EXPERIMENT)


if __name__ == "__main__":
    app()
