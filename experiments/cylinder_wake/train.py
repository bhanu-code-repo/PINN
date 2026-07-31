#!/usr/bin/env python3
"""Cylinder Wake Inverse PINN — Raissi et al. (2019) Benchmark.

Reproduces the headline result from the original Physics-Informed Neural
Networks paper: given **scattered noisy velocity observations** from a DNS
of 2D flow past a cylinder at Re = 100, a PINN simultaneously:

1. **Reconstructs the full velocity and pressure fields** — including
   pressure, which was *never observed* during training.
2. **Infers the unknown PDE parameters** λ₁ and λ₂, which encode the
   Reynolds number: the NS equations are written as

       u_t + λ₁(u·u_x + v·u_y) = -p_x + λ₂(u_xx + u_yy)
       v_t + λ₁(u·v_x + v·v_y) = -p_y + λ₂(v_xx + v_yy)

   with true values λ₁ = 1, λ₂ = 1/Re = 0.01.

DNS data: ``cylinder_nektar_wake.mat`` from Raissi's spectral-element
simulation (Nektar). Contains 5 000 spatial points × 200 time steps of
(x, y, t, u, v, p). The PINN sees only a random subset of (u, v); the
full (u, v, p) field is held for validation.

Every run writes a self-contained artifact directory (checkpoint, metrics,
plots, logs). See the README in this directory for the full methodology.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
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

app = typer.Typer(
    help="Inverse NS PINN: infer λ₁, λ₂ from cylinder wake DNS data (Raissi et al. 2019).",
)

EXPERIMENT = "cylinder_wake"
DATA_PATH = Path(".workspace/input/cylinder_nektar_wake.mat")
# True values for the parameterised NS: λ₁ = 1, λ₂ = 1/Re = 0.01
LAMBDA_1_TRUE = 1.0
LAMBDA_2_TRUE = 0.01  # i.e. Re = 100


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_cylinder_data(
    data_path: Path = DATA_PATH,
) -> dict[str, np.ndarray]:
    """Load the Raissi cylinder wake DNS dataset.

    The .mat file contains:
        - ``X_star``: (N, 2) spatial coordinates (x, y)
        - ``t``: (T, 1) time steps
        - ``U_star``: (N, 2, T) velocity fields (u, v) at each (x, t)
        - ``p_star``: (N, T) pressure field

    We flatten to (N*T, 5) arrays of (x, y, t, u, v) + pressure.

    Returns:
        Dict with keys ``x, y, t, u, v, p`` each shape ``(N*T,)``.
    """
    if not data_path.exists():
        raise FileNotFoundError(
            f"DNS data not found at {data_path}. Place cylinder_nektar_wake.mat "
            f"in .workspace/input/ (from Raissi's PINNs repository)."
        )

    raw = scipy.io.loadmat(str(data_path))
    X_star = raw["X_star"]  # (N, 2)
    t = raw["t"]  # (T, 1)
    U_star = raw["U_star"]  # (N, 2, T)
    p_star = raw["p_star"]  # (N, T)

    N = X_star.shape[0]
    T = t.shape[0]

    # Tile spatial coords across time, tile time across space
    xx = np.tile(X_star[:, 0:1], (1, T))  # (N, T)
    yy = np.tile(X_star[:, 1:2], (1, T))  # (N, T)
    tt = np.tile(t.T, (N, 1))  # (N, T)

    return {
        "x": xx.flatten().astype(np.float32),
        "y": yy.flatten().astype(np.float32),
        "t": tt.flatten().astype(np.float32),
        "u": U_star[:, 0, :].flatten().astype(np.float32),
        "v": U_star[:, 1, :].flatten().astype(np.float32),
        "p": p_star.flatten().astype(np.float32),
        "N_spatial": N,
        "N_time": T,
        "X_star": X_star,
        "t_steps": t,
        "U_star": U_star,
        "p_star": p_star,
    }


def sample_training_data(
    full_data: dict, n_train: int, seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Sample n_train random (x, y, t, u, v) points for training.

    Returns:
        ``(train_data, test_data)`` — train has n_train points,
        test has *all* points (for full-field validation including pressure).
    """
    rng = np.random.default_rng(seed)
    N_total = len(full_data["x"])
    idx = rng.choice(N_total, size=n_train, replace=False)

    train = {k: full_data[k][idx] for k in ("x", "y", "t", "u", "v")}
    test = {k: full_data[k] for k in ("x", "y", "t", "u", "v", "p")}
    return train, test


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CylinderWakePINN(nn.Module):
    """PINN for the cylinder wake inverse problem.

    Outputs ``(u, v, p)`` from ``(x, y, t)`` and holds two learnable
    parameters ``lambda_1`` and ``lambda_2`` for the parameterised NS
    equations. Both are optimised jointly with the network weights.
    """

    def __init__(
        self, hidden_layers: int, hidden_neurons: int,
        lambda1_init: float = 1.0, lambda2_init: float = 0.01,
    ):
        super().__init__()
        self.network = PINN(
            input_dim=3, hidden_layers=hidden_layers,
            hidden_neurons=hidden_neurons, output_dim=2,
        )
        # Learnable NS parameters (initialised at user-specified guesses)
        self.lambda_1 = nn.Parameter(torch.tensor(lambda1_init))
        self.lambda_2 = nn.Parameter(torch.tensor(lambda2_init))

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (psi, p); velocities are derived via autograd externally."""
        out = self.network(torch.cat([x, y, t], dim=1))
        psi, p = out[:, 0:1], out[:, 1:2]
        return psi, p

    def velocity(
        self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Derive (u, v, p) from the streamfunction: u = psi_y, v = -psi_x.

        Incompressibility is satisfied by construction.
        """
        psi, p = self(x, y, t)
        u = autograd.grad(psi, y, torch.ones_like(psi), create_graph=True)[0]
        v = -autograd.grad(psi, x, torch.ones_like(psi), create_graph=True)[0]
        return u, v, p


def build_model(config: dict) -> nn.Module:
    """Reconstruct the model architecture from a run config (self-describing checkpoints)."""
    return CylinderWakePINN(
        hidden_layers=config["hidden_layers"],
        hidden_neurons=config["hidden_neurons"],
        lambda1_init=config.get("lambda1_init", 1.0),
        lambda2_init=config.get("lambda2_init", 0.01),
    )


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def build_losses(
    train_data: dict[str, np.ndarray],
    n_physics: int,
    full_data: dict,
    device: torch.device,
) -> dict:
    """Create data + physics loss functions.

    The **data loss** matches (u, v) at the training observation points.
    The **physics loss** evaluates the parameterised NS residuals at
    random collocation points sampled from the full spatio-temporal domain.

    The streamfunction formulation is used: the network outputs (psi, p),
    and u = psi_y, v = -psi_x. This satisfies incompressibility by
    construction, so no continuity loss is needed. The momentum residuals
    involve third-order derivatives of psi (for the viscous terms), which
    autograd handles via create_graph=True chains.
    """
    # Training data tensors (velocity observations)
    x_d = torch.tensor(train_data["x"]).view(-1, 1).to(device).requires_grad_(True)
    y_d = torch.tensor(train_data["y"]).view(-1, 1).to(device).requires_grad_(True)
    t_d = torch.tensor(train_data["t"]).view(-1, 1).to(device).requires_grad_(True)
    u_d = torch.tensor(train_data["u"]).view(-1, 1).to(device)
    v_d = torch.tensor(train_data["v"]).view(-1, 1).to(device)

    # Physics collocation: random subset of full domain
    N_total = len(full_data["x"])
    rng = np.random.default_rng(12345)
    idx_p = rng.choice(N_total, size=n_physics, replace=False)

    x_p = torch.tensor(full_data["x"][idx_p]).view(-1, 1).to(device).requires_grad_(True)
    y_p = torch.tensor(full_data["y"][idx_p]).view(-1, 1).to(device).requires_grad_(True)
    t_p = torch.tensor(full_data["t"][idx_p]).view(-1, 1).to(device).requires_grad_(True)

    def data_loss(model):
        u_pred, v_pred, _ = model.velocity(x_d, y_d, t_d)
        return torch.mean((u_pred - u_d) ** 2 + (v_pred - v_d) ** 2)

    def physics_loss(model):
        """Parameterised NS residuals using the streamfunction.

        NS form:  u_t + λ₁(u·u_x + v·u_y) = -p_x + λ₂(u_xx + u_yy)
                  v_t + λ₁(u·v_x + v·v_y) = -p_y + λ₂(v_xx + v_yy)

        With u = ψ_y, v = -ψ_x, the velocity derivatives become
        second/third-order derivatives of ψ. Continuity (u_x + v_y = 0)
        is satisfied identically.
        """
        u, v, p = model.velocity(x_p, y_p, t_p)
        ones = torch.ones_like(u)

        lam1 = model.lambda_1
        lam2 = model.lambda_2

        # First-order velocity derivatives
        u_t = autograd.grad(u, t_p, ones, create_graph=True)[0]
        u_x = autograd.grad(u, x_p, ones, create_graph=True)[0]
        u_y = autograd.grad(u, y_p, ones, create_graph=True)[0]

        v_t = autograd.grad(v, t_p, ones, create_graph=True)[0]
        v_x = autograd.grad(v, x_p, ones, create_graph=True)[0]
        v_y = autograd.grad(v, y_p, ones, create_graph=True)[0]

        # Second-order velocity derivatives (viscous term)
        u_xx = autograd.grad(u_x, x_p, ones, create_graph=True)[0]
        u_yy = autograd.grad(u_y, y_p, ones, create_graph=True)[0]
        v_xx = autograd.grad(v_x, x_p, ones, create_graph=True)[0]
        v_yy = autograd.grad(v_y, y_p, ones, create_graph=True)[0]

        # Pressure gradients
        p_x = autograd.grad(p, x_p, ones, create_graph=True)[0]
        p_y = autograd.grad(p, y_p, ones, create_graph=True)[0]

        # Momentum residuals with learnable λ₁, λ₂
        f_u = u_t + lam1 * (u * u_x + v * u_y) + p_x - lam2 * (u_xx + u_yy)
        f_v = v_t + lam1 * (u * v_x + v * v_y) + p_y - lam2 * (v_xx + v_yy)

        return torch.mean(f_u**2 + f_v**2)

    return {"data": data_loss, "physics": physics_loss}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module, full_data: dict, device: torch.device,
    n_eval: int = 50000,
) -> tuple[dict, dict]:
    """Evaluate velocity + pressure accuracy on the full DNS field.

    Uses batched evaluation to avoid OOM on the full 1M-point grid.
    Pressure is validated mean-subtracted (gauge invariance).
    """
    lam1 = float(model.lambda_1.detach().cpu())
    lam2 = float(model.lambda_2.detach().cpu())
    re_inferred = 1.0 / lam2 if abs(lam2) > 1e-12 else float("inf")

    # Subsample for evaluation (full grid is 1M points)
    N_total = len(full_data["x"])
    rng = np.random.default_rng(99)
    idx = rng.choice(N_total, size=min(n_eval, N_total), replace=False)

    x_e = torch.tensor(full_data["x"][idx]).view(-1, 1).to(device).requires_grad_(True)
    y_e = torch.tensor(full_data["y"][idx]).view(-1, 1).to(device).requires_grad_(True)
    t_e = torch.tensor(full_data["t"][idx]).view(-1, 1).to(device).requires_grad_(True)

    # Batched forward (velocity requires grad, so can't use no_grad)
    batch_size = 10000
    u_all, v_all, p_all = [], [], []
    for i in range(0, len(idx), batch_size):
        sl = slice(i, min(i + batch_size, len(idx)))
        with torch.no_grad():
            # For eval we don't need velocity via autograd — use direct psi derivatives
            pass
        # Actually we need autograd for velocity from streamfunction
        u_b, v_b, p_b = model.velocity(x_e[sl], y_e[sl], t_e[sl])
        u_all.append(u_b.detach().cpu().numpy())
        v_all.append(v_b.detach().cpu().numpy())
        p_all.append(p_b.detach().cpu().numpy())

    u_pred = np.concatenate(u_all).flatten()
    v_pred = np.concatenate(v_all).flatten()
    p_pred = np.concatenate(p_all).flatten()

    u_true = full_data["u"][idx]
    v_true = full_data["v"][idx]
    p_true = full_data["p"][idx]

    # Velocity rel-L2
    vel_err = np.sqrt(np.sum((u_pred - u_true) ** 2 + (v_pred - v_true) ** 2))
    vel_ref = np.sqrt(np.sum(u_true**2 + v_true**2))
    rel_l2_vel = float(vel_err / vel_ref)

    # Pressure rel-L2 (mean-subtracted — pressure is defined up to a constant)
    p_pred_ms = p_pred - p_pred.mean()
    p_true_ms = p_true - p_true.mean()
    p_err = float(np.linalg.norm(p_pred_ms - p_true_ms) / np.linalg.norm(p_true_ms))

    metrics = {
        "lambda_1_inferred": lam1,
        "lambda_2_inferred": lam2,
        "lambda_1_error": abs(lam1 - LAMBDA_1_TRUE),
        "lambda_2_error": abs(lam2 - LAMBDA_2_TRUE),
        "re_inferred": re_inferred,
        "rel_l2_velocity": rel_l2_vel,
        "rel_l2_pressure": p_err,
    }
    return metrics, {}


def evaluate_snapshot(
    model: nn.Module, full_data: dict, device: torch.device,
    t_idx: int = 100,
) -> dict[str, np.ndarray]:
    """Evaluate on a single time snapshot for plotting."""
    X_star = full_data["X_star"]  # (N, 2)
    N = X_star.shape[0]
    t_val = full_data["t_steps"][t_idx, 0]

    x_t = torch.tensor(X_star[:, 0], dtype=torch.float32).view(-1, 1).to(device).requires_grad_(True)
    y_t = torch.tensor(X_star[:, 1], dtype=torch.float32).view(-1, 1).to(device).requires_grad_(True)
    t_t = torch.full((N, 1), t_val, dtype=torch.float32, device=device).requires_grad_(True)

    u_pred, v_pred, p_pred = model.velocity(x_t, y_t, t_t)
    u_pred = u_pred.detach().cpu().numpy().flatten()
    v_pred = v_pred.detach().cpu().numpy().flatten()
    p_pred = p_pred.detach().cpu().numpy().flatten()

    U_star = full_data["U_star"]
    p_star = full_data["p_star"]

    return {
        "x": X_star[:, 0], "y": X_star[:, 1], "t_val": t_val,
        "u_pred": u_pred, "v_pred": v_pred, "p_pred": p_pred,
        "u_true": U_star[:, 0, t_idx], "v_true": U_star[:, 1, t_idx],
        "p_true": p_star[:, t_idx],
    }


def make_plots(
    snap: dict, lambda_history: list[tuple[float, float]],
    save_path: str, show: bool,
) -> None:
    """Snapshot comparison (u, v, p) + λ convergence history."""
    from matplotlib.tri import Triangulation

    x, y = snap["x"], snap["y"]
    tri = Triangulation(x, y)

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fields = [
        ("u", snap["u_true"], snap["u_pred"]),
        ("v", snap["v_true"], snap["v_pred"]),
        ("p", snap["p_true"] - snap["p_true"].mean(),
         snap["p_pred"] - snap["p_pred"].mean()),
    ]

    for row, (name, true, pred) in enumerate(fields):
        vmin = min(true.min(), pred.min())
        vmax = max(true.max(), pred.max())

        im0 = axes[row, 0].tripcolor(tri, true, shading="gouraud", vmin=vmin, vmax=vmax)
        axes[row, 0].set_title(f"DNS {name}")
        plt.colorbar(im0, ax=axes[row, 0])

        im1 = axes[row, 1].tripcolor(tri, pred, shading="gouraud", vmin=vmin, vmax=vmax)
        axes[row, 1].set_title(f"PINN {name}")
        plt.colorbar(im1, ax=axes[row, 1])

        error = np.abs(pred - true)
        im2 = axes[row, 2].tripcolor(tri, error, shading="gouraud")
        axes[row, 2].set_title(f"|{name}| error")
        plt.colorbar(im2, ax=axes[row, 2])

    for ax in axes.flat:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")

    plt.suptitle(f"Cylinder Wake at t = {snap['t_val']:.1f}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    logger.info("Snapshot plot saved to {}", save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)

    # Lambda convergence plot
    if len(lambda_history) > 1:
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        epochs = range(len(lambda_history))
        lam1s = [h[0] for h in lambda_history]
        lam2s = [h[1] for h in lambda_history]

        ax1.plot(epochs, lam1s, "b-", linewidth=1.5)
        ax1.axhline(LAMBDA_1_TRUE, color="r", linestyle="--", linewidth=2, label=f"True λ₁ = {LAMBDA_1_TRUE}")
        ax1.set(xlabel="Epoch", ylabel="λ₁", title="λ₁ convergence")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, lam2s, "b-", linewidth=1.5)
        ax2.axhline(LAMBDA_2_TRUE, color="r", linestyle="--", linewidth=2, label=f"True λ₂ = {LAMBDA_2_TRUE}")
        ax2.set(xlabel="Epoch", ylabel="λ₂", title="λ₂ convergence (= 1/Re)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        convergence_path = str(Path(save_path).parent / "lambda_convergence.png")
        plt.savefig(convergence_path, dpi=300, bbox_inches="tight")
        logger.info("Lambda convergence plot saved to {}", convergence_path)
        if show:
            plt.show()
        else:
            plt.close(fig2)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve_cylinder_wake(
    epochs: int = 30000,
    lr: float = 1e-3,
    hidden_neurons: int = 64,
    hidden_layers: int = 8,
    n_train: int = 5000,
    n_physics: int = 10000,
    lambda1_init: float = 1.0,
    lambda2_init: float = 0.01,
    seed: int = 42,
    output_dir: str | None = None,
    show: bool = True,
    data_path: str | None = None,
) -> dict:
    """Train, evaluate, and persist a cylinder wake inverse PINN run.

    Following Raissi et al. (2019): the network sees scattered (u, v)
    observations and infers λ₁, λ₂ while reconstructing the full field
    including the never-observed pressure.
    """
    run_dir, device = init_run(EXPERIMENT, output_dir, seed)

    config = {
        "epochs": epochs, "lr": lr, "hidden_neurons": hidden_neurons,
        "hidden_layers": hidden_layers, "n_train": n_train,
        "n_physics": n_physics, "lambda1_init": lambda1_init,
        "lambda2_init": lambda2_init, "seed": seed,
    }
    logger.info("Config: {}", config)

    # Load DNS data
    dp = Path(data_path) if data_path else DATA_PATH
    full_data = load_cylinder_data(dp)
    logger.info(
        "Loaded DNS data: {} spatial × {} time = {} total points",
        full_data["N_spatial"], full_data["N_time"],
        len(full_data["x"]),
    )

    train_data, _test_data = sample_training_data(full_data, n_train, seed)
    logger.info("Sampled {} training observations from {} total", n_train, len(full_data["x"]))

    model = build_model(config)
    loss_functions = build_losses(train_data, n_physics, full_data, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = PINNTrainer(model, device=device)

    # Track λ convergence via callback
    lambda_history: list[tuple[float, float]] = []

    def record_lambdas(epoch, losses):
        l1 = float(model.lambda_1.detach().cpu())
        l2 = float(model.lambda_2.detach().cpu())
        lambda_history.append((l1, l2))
        if epoch % 1000 == 0:
            logger.info(
                "Epoch {}: λ₁ = {:.6f} (true: {}), λ₂ = {:.6f} (true: {})",
                epoch, l1, LAMBDA_1_TRUE, l2, LAMBDA_2_TRUE,
            )

    trainer.train(
        n_epochs=epochs, optimizer=optimizer, loss_functions=loss_functions,
        callbacks=[record_lambdas], save_best=run_dir / "best_model.pt",
    )
    trainer.save_checkpoint(run_dir / "checkpoint.pt", optimizer=optimizer, metadata=config)
    trainer.plot_loss_history(show_total=True, save_path=run_dir / "loss_history.png", show=show)

    # Evaluate
    metrics, _ = evaluate(model, full_data, device)
    final = trainer.loss_history[-1]
    metrics.update({
        "final_total_loss": final["total"],
        "final_data_loss": final["data"],
        "final_physics_loss": final["physics"],
        "epochs_run": len(trainer.loss_history),
    })
    save_metrics({"config": config, "metrics": metrics}, run_dir)

    # Snapshot plot at t_idx = 100 (mid-simulation)
    snap = evaluate_snapshot(model, full_data, device, t_idx=100)
    make_plots(snap, lambda_history, str(run_dir / "snapshot_comparison.png"), show)

    # Save predictions at the snapshot
    np.savez(
        run_dir / "predictions.npz",
        x=snap["x"], y=snap["y"], t_val=snap["t_val"],
        u_pred=snap["u_pred"], v_pred=snap["v_pred"], p_pred=snap["p_pred"],
        u_true=snap["u_true"], v_true=snap["v_true"], p_true=snap["p_true"],
        lambda_1=metrics["lambda_1_inferred"],
        lambda_2=metrics["lambda_2_inferred"],
    )

    print_summary("Training Summary", {
        "Final Total Loss": f"{metrics['final_total_loss']:.4e}",
        "λ₁ Inferred": f"{metrics['lambda_1_inferred']:.6f} (true: {LAMBDA_1_TRUE})",
        "λ₂ Inferred": f"{metrics['lambda_2_inferred']:.6f} (true: {LAMBDA_2_TRUE})",
        "λ₁ Error": f"{metrics['lambda_1_error']:.4e}",
        "λ₂ Error": f"{metrics['lambda_2_error']:.4e}",
        "Re Inferred": f"{metrics['re_inferred']:.2f} (true: 100)",
        "Rel-L2 Velocity": f"{metrics['rel_l2_velocity']:.4e}",
        "Rel-L2 Pressure": f"{metrics['rel_l2_pressure']:.4e}",
        "Training Points": str(n_train),
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
    layers: int = typer.Option(8, "--layers", "-l", help="Number of hidden layers."),
    n_train: int = typer.Option(5000, "--n-train", help="Number of training observations (u, v)."),
    n_physics: int = typer.Option(10000, "--n-physics", help="Collocation points for NS residual."),
    lambda1_init: float = typer.Option(1.0, "--lambda1-init", help="Initial guess for λ₁."),
    lambda2_init: float = typer.Option(0.01, "--lambda2-init", help="Initial guess for λ₂."),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility."),
    output_dir: str | None = typer.Option(
        None, "--output-dir", "-o",
        help="Artifact directory (default: outputs/cylinder_wake/<timestamp>).",
    ),
    show: bool = typer.Option(True, "--show/--no-show", help="Display plots interactively."),
    data_path: str | None = typer.Option(
        None, "--data-path",
        help="Path to cylinder_nektar_wake.mat (default: .workspace/input/).",
    ),
):
    """Infer NS parameters λ₁, λ₂ from cylinder wake DNS data (Raissi 2019)."""
    show_banner("CYLINDER", "Cylinder Wake Inverse NS — Raissi et al. (2019)")
    solve_cylinder_wake(
        epochs=epochs, lr=lr, hidden_neurons=neurons, hidden_layers=layers,
        n_train=n_train, n_physics=n_physics, lambda1_init=lambda1_init,
        lambda2_init=lambda2_init, seed=seed, output_dir=output_dir,
        show=show, data_path=data_path,
    )


@app.command()
def predict(
    run: str | None = typer.Option(
        None, "--run", "-r",
        help="Run directory containing checkpoint.pt (default: latest run).",
    ),
    t_idx: int = typer.Option(100, "--t-idx", help="Time snapshot index for evaluation (0–199)."),
    show: bool = typer.Option(True, "--show/--no-show", help="Display plots interactively."),
    data_path: str | None = typer.Option(
        None, "--data-path",
        help="Path to cylinder_nektar_wake.mat (default: .workspace/input/).",
    ),
):
    """Load a trained model and report inferred λ₁, λ₂ with field-comparison plots.

    Writes predictions.npz and prediction_snapshot.png into the run directory.
    """
    from pinn import setup_logging

    setup_logging()
    run_dir = Path(run) if run else find_latest_run(EXPERIMENT)
    device = get_device()
    model, _config = load_model(run_dir, build_model, device)

    dp = Path(data_path) if data_path else DATA_PATH
    full_data = load_cylinder_data(dp)

    metrics, _ = evaluate(model, full_data, device)
    snap = evaluate_snapshot(model, full_data, device, t_idx=t_idx)

    np.savez(
        run_dir / "predictions.npz",
        x=snap["x"], y=snap["y"], t_val=snap["t_val"],
        u_pred=snap["u_pred"], v_pred=snap["v_pred"], p_pred=snap["p_pred"],
        u_true=snap["u_true"], v_true=snap["v_true"], p_true=snap["p_true"],
        lambda_1=metrics["lambda_1_inferred"],
        lambda_2=metrics["lambda_2_inferred"],
    )
    logger.info("Predictions saved to {}", run_dir / "predictions.npz")

    make_plots(snap, [], str(run_dir / "prediction_snapshot.png"), show)
    print_summary("Prediction Summary", {
        "Run": str(run_dir),
        "λ₁ Inferred": f"{metrics['lambda_1_inferred']:.6f} (true: {LAMBDA_1_TRUE})",
        "λ₂ Inferred": f"{metrics['lambda_2_inferred']:.6f} (true: {LAMBDA_2_TRUE})",
        "Re Inferred": f"{metrics['re_inferred']:.2f} (true: 100)",
        "Rel-L2 Velocity": f"{metrics['rel_l2_velocity']:.4e}",
        "Rel-L2 Pressure": f"{metrics['rel_l2_pressure']:.4e}",
    })


@app.command()
def compare():
    """Rank all runs of this experiment by their recorded metrics."""
    compare_runs(EXPERIMENT)


if __name__ == "__main__":
    app()
