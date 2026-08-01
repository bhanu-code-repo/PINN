"""Convergence regression tests — the suite's real safety net.

These train small PINNs on problems with known solutions and assert accuracy.
They catch subtle breakage (wrong signs, broken autograd wiring, loss
mis-weighting) that unit tests cannot. Marked ``slow``; run with:

    uv run pytest -m slow
"""

import numpy as np
import pytest
import torch
from pinn import PINN, PINNTrainer, set_seed

pytestmark = pytest.mark.slow

CPU = torch.device("cpu")


def test_exponential_decay_converges():
    """u' = -u, u(0) = 1  ->  u(t) = exp(-t), rel-L2 under 5%."""
    set_seed(0)
    model = PINN(input_dim=1, hidden_layers=2, hidden_neurons=16)
    t = torch.linspace(0, 1, 50).view(-1, 1).requires_grad_(True)

    def physics_loss(m):
        u = m(t)
        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        return torch.mean((u_t + u) ** 2)

    def ic_loss(m):
        return (m(torch.zeros(1, 1)) - 1.0).pow(2).squeeze()

    trainer = PINNTrainer(model, device=CPU)
    trainer.train(
        n_epochs=2000,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-2),
        loss_functions={"physics": physics_loss, "ic": ic_loss},
        weights={"physics": 1.0, "ic": 10.0},
        verbose=False,
        log_every=0,
    )

    t_test = torch.linspace(0, 1, 100).view(-1, 1)
    with torch.no_grad():
        u_pred = model(t_test).numpy()
    u_exact = np.exp(-t_test.numpy())
    rel_l2 = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)
    assert rel_l2 < 0.05, f"relative L2 error {rel_l2:.3f} exceeds 5%"


def test_harmonic_ansatz_pipeline_loss_drops():
    """Low-frequency harmonic oscillator: total loss must fall by 100x.

    Exercises the full experiment pipeline (Ansatz model + build_losses)
    without the cost of a full high-frequency run.
    """
    from experiments.harmonic_oscillator.train import build_losses, build_model

    set_seed(0)
    w0, d = 5.0, 0.5
    config = {"hidden_layers": 2, "hidden_neurons": 16}
    model = build_model(config)
    losses = build_losses(mu=2 * d, k=w0**2, t_domain=(0.0, 1.0),
                          n_collocation=50, device=CPU)

    trainer = PINNTrainer(model, device=CPU)
    history = trainer.train(
        n_epochs=1500,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-2),
        loss_functions=losses,
        weights={"ic": 1.0, "physics": 1e-2},
        verbose=False,
        log_every=0,
    )
    assert history[-1]["total"] < history[0]["total"] * 1e-2, (
        f"loss only fell from {history[0]['total']:.3e} to {history[-1]['total']:.3e}"
    )


# ── Navier-Stokes convergence tests ───────────────────────────


def test_taylor_green_loss_drops():
    """Taylor-Green vortex: total loss must fall by 10x in 2000 epochs.

    Exercises the full NS pipeline (momentum + continuity residual, IC, periodic
    BCs) with a small network. The exact solution provides a ground truth, but
    we only assert loss reduction here — accuracy requires longer training.
    """
    from experiments.taylor_green.train import build_losses, build_model

    set_seed(0)
    config = {"hidden_layers": 2, "hidden_neurons": 16}
    model = build_model(config)
    losses = build_losses(n_physics=200, nu=0.01, device=CPU)

    trainer = PINNTrainer(model, device=CPU)
    history = trainer.train(
        n_epochs=2000,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        loss_functions=losses,
        verbose=False,
        log_every=0,
    )
    assert history[-1]["total"] < history[0]["total"] * 0.1, (
        f"loss only fell from {history[0]['total']:.3e} to {history[-1]['total']:.3e}"
    )


def test_lid_driven_cavity_loss_drops():
    """Lid-driven cavity: total loss must fall by 10x in 2000 epochs.

    Exercises the steady NS pipeline with hard-encoded wall BCs (mask-based)
    and the lid driving condition.
    """
    from experiments.lid_driven_cavity.train import build_losses, build_model

    set_seed(0)
    config = {"hidden_layers": 2, "hidden_neurons": 16}
    model = build_model(config)
    losses = build_losses(n_physics=200, re=100.0, device=CPU)

    trainer = PINNTrainer(model, device=CPU)
    history = trainer.train(
        n_epochs=2000,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        loss_functions=losses,
        verbose=False,
        log_every=0,
    )
    assert history[-1]["total"] < history[0]["total"] * 0.1, (
        f"loss only fell from {history[0]['total']:.3e} to {history[-1]['total']:.3e}"
    )


def test_navier_stokes_inverse_re_moves_toward_truth():
    """Inverse NS (Kovasznay): Re must move closer to truth in 3000 epochs.

    The model starts with Re_init=10 and the true value is Re=20. After
    training, the inferred Re should be closer to 20 than the initial guess.
    We don't require high accuracy — just that the gradient signal through
    the physics loss is correctly driving Re toward the true value.
    """
    from experiments.navier_stokes_inverse.train import (
        InverseNavierStokesPINN,
        build_losses,
        generate_observations,
    )

    set_seed(0)
    re_true = 20.0
    re_init = 10.0

    # Generate synthetic noisy observations from the exact Kovasznay solution
    x_obs, y_obs, u_obs, v_obs = generate_observations(
        n_obs=100, re_true=re_true, noise=0.01, seed=0,
    )

    model = InverseNavierStokesPINN(
        hidden_layers=3, hidden_neurons=32,
        log_re_init=np.log(re_init),
    )
    losses = build_losses(
        n_physics=500,
        x_obs=x_obs, y_obs=y_obs, u_obs=u_obs, v_obs=v_obs,
        device=CPU,
    )

    trainer = PINNTrainer(model, device=CPU)
    trainer.train(
        n_epochs=5000,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        loss_functions=losses,
        weights={"data": 10.0, "physics": 1.0},
        verbose=False,
        log_every=0,
    )

    re_inferred = model.re.item()
    initial_error = abs(re_init - re_true)
    final_error = abs(re_inferred - re_true)
    assert final_error < initial_error, (
        f"Re did not move toward truth: initial error {initial_error:.2f}, "
        f"final error {final_error:.2f} (Re inferred = {re_inferred:.2f})"
    )
