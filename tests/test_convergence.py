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
