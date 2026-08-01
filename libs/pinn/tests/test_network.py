"""Tests for the PINN network backbone.

Copyright 2026 Bhanu Thakur. All rights reserved.
"""

import pytest
import torch
import torch.nn as nn
from pinn import PINN


def test_output_shape_scalar_field():
    model = PINN(input_dim=1, hidden_layers=3, hidden_neurons=16)
    x = torch.rand(50, 1)
    assert model(x).shape == (50, 1)


def test_output_shape_multi_input_multi_output():
    model = PINN(input_dim=2, hidden_layers=2, hidden_neurons=16, output_dim=2)
    xt = torch.rand(50, 2)
    assert model(xt).shape == (50, 2)


def test_single_hidden_layer_edge_case():
    model = PINN(input_dim=1, hidden_layers=1, hidden_neurons=4)
    assert model(torch.rand(3, 1)).shape == (3, 1)


def test_layer_count_matches_config():
    hidden_layers = 4
    model = PINN(input_dim=1, hidden_layers=hidden_layers, hidden_neurons=8)
    linears = [m for m in model.network if isinstance(m, nn.Linear)]
    tanhs = [m for m in model.network if isinstance(m, nn.Tanh)]
    assert len(linears) == hidden_layers + 1  # hidden layers + output layer
    assert len(tanhs) == hidden_layers


def test_gradients_flow_to_inputs():
    """PINN losses differentiate outputs w.r.t. inputs — twice. Both must work."""
    model = PINN(input_dim=1, hidden_layers=2, hidden_neurons=8)
    t = torch.rand(10, 1, requires_grad=True)
    u = model(t)
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t, torch.ones_like(u_t), create_graph=True)[0]
    assert u_t.shape == t.shape
    assert u_tt.shape == t.shape
    assert not torch.isnan(u_tt).any()


def test_gradients_flow_to_parameters():
    model = PINN(input_dim=1, hidden_layers=2, hidden_neurons=8)
    loss = model(torch.rand(10, 1)).pow(2).mean()
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())


# ── Input validation ───────────────────────────────────────────

def test_invalid_input_dim():
    with pytest.raises(ValueError, match="input_dim must be >= 1"):
        PINN(input_dim=0, hidden_layers=2, hidden_neurons=8)


def test_invalid_hidden_layers():
    with pytest.raises(ValueError, match="hidden_layers must be >= 1"):
        PINN(input_dim=1, hidden_layers=0, hidden_neurons=8)


def test_invalid_hidden_neurons():
    with pytest.raises(ValueError, match="hidden_neurons must be >= 1"):
        PINN(input_dim=1, hidden_layers=2, hidden_neurons=0)


def test_invalid_output_dim():
    with pytest.raises(ValueError, match="output_dim must be >= 1"):
        PINN(input_dim=1, hidden_layers=2, hidden_neurons=8, output_dim=0)


def test_invalid_activation():
    with pytest.raises(ValueError, match="Unknown activation"):
        PINN(input_dim=1, hidden_layers=2, hidden_neurons=8, activation="relu")


# ── Activation support ─────────────────────────────────────────

def test_silu_activation():
    model = PINN(input_dim=1, hidden_layers=2, hidden_neurons=8, activation="silu")
    acts = [m for m in model.network if isinstance(m, nn.SiLU)]
    assert len(acts) == 2
    assert model(torch.rand(5, 1)).shape == (5, 1)


def test_gelu_activation():
    model = PINN(input_dim=1, hidden_layers=2, hidden_neurons=8, activation="gelu")
    acts = [m for m in model.network if isinstance(m, nn.GELU)]
    assert len(acts) == 2
    assert model(torch.rand(5, 1)).shape == (5, 1)


def test_activation_case_insensitive():
    model = PINN(input_dim=1, hidden_layers=2, hidden_neurons=8, activation="Tanh")
    assert model(torch.rand(5, 1)).shape == (5, 1)


# ── Utilities ──────────────────────────────────────────────────

def test_count_parameters():
    model = PINN(input_dim=1, hidden_layers=2, hidden_neurons=8)
    expected = sum(p.numel() for p in model.parameters())
    assert model.count_parameters() == expected
    assert model.count_parameters() > 0


def test_repr():
    model = PINN(input_dim=2, hidden_layers=3, hidden_neurons=16)
    r = repr(model)
    assert "PINN(" in r
    assert "Tanh" in r
    assert "params=" in r


def test_version():
    from pinn import __version__
    assert isinstance(__version__, str)
    assert "." in __version__
