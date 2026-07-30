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
