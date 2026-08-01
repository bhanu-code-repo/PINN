import matplotlib

matplotlib.use("Agg")  # headless backend for all plotting tests

import pytest
import torch
from pinn import PINN, set_seed


@pytest.fixture(autouse=True)
def _seeded():
    """Every test starts from the same RNG state."""
    set_seed(0)


@pytest.fixture()
def tiny_model() -> PINN:
    return PINN(input_dim=1, hidden_layers=2, hidden_neurons=8)


@pytest.fixture()
def cpu() -> torch.device:
    return torch.device("cpu")
