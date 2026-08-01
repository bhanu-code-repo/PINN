import random

import numpy as np
import torch
from pinn import PINN, set_seed


def test_same_seed_same_model_init():
    set_seed(123)
    w1 = PINN(1, 2, 8).network[0].weight.detach().clone()
    set_seed(123)
    w2 = PINN(1, 2, 8).network[0].weight.detach().clone()
    assert torch.equal(w1, w2)


def test_different_seed_different_model_init():
    set_seed(123)
    w1 = PINN(1, 2, 8).network[0].weight.detach().clone()
    set_seed(456)
    w2 = PINN(1, 2, 8).network[0].weight.detach().clone()
    assert not torch.equal(w1, w2)


def test_seeds_all_generators():
    set_seed(7)
    values1 = (random.random(), np.random.rand(), torch.rand(1).item())
    set_seed(7)
    values2 = (random.random(), np.random.rand(), torch.rand(1).item())
    assert values1 == values2
