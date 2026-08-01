import matplotlib

matplotlib.use("Agg")  # headless backend for all CLI/plotting tests

import pytest
from pinn import set_seed


@pytest.fixture(autouse=True)
def _seeded():
    """Every test starts from the same RNG state."""
    set_seed(0)
