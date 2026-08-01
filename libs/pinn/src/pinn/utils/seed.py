"""Reproducibility — seed all random number generators in one call.

Copyright 2026 Bhanu Thakur. All rights reserved.
"""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed all relevant random number generators for reproducible runs.

    Seeds Python's ``random``, NumPy, and PyTorch (CPU and all CUDA devices).

    Note:
        This makes runs *repeatable*, not bit-identical across hardware or
        library versions. For stricter determinism see
        ``torch.use_deterministic_algorithms``.

    Args:
        seed: The seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
