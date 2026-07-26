"""
Reproducibility utilities.

Import set_seed() and call it once at the top of any script that trains,
evaluates, or benchmarks the model, BEFORE any dataloader/model is created.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Do NOT set torch.backends.cudnn.deterministic=True here for the
    # benchmarking scripts — it can change kernel selection and distort
    # latency measurements. Use plain seeding for benchmarks; add
    # deterministic mode separately only in accuracy-reproduction runs
    # if you need bit-exact reproducibility rather than fast kernels.


def worker_init_fn(worker_id: int) -> None:
    """Pass to DataLoader(worker_init_fn=...) for reproducible shuffling
    across multiple dataloader workers."""
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)
