"""Deterministic seeding.

Fixing seeds makes steering vectors, sampled generations, and metrics
reproducible. Note on nondeterminism: some GPU/MPS kernels are not bit-for-bit
reproducible even with a fixed seed. We set the deterministic flags on a
best-effort basis and document the residual nondeterminism in the README.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

DEFAULT_SEED = 0


def set_seed(seed: int = DEFAULT_SEED, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch RNGs.

    Args:
        seed: The seed applied to every RNG.
        deterministic: If True, request deterministic cuDNN/algorithm behaviour
            where the platform supports it. This can slow things down and is a
            no-op for kernels without a deterministic implementation.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def torch_generator(seed: int = DEFAULT_SEED, device: str = "cpu") -> torch.Generator:
    """Return a fresh, explicitly seeded :class:`torch.Generator`.

    Prefer this over the global RNG when a specific tensor (e.g. a random
    baseline steering vector) must be reproducible independent of surrounding
    RNG consumption.
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator
