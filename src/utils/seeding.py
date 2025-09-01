# src/utils/seeding.py
from __future__ import annotations

import os
import random


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set seeds for Python, NumPy, and PyTorch (CPU + CUDA) and enable deterministic flags.

    Args:
        seed: Non-negative integer seed.
        deterministic: If True, enable deterministic algorithms and CuDNN settings.
    """
    if seed is None:
        return

    # Python hash seed for true determinism across dict/set iteration
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Python & NumPy
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    # PyTorch
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            # May reduce performance but improves reproducibility
            torch.use_deterministic_algorithms(True, warn_only=True)
            try:
                import torch.backends.cudnn as cudnn

                cudnn.deterministic = True
                cudnn.benchmark = False
            except Exception:
                pass
        else:
            # Default fast path
            try:
                import torch.backends.cudnn as cudnn

                cudnn.deterministic = False
                cudnn.benchmark = True
            except Exception:
                pass
    except Exception:
        pass


def version_banner() -> str:
    """Return a one-line banner with Python, torch, and gymnasium versions."""
    import platform

    py = platform.python_version()
    try:
        import torch

        tv = torch.__version__
    except Exception:
        tv = "not-installed"

    try:
        import gymnasium as gym

        gv = gym.__version__
    except Exception:
        gv = "not-installed"

    return f"[Versions] Python {py} | torch {tv} | gymnasium {gv}"
