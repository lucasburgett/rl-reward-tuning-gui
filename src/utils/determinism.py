# src/utils/determinism.py
"""Determinism utilities for reproducible RL experiments."""

from __future__ import annotations

import importlib
import os
import platform
import random
import sys
import warnings

import numpy as np
import torch


def _get_version(pkg: str) -> str:
    """Get package version safely."""
    try:
        module = importlib.import_module(pkg)
        return str(module.__version__)
    except Exception:
        return "not installed"


def configure_determinism(seed: int, deterministic: bool) -> None:
    """
    Configure deterministic settings for reproducible experiments.

    Args:
        seed: Random seed to use across all libraries
        deterministic: Whether to enable strict deterministic algorithms

    Note:
        When deterministic=True and CUDA is available, some operations may
        be slower but will be fully reproducible. Set CUBLAS_WORKSPACE_CONFIG
        environment variable for operations requiring workspace.
    """
    # Set environment variables first
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Set seeds for all random number generators
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Configure PyTorch deterministic algorithms
    if deterministic:
        # Enable deterministic algorithms with warning for unsupported ops
        torch.use_deterministic_algorithms(True)

        # Configure CuDNN for deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Set CUBLAS workspace config for deterministic CUDA operations
        if torch.cuda.is_available():
            # Use smaller workspace size by default; may need :4096:8 for some ops
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

        # Warn about potential performance impact
        if torch.cuda.is_available():
            warnings.warn(
                "Deterministic mode enabled with CUDA. This may significantly "
                "reduce performance but ensures reproducible results.",
                UserWarning,
                stacklevel=2,
            )
    else:
        # Disable deterministic algorithms for better performance
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def print_determinism_report(
    seed: int | None = None, deterministic: bool | None = None
) -> None:
    """
    Print comprehensive determinism and environment report.

    Args:
        seed: Seed value to display (optional override)
        deterministic: Deterministic flag to display (optional override)
    """
    # System information
    py_version = sys.version.split()[0]
    platform_info = platform.platform()

    # Package versions
    torch_version = torch.__version__
    cuda_available = torch.cuda.is_available()
    cuda_version = torch.version.cuda if cuda_available else "N/A"
    gym_version = _get_version("gymnasium")
    mujoco_version = _get_version("mujoco")

    # Environment variables
    cublas_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG", "unset")
    pythonhashseed = os.environ.get("PYTHONHASHSEED", "unset")

    # PyTorch deterministic state
    deterministic_algos: str | bool = "unknown"
    cudnn_deterministic: str | bool = "unknown"
    cudnn_benchmark: str | bool = "unknown"

    try:
        deterministic_algos = torch.are_deterministic_algorithms_enabled()
        cudnn_deterministic = torch.backends.cudnn.deterministic
        cudnn_benchmark = torch.backends.cudnn.benchmark
    except Exception:
        pass

    # CUDA device info
    cuda_device_info = "N/A"
    if cuda_available:
        try:
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            cuda_device_info = f"{device_count} devices, current: {device_name}"
        except Exception:
            cuda_device_info = "available but info unavailable"

    print(
        "[Determinism Report]\n"
        f"  seed: {seed}\n"
        f"  deterministic: {deterministic}\n"
        f"  python: {py_version}\n"
        f"  platform: {platform_info}\n"
        f"  torch: {torch_version}\n"
        f"  torch_deterministic_algorithms: {deterministic_algos}\n"
        f"  cudnn_deterministic: {cudnn_deterministic}\n"
        f"  cudnn_benchmark: {cudnn_benchmark}\n"
        f"  cuda_available: {cuda_available}\n"
        f"  cuda_version: {cuda_version}\n"
        f"  cuda_devices: {cuda_device_info}\n"
        f"  gymnasium: {gym_version}\n"
        f"  mujoco: {mujoco_version}\n"
        f"  CUBLAS_WORKSPACE_CONFIG: {cublas_config}\n"
        f"  PYTHONHASHSEED: {pythonhashseed}"
    )
