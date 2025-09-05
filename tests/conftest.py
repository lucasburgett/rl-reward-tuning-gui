"""Common fixtures for testing."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

# Keep math single-threaded and deterministic-ish
os.environ.setdefault("PYTHONHASHSEED", "0")


@pytest.fixture(scope="session")  # type: ignore[misc]
def seed() -> int:
    return 123


@pytest.fixture(scope="session")  # type: ignore[misc]
def fast_ppo_cfg() -> dict[str, Any]:
    """Minimal PPO hyperparams that learn CartPole quickly in tests."""
    return {
        "learning_rate": 3e-4,
        "rollout_len": 512,
        "num_minibatches": 4,
        "update_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_ratio": 0.2,
        "entropy_coef": 0.0,
        "value_coef": 0.5,
        "max_grad_norm": 0.5,
        "normalize_adv": True,
        "checkpoint_every": 2000,
        "eval_every": 0,  # Disable eval callback in tests
        "policy_kwargs": {},
    }


@pytest.fixture()  # type: ignore[misc]
def tmp_log_dir(tmp_path: Path, seed: int) -> Path:
    """Create temporary log directory for test artifacts."""
    d = tmp_path / "artifacts" / "cartpole" / "ppo" / str(seed) / "test-run"
    d.mkdir(parents=True, exist_ok=True)
    return d
