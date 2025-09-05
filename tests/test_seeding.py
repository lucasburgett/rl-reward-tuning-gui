"""Test deterministic seeding and reproducibility."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from src.agents.ppo import PPOAgent
from src.utils.seeding import set_seed


@pytest.mark.timeout(90)  # type: ignore[misc]
def test_deterministic_eval_same_seed_identical_returns(
    seed: int, fast_ppo_cfg: dict[str, Any], tmp_log_dir: Path
) -> None:
    """Test that same config + seed => identical returns (tiny tolerance)."""
    set_seed(seed, deterministic=True)
    torch.set_num_threads(1)

    # Train once (small step budget) and checkpoint
    agent = PPOAgent(
        env_id="CartPole-v1",
        seed=seed,
        device="cpu",
        log_dir=str(tmp_log_dir),
        cfg=fast_ppo_cfg,
        impl="sb3",
    )
    agent.train(total_timesteps=10_000, checkpoint_every=5_000)

    # Resolve latest checkpoint
    ckpts = sorted((tmp_log_dir / "checkpoints").glob("*.zip"))
    assert ckpts, "Expected at least one checkpoint to exist"
    ckpt = str(ckpts[-1])

    # Evaluate twice with the same seed & deterministic=True
    set_seed(seed, deterministic=True)
    r1 = agent.evaluate(n_episodes=5, deterministic=True, record_video=False)

    # Reload to ensure fresh eval state
    agent2 = PPOAgent(
        env_id="CartPole-v1",
        seed=seed,
        device="cpu",
        log_dir=str(tmp_log_dir),
        cfg=fast_ppo_cfg,
        impl="sb3",
    )
    agent2.load(ckpt)
    set_seed(seed, deterministic=True)
    r2 = agent2.evaluate(n_episodes=5, deterministic=True, record_video=False)

    # Compare means with tiny tolerance
    m1, m2 = r1["mean_return"], r2["mean_return"]
    assert np.isfinite(m1) and np.isfinite(m2)
    assert abs(m1 - m2) <= 1e-6, f"Expected identical returns, got {m1} vs {m2}"
