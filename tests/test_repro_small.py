"""Test small reproducible training runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from src.agents.ppo import PPOAgent
from src.utils.seeding import set_seed


@pytest.mark.timeout(90)  # type: ignore[misc]
def test_small_cpu_train_hits_threshold(
    seed: int, fast_ppo_cfg: dict[str, Any], tmp_log_dir: Path
) -> None:
    """Test that 5k steps on CPU completes and average return >= threshold."""
    set_seed(seed, deterministic=True)
    torch.set_num_threads(1)

    agent = PPOAgent(
        env_id="CartPole-v1",
        seed=seed,
        device="cpu",
        log_dir=str(tmp_log_dir),
        cfg=fast_ppo_cfg,
        impl="sb3",
    )

    # Train for a small number of steps
    agent.train(total_timesteps=5_000, checkpoint_every=5_000)

    # Evaluate performance
    result = agent.evaluate(n_episodes=5, deterministic=True, record_video=False)

    mean_ret = result["mean_return"]
    assert np.isfinite(mean_ret), f"Expected finite return, got {mean_ret}"
    assert mean_ret >= 100.0, (
        f"Expected >= 100 average return after 5k steps, got {mean_ret}. "
        f"This may indicate training instability or insufficient steps."
    )
