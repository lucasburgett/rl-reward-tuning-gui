"""Test environment wrappers and basic environment functionality."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest


@pytest.mark.timeout(30)  # type: ignore[misc]
def test_shapes_time_limit_and_no_nans() -> None:
    """Test obs/act shapes correct, time-limit truncation works, no NaNs."""
    env = gym.make("CartPole-v1", max_episode_steps=5)  # force quick truncation
    try:
        obs, info = env.reset(seed=0)

        # Check observation space and shapes
        assert obs.shape == env.observation_space.shape
        assert env.observation_space.shape == (4,)
        assert hasattr(env.action_space, "n") and env.action_space.n == 2

        # Step a few times with random actions; check finiteness
        total_steps = 0
        terminated = truncated = False

        while not (terminated or truncated):
            action = env.action_space.sample()
            assert isinstance(action, int | np.integer)

            obs, reward, terminated, truncated, info = env.step(action)
            total_steps += 1

            assert np.all(np.isfinite(obs)), "NaN/Inf in observation"
            assert np.isfinite(float(reward)), "NaN/Inf in reward"

        # Because of max_episode_steps=5, we expect truncation to trigger
        assert truncated is True and terminated is False, (
            f"Expected truncation after 5 steps, got terminated={terminated}, "
            f"truncated={truncated} after {total_steps} steps"
        )
        assert total_steps == 5

    finally:
        env.close()  # type: ignore[no-untyped-call]
