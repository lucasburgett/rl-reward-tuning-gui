from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

import gymnasium as gym
import numpy as np


class RunningMeanStd:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        x = x.astype(np.float64)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0] if x.ndim > 1 else 1.0
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / tot_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * (self.count * batch_count / tot_count)
        new_var = M2 / tot_count
        self.mean, self.var, self.count = new_mean, new_var, tot_count


class NormalizeObs(gym.ObservationWrapper):  # type: ignore[type-arg]
    """
    Running mean/std normalization for observations.
    Applies y = clip((x - mean) / sqrt(var + eps), [-clip, clip]).
    """

    def __init__(
        self,
        env: gym.Env[Any, Any],
        clip: float = 5.0,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__(env)
        assert isinstance(
            env.observation_space, gym.spaces.Box
        ), "NormalizeObs requires Box observations."
        shape = env.observation_space.shape
        assert shape is not None
        self.rms = RunningMeanStd(shape)
        self.clip = float(clip)
        self.epsilon = float(epsilon)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32)
        # Update stats per-step using current obs
        self.rms.update(obs)
        norm = (obs - self.rms.mean) / np.sqrt(self.rms.var + self.epsilon)
        if self.clip is not None:
            norm = np.clip(norm, -self.clip, self.clip)
        return norm.astype(np.float32)  # type: ignore[no-any-return]
