"""PPO Agent wrapper for Stable-Baselines3 and CleanRL implementations."""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO as SB3PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)

from src.wrappers import NormalizeObs, RewardComponentsLogger


class PPOAgent:
    """PPO Agent wrapper with support for SB3 and CleanRL implementations."""

    def __init__(
        self,
        env_id: str,
        seed: int,
        device: str,
        log_dir: str,
        cfg: dict[str, Any],
        impl: str = "sb3",
        env_cfg: dict[str, Any] | None = None,
    ):
        """
        Initialize PPO agent.

        Args:
            env_id: Gymnasium environment ID (e.g., "CartPole-v1")
            seed: Random seed for reproducibility
            device: Device to use ("cpu", "cuda", or "auto")
            log_dir: Directory for logs and checkpoints
            cfg: Configuration dictionary with PPO hyperparameters
            impl: Implementation to use ("sb3" or "cleanrl")
            env_cfg: Environment configuration for wrappers
        """
        self.env_id = env_id
        self.seed = seed
        self.device = device if device != "auto" else "cpu"
        self.log_dir = Path(log_dir)
        self.cfg = cfg
        self.impl = impl
        self.env_cfg = env_cfg or {}

        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.log_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Create training environment
        self.env = self._create_env(render_mode=None)

        # Create evaluation environment (separate instance for proper evaluation)
        self.eval_env = self._create_env(render_mode=None)

        # Initialize agent
        if impl == "sb3":
            self._init_sb3_agent()
        elif impl == "cleanrl":
            self._init_cleanrl_agent()
        else:
            raise ValueError(f"Unknown implementation: {impl}")

    def _create_env(self, render_mode: str | None = None) -> gym.Env[Any, Any]:
        """Create a Gymnasium environment with proper seeding and wrappers."""
        env = gym.make(self.env_id, render_mode=render_mode)
        env.reset(seed=self.seed)

        # Apply wrappers based on env_cfg
        # TimeLimit wrapper
        max_steps = self.env_cfg.get("max_episode_steps")
        if max_steps:
            env = TimeLimit(env, max_episode_steps=int(max_steps))

        # Normalize observations wrapper
        norm_cfg = self.env_cfg.get("normalize_obs", {})
        if norm_cfg.get("enabled", False):
            clip = float(norm_cfg.get("clip", 5.0))
            epsilon = float(norm_cfg.get("epsilon", 1e-8))
            env = NormalizeObs(env, clip=clip, epsilon=epsilon)

        # Reward components logging wrapper
        rc_cfg = self.env_cfg.get("reward_components", {})
        if rc_cfg.get("enabled", False):
            csv_path = rc_cfg.get("csv_path")
            if csv_path:
                # Resolve ${log_dir} template
                csv_path = str(csv_path).replace("${log_dir}", str(self.log_dir))
            env = RewardComponentsLogger(env, csv_path=csv_path)

        return env

    def _init_sb3_agent(self) -> None:
        """Initialize Stable-Baselines3 PPO agent."""
        # Map config keys to SB3 parameter names
        sb3_kwargs = {
            "policy": "MlpPolicy",
            "env": self.env,
            "learning_rate": self.cfg.get("learning_rate", 3e-4),
            "n_steps": self.cfg.get("rollout_len", 2048),
            "batch_size": self.cfg.get("rollout_len", 2048)
            // self.cfg.get("num_minibatches", 4),
            "n_epochs": self.cfg.get("update_epochs", 4),
            "gamma": self.cfg.get("gamma", 0.99),
            "gae_lambda": self.cfg.get("gae_lambda", 0.95),
            "clip_range": self.cfg.get("clip_ratio", 0.2),
            "ent_coef": self.cfg.get("entropy_coef", 0.0),
            "vf_coef": self.cfg.get("value_coef", 0.5),
            "max_grad_norm": self.cfg.get("max_grad_norm", 0.5),
            "seed": self.seed,
            "device": self.device,
            "verbose": 1,
        }

        # Add policy kwargs if provided
        if "policy_kwargs" in self.cfg and self.cfg["policy_kwargs"]:
            sb3_kwargs["policy_kwargs"] = self.cfg["policy_kwargs"]

        self.model = SB3PPO(**sb3_kwargs)

    def _init_cleanrl_agent(self) -> None:
        """Initialize CleanRL PPO agent (placeholder for now)."""
        raise NotImplementedError(
            "CleanRL implementation is not yet available. "
            "TODO: Integrate CleanRL PPO training loop as a function."
        )

    def train(self, total_timesteps: int, checkpoint_every: int = 10000) -> None:
        """
        Train the agent and save periodic checkpoints.

        Args:
            total_timesteps: Total number of environment steps to train
            checkpoint_every: Save checkpoint every N steps
        """
        if self.impl != "sb3":
            raise NotImplementedError(f"Training not implemented for {self.impl}")

        # Setup callbacks
        callbacks: list[BaseCallback] = []

        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_every,
            save_path=str(self.checkpoint_dir),
            name_prefix="ckpt_step",
        )
        callbacks.append(checkpoint_callback)

        # Evaluation callback (optional, for best model saving)
        eval_freq = self.cfg.get("eval_every", 20000)
        if eval_freq > 0:
            eval_callback = EvalCallback(
                eval_env=self.eval_env,
                best_model_save_path=str(self.checkpoint_dir),
                log_path=str(self.log_dir),
                eval_freq=eval_freq,
                deterministic=True,
                render=False,
                n_eval_episodes=10,
            )
            callbacks.append(eval_callback)

        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=False,  # Disable progress bar to avoid dependency issues
        )

        # Save final model
        self.save(str(self.checkpoint_dir / "final_model.zip"))

    def save(self, path: str) -> None:
        """Save the trained model."""
        if self.impl == "sb3":
            self.model.save(path)
        else:
            raise NotImplementedError(f"Save not implemented for {self.impl}")

    def load(self, path: str) -> None:
        """Load a trained model."""
        if self.impl == "sb3":
            self.model = SB3PPO.load(path, env=self.env)
        else:
            raise NotImplementedError(f"Load not implemented for {self.impl}")

    def evaluate(
        self,
        n_episodes: int,
        deterministic: bool = True,
        record_video: bool = False,
        video_dir: str | None = None,
    ) -> dict[str, float]:
        """
        Evaluate the agent and optionally record videos.

        Args:
            n_episodes: Number of episodes to evaluate
            deterministic: Whether to use deterministic policy
            record_video: Whether to record evaluation videos
            video_dir: Directory to save videos (if record_video=True)

        Returns:
            Dictionary with evaluation metrics
        """
        if self.impl != "sb3":
            raise NotImplementedError(f"Evaluation not implemented for {self.impl}")

        # Create evaluation environment
        if record_video and video_dir:
            from gymnasium.wrappers import RecordVideo

            os.makedirs(video_dir, exist_ok=True)
            eval_env = gym.make(self.env_id, render_mode="rgb_array")
            eval_env = RecordVideo(
                eval_env,
                video_folder=video_dir,
                episode_trigger=lambda ep: True,
                name_prefix="eval",
            )
        else:
            eval_env = gym.make(self.env_id)

        eval_env.reset(seed=self.seed)

        episode_returns = []

        for _episode in range(n_episodes):
            obs, _ = eval_env.reset()
            episode_return = 0.0
            done = False

            while not done:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                episode_return += float(reward)
                done = terminated or truncated

            episode_returns.append(episode_return)

        eval_env.close()  # type: ignore[no-untyped-call]

        return {
            "mean_return": float(np.mean(episode_returns)),
            "std_return": float(np.std(episode_returns)),
        }

    @staticmethod
    def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
        """Find the latest checkpoint by modification time."""
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            return None

        # Look for checkpoint files
        pattern = str(checkpoint_path / "ckpt_step_*.zip")
        checkpoints = glob.glob(pattern)

        # Also check for best_model.zip and final_model.zip
        for name in ["best_model.zip", "final_model.zip"]:
            candidate = checkpoint_path / name
            if candidate.exists():
                checkpoints.append(str(candidate))

        if not checkpoints:
            return None

        # Return the most recently modified checkpoint
        latest = max(checkpoints, key=os.path.getmtime)
        return latest
