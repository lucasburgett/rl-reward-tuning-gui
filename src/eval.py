# src/eval.py
from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from src.agents import PPOAgent
from src.utils.seeding import set_seed, version_banner


def resolve_checkpoint_path(cfg: DictConfig) -> str:
    """Resolve checkpoint path from config."""
    load_checkpoint = cfg.eval.get("load_checkpoint", "latest")

    if load_checkpoint == "latest":
        # Find latest checkpoint in log_dir/checkpoints/
        checkpoint_dir = Path(cfg.log_dir) / "checkpoints"
        latest_path = PPOAgent.find_latest_checkpoint(str(checkpoint_dir))
        if latest_path is None:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        return str(latest_path)
    else:
        # Explicit path provided
        if not Path(load_checkpoint).exists():
            raise FileNotFoundError(f"Checkpoint not found: {load_checkpoint}")
        return str(load_checkpoint)


def create_agent_from_config(cfg: DictConfig, checkpoint_path: str) -> PPOAgent:
    """Create and load agent from config and checkpoint."""
    # Resolve device
    device = cfg.device if cfg.device != "auto" else "cpu"

    # Get environment config
    if hasattr(cfg, "env") and hasattr(cfg.env, "id"):
        env_id = cfg.env.id
    else:
        # Fallback - map common env names
        env_map = {"cartpole": "CartPole-v1", "lunarlander": "LunarLander-v2"}
        env_id = env_map.get(cfg.env, cfg.env)

    # Create agent with dummy config (we'll load from checkpoint)
    agent = PPOAgent(
        env_id=env_id,
        seed=cfg.seed,
        device=device,
        log_dir=cfg.log_dir,
        cfg=cfg.algo,
        impl="sb3",
    )

    # Load the checkpoint
    agent.load(checkpoint_path)

    return agent


@hydra.main(version_base=None, config_path="../configs", config_name="eval_config")
def main(cfg: DictConfig) -> None:
    """Main evaluation function."""
    # Seed + banner
    set_seed(cfg.seed, cfg.deterministic)
    print(version_banner())
    print("[Seed] set_seed:", cfg.seed, "| deterministic:", bool(cfg.deterministic))

    # Show composed config
    print("\n[Composed Config]")
    print(OmegaConf.to_yaml(cfg))

    try:
        # Resolve checkpoint path
        checkpoint_path = resolve_checkpoint_path(cfg)
        print(f"\n[Loading] Checkpoint: {checkpoint_path}")

        # Get environment ID for later use
        if hasattr(cfg, "env") and hasattr(cfg.env, "id"):
            env_id = cfg.env.id
        else:
            # Fallback - map common env names
            env_map = {"cartpole": "CartPole-v1", "lunarlander": "LunarLander-v2"}
            env_id = env_map.get(cfg.env, cfg.env)

        # Create and load agent
        agent = create_agent_from_config(cfg, checkpoint_path)

        # Setup video recording if requested
        video_dir = None
        if cfg.eval.get("record_video", False):
            video_dir = cfg.eval.get("video_dir", str(Path(cfg.log_dir) / "videos"))
            print(f"[Video] Recording to: {video_dir}")

        # Run evaluation
        print(f"\n[Evaluating] Running {cfg.eval.n_episodes} episodes...")
        results = agent.evaluate(
            n_episodes=cfg.eval.n_episodes,
            deterministic=True,
            record_video=cfg.eval.get("record_video", False),
            video_dir=video_dir,
        )

        # Print results
        print("\n[Results]")
        print(
            f"Mean return: {results['mean_return']:.2f} ± {results['std_return']:.2f}"
        )
        print(f"Episodes: {cfg.eval.n_episodes}")

        # Check if we achieved the target score for CartPole
        if env_id == "CartPole-v1" and results["mean_return"] >= 475.0:
            print("🎉 Successfully achieved ≥475/500 on CartPole!")

        print("✅ Evaluation complete.")

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
