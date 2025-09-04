# src/train.py
from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from src.agents import PPOAgent
from src.utils.seeding import set_seed, version_banner


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Seed + banner
    set_seed(cfg.seed, cfg.deterministic)
    print(version_banner())
    print("[Seed] set_seed:", cfg.seed, "| deterministic:", bool(cfg.deterministic))

    # Show composed config (including CLI overrides like env=cartpole algo=ppo)
    print("\n[Composed Config]")
    print(OmegaConf.to_yaml(cfg))

    try:
        # Resolve device
        device = cfg.device if cfg.device != "auto" else "cpu"
        print(f"[Device] Using: {device}")

        # Get environment config
        if hasattr(cfg, "env") and hasattr(cfg.env, "id"):
            env_id = cfg.env.id
        else:
            # Fallback - map common env names
            env_map = {"cartpole": "CartPole-v1", "lunarlander": "LunarLander-v2"}
            env_id = env_map.get(cfg.env, cfg.env)

        # Create PPO agent
        print(f"[Agent] Initializing PPO agent for {env_id}")
        agent = PPOAgent(
            env_id=env_id,
            seed=cfg.seed,
            device=device,
            log_dir=cfg.log_dir,
            cfg=cfg.algo,
            impl="sb3",
        )

        # Train the agent
        print(f"[Training] Starting training for {cfg.total_steps} steps...")
        checkpoint_every = cfg.algo.get("checkpoint_every", 10000)
        agent.train(total_timesteps=cfg.total_steps, checkpoint_every=checkpoint_every)

        print(
            f"[Training] Complete! Checkpoints saved to: {Path(cfg.log_dir) / 'checkpoints'}"
        )

        # Run evaluation after training if configured
        if cfg.get("eval", {}).get("run_after_train", False):
            print("\n[Post-Training Evaluation]")
            eval_episodes = cfg.eval.get("n_episodes", 10)
            results = agent.evaluate(n_episodes=eval_episodes, deterministic=True)

            print(f"Final evaluation ({eval_episodes} episodes):")
            print(
                f"Mean return: {results['mean_return']:.2f} ± {results['std_return']:.2f}"
            )

            # Check if we achieved the target score for CartPole
            if env_id == "CartPole-v1" and results["mean_return"] >= 475.0:
                print("🎉 Successfully achieved ≥475/500 on CartPole!")

        print("✅ Training complete!")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
