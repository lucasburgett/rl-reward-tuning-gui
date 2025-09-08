# src/train.py
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from src.agents import PPOAgent
from src.utils.logger import CSVLogger, create_training_metrics_row
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
        # Create artifacts directory structure
        run_id = f"{datetime.now():%Y-%m-%d/%H-%M-%S}"
        env_name = (
            cfg.env
            if isinstance(cfg.env, str)
            else getattr(cfg.env, "id", "unknown").split("-")[0].lower()
        )
        algo_name = getattr(cfg.algo, "algo_name", cfg.get("algo", "ppo"))
        artifacts_dir = (
            Path("artifacts") / env_name / algo_name / str(cfg.seed) / run_id
        )
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (artifacts_dir / "checkpoints").mkdir(exist_ok=True)
        (artifacts_dir / "videos").mkdir(exist_ok=True)
        (artifacts_dir / "logs").mkdir(exist_ok=True)
        (artifacts_dir / "config").mkdir(exist_ok=True)

        # Save composed config
        with open(artifacts_dir / "config" / "train.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(cfg))

        print(f"[Artifacts] Saving to: {artifacts_dir}")

        # Resolve device
        device = cfg.device if cfg.device != "auto" else "cpu"
        print(f"[Device] Using: {device}")

        # Get environment config
        if hasattr(cfg, "env") and hasattr(cfg.env, "id"):
            env_id = cfg.env.id
        elif hasattr(cfg, "env") and hasattr(cfg.env, "env_id"):
            env_id = cfg.env.env_id
        else:
            # Fallback - map common env names
            env_map = {"cartpole": "CartPole-v1", "lunarlander": "LunarLander-v3"}
            env_id = env_map.get(cfg.env, cfg.env)

        # Initialize CSV logger for training metrics
        train_logger = CSVLogger(artifacts_dir / "logs" / "train_metrics.csv")

        # Optional W&B setup
        wandb_run = None
        if cfg.get("use_wandb", False):
            try:
                import wandb

                wandb_config = cfg.get("wandb", {})
                wandb_run = wandb.init(
                    project=wandb_config.get("project", "rl-template"),
                    entity=wandb_config.get("entity"),
                    group=wandb_config.get("group", f"{env_name}-{algo_name}"),
                    job_type=wandb_config.get("job_type", "train"),
                    mode=wandb_config.get("mode", "online"),
                    config=OmegaConf.to_container(cfg, resolve=True),
                    name=f"{env_name}-{algo_name}-seed{cfg.seed}",
                )
                print(f"[W&B] Initialized run: {wandb_run.name}")
            except ImportError:
                print("[W&B] wandb not installed, skipping W&B logging")
            except Exception as e:
                print(f"[W&B] Failed to initialize: {e}")

        # Create PPO agent with artifacts directory
        print(f"[Agent] Initializing PPO agent for {env_id}")

        # Pass env config for wrapper settings
        env_cfg = cfg.env if hasattr(cfg, "env") and isinstance(cfg.env, dict) else {}

        agent = PPOAgent(
            env_id=env_id,
            seed=cfg.seed,
            device=device,
            log_dir=str(artifacts_dir),
            cfg=cfg.algo,
            impl="sb3",
            env_cfg=env_cfg,
        )

        # Train the agent with periodic evaluation and logging
        print(f"[Training] Starting training for {cfg.total_steps} steps...")
        checkpoint_every = cfg.algo.get("checkpoint_every", 10000)
        eval_every = cfg.algo.get("eval_every", 20000)

        # Note: SB3 automatically logs episode metrics, no need for explicit Monitor wrapper

        start_time = time.time()

        # Simple training loop with periodic evaluation
        timesteps_done = 0
        eval_idx = 0

        while timesteps_done < cfg.total_steps:
            # Train for eval_every steps or remaining steps
            steps_to_train = min(eval_every, cfg.total_steps - timesteps_done)

            # Train
            agent.model.learn(
                total_timesteps=steps_to_train,
                reset_num_timesteps=False,
                progress_bar=False,
            )

            timesteps_done += steps_to_train

            # Periodic evaluation and logging
            if timesteps_done % eval_every == 0 or timesteps_done >= cfg.total_steps:
                wall_time = time.time() - start_time

                print(f"\n[Eval] Evaluating at {timesteps_done} timesteps...")
                eval_results = agent.evaluate(
                    n_episodes=10, deterministic=True, record_video=False
                )

                # Log training metrics
                metrics_row = create_training_metrics_row(
                    global_step=timesteps_done,
                    timesteps=timesteps_done,
                    mean_return=eval_results["mean_return"],
                    std_return=eval_results["std_return"],
                    wall_time_s=wall_time,
                )
                train_logger.log(metrics_row)

                # Log to W&B if enabled
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "timesteps": timesteps_done,
                            "eval/mean_return": eval_results["mean_return"],
                            "eval/std_return": eval_results["std_return"],
                            "time/wall_time_s": wall_time,
                        },
                        step=timesteps_done,
                    )

                print(
                    f"[Eval] Mean return: {eval_results['mean_return']:.2f} ± {eval_results['std_return']:.2f}"
                )
                eval_idx += 1

            # Save checkpoint
            if timesteps_done % checkpoint_every == 0:
                checkpoint_path = (
                    artifacts_dir / "checkpoints" / f"ckpt_step_{timesteps_done}.zip"
                )
                agent.save(str(checkpoint_path))
                print(f"[Checkpoint] Saved: {checkpoint_path}")

        # Save final model
        final_path = artifacts_dir / "checkpoints" / "final_model.zip"
        agent.save(str(final_path))
        print(f"[Training] Complete! Final model saved to: {final_path}")

        # Final evaluation with video recording if configured
        if cfg.get("eval", {}).get("run_after_train", True):
            print("\n[Final Evaluation]")
            eval_episodes = cfg.eval.get("n_episodes", 10)
            record_video = cfg.eval.get("record_video", False)

            results = agent.evaluate(
                n_episodes=eval_episodes,
                deterministic=True,
                record_video=record_video,
                video_dir=str(artifacts_dir / "videos") if record_video else None,
            )

            print(f"Final evaluation ({eval_episodes} episodes):")
            print(
                f"Mean return: {results['mean_return']:.2f} ± {results['std_return']:.2f}"
            )

            # Log final evaluation
            from src.utils.logger import create_eval_metrics_row

            eval_logger = CSVLogger(artifacts_dir / "logs" / "eval_metrics.csv")
            eval_row = create_eval_metrics_row(
                eval_idx=0,
                n_episodes=eval_episodes,
                mean_return=results["mean_return"],
                std_return=results["std_return"],
                video_written=record_video,
                checkpoint_path=str(final_path),
            )
            eval_logger.log(eval_row)

            # Log final results to W&B
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "final_eval/mean_return": results["mean_return"],
                        "final_eval/std_return": results["std_return"],
                        "final_eval/episodes": eval_episodes,
                    }
                )

                # Upload artifacts to W&B if configured
                try:
                    # Upload final checkpoint
                    wandb_run.save(str(final_path))
                    # Upload CSV logs
                    wandb_run.save(str(artifacts_dir / "logs" / "*.csv"))
                    # Upload sample videos if they exist
                    if record_video:
                        video_files = list((artifacts_dir / "videos").glob("*.mp4"))
                        for video_file in video_files[:2]:  # Upload first 2 videos
                            wandb_run.save(str(video_file))
                except Exception as e:
                    print(f"[W&B] Failed to upload artifacts: {e}")

            # Check target scores
            if env_id == "CartPole-v1" and results["mean_return"] >= 475.0:
                print("Successfully achieved ≥475/500 on CartPole!")
            elif env_id == "LunarLander-v3" and results["mean_return"] >= 200.0:
                print("Successfully achieved ≥200 on LunarLander!")

        print("Training complete!")

        # Clean up W&B
        if wandb_run is not None:
            wandb_run.finish()

    except Exception as e:
        print(f"Training failed: {e}")
        # Clean up W&B on error
        try:
            if "wandb_run" in locals() and wandb_run is not None:
                wandb_run.finish()
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
