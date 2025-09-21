# src/eval.py
from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from src.agents import PPOAgent
from src.utils.determinism import configure_determinism, print_determinism_report
from src.utils.logger import CSVLogger, create_eval_metrics_row


def resolve_artifacts_dir(cfg: DictConfig) -> Path:
    """Resolve artifacts directory based on config or explicit log_dir."""
    if hasattr(cfg, "log_dir") and cfg.log_dir:
        # Explicit log_dir provided
        return Path(cfg.log_dir)
    else:
        # Derive from env/algo/seed structure
        env_name = (
            cfg.env
            if isinstance(cfg.env, str)
            else getattr(cfg.env, "env_id", getattr(cfg.env, "id", "unknown"))
            .split("-")[0]
            .lower()
        )
        algo_name = getattr(cfg.algo, "algo_name", cfg.get("algo", "ppo"))

        # Find the latest run for this combination
        base_path = Path("artifacts") / env_name / algo_name / str(cfg.seed)
        if not base_path.exists():
            raise FileNotFoundError(
                f"No artifacts found for {env_name}/{algo_name}/seed{cfg.seed}"
            )

        # Find all timestamped run directories (recursively search through date directories)
        all_run_dirs = []
        for date_dir in base_path.iterdir():
            if date_dir.is_dir():
                # Check if this is a timestamped run directory (has checkpoints)
                if (date_dir / "checkpoints").exists():
                    all_run_dirs.append(date_dir)
                else:
                    # Look one level deeper for timestamped directories
                    for time_dir in date_dir.iterdir():
                        if time_dir.is_dir() and (time_dir / "checkpoints").exists():
                            all_run_dirs.append(time_dir)

        if not all_run_dirs:
            raise FileNotFoundError(
                f"No run directories with checkpoints found in {base_path}"
            )

        # Return the most recently modified run directory
        latest_run: Path = max(all_run_dirs, key=lambda x: x.stat().st_mtime)
        return latest_run


def resolve_checkpoint_path(cfg: DictConfig, artifacts_dir: Path) -> str:
    """Resolve checkpoint path from config."""
    load_checkpoint = cfg.eval.get("load_checkpoint", "latest")

    if load_checkpoint == "latest":
        # Find latest checkpoint in artifacts_dir/checkpoints/
        checkpoint_dir = artifacts_dir / "checkpoints"
        latest_path = PPOAgent.find_latest_checkpoint(str(checkpoint_dir))
        if latest_path is None:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        return str(latest_path)
    else:
        # Explicit path provided
        if not Path(load_checkpoint).exists():
            raise FileNotFoundError(f"Checkpoint not found: {load_checkpoint}")
        return str(load_checkpoint)


def create_agent_from_config(
    cfg: DictConfig, checkpoint_path: str, artifacts_dir: Path
) -> PPOAgent:
    """Create and load agent from config and checkpoint."""
    # Resolve device
    device = cfg.device if cfg.device != "auto" else "cpu"

    # Get environment config
    if hasattr(cfg, "env") and hasattr(cfg.env, "id"):
        env_id = cfg.env.id
    elif hasattr(cfg, "env") and hasattr(cfg.env, "env_id"):
        env_id = cfg.env.env_id
    else:
        # Fallback - map common env names
        env_map = {"cartpole": "CartPole-v1", "lunarlander": "LunarLander-v3"}
        env_id = env_map.get(cfg.env, cfg.env)

    # Pass env config for wrapper settings
    env_cfg = cfg.env if hasattr(cfg, "env") and isinstance(cfg.env, dict) else {}

    # Create agent with dummy config (we'll load from checkpoint)
    agent = PPOAgent(
        env_id=env_id,
        seed=cfg.seed,
        device=device,
        log_dir=str(artifacts_dir),
        cfg=cfg.algo,
        impl="sb3",
        env_cfg=env_cfg,
    )

    # Load the checkpoint
    agent.load(checkpoint_path)

    return agent


@hydra.main(version_base=None, config_path="../configs", config_name="eval_config")
def main(cfg: DictConfig) -> None:
    """Main evaluation function."""
    # Configure determinism and print report
    configure_determinism(cfg.seed, cfg.deterministic)
    print_determinism_report(cfg.seed, cfg.deterministic)

    # Show composed config
    print("\n[Composed Config]")
    print(OmegaConf.to_yaml(cfg))

    try:
        # Resolve artifacts directory and checkpoint path
        artifacts_dir = resolve_artifacts_dir(cfg)
        print(f"\n[Artifacts] Using directory: {artifacts_dir}")

        checkpoint_path = resolve_checkpoint_path(cfg, artifacts_dir)
        print(f"[Loading] Checkpoint: {checkpoint_path}")

        # Get environment ID for later use
        if hasattr(cfg, "env") and hasattr(cfg.env, "id"):
            env_id = cfg.env.id
        else:
            # Fallback - map common env names
            env_map = {"cartpole": "CartPole-v1", "lunarlander": "LunarLander-v2"}
            env_id = env_map.get(cfg.env, cfg.env)

        # Create and load agent
        agent = create_agent_from_config(cfg, checkpoint_path, artifacts_dir)

        # Setup video recording if requested
        video_dir = None
        if cfg.eval.get("record_video", False):
            # Auto-resolve video directory to artifacts/videos
            video_dir = str(artifacts_dir / "videos")
            (artifacts_dir / "videos").mkdir(exist_ok=True)
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

        # Log evaluation results to CSV
        (artifacts_dir / "logs").mkdir(exist_ok=True)
        eval_logger = CSVLogger(artifacts_dir / "logs" / "eval_metrics.csv")
        eval_row = create_eval_metrics_row(
            eval_idx=0,  # Single evaluation run
            n_episodes=cfg.eval.n_episodes,
            mean_return=results["mean_return"],
            std_return=results["std_return"],
            video_written=cfg.eval.get("record_video", False),
            checkpoint_path=checkpoint_path,
        )
        eval_logger.log(eval_row)
        print(
            f"[Logging] Results saved to: {artifacts_dir / 'logs' / 'eval_metrics.csv'}"
        )

        # Save eval config
        (artifacts_dir / "config").mkdir(exist_ok=True)
        with open(artifacts_dir / "config" / "eval.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(cfg))

        # Check target scores
        if env_id == "CartPole-v1" and results["mean_return"] >= 475.0:
            print("Successfully achieved ≥475/500 on CartPole!")
        elif env_id == "LunarLander-v3" and results["mean_return"] >= 200.0:
            print("Successfully achieved ≥200 on LunarLander!")

        print("Evaluation complete.")

    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
