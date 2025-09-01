# src/train.py
from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from src.utils.seeding import set_seed, version_banner


@hydra.main(version_base=None, config_path="../configs/train", config_name="default")  # type: ignore[misc]
def main(cfg: DictConfig) -> None:
    # Seed + banner
    set_seed(cfg.seed, cfg.deterministic)
    print(version_banner())
    print("[Seed] set_seed:", cfg.seed, "| deterministic:", bool(cfg.deterministic))

    # Show composed config (including CLI overrides like env=cartpole algo=ppo)
    print("\n[Composed Config]")
    # Merge in selected env/algo files via Hydra's group overrides
    # Access them as cfg.env, cfg.algo if needed later; for Day 2 just show full cfg
    print(OmegaConf.to_yaml(cfg))

    # Day 2 exit point (no training loop yet)
    print("✅ Train stub complete; exiting.")
    return


if __name__ == "__main__":
    main()
