# src/eval.py
from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from src.utils.seeding import set_seed, version_banner


@hydra.main(version_base=None, config_path="../configs/eval", config_name="default")  # type: ignore[misc]
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed, cfg.deterministic)
    print(version_banner())
    print("[Seed] set_seed:", cfg.seed, "| deterministic:", bool(cfg.deterministic))

    print("\n[Composed Config]")
    print(OmegaConf.to_yaml(cfg))

    # Day 2 exit point
    print("✅ Eval stub complete; exiting.")
    return


if __name__ == "__main__":
    main()
