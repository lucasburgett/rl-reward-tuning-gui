from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

import gymnasium as gym
import numpy as np


class RewardComponentsLogger(gym.Wrapper):  # type: ignore[type-arg]
    """
    Logs per-episode sums of any reward components exposed via info dict.
    Components are inferred by keys like: 'reward_*', 'r_*', or nested 'reward' dicts.
    Writes a row at episode end: step_count and component sums.

    If no components are present, writes only total reward.
    """

    def __init__(
        self,
        env: gym.Env[Any, Any],
        csv_path: str | None = None,
    ) -> None:
        super().__init__(env)
        self.csv_path = Path(csv_path) if csv_path else None
        self._episode_sums: dict[str, float] = {}
        self._episode_len = 0
        if self.csv_path:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.csv_path.exists():
                with self.csv_path.open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["episode_len", "return"])
                    writer.writeheader()

    def reset(self, **kwargs: Any) -> Any:
        if self._episode_len > 0:
            self._flush()
        self._episode_sums = {}
        self._episode_len = 0
        return self.env.reset(**kwargs)

    def step(self, action: Any) -> Any:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._episode_len += 1
        # sum total reward
        self._episode_sums["return"] = self._episode_sums.get("return", 0.0) + float(
            reward
        )

        # discover components
        keys = []
        for k, v in info.items():
            if isinstance(v, int | float | np.floating) and (
                k.startswith("reward_") or k.startswith("r_")
            ):
                keys.append(k)
            elif k in ("reward", "rewards") and isinstance(v, dict):
                for ck, cv in v.items():
                    if isinstance(cv, int | float | np.floating):
                        keys.append(f"reward.{ck}")
                        self._episode_sums[f"reward.{ck}"] = self._episode_sums.get(
                            f"reward.{ck}", 0.0
                        ) + float(cv)
        # flat numeric keys collected above
        for k in keys:
            self._episode_sums[k] = self._episode_sums.get(k, 0.0) + float(
                info[k] if k in info else 0.0
            )

        if terminated or truncated:
            self._flush()
        return obs, reward, terminated, truncated, info

    def _flush(self) -> None:
        if not self.csv_path:
            # nothing to write
            self._episode_len = 0
            self._episode_sums = {}
            return
        row = {"episode_len": self._episode_len}
        row.update({k: round(float(v), 6) for k, v in self._episode_sums.items()})  # type: ignore[misc]
        # ensure header includes all fields
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
                writer.writerow(row)
        else:
            # grow header if new keys appear
            with self.csv_path.open("r", newline="") as f:
                reader = csv.DictReader(f)
                existing_fields = reader.fieldnames or []
            if set(row.keys()) - set(existing_fields):
                # rewrite with expanded header
                rows = []
                with self.csv_path.open("r", newline="") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                new_fields = [k for k in row.keys() if k not in existing_fields]
                fields = list(dict.fromkeys(list(existing_fields) + new_fields))
                with self.csv_path.open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fields)
                    writer.writeheader()
                    for r in rows:
                        writer.writerow(r)
                    writer.writerow(row)
            else:
                with self.csv_path.open("a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=existing_fields)
                    writer.writerow(row)
        self._episode_len = 0
        self._episode_sums = {}
