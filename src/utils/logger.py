"""CSV logging utilities for training and evaluation metrics."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class CSVLogger:
    """Simple CSV logger that appends rows with automatic header creation."""

    def __init__(self, path: Path | str) -> None:
        """Initialize CSV logger.

        Args:
            path: Path to the CSV file
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._header_written = self.path.exists() and self.path.stat().st_size > 0

    def log(self, row: dict[str, float | int | str | bool]) -> None:
        """Log a row to the CSV file.

        Args:
            row: Dictionary of column name to value mappings
        """
        with self.path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)


def create_training_metrics_row(
    global_step: int,
    timesteps: int,
    mean_return: float,
    std_return: float,
    wall_time_s: float,
    **kwargs: Any,
) -> dict[str, Any]:
    """Create a standardized training metrics row.

    Args:
        global_step: Current training step
        timesteps: Total environment timesteps
        mean_return: Mean episode return
        std_return: Standard deviation of episode returns
        wall_time_s: Wall clock time in seconds
        **kwargs: Additional metrics to log

    Returns:
        Dictionary ready for CSV logging
    """
    row = {
        "global_step": global_step,
        "timesteps": timesteps,
        "episodic_return_mean": mean_return,
        "episodic_return_std": std_return,
        "wall_time_s": wall_time_s,
    }
    row.update(kwargs)
    return row


def create_eval_metrics_row(
    eval_idx: int,
    n_episodes: int,
    mean_return: float,
    std_return: float,
    video_written: bool,
    checkpoint_path: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Create a standardized evaluation metrics row.

    Args:
        eval_idx: Evaluation index/iteration
        n_episodes: Number of evaluation episodes
        mean_return: Mean episode return
        std_return: Standard deviation of episode returns
        video_written: Whether videos were recorded
        checkpoint_path: Path to checkpoint used (if any)
        **kwargs: Additional metrics to log

    Returns:
        Dictionary ready for CSV logging
    """
    row: dict[str, Any] = {
        "eval_idx": eval_idx,
        "n_episodes": n_episodes,
        "mean_return": mean_return,
        "std_return": std_return,
        "video_written": video_written,
    }
    if checkpoint_path is not None:
        row["checkpoint_path"] = checkpoint_path
    row.update(kwargs)
    return row
