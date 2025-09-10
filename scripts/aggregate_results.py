#!/usr/bin/env python3
"""
Day 11: Comprehensive results aggregation with version tracking.
Reads run manifest and produces detailed CSV/Markdown reports with system versions.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import pandas as pd


def get_system_versions() -> dict[str, str]:
    """Get versions of all relevant packages and system info."""
    versions = {}

    # Python version
    versions["python"] = platform.python_version()

    # Git commit hash
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).parent.parent,
        )
        versions["commit"] = (
            result.stdout.strip() if result.returncode == 0 else "unknown"
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        versions["commit"] = "unknown"

    # Package versions
    try:
        import torch

        versions["torch"] = torch.__version__
    except ImportError:
        versions["torch"] = "not installed"

    try:
        import gymnasium as gym

        versions["gymnasium"] = gym.__version__
    except ImportError:
        versions["gymnasium"] = "not installed"

    try:
        import mujoco

        versions["mujoco"] = mujoco.__version__
    except ImportError:
        versions["mujoco"] = "not installed"

    try:
        import stable_baselines3

        versions["sb3"] = stable_baselines3.__version__
    except ImportError:
        versions["sb3"] = "not installed"

    return versions


def load_run_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    """Load run manifest JSON file."""
    if not manifest_path.exists():
        warnings.warn(f"Manifest not found: {manifest_path}", stacklevel=2)
        return []

    try:
        with open(manifest_path) as f:
            data = json.load(f)

        runs = data.get("runs", [])
        return runs if isinstance(runs, list) else []
    except Exception as e:
        warnings.warn(f"Failed to load manifest: {e}", stacklevel=2)
        return []


def scan_artifacts_for_runs() -> list[dict[str, Any]]:
    """Fallback: scan artifacts directory for runs if manifest is missing."""
    runs: list[dict[str, Any]] = []
    artifacts_dir = Path("artifacts")

    if not artifacts_dir.exists():
        return runs

    for env_dir in artifacts_dir.iterdir():
        if not env_dir.is_dir():
            continue

        for algo_dir in env_dir.iterdir():
            if not algo_dir.is_dir() or algo_dir.name != "ppo":
                continue

            for seed_dir in algo_dir.iterdir():
                if not seed_dir.is_dir():
                    continue

                try:
                    seed = int(seed_dir.name)
                except ValueError:
                    continue

                # Find timestamped run directories
                for date_dir in seed_dir.iterdir():
                    if not date_dir.is_dir():
                        continue

                    for time_dir in date_dir.iterdir():
                        if not time_dir.is_dir():
                            continue

                        if (time_dir / "logs" / "eval_metrics.csv").exists():
                            runs.append(
                                {
                                    "env": env_dir.name,
                                    "seed": seed,
                                    "run_dir": str(time_dir),
                                }
                            )

    return runs


def load_eval_metrics_from_run(run_info: dict[str, Any]) -> dict[str, Any] | None:
    """Load evaluation metrics from a single run."""
    run_dir = Path(run_info["run_dir"])
    eval_csv = run_dir / "logs" / "eval_metrics.csv"

    if not eval_csv.exists():
        warnings.warn(f"Missing eval CSV: {eval_csv}", stacklevel=2)
        return None

    try:
        df = pd.read_csv(eval_csv)
        if df.empty:
            warnings.warn(f"Empty eval CSV: {eval_csv}", stacklevel=2)
            return None

        # Take the final evaluation row
        final_row = df.iloc[-1]

        # Robust column name handling
        mean_return_col = None
        for col in ["mean_return", "avg_return", "return_mean"]:
            if col in df.columns:
                mean_return_col = col
                break

        if mean_return_col is None:
            warnings.warn(f"No return column found in {eval_csv}", stacklevel=2)
            return None

        std_return_col = None
        for col in ["std_return", "return_std", "std", "return_std_dev"]:
            if col in df.columns:
                std_return_col = col
                break

        return {
            "env": run_info["env"],
            "seed": run_info["seed"],
            "mean_return": float(final_row[mean_return_col]),
            "std_return": float(final_row[std_return_col]) if std_return_col else 0.0,
            "n_episodes": int(final_row.get("n_episodes", 10)),
            "run_dir": run_info["run_dir"],
            "csv_path": str(eval_csv),
        }

    except Exception as e:
        warnings.warn(f"Failed to load {eval_csv}: {e}", stacklevel=2)
        return None


def compute_environment_stats(env_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute mean±std across seeds for one environment."""
    if not env_results:
        return {}

    # Extract mean returns across seeds
    mean_returns = [r["mean_return"] for r in env_results]
    n_episodes = [r["n_episodes"] for r in env_results]

    # Compute cross-seed statistics
    import statistics

    mean_of_means = statistics.mean(mean_returns)
    std_of_means = statistics.stdev(mean_returns) if len(mean_returns) > 1 else 0.0

    # Environment info
    env_name = env_results[0]["env"]
    seeds = [r["seed"] for r in env_results]
    avg_episodes = statistics.mean(n_episodes)

    # Example paths
    example_run_dir = env_results[0]["run_dir"]
    csv_paths = [r["csv_path"] for r in env_results]

    # Find checkpoint path
    checkpoint_path = "n/a"
    run_path = Path(example_run_dir)
    checkpoints_dir = run_path / "checkpoints"
    if checkpoints_dir.exists():
        # Look for latest checkpoint
        checkpoint_files = list(checkpoints_dir.glob("*.zip"))
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
            try:
                checkpoint_path = str(latest_checkpoint.relative_to(Path.cwd()))
            except ValueError:
                # Fallback if relative path fails
                checkpoint_path = str(latest_checkpoint)
        else:
            try:
                checkpoint_path = str(checkpoints_dir.relative_to(Path.cwd()))
            except ValueError:
                checkpoint_path = str(checkpoints_dir)

    return {
        "env": env_name,
        "algo": "ppo",
        "seeds": seeds,
        "n_seeds": len(seeds),
        "mean_return": mean_of_means,
        "std_return": std_of_means,
        "n_episodes": int(avg_episodes),
        "example_run_dir": example_run_dir,
        "checkpoint_path": checkpoint_path,
        "csv_paths": csv_paths,
    }


def get_steps_for_env(env: str) -> str:
    """Get step count for each environment."""
    steps_map = {
        "cartpole": "100k",
        "lunarlander": "1M",
        "reacher_mujoco": "800k",
        "reacher": "800k",
    }
    return steps_map.get(env.lower(), "Unknown")


def get_env_display_name(env: str) -> str:
    """Get display name for environment."""
    display_map = {
        "cartpole": "CartPole-v1",
        "lunarlander": "LunarLander-v2",
        "reacher_mujoco": "Reacher-v5",
        "reacher": "Reacher-v5",
    }
    return display_map.get(env.lower(), env)


def format_versions_string(versions: dict[str, str]) -> str:
    """Format versions as a compact string."""
    key_versions = ["python", "torch", "gymnasium", "mujoco", "sb3"]
    version_parts = []

    for key in key_versions:
        if key in versions and versions[key] != "not installed":
            version_parts.append(f"{key}={versions[key]}")

    if versions.get("commit", "unknown") != "unknown":
        version_parts.append(f"git={versions['commit']}")

    return " | ".join(version_parts)


def main() -> None:
    """Main aggregation function."""
    print("=== Day 11: Comprehensive Results Aggregation ===")

    # Get system versions
    versions = get_system_versions()
    print(f"System: Python {versions['python']}, Git {versions['commit']}")

    # Load run data
    manifest_path = Path("reports/run_manifest.json")
    runs = load_run_manifest(manifest_path)

    if not runs:
        print("No runs found in manifest, scanning artifacts directory...")
        runs = scan_artifacts_for_runs()

    print(f"Found {len(runs)} runs")

    if not runs:
        print("ERROR: No runs found in manifest or artifacts directory")
        sys.exit(1)

    # Group runs by environment
    env_groups: dict[str, list[dict[str, Any]]] = {}

    for run_info in runs:
        env = run_info["env"]
        if env not in env_groups:
            env_groups[env] = []

        # Load evaluation metrics for this run
        eval_result = load_eval_metrics_from_run(run_info)
        if eval_result:
            env_groups[env].append(eval_result)

    if not env_groups:
        print("ERROR: No evaluation results found")
        sys.exit(1)

    # Compute stats for each environment
    env_stats = []
    for env, results in env_groups.items():
        if len(results) != 3:
            warnings.warn(
                f"Expected 3 seeds for {env}, got {len(results)}", stacklevel=2
            )

        stats = compute_environment_stats(results)
        if stats:
            env_stats.append(stats)

    # Sort by environment name for consistent output
    env_stats.sort(key=lambda x: x["env"])

    # Create reports directory
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    # Write CSV results
    csv_output = reports_dir / "results_seed_stats.csv"
    csv_rows = []

    for stats in env_stats:
        csv_rows.append(
            {
                "env": stats["env"],
                "algo": stats["algo"],
                "steps": get_steps_for_env(stats["env"]),
                "seeds": f"{stats['n_seeds']} ({min(stats['seeds'])}-{max(stats['seeds'])})",
                "mean_return": f"{stats['mean_return']:.2f}",
                "std_return": f"{stats['std_return']:.3f}",
                "n_episodes": stats["n_episodes"],
                "example_run_dir": stats["example_run_dir"],
                "eval_csv_paths": "; ".join(stats["csv_paths"]),
                "commit": versions["commit"],
                "python": versions["python"],
                "torch": versions["torch"],
                "gymnasium": versions["gymnasium"],
                "mujoco": versions["mujoco"],
                "sb3": versions["sb3"],
            }
        )

    # Write CSV
    import csv

    with open(csv_output, "w", newline="") as f:
        if csv_rows:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)

    print(f"CSV results written to: {csv_output}")

    # Write Markdown results
    md_output = reports_dir / "results.md"
    versions_str = format_versions_string(versions)

    with open(md_output, "w") as f:
        f.write("# Day 11: Complete Reproducible RL Results\n\n")
        f.write(
            "Comprehensive 3-seed variance analysis across environments with system tracking.\n\n"
        )

        # Versions block
        f.write("## System Versions\n\n")
        f.write("| Component | Version |\n")
        f.write("|-----------|----------|\n")
        for key, value in versions.items():
            key_display = key.replace("_", " ").title()
            if key == "sb3":
                key_display = "Stable-Baselines3"
            elif key == "commit":
                key_display = "Git Commit"
            f.write(f"| {key_display} | `{value}` |\n")
        f.write("\n")

        # Results table
        f.write("## Results\n\n")
        f.write(
            "| Env | Algo | Steps | Seeds | Mean ± Std | Checkpoint path | Versions |\n"
        )
        f.write(
            "|-----|------|-------|-------|------------|-----------------|----------|\n"
        )

        for stats in env_stats:
            env_display = get_env_display_name(stats["env"])
            steps = get_steps_for_env(stats["env"])
            seeds = stats["n_seeds"]
            mean_std = f"{stats['mean_return']:.1f} ± {stats['std_return']:.2f}"
            checkpoint_display = stats["checkpoint_path"]

            # Truncate long checkpoint paths
            if len(checkpoint_display) > 40:
                checkpoint_display = "..." + checkpoint_display[-37:]

            f.write(
                f"| {env_display} | PPO | {steps} | {seeds} | {mean_std} | `{checkpoint_display}` | {versions_str} |\n"
            )

        # Run details
        f.write("\n## Run Details\n\n")
        f.write(
            f"**Total runs processed**: {sum(len(env_groups[env]) for env in env_groups)}\n"
        )
        f.write(f"**Environments**: {len(env_stats)}\n")
        f.write(
            f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n"
        )

        for stats in env_stats:
            env_display = get_env_display_name(stats["env"])
            f.write(f"### {env_display}\n")
            f.write(f"- **Seeds**: {stats['seeds']}\n")
            f.write(f"- **Episodes per eval**: {stats['n_episodes']}\n")
            f.write(f"- **Mean return**: {stats['mean_return']:.3f}\n")
            f.write(f"- **Std across seeds**: {stats['std_return']:.3f}\n")
            f.write(f"- **Example run**: `{stats['example_run_dir']}`\n")
            f.write(f"- **Checkpoint**: `{stats['checkpoint_path']}`\n")
            f.write("\n")

    print(f"Markdown results written to: {md_output}")

    # Print summary table
    print("\n=== Final Results ===")
    print("| Env            | Algo | Steps | Seeds | Mean ± Std |")
    print("|----------------|------|-------|-------|------------|")

    for stats in env_stats:
        env_display = get_env_display_name(stats["env"])
        steps = get_steps_for_env(stats["env"])
        seeds = stats["n_seeds"]
        mean_std = f"{stats['mean_return']:.1f} ± {stats['std_return']:.2f}"
        print(
            f"| {env_display:<14} | PPO  | {steps:<5} | {seeds}     | {mean_std:<10} |"
        )

    print("\nResults saved:")
    print(f"  - CSV: {csv_output.absolute()}")
    print(f"  - Markdown: {md_output.absolute()}")


if __name__ == "__main__":
    main()
