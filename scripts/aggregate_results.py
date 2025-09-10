#!/usr/bin/env python3
"""
Day 10: Aggregate results across exactly 3 seeds per environment.
Reads from reports/run_manifest.json and artifacts logs to compute mean±std.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any

import pandas as pd


def load_run_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    """Load run manifest JSON file."""
    if not manifest_path.exists():
        print(f"ERROR: Manifest not found: {manifest_path}")
        print("Run scripts/run_seeds.sh first to generate the manifest")
        sys.exit(1)

    with open(manifest_path) as f:
        data = json.load(f)

    runs = data.get("runs", [])
    return runs if isinstance(runs, list) else []


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

        return {
            "env": run_info["env"],
            "seed": run_info["seed"],
            "mean_return": float(final_row["mean_return"]),
            "std_return": float(final_row["std_return"]),
            "n_episodes": int(final_row["n_episodes"]),
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

    return {
        "env": env_name,
        "algo": "ppo",
        "seeds": seeds,
        "n_seeds": len(seeds),
        "mean_return": mean_of_means,
        "std_return": std_of_means,
        "n_episodes": int(avg_episodes),
        "example_run_dir": example_run_dir,
        "csv_paths": csv_paths,
    }


def get_steps_for_env(env: str) -> str:
    """Get step count for each environment."""
    steps_map = {
        "cartpole": "100k",
        "lunarlander": "1.0M",
        "reacher": "0.8M",
    }
    return steps_map.get(env.lower(), "Unknown")


def get_env_display_name(env: str) -> str:
    """Get display name for environment."""
    display_map = {
        "cartpole": "CartPole-v1",
        "lunarlander": "LunarLander-v2",
        "reacher": "Reacher-v5",
    }
    return display_map.get(env.lower(), env)


def main() -> None:
    """Main aggregation function."""
    print("=== Day 10: Aggregating Clean 3-Seed Results ===")

    # Load run manifest
    manifest_path = Path("reports/run_manifest.json")
    runs = load_run_manifest(manifest_path)
    print(f"Found {len(runs)} runs in manifest")

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

    with open(md_output, "w") as f:
        f.write("# Day 10: Clean 3-Seed Results\n\n")
        f.write(
            "Mean ± std performance across exactly 3 seeds per environment with optimized PPO settings.\n\n"
        )
        f.write("| Env | Algo | Steps | Seeds | Mean ± Std |\n")
        f.write("|-----|------|-------|-------|------------|\n")

        for stats in env_stats:
            env_display = get_env_display_name(stats["env"])
            steps = get_steps_for_env(stats["env"])
            seeds = stats["n_seeds"]
            mean_std = f"{stats['mean_return']:.1f} ± {stats['std_return']:.2f}"

            f.write(f"| {env_display} | PPO | {steps} | {seeds} | {mean_std} |\n")

        f.write("\n## Run Details\n\n")
        f.write(
            f"Total runs processed: {sum(len(env_groups[env]) for env in env_groups)}\n"
        )
        f.write(f"Environments: {len(env_stats)}\n\n")

        for stats in env_stats:
            env_display = get_env_display_name(stats["env"])
            f.write(f"### {env_display}\n")
            f.write(f"- **Seeds**: {stats['seeds']}\n")
            f.write(f"- **Episodes per eval**: {stats['n_episodes']}\n")
            f.write(f"- **Mean return**: {stats['mean_return']:.3f}\n")
            f.write(f"- **Std across seeds**: {stats['std_return']:.3f}\n")
            f.write(f"- **Example run**: `{stats['example_run_dir']}`\n")
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
    print(f"  - CSV: {csv_output}")
    print(f"  - Markdown: {md_output}")


if __name__ == "__main__":
    main()
