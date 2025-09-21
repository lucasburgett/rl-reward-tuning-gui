#!/usr/bin/env python3
"""Determinism verification script for Day 12 validation."""

import subprocess
import tempfile
from pathlib import Path


def run_command(cmd: str) -> tuple[int, str, str]:
    """Run shell command and return output."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=300
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Timeout"


def main() -> int:
    print("=== Day 12 Determinism Verification ===\n")

    # Check A: Dependency pinning
    print("A. Checking dependency pinning...")
    rc, out, err = run_command("grep -nE '(>=|~=)' requirements.txt")
    pinning_ok = rc != 0  # grep returns non-zero when no matches found
    print(f"   Pinned deps: {'PASS' if pinning_ok else 'FAIL'}")

    lock_exists = Path("requirements-lock.txt").exists()
    print(f"   Lock file exists: {'PASS' if lock_exists else 'FAIL'}")

    rc, out, err = run_command("grep -n 'Deterministic Setup' README.md")
    readme_ok = rc == 0
    print(f"   README section: {'PASS' if readme_ok else 'FAIL'}")

    # Check B: Startup report
    print("\nB. Checking startup determinism report...")
    rc, out, err = run_command(
        "python3 -m src.train env=cartpole algo=ppo seed=42 deterministic=true total_steps=10"
    )
    report_fields = [
        "seed: 42",
        "deterministic: True",
        "python:",
        "torch:",
        "gymnasium:",
        "mujoco:",
        "PYTHONHASHSEED: 42",
    ]

    all_fields_present = all(field in out for field in report_fields)
    print(f"   Report fields: {'PASS' if all_fields_present else 'FAIL'}")

    # Check C: Tests pass
    print("\nC. Checking determinism tests...")
    rc, out, err = run_command("python3 -m pytest tests/test_determinism_flags.py -q")
    tests_pass = rc == 0
    print(f"   Tests pass: {'PASS' if tests_pass else 'FAIL'}")

    # Check D: Same-seed reproducibility
    print("\nD. Checking same-seed reproducibility...")
    with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
        # Run twice with same seed
        rc1, out1, err1 = run_command(
            f"python3 -m src.train env=cartpole algo=ppo seed=42 deterministic=true total_steps=200 log_dir={tmp_a}"
        )
        rc2, out2, err2 = run_command(
            f"python3 -m src.train env=cartpole algo=ppo seed=42 deterministic=true total_steps=200 log_dir={tmp_b}"
        )

        if rc1 == 0 and rc2 == 0:
            # Compare determinism reports
            report1 = extract_determinism_report(out1)
            report2 = extract_determinism_report(out2)
            reports_match = report1 == report2

            # Compare metrics
            metrics1 = extract_metrics(out1)
            metrics2 = extract_metrics(out2)
            metrics_match = metrics1 == metrics2

            reproducible = reports_match and metrics_match
            print(f"   Reproducibility: {'PASS' if reproducible else 'FAIL'}")
        else:
            print("   Reproducibility: FAIL (training failed)")

    # Check E: Different seed divergence
    print("\nE. Checking different-seed divergence...")
    with tempfile.TemporaryDirectory() as tmp_c:
        rc3, out3, err3 = run_command(
            f"python3 -m src.train env=cartpole algo=ppo seed=123 deterministic=true total_steps=200 log_dir={tmp_c}"
        )

        if rc3 == 0:
            report3 = extract_determinism_report(out3)
            metrics3 = extract_metrics(out3)

            # Should differ from seed=42 runs
            differs = (report3 != report1) or (metrics3 != metrics1)
            print(f"   Divergence: {'PASS' if differs else 'FAIL'}")
        else:
            print("   Divergence: FAIL (training failed)")

    # Check F: Eval honors flags
    print("\nF. Checking eval determinism...")
    rc, out, err = run_command(
        "python3 -m src.eval env=cartpole algo=ppo seed=42 deterministic=true eval.n_episodes=2"
    )
    eval_report_ok = "seed: 42" in out and "deterministic: True" in out
    print(f"   Eval report: {'PASS' if eval_report_ok else 'FAIL'}")

    print("\n=== Summary ===")
    all_checks = [
        pinning_ok,
        lock_exists,
        readme_ok,
        all_fields_present,
        tests_pass,
        reproducible,
        differs,
        eval_report_ok,
    ]
    overall = "PASS" if all(all_checks) else "FAIL"
    print(f"Overall: {overall}")

    return 0 if overall == "PASS" else 1


def extract_determinism_report(output: str) -> str:
    """Extract determinism report section from output."""
    lines = output.split("\n")
    report_lines = []
    capturing = False

    for line in lines:
        if line.startswith("[Determinism Report]"):
            capturing = True
            report_lines.append(line)
        elif capturing:
            if line.strip() == "" or line.startswith("["):
                break
            report_lines.append(line)

    return "\n".join(report_lines)


def extract_metrics(output: str) -> dict[str, float]:
    """Extract key metrics from training output."""
    import re

    metrics = {}

    # Extract mean return
    match = re.search(r"Mean return: ([\d.]+)", output)
    if match:
        metrics["mean_return"] = float(match.group(1))

    return metrics


if __name__ == "__main__":
    exit(main())
