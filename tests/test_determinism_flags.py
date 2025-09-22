# tests/test_determinism_flags.py
"""Tests for determinism configuration and reproducibility."""

from __future__ import annotations

import os
import warnings
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from src.utils.determinism import configure_determinism, print_determinism_report


class TestDeterminismConfiguration:
    """Test determinism configuration functions."""

    def test_configure_determinism_seeds_set(self) -> None:
        """Test that configure_determinism sets all required seeds."""
        seed = 12345
        configure_determinism(seed, deterministic=True)

        # Check environment variable
        assert os.environ["PYTHONHASHSEED"] == str(seed)

        # Check that PyTorch deterministic algorithms are enabled
        assert torch.are_deterministic_algorithms_enabled()

        # Check CuDNN settings
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False

    def test_configure_determinism_non_deterministic(self) -> None:
        """Test non-deterministic mode configuration."""
        seed = 54321
        configure_determinism(seed, deterministic=False)

        # Check that deterministic algorithms are disabled
        assert not torch.are_deterministic_algorithms_enabled()

        # Check CuDNN settings for performance
        assert torch.backends.cudnn.deterministic is False
        assert torch.backends.cudnn.benchmark is True

    def test_cuda_workspace_config_set_when_deterministic(self) -> None:
        """Test that CUBLAS_WORKSPACE_CONFIG is set when deterministic and CUDA available."""
        seed = 11111
        configure_determinism(seed, deterministic=True)

        if torch.cuda.is_available():
            # Should have CUBLAS_WORKSPACE_CONFIG set
            assert "CUBLAS_WORKSPACE_CONFIG" in os.environ
            assert os.environ["CUBLAS_WORKSPACE_CONFIG"] in [":16:8", ":4096:8"]

    def test_print_determinism_report_no_crash(self, capsys: Any) -> None:
        """Test that print_determinism_report runs without crashing."""
        # Should not crash regardless of system configuration
        print_determinism_report(seed=42, deterministic=True)

        captured = capsys.readouterr()
        assert "[Determinism Report]" in captured.out
        assert "seed: 42" in captured.out
        assert "deterministic: True" in captured.out


class TestReproducibility:
    """Test actual reproducibility of operations."""

    def test_torch_operations_reproducible(self) -> None:
        """Test that torch operations are reproducible with same seed."""

        def run_torch_operation(seed: int) -> torch.Tensor:
            configure_determinism(seed, deterministic=True)

            # Create a simple model
            torch.manual_seed(seed)  # Extra manual seed for model init
            model = torch.nn.Linear(4, 2)

            # Generate some data
            torch.manual_seed(seed)  # Reset for data generation
            x = torch.randn(10, 4)

            # Forward pass
            with torch.no_grad():
                output: torch.Tensor = model(x)

            return output

        seed = 9999

        # Run twice with same seed
        result1 = run_torch_operation(seed)
        result2 = run_torch_operation(seed)

        # Results should be identical
        torch.testing.assert_close(result1, result2, atol=1e-7, rtol=1e-7)

    def test_numpy_operations_reproducible(self) -> None:
        """Test that numpy operations are reproducible."""

        def run_numpy_operation(seed: int) -> np.ndarray:
            configure_determinism(seed, deterministic=True)
            return np.random.randn(5, 3)

        seed = 7777

        result1 = run_numpy_operation(seed)
        result2 = run_numpy_operation(seed)

        np.testing.assert_array_equal(result1, result2)

    def test_gym_env_reproducible(self) -> None:
        """Test that gymnasium environment is reproducible."""

        def run_env_steps(seed: int) -> list[float]:
            configure_determinism(seed, deterministic=True)

            # Create environment
            env = gym.make("CartPole-v1")
            env.reset(seed=seed)

            rewards = []
            for _ in range(5):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                rewards.append(reward)

                if terminated or truncated:
                    env.reset(seed=seed)

            env.close()  # type: ignore[no-untyped-call]
            return [float(r) for r in rewards]

        seed = 3333

        # Suppress any gym warnings during test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rewards1 = run_env_steps(seed)
            rewards2 = run_env_steps(seed)

        # Should be identical
        assert rewards1 == rewards2

    def test_mixed_operations_reproducible(self) -> None:
        """Test reproducibility of mixed torch/numpy/gym operations."""

        def run_mixed_operation(seed: int) -> dict[str, float]:
            configure_determinism(seed, deterministic=True)

            # Torch operation
            torch_result = torch.randn(3, 3).sum().item()

            # Numpy operation
            numpy_result = np.random.rand(2, 2).mean()

            # Simple calculation
            combined = torch_result + numpy_result

            return {"torch": torch_result, "numpy": numpy_result, "combined": combined}

        seed = 1234

        result1 = run_mixed_operation(seed)
        result2 = run_mixed_operation(seed)

        # All values should be identical
        assert result1["torch"] == result2["torch"]
        assert result1["numpy"] == result2["numpy"]
        assert result1["combined"] == result2["combined"]


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_determinism_with_different_seeds_different_results(self) -> None:
        """Test that different seeds produce different results."""

        def run_operation(seed: int) -> float:
            configure_determinism(seed, deterministic=True)
            return torch.randn(1).item()

        result1 = run_operation(1111)
        result2 = run_operation(2222)

        # Different seeds should produce different results
        assert result1 != result2

    def test_configure_determinism_handles_missing_cuda(self) -> None:
        """Test that function works even if CUDA operations fail."""
        # This should not crash even on systems without CUDA
        configure_determinism(42, deterministic=True)
        configure_determinism(42, deterministic=False)

        # If we get here without exception, the test passes
        assert True
