"""Basic tests to verify project setup."""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


def test_import_rrl() -> None:
    """Test that the main rrl package can be imported."""
    import rrl

    assert rrl.__version__ == "0.1.0"


def test_import_agents() -> None:
    """Test that agents package can be imported."""
    from rrl import agents

    assert hasattr(agents, "__all__")


def test_import_envs() -> None:
    """Test that envs package can be imported."""
    from rrl import envs

    assert hasattr(envs, "__all__")


def test_import_utils() -> None:
    """Test that utils package can be imported."""
    from rrl import utils

    assert hasattr(utils, "__all__")


def test_train_script_exists() -> None:
    """Test that train.py exists and can be imported."""
    from rrl import train

    assert hasattr(train, "main")


def test_eval_script_exists() -> None:
    """Test that eval.py exists and can be imported."""
    from rrl import eval

    assert hasattr(eval, "main")


def test_config_files_exist() -> None:
    """Test that config files exist."""
    config_dir = Path(__file__).parent.parent / "configs"

    assert (config_dir / "defaults.yaml").exists()
    assert (config_dir / "agent" / "ppo.yaml").exists()
    assert (config_dir / "env" / "cartpole.yaml").exists()
    assert (config_dir / "run" / "debug.yaml").exists()
    assert (config_dir / "run" / "paper.yaml").exists()
