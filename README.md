# Reproducible RL Template

[![CI](https://github.com/YOUR_GH_USER/YOUR_REPO/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_GH_USER/YOUR_REPO/actions/workflows/ci.yml)

A clean, reproducible template for deep RL experiments with deterministic runs, clear configs, and one-command training/evaluation.

## Features

- **Multi-Environment PPO**: Stable-Baselines3 PPO integration with CartPole-v1 and LunarLander-v3
- **Reproducible**: Deterministic seeding and configuration management via Hydra
- **Advanced Logging**: CSV metrics logging, optional Weights & Biases integration, video recording
- **Organized Artifacts**: Structured output layout under `artifacts/{env}/{algo}/{seed}/{timestamp}/`
- **Fast Setup**: One-command training and evaluation with convenience scripts

## Quick Start

### Install Dependencies
```bash
pip install -r requirements.txt
```

**Note for macOS users**: If you encounter Box2D build issues for LunarLander, install swig:
```bash
brew install swig
```

### Train PPO on CartPole (Fast)
```bash
./scripts/run_cartpole.sh
```

### Train PPO on LunarLander (Day 4 - New!)
```bash
./scripts/run_lunarlander.sh
```

### Manual Training Examples
```bash
# CartPole (100k steps, ~2-3 minutes)
python -m src.train env=cartpole algo=ppo total_steps=100000

# LunarLander-v3 (1M steps, ~30-60 minutes) 
python -m src.train env=lunarlander algo=ppo total_steps=1000000

# LunarLander with optimized hyperparameters
python -m src.train env=lunarlander algo=ppo total_steps=1000000 \
    algo.ppo.rollout_len=4096 algo.ppo.num_minibatches=64

# Enable Weights & Biases logging
python -m src.train env=lunarlander algo=ppo use_wandb=true
```

### Manual Evaluation
```bash
# Evaluate latest checkpoint (auto-finds artifacts)
python -m src.eval env=lunarlander algo=ppo

# Evaluate with video recording
python -m src.eval env=lunarlander algo=ppo eval.record_video=true
```

## Performance Results

The PPO implementation achieves:

### CartPole-v1
- **500/500 perfect score** consistently 
- **Fast convergence** in ~20k training steps
- **Sub-5 minute training** on CPU

### LunarLander-v3 (Day 4)
- **≥200 mean return** (target achieved with default settings)
- **Convergence** in 0.5-1.5M training steps (~30-60 minutes)
- **Deterministic training** with proper seeding across environments

## Configuration

All hyperparameters are configurable via YAML files:

- `configs/algo/ppo.yaml` - PPO hyperparameters with environment presets
- `configs/env/cartpole.yaml` & `configs/env/lunarlander.yaml` - Environment settings  
- `configs/config.yaml` - Training configuration with W&B options
- `configs/eval_config.yaml` - Evaluation configuration

Override any parameter via CLI:
```bash
# Tune hyperparameters
python -m src.train env=lunarlander total_steps=500000 algo.ppo.learning_rate=1e-4

# Enable Weights & Biases with custom project
python -m src.train env=lunarlander use_wandb=true wandb.project=my-rl-experiments
```

## Project Structure

```
├── src/
│   ├── agents/
│   │   └── ppo.py          # PPO agent wrapper (SB3 + CleanRL support)
│   ├── utils/
│   │   ├── seeding.py      # Deterministic seeding utilities
│   │   └── logger.py       # CSV logging utilities (Day 4)
│   ├── train.py            # Training entrypoint with Hydra
│   └── eval.py             # Evaluation entrypoint with Hydra  
├── configs/
│   ├── config.yaml         # Main training config with W&B options
│   ├── eval_config.yaml    # Evaluation configuration
│   ├── algo/ppo.yaml       # PPO hyperparameters + environment presets
│   └── env/
│       ├── cartpole.yaml   # CartPole environment settings
│       └── lunarlander.yaml # LunarLander environment settings (Day 4)
├── scripts/
│   ├── run_cartpole.sh     # CartPole training script
│   └── run_lunarlander.sh  # LunarLander training script (Day 4)
├── artifacts/              # Organized training outputs (Day 4)
│   └── {env}/{algo}/{seed}/{timestamp}/
│       ├── checkpoints/    # Model checkpoints (.zip files)
│       ├── videos/         # Evaluation videos (.mp4 files)
│       ├── logs/           # CSV metrics (train_metrics.csv, eval_metrics.csv, monitor.csv)
│       └── config/         # Saved configurations (train.yaml, eval.yaml)
└── requirements.txt        # Python dependencies (includes Box2D)
```

## Day 4 - LunarLander + Advanced Logging

Day 4 adds LunarLander-v3 support with enhanced logging and artifacts management:

### Key Day 4 Features
- **LunarLander-v3** environment with Box2D physics
- **CSV Logging**: Training/eval metrics saved to structured CSV files
- **Weights & Biases**: Optional W&B integration for experiment tracking  
- **Artifacts Layout**: Organized structure under `artifacts/{env}/{algo}/{seed}/{timestamp}/`
- **Enhanced Video Recording**: Automatic video capture during evaluations
- **Training Optimizations**: Environment-specific hyperparameter presets

### LunarLander Performance Tips
- **Target score**: ≥200 mean return (typically achieved in 500k-1.5M steps)
- **Recommended hyperparameters**: Use `algo.ppo.rollout_len=4096` and `algo.ppo.num_minibatches=64` for better performance
- **Training time**: ~30-60 minutes on modern CPU, ~10-20 minutes on GPU
- **Box2D installation**: May require `brew install swig` on macOS

### Weights & Biases Setup
```bash
# Install wandb (optional)
pip install wandb

# Enable W&B logging
python -m src.train env=lunarlander use_wandb=true wandb.project=my-experiments

# Configure W&B settings in configs/config.yaml:
# wandb:
#   project: rl-template
#   entity: your-wandb-username  
#   group: ${env}-${algo}
```

## Troubleshooting

- **Box2D build errors**: Install swig first: `brew install swig` (macOS) or `apt install swig` (Ubuntu)
- **ffmpeg not found**: Install ffmpeg for video recording
  - macOS: `brew install ffmpeg`
  - Ubuntu: `apt install ffmpeg`
- **Memory issues**: Reduce `total_steps` or use `device=cpu`
- **LunarLander not converging**: Increase `total_steps` to 1M+ or try the optimized hyperparameters
- **Missing dependencies**: Run `pip install -r requirements.txt`
- **Config errors**: Ensure you're using the root `configs/` directory structure
- **Artifacts not found**: Check the `artifacts/` directory structure matches `{env}/{algo}/{seed}/{timestamp}/`

## Day 5 — Testing

A minimal but robust test suite with fast runtime and strong determinism guarantees.

### Running Tests

```bash
# Install testing dependencies (if not already installed)
pip install pytest>=8.2 pytest-timeout>=2.3

# Run all tests
pytest -q

# Run specific test modules
pytest tests/test_seeding.py -v
pytest tests/test_env_wrappers.py -v
pytest tests/test_repro_small.py -v
```

### Test Coverage

- **`test_seeding.py`**: Ensures deterministic evaluation with same seed & config
- **`test_env_wrappers.py`**: Validates environment shapes, truncation, and finite values
- **`test_repro_small.py`**: Verifies 5k step training achieves ≥100 average return on CartPole

### Performance Notes

- **Expected runtime**: ~1-2 minutes on CPU
- **Fast configs**: Tests use minimal PPO hyperparameters for quick execution
- **No videos/W&B**: Tests disable video recording and Weights & Biases for speed
- **Determinism**: All tests use single-threaded math and fixed seeds

### Troubleshooting

- **Slow tests**: Tests use `@pytest.mark.timeout(90)` to prevent hanging
- **Box2D not needed**: CartPole tests don't require Box2D/LunarLander dependencies
- **Temporary files**: Tests use `tmp_path` fixtures, no cleanup needed

## Docker (CPU)

Build the CPU image:
```bash
docker build -f docker/CPU.Dockerfile -t rl-template-cpu:latest .
```

Run a tiny CartPole train inside the container:
```bash
docker run --rm -it -v "$PWD:/app" rl-template-cpu:latest python -m src.train env=cartpole algo=ppo total_steps=5000 use_wandb=false
```

Or use the convenience scripts:
```bash
./scripts/docker_build_cpu.sh
./scripts/docker_run_cpu.sh python -m src.train env=cartpole algo=ppo total_steps=5000 use_wandb=false
```

## Development Notes

This template follows the patterns defined in `CLAUDE.md`:
- Type-hinted Python with Black formatting
- Modular design with clear separation of concerns  
- Hydra configuration management
- Comprehensive error handling and logging
- Deterministic training for reproducible results
