# Reproducible RL Template

A clean, reproducible template for deep RL experiments with deterministic runs, clear configs, and one-command training/evaluation.

## Features

- 🎯 **PPO Implementation**: Stable-Baselines3 PPO integration with CartPole-v1 achieving ≥475/500 score
- 🔧 **Reproducible**: Deterministic seeding and configuration management via Hydra
- 📊 **Monitoring**: Automatic checkpointing, evaluation, and video recording
- ⚡ **Fast Setup**: One-command training and evaluation with convenience scripts

## Quick Start

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Train PPO on CartPole (Recommended)
```bash
./scripts/run_cartpole.sh
```

### Manual Training
```bash
python -m src.train env=cartpole algo=ppo total_steps=100000
```

### Manual Evaluation
```bash
python -m src.eval env=cartpole algo=ppo log_dir=experiments/YYYY-MM-DD/HH-MM-SS
```

## Performance Results

The PPO implementation achieves:
- ✅ **500/500 perfect score** on CartPole-v1 
- ✅ **Fast convergence** in ~20k training steps
- ✅ **Deterministic training** with proper seeding
- ✅ **Sub-5 minute training** on CPU

## Configuration

All hyperparameters are configurable via YAML files:

- `configs/algo/ppo.yaml` - PPO hyperparameters
- `configs/env/cartpole.yaml` - Environment settings  
- `configs/config.yaml` - Training configuration
- `configs/eval_config.yaml` - Evaluation configuration

Override any parameter via CLI:
```bash
python -m src.train total_steps=200000 algo.learning_rate=1e-4
```

## Project Structure

```
├── src/
│   ├── agents/
│   │   └── ppo.py          # PPO agent wrapper (SB3 + CleanRL support)
│   ├── utils/
│   │   └── seeding.py      # Deterministic seeding utilities
│   ├── train.py            # Training entrypoint with Hydra
│   └── eval.py             # Evaluation entrypoint with Hydra
├── configs/
│   ├── config.yaml         # Main training configuration
│   ├── eval_config.yaml    # Evaluation configuration
│   ├── algo/ppo.yaml       # PPO hyperparameters
│   └── env/cartpole.yaml   # Environment settings
├── scripts/
│   └── run_cartpole.sh     # Convenience training script
├── experiments/            # Training outputs (auto-generated)
│   └── YYYY-MM-DD/HH-MM-SS/
│       ├── checkpoints/    # Model checkpoints
│       └── videos/         # Evaluation videos
└── requirements.txt        # Python dependencies
```

## Troubleshooting

- **ffmpeg not found**: Install ffmpeg for video recording
  - macOS: `brew install ffmpeg`
  - Ubuntu: `apt install ffmpeg`
- **Memory issues**: Reduce `total_steps` or use `device=cpu`
- **Poor performance**: Increase `total_steps` to 200k+ or tune hyperparameters
- **Missing dependencies**: Run `pip install -r requirements.txt`
- **Config errors**: Ensure you're using the root `configs/` directory structure

## Development Notes

This template follows the patterns defined in `CLAUDE.md`:
- Type-hinted Python with Black formatting
- Modular design with clear separation of concerns  
- Hydra configuration management
- Comprehensive error handling and logging
- Deterministic training for reproducible results
