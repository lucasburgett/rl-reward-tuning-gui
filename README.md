# Reproducible RL Template

[![CI](https://github.com/lucasburgett/rl-reward-tuning-gui/actions/workflows/ci.yml/badge.svg)](https://github.com/lucasburgett/rl-reward-tuning-gui/actions/workflows/ci.yml)

## Project Goals

A minimal, reproducible RL template for deep reinforcement learning experiments. Features PPO baselines on CartPole and LunarLander with deterministic seeds, structured artifacts, and CPU-friendly training. Designed for researchers who want clean, reproducible results without complex setup.

**Key Features**: Deterministic training, Hydra configs, optional W&B logging, automated testing, CI/Docker support.

## Getting Started

**Requirements**: Python 3.11+ recommended, ffmpeg only needed for videos.

```bash
# Clone and setup in 3 commands
git clone https://github.com/lucasburgett/rl-reward-tuning-gui
cd rl-reward-tuning-gui
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

## Reproduce in One Command

**For new users**: Clone the repo and run CartPole PPO training + evaluation with a single command:

```bash
chmod +x scripts/reproduce_all.sh
./scripts/reproduce_all.sh
```

This creates the venv, installs dependencies, trains CartPole (100k steps, ~2-3 min), and evaluates deterministically. No W&B or video by default → works on a clean laptop.

## One-Liners

```bash
# Train CartPole PPO (100k steps, ~2-3 min)
python -m src.train env=cartpole algo=ppo total_steps=100000

# Train LunarLander PPO (500k steps, ~15-30 min)  
python -m src.train env=lunarlander algo=ppo total_steps=500000

# Evaluate (no video, deterministic)
python -m src.eval env=cartpole algo=ppo eval.record_video=false

# Evaluate (with video, if ffmpeg installed)
python -m src.eval env=cartpole algo=ppo eval.record_video=true

# Enable Weights & Biases logging
python -m src.train env=cartpole algo=ppo use_wandb=true
```

## Supported Environments & Algorithms

| Environment | Algorithm | Status | Performance Target |
|-------------|-----------|--------|--------------------|
| CartPole-v1 | PPO | ✅ | 500/500 (perfect score) |
| LunarLander-v3 | PPO | ✅ | ≥200 mean return |
| *More coming...* | *PPO, DQN* | 🔄 | *TBD* |

## Determinism Checklist

✅ **Fixed seed**: `seed: 42` (or CLI override: `seed=123`)  
✅ **Deterministic mode**: `deterministic: true` in configs  
✅ **Single-threaded BLAS**: Set in scripts/tests via env vars  
✅ **Config snapshotting**: All hyperparameters saved with artifacts  
✅ **Deterministic evaluation**: No exploration noise during eval  
✅ **Structured artifacts**: Reproducible paths under `artifacts/{env}/{algo}/{seed}/{timestamp}/`

## Artifacts & Logs

**Artifact structure**: `artifacts/{env}/{algo}/{seed}/{timestamp}/`
- `checkpoints/`: Model checkpoints (`.zip` files)
- `videos/`: Evaluation videos (`.mp4` files, if enabled)  
- `logs/`: CSV metrics (`train_metrics.csv`, `eval_metrics.csv`, `monitor.csv`)
- `config/`: Saved configurations (`train.yaml`, `eval.yaml`)

## Troubleshooting

**Missing ffmpeg**: Disable video in eval (`eval.record_video=false`) or install:
- macOS: `brew install ffmpeg`
- Ubuntu: `apt install ffmpeg`

**Box2D issues (LunarLander)**: Install swig first:
- macOS: `brew install swig`  
- Ubuntu: `apt install swig`

**Pre-commit/mypy**: Run `pre-commit run --all-files` to fix formatting  
**Tests failing**: Run `pytest -q` and check CI status  
**Import errors**: Ensure `pip install -r requirements.txt` completed successfully

---

## Advanced Usage

<details>
<summary>Configuration & Hyperparameters</summary>

Override any parameter via CLI:
```bash
# Tune hyperparameters
python -m src.train env=lunarlander total_steps=500000 algo.ppo.learning_rate=1e-4

# Enable Weights & Biases
python -m src.train env=lunarlander use_wandb=true wandb.project=my-experiments
```

**Config files**: `configs/algo/ppo.yaml`, `configs/env/*.yaml`, `configs/config.yaml`
</details>

<details>
<summary>Project Structure</summary>

```
├── src/
│   ├── agents/ppo.py       # PPO agent wrapper (SB3)
│   ├── utils/              # Seeding, logging utilities
│   ├── train.py            # Training entrypoint with Hydra
│   └── eval.py             # Evaluation entrypoint with Hydra  
├── configs/                # YAML configuration files
├── scripts/                # Training/reproduction scripts
├── artifacts/              # Structured training outputs
├── tests/                  # pytest test suite
└── docker/                 # CPU Dockerfile
```
</details>

<details>
<summary>Testing & CI</summary>

```bash
# Run tests
pytest -q

# Run linting
pre-commit run --all-files  

# Build Docker image
docker build -f docker/CPU.Dockerfile -t rl-template-cpu .
```

**CI Status**: Tests run on Python 3.11 with deterministic settings. All jobs (lint, test, smoke) must pass.
</details>

<details>
<summary>Docker Usage</summary>

```bash
# Build CPU image
docker build -f docker/CPU.Dockerfile -t rl-template-cpu .

# Run training in container
docker run --rm -it -v "$PWD:/app" rl-template-cpu \
  python -m src.train env=cartpole algo=ppo total_steps=5000 use_wandb=false
```
</details>
