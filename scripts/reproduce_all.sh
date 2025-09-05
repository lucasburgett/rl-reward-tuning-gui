#!/usr/bin/env bash
set -euo pipefail

# Reproducible RL Template - Complete Reproduction Script
# Clone → Install → Train → Eval in one command

# Repo root
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Deterministic CPU settings for reproducible results
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0

echo "[Reproduce] Setting up Python environment..."

# Create Python virtual environment if missing
if [ ! -d ".venv" ]; then
  echo "[Reproduce] Creating .venv..."
  python3 -m venv .venv
fi

# Activate virtual environment
# shellcheck source=/dev/null
source .venv/bin/activate

# Install dependencies
echo "[Reproduce] Installing dependencies..."
python -m pip install --upgrade pip
pip install -r requirements.txt

# Train CartPole with PPO (100k steps, deterministic, CPU-only)
echo "[Reproduce] Training CartPole with PPO (100k steps, ~2-3 min)..."
python -m src.train env=cartpole algo=ppo total_steps=100000 use_wandb=false

# Deterministic evaluation (no video to avoid ffmpeg requirement)  
echo "[Reproduce] Evaluating deterministically (10 episodes, no video)..."
python -m src.eval env=cartpole algo=ppo eval.record_video=false eval.n_episodes=10

# Success message with artifact location
echo ""
echo "[Reproduce] ✅ Success! CartPole PPO training and evaluation complete."
echo "[Reproduce] Check results in: ./artifacts/cartpole/ppo/42/<timestamp>/"
echo ""
echo "Next steps:"
echo "  • Enable video: python -m src.eval env=cartpole algo=ppo eval.record_video=true"
echo "  • Try LunarLander: python -m src.train env=lunarlander algo=ppo total_steps=500000"
echo "  • Enable W&B logging: python -m src.train env=cartpole use_wandb=true"
echo ""