#!/usr/bin/env bash
set -euo pipefail

# LunarLander training script
# This script trains a PPO agent on LunarLander-v3 using the RL template

echo "🚀 Starting LunarLander training..."

# Ensure venv exists and is activated
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

echo "🔧 Activating virtual environment..."
source .venv/bin/activate

echo "📋 Installing/updating dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "🎯 Starting PPO training on LunarLander-v3..."
echo "   - Environment: LunarLander-v3"
echo "   - Algorithm: PPO"  
echo "   - Total steps: 1M (default, adjust with total_steps=N)"
echo "   - Artifacts will be saved to: artifacts/lunarlander/ppo/{seed}/{timestamp}/"

# Train with default settings (1M steps)
# Users can override settings via CLI, e.g.:
# ./scripts/run_lunarlander.sh total_steps=500000 algo.ppo.rollout_len=4096
python -m src.train env=lunarlander algo=ppo use_wandb=false total_steps=100000 "$@"

echo "✅ Training complete!"
echo ""
echo "🎬 Running evaluation with video recording..."

# Evaluate with video recording
python -m src.eval env=lunarlander algo=ppo eval.record_video=true eval.n_episodes=5

echo "✅ Evaluation complete!"
echo ""
echo "📁 Check the artifacts/ directory for:"
echo "   - Checkpoints: artifacts/lunarlander/ppo/{seed}/{timestamp}/checkpoints/"
echo "   - Videos: artifacts/lunarlander/ppo/{seed}/{timestamp}/videos/"
echo "   - Logs: artifacts/lunarlander/ppo/{seed}/{timestamp}/logs/"
echo "   - Configs: artifacts/lunarlander/ppo/{seed}/{timestamp}/config/"
echo ""
echo "🎯 Target: Mean return ≥ 200 (typically achieved around 500k-1.5M steps)"