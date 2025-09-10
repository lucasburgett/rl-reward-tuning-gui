#!/usr/bin/env bash
# Day 10: Clean 3-seed runs for CartPole, LunarLander, Reacher with proper PPO settings
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "=== Day 10: Clean 3-Seed Variance Analysis ==="
echo "Running 3 seeds × 3 environments with optimized PPO settings"

# Deterministic, CPU-friendly defaults
export OMP_NUM_THREADS=1 
export MKL_NUM_THREADS=1 
export OPENBLAS_NUM_THREADS=1 
export NUMEXPR_NUM_THREADS=1 
export PYTHONHASHSEED=0

# Ensure venv & deps (idempotent)
if [ ! -d ".venv" ]; then 
    echo "[Setup] Creating virtual environment..."
    python3 -m venv .venv
fi
source .venv/bin/activate
echo "[Setup] Installing/updating dependencies..."
python -m pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt > /dev/null 2>&1

# Create reports directory and initialize manifest
mkdir -p reports
echo '{"runs": []}' > reports/run_manifest.json

# Function to add run to manifest
add_to_manifest() {
    local env="$1"
    local seed="$2"
    local run_dir="$3"
    
    python3 -c "
import json
with open('reports/run_manifest.json', 'r') as f:
    data = json.load(f)
data['runs'].append({'env': '$env', 'seed': $seed, 'run_dir': '$run_dir'})
with open('reports/run_manifest.json', 'w') as f:
    json.dump(data, f, indent=2)
"
}

# Function to find latest run directory
find_latest_run_dir() {
    local env="$1"
    local seed="$2"
    find "artifacts/${env}/ppo/${seed}" -type d -name "checkpoints" -exec dirname {} \; 2>/dev/null | sort -r | head -1
}

echo
echo "=== Environment 1: CartPole-v1 (100k steps, fast PPO) ==="

for seed in 1 2 3; do
    echo "--- Training CartPole seed $seed ---"
    python -m src.train \
        env=cartpole \
        algo=ppo \
        seed="$seed" \
        total_steps=100000 \
        deterministic=true \
        use_wandb=false \
        algo.learning_rate=3e-4 \
        algo.rollout_len=1024 \
        algo.num_minibatches=16 \
        algo.update_epochs=10 \
        algo.gamma=0.99 \
        algo.gae_lambda=0.95 \
        eval.run_after_train=true \
        eval.record_video=false
    
    # Find and register run directory
    run_dir=$(find_latest_run_dir "cartpole" "$seed")
    echo "Run directory: $run_dir"
    add_to_manifest "cartpole" "$seed" "$run_dir"
    
    # Run deterministic eval
    echo "--- Evaluating CartPole seed $seed ---"
    python -m src.eval \
        env=cartpole \
        algo=ppo \
        seed="$seed" \
        deterministic=true \
        eval.n_episodes=10 \
        eval.record_video=false \
        log_dir="$run_dir"
done

# Export one video for CartPole (seed 1)
echo "--- Recording CartPole video (seed 1) ---"
cartpole_run_dir=$(find_latest_run_dir "cartpole" "1")
python -m src.eval \
    env=cartpole \
    algo=ppo \
    seed=1 \
    deterministic=true \
    eval.n_episodes=1 \
    eval.record_video=true \
    log_dir="$cartpole_run_dir"

echo
echo "=== Environment 2: LunarLander-v2 (1M steps, robust PPO) ==="

for seed in 1 2 3; do
    echo "--- Training LunarLander seed $seed ---"
    python -m src.train \
        env=lunarlander \
        algo=ppo \
        seed="$seed" \
        total_steps=1000000 \
        deterministic=true \
        use_wandb=false \
        algo.learning_rate=3e-4 \
        algo.rollout_len=2048 \
        algo.num_minibatches=32 \
        algo.update_epochs=10 \
        algo.gamma=0.99 \
        algo.gae_lambda=0.95 \
        eval.run_after_train=true \
        eval.record_video=false
    
    # Find and register run directory
    run_dir=$(find_latest_run_dir "lunarlander" "$seed")
    echo "Run directory: $run_dir"
    add_to_manifest "lunarlander" "$seed" "$run_dir"
    
    # Run deterministic eval
    echo "--- Evaluating LunarLander seed $seed ---"
    python -m src.eval \
        env=lunarlander \
        algo=ppo \
        seed="$seed" \
        deterministic=true \
        eval.n_episodes=10 \
        eval.record_video=false \
        log_dir="$run_dir"
done

# Export one video for LunarLander (seed 1)
echo "--- Recording LunarLander video (seed 1) ---"
lunarlander_run_dir=$(find_latest_run_dir "lunarlander" "1")
python -m src.eval \
    env=lunarlander \
    algo=ppo \
    seed=1 \
    deterministic=true \
    eval.n_episodes=1 \
    eval.record_video=true \
    log_dir="$lunarlander_run_dir"

echo
echo "=== Environment 3: Reacher-v5 (800k steps, MuJoCo + wrappers) ==="

for seed in 1 2 3; do
    echo "--- Training Reacher seed $seed ---"
    python -m src.train \
        env=reacher_mujoco \
        algo=ppo \
        seed="$seed" \
        total_steps=800000 \
        deterministic=true \
        use_wandb=false \
        algo.learning_rate=3e-4 \
        algo.rollout_len=2048 \
        algo.num_minibatches=32 \
        algo.update_epochs=10 \
        algo.gamma=0.99 \
        algo.gae_lambda=0.95 \
        eval.run_after_train=true \
        eval.record_video=false
    
    # Find and register run directory
    run_dir=$(find_latest_run_dir "reacher" "$seed")
    echo "Run directory: $run_dir"
    add_to_manifest "reacher" "$seed" "$run_dir"
    
    # Run deterministic eval
    echo "--- Evaluating Reacher seed $seed ---"
    python -m src.eval \
        env=reacher_mujoco \
        algo=ppo \
        seed="$seed" \
        deterministic=true \
        eval.n_episodes=10 \
        eval.record_video=false \
        log_dir="$run_dir"
done

# Export one video for Reacher (seed 1)
echo "--- Recording Reacher video (seed 1) ---"
reacher_run_dir=$(find_latest_run_dir "reacher" "1")
python -m src.eval \
    env=reacher_mujoco \
    algo=ppo \
    seed=1 \
    deterministic=true \
    eval.n_episodes=1 \
    eval.record_video=true \
    log_dir="$reacher_run_dir"

echo
echo "=== All runs completed! ==="
echo "Run manifest: reports/run_manifest.json"
echo "Next: python scripts/aggregate_results.py"
echo
echo "Summary:"
echo "- 9 training runs (3 seeds × 3 environments)"  
echo "- 9 deterministic evaluations"
echo "- 3 video exports (1 per environment)"
cat reports/run_manifest.json