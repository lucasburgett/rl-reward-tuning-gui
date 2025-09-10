#!/usr/bin/env bash
# Day 10: Demo 3-seed runs with reduced steps for verification
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "=== Day 10: Demo 3-Seed Variance Analysis ==="
echo "Running 3 seeds × 3 environments (DEMO with reduced steps)"

# Deterministic settings
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0

source .venv/bin/activate

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
echo "=== Environment 1: CartPole-v1 (20k steps DEMO) ==="

for seed in 1 2 3; do
    echo "--- Training CartPole seed $seed ---"
    python -m src.train \
        env=cartpole \
        algo=ppo \
        seed="$seed" \
        total_steps=20000 \
        deterministic=true \
        use_wandb=false \
        algo.learning_rate=3e-4 \
        algo.rollout_len=1024 \
        algo.num_minibatches=16 \
        algo.update_epochs=10 \
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
echo "=== Demo completed! ==="
echo "Run manifest: reports/run_manifest.json"
echo
cat reports/run_manifest.json