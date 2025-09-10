#!/usr/bin/env bash
# Day 11: Complete reproducible RL runs - one command for fresh 3×3 seed matrix
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "=== Day 11: Complete Reproducible RL Pipeline ==="
echo "Fresh 3-seed runs × 3 environments with comprehensive reporting"

# Environment variables for configuration
SEEDS="${SEEDS:-"1 2 3"}"
STEPS_CARTPOLE="${STEPS_CARTPOLE:-100000}"
STEPS_LUNAR="${STEPS_LUNAR:-1000000}"
STEPS_REACHER="${STEPS_REACHER:-800000}"
USE_WANDB="${USE_WANDB:-false}"
WANDB_MODE="${WANDB_MODE:-offline}"

echo "Configuration:"
echo "  Seeds: $SEEDS"
echo "  Steps - CartPole: $STEPS_CARTPOLE, LunarLander: $STEPS_LUNAR, Reacher: $STEPS_REACHER"
echo "  W&B enabled: $USE_WANDB (mode: $WANDB_MODE)"

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

echo "[Setup] Activating virtual environment..."
source .venv/bin/activate

echo "[Setup] Checking dependencies..."
if ! python -c "import torch, gymnasium, mujoco, stable_baselines3" &>/dev/null; then
    echo "[Setup] Installing dependencies..."
    python -m pip install --upgrade pip --quiet
    pip install -r requirements.txt --quiet
else
    echo "[Setup] Dependencies already installed"
fi

# Clean up old reports and create fresh structure
echo "[Setup] Preparing reports directory..."
rm -rf reports
mkdir -p reports

# Initialize run manifest
cat > reports/run_manifest.json << 'EOF'
{
  "generated_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "config": {
    "seeds": "PLACEHOLDER_SEEDS",
    "steps": {
      "cartpole": $STEPS_CARTPOLE,
      "lunarlander": $STEPS_LUNAR,
      "reacher": $STEPS_REACHER
    }
  },
  "runs": []
}
EOF

# Replace placeholders with actual values using simple sed commands
sed -i.bak "s/\$(date -u +\"%Y-%m-%dT%H:%M:%SZ\")/$(date -u +"%Y-%m-%dT%H:%M:%SZ")/g" reports/run_manifest.json
sed -i.bak "s/PLACEHOLDER_SEEDS/$SEEDS/g" reports/run_manifest.json
sed -i.bak "s/\$STEPS_CARTPOLE/$STEPS_CARTPOLE/g" reports/run_manifest.json
sed -i.bak "s/\$STEPS_LUNAR/$STEPS_LUNAR/g" reports/run_manifest.json
sed -i.bak "s/\$STEPS_REACHER/$STEPS_REACHER/g" reports/run_manifest.json
rm reports/run_manifest.json.bak

# Function to add run to manifest
add_to_manifest() {
    local env="$1"
    local seed="$2"
    local run_dir="$3"
    
    python3 -c "
import json
import sys

try:
    with open('reports/run_manifest.json', 'r') as f:
        data = json.load(f)
    
    data['runs'].append({
        'env': '$env',
        'seed': $seed,
        'run_dir': '$run_dir',
        'completed_at': '$(date -u +"%Y-%m-%dT%H:%M:%SZ")'
    })
    
    with open('reports/run_manifest.json', 'w') as f:
        json.dump(data, f, indent=2)
        
    print('Added to manifest: $env/seed$seed -> $run_dir')
except Exception as e:
    print(f'Warning: Failed to update manifest: {e}', file=sys.stderr)
"
}

# Function to find latest run directory
find_latest_run_dir() {
    local env="$1"
    local seed="$2"
    find "artifacts/${env}/ppo/${seed}" -type d -name "checkpoints" -exec dirname {} \; 2>/dev/null | sort -r | head -1
}

# Function to get steps for environment
get_env_steps() {
    case "$1" in
        "cartpole") echo "$STEPS_CARTPOLE" ;;
        "lunarlander") echo "$STEPS_LUNAR" ;;
        "reacher_mujoco") echo "$STEPS_REACHER" ;;
        *) echo "100000" ;;
    esac
}

# Function to get display name for environment
get_env_display() {
    case "$1" in
        "cartpole") echo "CartPole-v1" ;;
        "lunarlander") echo "LunarLander-v2" ;;
        "reacher_mujoco") echo "Reacher-v5" ;;
        *) echo "$1" ;;
    esac
}

# Main training loop
total_runs=0
completed_runs=0

for env in cartpole lunarlander reacher_mujoco; do
    steps=$(get_env_steps "$env")
    env_display=$(get_env_display "$env")
    
    echo
    echo "=== Environment: $env_display ($steps steps) ==="
    
    for seed in $SEEDS; do
        echo "--- Training $env_display with seed $seed ---"
        ((total_runs++))
        
        # Configure W&B settings if enabled
        wandb_args=""
        if [ "$USE_WANDB" = "true" ]; then
            wandb_args="use_wandb=true wandb.mode=$WANDB_MODE"
        fi
        
        # Train the agent
        if python -m src.train \
            env="$env" \
            algo=ppo \
            seed="$seed" \
            total_steps="$steps" \
            deterministic=true \
            $wandb_args \
            eval.run_after_train=true \
            eval.record_video=false; then
            
            # Find the run directory that was just created
            run_dir=$(find_latest_run_dir "$env" "$seed")
            if [ -n "$run_dir" ] && [ -d "$run_dir" ]; then
                echo "✓ Training completed: $run_dir"
                add_to_manifest "$env" "$seed" "$run_dir"
                
                # Run deterministic evaluation
                echo "--- Evaluating $env_display with seed $seed ---"
                if python -m src.eval \
                    env="$env" \
                    algo=ppo \
                    seed="$seed" \
                    deterministic=true \
                    eval.n_episodes=10 \
                    eval.record_video=false \
                    log_dir="$run_dir"; then
                    
                    echo "✓ Evaluation completed for $env_display/seed$seed"
                    ((completed_runs++))
                else
                    echo "⚠ Evaluation failed for $env_display/seed$seed"
                fi
            else
                echo "⚠ Could not find run directory for $env_display/seed$seed"
            fi
        else
            echo "⚠ Training failed for $env_display/seed$seed"
        fi
    done
done

echo
echo "=== Training Summary ==="
echo "Completed runs: $completed_runs/$total_runs"

if [ $completed_runs -eq 0 ]; then
    echo "❌ No runs completed successfully"
    exit 1
fi

# Run aggregation
echo
echo "=== Aggregating Results ==="
if python scripts/aggregate_results.py; then
    echo "✓ Results aggregation completed"
else
    echo "⚠ Results aggregation failed, but continuing..."
fi

# Display final results
echo
echo "=== Day 11 Pipeline Complete ==="
echo
echo "Generated files:"

if [ -f "reports/results_seed_stats.csv" ]; then
    echo "  📊 $(realpath reports/results_seed_stats.csv)"
fi

if [ -f "reports/results.md" ]; then
    echo "  📄 $(realpath reports/results.md)"
    echo
    echo "=== Results Preview ==="
    head -20 reports/results.md
fi

if [ -f "reports/run_manifest.json" ]; then
    echo
    echo "  📋 $(realpath reports/run_manifest.json)"
fi

echo
echo "🎉 Reproducible RL pipeline completed with $completed_runs runs!"
echo "   View full results: cat reports/results.md"