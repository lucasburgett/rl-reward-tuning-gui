#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Deterministic, CPU-friendly
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0

# Ensure venv & deps (just MuJoCo, skip Box2D)
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
# shellcheck source=/dev/null
source .venv/bin/activate
python -m pip install --upgrade pip
pip install mujoco>=2.3.0 gymnasium[mujoco]>=0.29.1

# Import & random-step smoke
python -c "
import gymnasium as gym
env = gym.make('Ant-v5')
obs, info = env.reset(seed=0)
steps = 0
terminated = truncated = False
while steps < 50 and not (terminated or truncated):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    steps += 1
print('OK: Ant-v5 reset/step for', steps, 'steps; obs shape =', obs.shape)
env.close()
print('[SMOKE] MuJoCo robotics OK.')
"

echo "[SMOKE] MuJoCo robotics OK."