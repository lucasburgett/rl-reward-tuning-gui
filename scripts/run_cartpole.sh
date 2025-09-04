#!/usr/bin/env bash
# Convenience script to train and evaluate PPO on CartPole
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting CartPole PPO Training & Evaluation${NC}"

# Check if virtual environment exists, create if missing
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install -r requirements-dev.txt
else
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source .venv/bin/activate
fi

echo -e "${GREEN}📊 Training PPO on CartPole-v1${NC}"

# Train PPO with optimized hyperparameters for CartPole
python -m src.train \
  env=cartpole \
  algo=ppo \
  total_steps=200000 \
  eval.run_after_train=true \
  eval.n_episodes=10 \
  "$@"

# Check if training was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Training completed successfully!${NC}"
    
    # Run standalone evaluation with video recording
    echo -e "${GREEN}🎬 Running evaluation with video recording...${NC}"
    python -m src.eval \
      env=cartpole \
      algo=ppo \
      eval.n_episodes=10 \
      eval.record_video=true
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Evaluation completed successfully!${NC}"
        echo -e "${YELLOW}📁 Check artifacts/cartpole/ppo/ for results${NC}"
    else
        echo -e "${RED}❌ Evaluation failed${NC}"
        exit 1
    fi
else
    echo -e "${RED}❌ Training failed${NC}"
    exit 1
fi

echo -e "${GREEN}🎉 All done! Check artifacts/cartpole/ppo/ for results.${NC}"