# rl-reward-tuning-gui

Interactive tool to visualize and tweak reinforcement learning reward terms and inspect induced behaviors in near-real time.

## Day 2 - Seeds + Config Plumbing

### Usage

Training with configuration composition:
```bash
python -m src.train env=cartpole algo=ppo
```

Expected output:
```
[Versions] Python 3.12.7 | torch not-installed | gymnasium not-installed
[Seed] set_seed: 42 | deterministic: True

[Composed Config]
seed: 42
deterministic: true
device: auto
total_steps: 100000
log_dir: experiments/${now:%Y-%m-%d}/${now:%H-%M-%S}
use_wandb: false
env: cartpole
algo: ppo

✅ Train stub complete; exiting.
```

Evaluation with configuration composition:
```bash
python -m src.eval env=cartpole algo=ppo
```

Expected output:
```
[Versions] Python 3.12.7 | torch not-installed | gymnasium not-installed
[Seed] set_seed: 42 | deterministic: True

[Composed Config]
seed: 42
deterministic: true
device: auto
episodes: 10
render: false
checkpoint: null
env: cartpole
algo: ppo

✅ Eval stub complete; exiting.
```

### Project Structure

```
configs/
  env/cartpole.yaml      # Environment configurations
  algo/ppo.yaml          # Algorithm configurations  
  train/default.yaml     # Training session defaults
  eval/default.yaml      # Evaluation session defaults
src/
  train.py               # Training entrypoint with Hydra
  eval.py                # Evaluation entrypoint with Hydra
  utils/
    seeding.py           # Deterministic seeding utilities
```
