# crazyflow_MARL

Multi-Agent Reinforcement Learning experiments using [Crazyflow](https://github.com/utiasDSL/crazyflow) drone environments.

## Setup

```bash
# Install crazyflow (from source)
pip install -e /path/to/crazyflow

# Install this project
pip install -e .
```

## Training

### PPO - Reach Position
```bash
python scripts/train_ppo.py --env_id DroneReachPos-v0 --num_envs 64 --total_timesteps 2000000
```

### PPO - Figure-8 Trajectory
```bash
python scripts/train_ppo.py --env_id DroneFigureEightTrajectory-v0 --num_envs 64 --total_timesteps 2000000 --max_episode_time 10.0
```

### Key Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--env_id` | `DroneReachPos-v0` | Gymnasium environment ID |
| `--num_envs` | `64` | Number of parallel environments |
| `--total_timesteps` | `2000000` | Total training timesteps |
| `--learning_rate` | `3e-4` | Learning rate |
| `--num_steps` | `128` | Rollout length per update |
| `--gamma` | `0.99` | Discount factor |
| `--clip_eps` | `0.2` | PPO clipping epsilon |
| `--save_path` | `""` | Path to save model checkpoints |

## Project Structure
```
crazyflow_MARL/
├── scripts/          # Training scripts
│   └── train_ppo.py  # PPO implementation (JAX/Flax)
├── configs/          # Experiment configurations
├── pyproject.toml    # Project dependencies
└── README.md
```

## Available Environments
- `DroneReachPos-v0` — Reach a random goal position
- `DroneReachVel-v0` — Reach a target velocity
- `DroneLanding-v0` — Land at a target position
- `DroneFigureEightTrajectory-v0` — Follow a figure-8 trajectory
