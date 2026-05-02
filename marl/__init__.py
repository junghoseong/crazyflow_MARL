"""MARL (Multi-Agent Reinforcement Learning) package for task allocation.

Provides LIA_MADDPG implementation for robot swarm dynamic task allocation.
"""

from marl.envs.task_allocation import EnvConfig, EnvState, TaskAllocationEnv
from marl.lia import compute_lia
from marl.maddpg import MADDPGConfig
