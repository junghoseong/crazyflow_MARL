"""2D Task Allocation environment for robot swarm dynamic task allocation.

Implements the Dec-POMDP formulation from:
  "A Local Information Aggregation based Multi-Agent Reinforcement Learning
   for Robot Swarm Dynamic Task Allocation" (Lv, Lei, Yi)

Environment: N robots navigate a 2D space to reach M movable tasks.
Each robot selects a target task at each timestep and moves toward it.
Binding occurs when a robot is within d_bind distance of its target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array


@dataclass
class EnvConfig:
    """Configuration for the Task Allocation environment."""

    n_robots: int = 60
    n_tasks: int = 10
    area_size: float = 1000.0

    robot_speed_min: float = 2.0
    robot_speed_max: float = 5.0
    task_speed_min: float = 0.5
    task_speed_max: float = 1.0

    d_bind: float = 30.0
    max_neighbors: int = 10
    max_steps: int = 150
    tau: float = 1.0  # decision time step

    phi_1: float = 10.0
    phi_2: float = 0.001  # stored as positive; applied as -phi_2 per step
    phi_3: float = 1.0

    @property
    def task_max_demand(self) -> int:
        return math.ceil(self.n_robots / self.n_tasks)

    @property
    def obs_self_dim(self) -> int:
        return 5  # x, y, v, direction, status

    @property
    def obs_task_dim(self) -> int:
        return 6 * self.n_tasks  # (dx, dy, dv, d_theta, h_j, kappa) per task

    @property
    def obs_neighbor_dim(self) -> int:
        return 5 * self.max_neighbors  # (dx, dy, dv, d_theta, prev_target) per neighbor

    @property
    def obs_dim(self) -> int:
        return self.obs_self_dim + self.obs_task_dim + self.obs_neighbor_dim


class EnvState(NamedTuple):
    """Full environment state (JAX-compatible pytree)."""

    robot_pos: Array       # (N, 2)
    robot_speed: Array     # (N,)
    robot_dir: Array       # (N,)   current heading angle
    robot_status: Array    # (N,)   1=active, 0=bound
    robot_target: Array    # (N,)   int, current target task index
    task_pos: Array        # (M, 2)
    task_speed: Array      # (M,)
    task_angle: Array      # (M,)
    task_bound_count: Array  # (M,) int, number of robots bound
    task_rewards: Array    # (M,)   reward value r_{i,j} for each task (U[0,1])
    step_count: Array      # scalar int
    rng_key: Array         # JAX PRNG key


class StepResult(NamedTuple):
    """Result of a single environment step."""

    state: EnvState
    obs: Array          # (N, obs_dim)
    rewards: Array      # (N,)
    dones: Array        # (N,) per-robot done flags
    info: dict


class TaskAllocationEnv:
    """Pure-JAX 2D Task Allocation environment."""

    def __init__(self, config: EnvConfig | None = None):
        self.cfg = config or EnvConfig()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: Array) -> tuple[EnvState, Array]:
        """Reset environment and return initial (state, observations)."""
        cfg = self.cfg
        k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)

        robot_pos = jax.random.uniform(k1, (cfg.n_robots, 2), minval=0.0, maxval=cfg.area_size)
        robot_speed = jax.random.uniform(
            k2, (cfg.n_robots,), minval=cfg.robot_speed_min, maxval=cfg.robot_speed_max
        )
        robot_dir = jnp.zeros(cfg.n_robots)
        robot_status = jnp.ones(cfg.n_robots)  # all active
        robot_target = jnp.zeros(cfg.n_robots, dtype=jnp.int32)

        task_pos = jax.random.uniform(k3, (cfg.n_tasks, 2), minval=0.0, maxval=cfg.area_size)
        task_speed = jax.random.uniform(
            k4, (cfg.n_tasks,), minval=cfg.task_speed_min, maxval=cfg.task_speed_max
        )
        task_angle = jax.random.uniform(k5, (cfg.n_tasks,), minval=-jnp.pi, maxval=jnp.pi)
        task_bound_count = jnp.zeros(cfg.n_tasks, dtype=jnp.int32)
        task_rewards = jax.random.uniform(k6, (cfg.n_tasks,), minval=0.0, maxval=1.0)

        state = EnvState(
            robot_pos=robot_pos,
            robot_speed=robot_speed,
            robot_dir=robot_dir,
            robot_status=robot_status,
            robot_target=robot_target,
            task_pos=task_pos,
            task_speed=task_speed,
            task_angle=task_angle,
            task_bound_count=task_bound_count,
            task_rewards=task_rewards,
            step_count=jnp.int32(0),
            rng_key=k7,
        )
        obs = self._get_obs(state)
        return state, obs

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: EnvState, actions: Array) -> StepResult:
        """Execute one environment step.

        Args:
            state: Current environment state.
            actions: (N, M) action logits or one-hot vectors per robot.

        Returns:
            StepResult with new state, observations, rewards, dones, info.
        """
        cfg = self.cfg
        key, k_task = jax.random.split(state.rng_key)

        # 1. Decode actions -> target task indices (argmax of logits)
        new_targets = jnp.argmax(actions, axis=-1).astype(jnp.int32)  # (N,)
        # Bound robots keep their old target
        robot_target = jnp.where(state.robot_status > 0.5, new_targets, state.robot_target)

        # 2. Update task positions (Eq. 1): random angle perturbation
        angle_perturb = jax.random.uniform(
            k_task, (cfg.n_tasks,), minval=-jnp.pi / 4, maxval=jnp.pi / 4
        )
        new_task_angle = state.task_angle + angle_perturb
        task_dx = state.task_speed * cfg.tau * jnp.cos(new_task_angle)
        task_dy = state.task_speed * cfg.tau * jnp.sin(new_task_angle)
        new_task_pos = state.task_pos + jnp.stack([task_dx, task_dy], axis=-1)
        # Clamp tasks within area
        new_task_pos = jnp.clip(new_task_pos, 0.0, cfg.area_size)

        # 3. Update robot positions (Eq. 2)
        target_pos = new_task_pos[robot_target]  # (N, 2)
        diff = target_pos - state.robot_pos       # (N, 2)
        direction = jnp.arctan2(diff[:, 1], diff[:, 0])  # (N,)

        move_dx = state.robot_speed * cfg.tau * jnp.cos(direction)
        move_dy = state.robot_speed * cfg.tau * jnp.sin(direction)
        new_robot_pos_candidate = state.robot_pos + jnp.stack([move_dx, move_dy], axis=-1)

        # Bound robots sync with their task
        bound_task_pos = new_task_pos[state.robot_target]
        new_robot_pos = jnp.where(
            (state.robot_status > 0.5)[:, None],
            new_robot_pos_candidate,
            bound_task_pos,
        )

        # 4. Check binding: active robots within d_bind of target
        dist_to_target = jnp.linalg.norm(new_robot_pos - new_task_pos[robot_target], axis=-1)
        newly_bound = (state.robot_status > 0.5) & (dist_to_target <= cfg.d_bind)
        new_robot_status = jnp.where(newly_bound, 0.0, state.robot_status)

        # Update task bound counts
        h_bar = cfg.task_max_demand
        # Count new bindings per task
        new_bind_counts = jnp.zeros(cfg.n_tasks, dtype=jnp.int32)
        new_bind_counts = new_bind_counts.at[robot_target].add(
            newly_bound.astype(jnp.int32)
        )
        new_task_bound_count = state.task_bound_count + new_bind_counts

        # 5. Compute rewards (Eq. 7-9)
        rewards = self._compute_rewards(
            state, robot_target, newly_bound, new_task_bound_count, dist_to_target
        )

        # 6. Build new state
        new_step = state.step_count + 1
        done = new_step >= cfg.max_steps
        all_bound = jnp.all(new_robot_status < 0.5)
        episode_done = done | all_bound

        new_state = EnvState(
            robot_pos=new_robot_pos,
            robot_speed=state.robot_speed,
            robot_dir=direction,
            robot_status=new_robot_status,
            robot_target=robot_target,
            task_pos=new_task_pos,
            task_speed=state.task_speed,
            task_angle=new_task_angle,
            task_bound_count=new_task_bound_count,
            task_rewards=state.task_rewards,
            step_count=new_step,
            rng_key=key,
        )

        obs = self._get_obs(new_state)
        dones = jnp.full(cfg.n_robots, episode_done, dtype=jnp.bool_)

        return StepResult(
            state=new_state,
            obs=obs,
            rewards=rewards,
            dones=dones,
            info={"episode_done": episode_done},
        )

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    @partial(jax.jit, static_argnums=(0,))
    def _get_obs(self, state: EnvState) -> Array:
        """Compute per-robot observations. Returns (N, obs_dim)."""
        cfg = self.cfg
        N, M = cfg.n_robots, cfg.n_tasks

        # Normalize positions to [0, 1]
        norm_robot_pos = state.robot_pos / cfg.area_size
        norm_task_pos = state.task_pos / cfg.area_size

        # o_self: (x, y, v, direction, status) per robot - (N, 5)
        norm_speed = (state.robot_speed - cfg.robot_speed_min) / (
            cfg.robot_speed_max - cfg.robot_speed_min + 1e-8
        )
        o_self = jnp.stack([
            norm_robot_pos[:, 0],
            norm_robot_pos[:, 1],
            norm_speed,
            state.robot_dir / jnp.pi,
            state.robot_status,
        ], axis=-1)  # (N, 5)

        # o_task: relative info between each robot and all tasks - (N, 6*M)
        # For each robot i, for each task j: (dx, dy, dv, d_theta, h_j, kappa)
        dx = norm_task_pos[None, :, 0] - norm_robot_pos[:, None, 0]  # (N, M)
        dy = norm_task_pos[None, :, 1] - norm_robot_pos[:, None, 1]  # (N, M)

        norm_task_speed = (state.task_speed - cfg.task_speed_min) / (
            cfg.task_speed_max - cfg.task_speed_min + 1e-8
        )
        dv = norm_task_speed[None, :] - norm_speed[:, None]  # (N, M)
        d_theta = state.task_angle[None, :] / jnp.pi - state.robot_dir[:, None] / jnp.pi  # (N, M)
        h_j = state.task_bound_count[None, :].astype(jnp.float32) / max(cfg.task_max_demand, 1)
        h_j = jnp.broadcast_to(h_j, (N, M))
        # kappa: normalized task reward weight
        kappa = state.task_rewards[None, :] / (jnp.max(state.task_rewards) + 1e-8)
        kappa = jnp.broadcast_to(kappa, (N, M))

        o_task = jnp.concatenate([dx, dy, dv, d_theta, h_j, kappa], axis=-1)  # (N, 6*M)

        # o_neighbor: relative info between each robot and its alpha nearest neighbors
        # Compute pairwise distances
        robot_dists = jnp.linalg.norm(
            state.robot_pos[:, None, :] - state.robot_pos[None, :, :], axis=-1
        )  # (N, N)
        # Mask self
        robot_dists = robot_dists + jnp.eye(N) * 1e9

        # Find alpha nearest neighbors per robot
        alpha = cfg.max_neighbors
        _, neighbor_indices = jax.lax.top_k(-robot_dists, k=alpha)  # (N, alpha), smallest dists

        # Gather neighbor info
        nb_pos = norm_robot_pos[neighbor_indices]           # (N, alpha, 2)
        nb_speed = norm_speed[neighbor_indices]              # (N, alpha)
        nb_dir = state.robot_dir[neighbor_indices]           # (N, alpha)
        nb_target = state.robot_target[neighbor_indices]     # (N, alpha)

        nb_dx = nb_pos[:, :, 0] - norm_robot_pos[:, None, 0]  # (N, alpha)
        nb_dy = nb_pos[:, :, 1] - norm_robot_pos[:, None, 1]  # (N, alpha)
        nb_dv = nb_speed - norm_speed[:, None]                 # (N, alpha)
        nb_dtheta = nb_dir / jnp.pi - state.robot_dir[:, None] / jnp.pi  # (N, alpha)
        # Encode prev target as float normalized by n_tasks
        nb_prev_target = nb_target.astype(jnp.float32) / max(M, 1)  # (N, alpha)

        o_neighbor = jnp.concatenate(
            [nb_dx, nb_dy, nb_dv, nb_dtheta, nb_prev_target], axis=-1
        )  # (N, 5*alpha)

        obs = jnp.concatenate([o_self, o_task, o_neighbor], axis=-1)  # (N, obs_dim)
        return obs

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    def _compute_rewards(
        self,
        state: EnvState,
        robot_target: Array,
        newly_bound: Array,
        task_bound_count: Array,
        dist_to_target: Array,
    ) -> Array:
        """Compute per-robot rewards (Eq. 7-9)."""
        cfg = self.cfg
        h_bar = cfg.task_max_demand
        target_h = task_bound_count[robot_target]  # (N,)
        target_reward = state.task_rewards[robot_target]  # (N,)

        # r_dis: constant negative for active robots
        r_dis = -cfg.phi_2 * state.robot_status

        # r_step: penalize overcrowded tasks (Eq. 8)
        # h_bar_j - current count for chosen task (can be negative = penalty)
        h_diff = h_bar - task_bound_count[robot_target].astype(jnp.float32)
        r_step = cfg.phi_3 * h_diff * state.robot_status

        # r_final: reward for newly bound robots if task not over capacity (Eq. 9)
        task_under_capacity = target_h <= h_bar
        r_final = cfg.phi_1 * target_reward * newly_bound.astype(jnp.float32) * task_under_capacity

        rewards = r_dis + r_step + r_final
        return rewards

    # ------------------------------------------------------------------
    # Neighbor graph (for external use by LIA)
    # ------------------------------------------------------------------
    @partial(jax.jit, static_argnums=(0,))
    def get_neighbor_indices(self, state: EnvState) -> Array:
        """Get alpha-nearest neighbor indices for each robot. Returns (N, alpha)."""
        N = self.cfg.n_robots
        alpha = self.cfg.max_neighbors
        robot_dists = jnp.linalg.norm(
            state.robot_pos[:, None, :] - state.robot_pos[None, :, :], axis=-1
        )
        robot_dists = robot_dists + jnp.eye(N) * 1e9
        _, indices = jax.lax.top_k(-robot_dists, k=alpha)
        return indices

    @partial(jax.jit, static_argnums=(0,))
    def get_pairwise_distances(self, state: EnvState) -> Array:
        """Get pairwise distance matrix between all robots. Returns (N, N)."""
        return jnp.linalg.norm(
            state.robot_pos[:, None, :] - state.robot_pos[None, :, :], axis=-1
        )
