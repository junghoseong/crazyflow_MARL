"""Experience Replay Buffer for LIA_MADDPG.

Stores multi-agent transitions including LIA-aggregated information.
Supports uniform and prioritized sampling.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array


class Transition(NamedTuple):
    """A single multi-agent transition."""

    obs: np.ndarray           # (N, obs_dim)
    actions: np.ndarray       # (N, M) one-hot
    actions_idx: np.ndarray   # (N,) int target indices
    phi_obs: np.ndarray       # (N, obs_dim) LIA aggregated obs
    phi_actions: np.ndarray   # (N, M) LIA aggregated actions
    rewards: np.ndarray       # (N,)
    next_obs: np.ndarray      # (N, obs_dim)
    dones: np.ndarray         # (N,) bool
    robot_pos: np.ndarray     # (N, 2) for LIA recomputation


class ReplayBuffer:
    """Numpy-based replay buffer with optional priority sampling.

    Stores full multi-agent transitions. Uses numpy for storage to avoid
    JAX memory fragmentation, converts to JAX arrays on sample.
    """

    def __init__(
        self,
        capacity: int,
        n_robots: int,
        obs_dim: int,
        n_tasks: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_steps: int = 100_000,
    ):
        self.capacity = capacity
        self.n_robots = n_robots
        self.obs_dim = obs_dim
        self.n_tasks = n_tasks
        self.alpha = alpha  # priority exponent
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_steps = beta_steps

        self._ptr = 0
        self._size = 0
        self._total_steps = 0

        # Pre-allocate storage arrays
        self._obs = np.zeros((capacity, n_robots, obs_dim), dtype=np.float32)
        self._actions = np.zeros((capacity, n_robots, n_tasks), dtype=np.float32)
        self._actions_idx = np.zeros((capacity, n_robots), dtype=np.int32)
        self._phi_obs = np.zeros((capacity, n_robots, obs_dim), dtype=np.float32)
        self._phi_actions = np.zeros((capacity, n_robots, n_tasks), dtype=np.float32)
        self._rewards = np.zeros((capacity, n_robots), dtype=np.float32)
        self._next_obs = np.zeros((capacity, n_robots, obs_dim), dtype=np.float32)
        self._dones = np.zeros((capacity, n_robots), dtype=np.float32)
        self._robot_pos = np.zeros((capacity, n_robots, 2), dtype=np.float32)

        # Priority tree (sum-tree for efficient sampling)
        self._priorities = np.ones(capacity, dtype=np.float64)
        self._max_priority = 1.0

    @property
    def size(self) -> int:
        return self._size

    @property
    def beta(self) -> float:
        """Linearly annealed importance sampling exponent."""
        frac = min(self._total_steps / max(self.beta_steps, 1), 1.0)
        return self.beta_start + frac * (self.beta_end - self.beta_start)

    def add(self, transition: Transition) -> None:
        """Add a transition to the buffer."""
        idx = self._ptr

        self._obs[idx] = np.asarray(transition.obs)
        self._actions[idx] = np.asarray(transition.actions)
        self._actions_idx[idx] = np.asarray(transition.actions_idx)
        self._phi_obs[idx] = np.asarray(transition.phi_obs)
        self._phi_actions[idx] = np.asarray(transition.phi_actions)
        self._rewards[idx] = np.asarray(transition.rewards)
        self._next_obs[idx] = np.asarray(transition.next_obs)
        self._dones[idx] = np.asarray(transition.dones)
        self._robot_pos[idx] = np.asarray(transition.robot_pos)

        self._priorities[idx] = self._max_priority ** self.alpha

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        self._total_steps += 1

    def sample(
        self, batch_size: int, rng: np.random.Generator | None = None
    ) -> tuple[dict[str, Array], np.ndarray, np.ndarray]:
        """Sample a batch of transitions.

        Args:
            batch_size: number of transitions to sample.
            rng: numpy random generator.

        Returns:
            batch: dict of JAX arrays with keys matching Transition fields.
            indices: sampled indices (for priority updates).
            weights: importance sampling weights.
        """
        if rng is None:
            rng = np.random.default_rng()

        priorities = self._priorities[:self._size]
        probs = priorities / priorities.sum()
        indices = rng.choice(self._size, size=batch_size, p=probs, replace=False)

        # Importance sampling weights
        beta = self.beta
        weights = (self._size * probs[indices]) ** (-beta)
        weights = weights / weights.max()

        batch = {
            "obs": jnp.asarray(self._obs[indices]),
            "actions": jnp.asarray(self._actions[indices]),
            "actions_idx": jnp.asarray(self._actions_idx[indices]),
            "phi_obs": jnp.asarray(self._phi_obs[indices]),
            "phi_actions": jnp.asarray(self._phi_actions[indices]),
            "rewards": jnp.asarray(self._rewards[indices]),
            "next_obs": jnp.asarray(self._next_obs[indices]),
            "dones": jnp.asarray(self._dones[indices]),
            "robot_pos": jnp.asarray(self._robot_pos[indices]),
        }

        return batch, indices, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities based on TD errors."""
        priorities = (np.abs(td_errors) + 1e-6) ** self.alpha
        self._priorities[indices] = priorities
        self._max_priority = max(self._max_priority, priorities.max())

    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples."""
        return self._size >= batch_size
