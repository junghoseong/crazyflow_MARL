"""Local Information Aggregation (LIA) module.

Implements the LIA mechanism from Section III-A of:
  "A Local Information Aggregation based Multi-Agent Reinforcement Learning
   for Robot Swarm Dynamic Task Allocation"

The LIA module:
1. Identifies "locally related robots" G_i for each robot i:
   - N_i: spatially nearest neighbors
   - L_i: robots performing the same action
   - G_i = N_i ∪ L_i (excluding self)
2. Aggregates their observations and actions using distance-dependent weights (Eq. 14-16).
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array


def find_locally_related(
    robot_pos: Array,
    actions: Array,
    max_neighbors: int,
    n_robots: int,
) -> tuple[Array, Array]:
    """Find locally related robot set G_i for each robot.

    G_i = N_i (spatial neighbors) ∪ L_i (same-action robots), excluding self.

    Args:
        robot_pos: (N, 2) robot positions.
        actions: (N,) int target task indices per robot.
        max_neighbors: alpha, max spatial neighbors.
        n_robots: N total robots.

    Returns:
        related_mask: (N, N) bool mask where related_mask[i, k] = True if k ∈ G_i.
        distances: (N, N) pairwise distance matrix.
    """
    # Pairwise distances
    distances = jnp.linalg.norm(
        robot_pos[:, None, :] - robot_pos[None, :, :], axis=-1
    )  # (N, N)

    # Self-mask (exclude self from all sets)
    self_mask = ~jnp.eye(n_robots, dtype=jnp.bool_)

    # N_i: alpha nearest neighbors (spatial proximity)
    dists_no_self = distances + jnp.eye(n_robots) * 1e9
    _, neighbor_idx = jax.lax.top_k(-dists_no_self, k=max_neighbors)  # (N, alpha)
    spatial_mask = jnp.zeros((n_robots, n_robots), dtype=jnp.bool_)
    # Scatter True at neighbor positions
    row_idx = jnp.arange(n_robots)[:, None]  # (N, 1)
    spatial_mask = spatial_mask.at[row_idx, neighbor_idx].set(True)  # (N, N)

    # L_i: robots with same action
    same_action_mask = actions[:, None] == actions[None, :]  # (N, N)

    # G_i = (N_i ∪ L_i) \ {i}
    related_mask = (spatial_mask | same_action_mask) & self_mask

    return related_mask, distances


def compute_lia_weights(
    distances: Array,
    related_mask: Array,
    beta: float = -1.0,
) -> Array:
    """Compute distance-dependent LIA weight coefficients (Eq. 16).

    w_{i,k} = exp(beta * ln(d_{i,k})) / sum_{k'∈G_i} exp(beta * ln(d_{i,k'}))
            = d_{i,k}^beta / sum_{k'} d_{i,k'}^beta   (for k ∈ G_i)

    Negative beta gives higher weight to closer robots.

    Args:
        distances: (N, N) pairwise distance matrix.
        related_mask: (N, N) bool mask for G_i.
        beta: scaling factor (default -1.0, closer = higher weight).

    Returns:
        weights: (N, N) weight matrix, weights[i,k] = w_{i,k} (0 if k ∉ G_i).
    """
    # Avoid log(0) by clamping distances
    safe_dist = jnp.clip(distances, min=1e-6)
    log_dist = jnp.log(safe_dist)

    # exp(beta * ln(d)) = d^beta
    raw_weights = jnp.exp(beta * log_dist)  # (N, N)

    # Zero out non-related
    raw_weights = raw_weights * related_mask.astype(jnp.float32)

    # Normalize per robot (softmax over G_i)
    denom = jnp.sum(raw_weights, axis=-1, keepdims=True) + 1e-8  # (N, 1)
    weights = raw_weights / denom  # (N, N)

    return weights


def aggregate_lia(
    weights: Array,
    values: Array,
) -> Array:
    """Aggregate values using LIA weights (Eq. 14-15).

    phi_i(v_k) = sum_{k ∈ G_i} w_{i,k} * v_k

    Args:
        weights: (N, N) LIA weight matrix.
        values: (N, D) per-robot values (observations or actions).

    Returns:
        aggregated: (N, D) aggregated values per robot.
    """
    return weights @ values  # (N, N) @ (N, D) -> (N, D)


def compute_lia(
    robot_pos: Array,
    actions_idx: Array,
    obs: Array,
    actions_onehot: Array,
    max_neighbors: int,
    n_robots: int,
    beta: float = -1.0,
) -> tuple[Array, Array]:
    """Full LIA computation: find related robots, compute weights, aggregate.

    Args:
        robot_pos: (N, 2) positions.
        actions_idx: (N,) int target task indices.
        obs: (N, obs_dim) observations.
        actions_onehot: (N, M) one-hot actions.
        max_neighbors: alpha.
        n_robots: N.
        beta: LIA distance weighting factor.

    Returns:
        phi_obs: (N, obs_dim) aggregated observations.
        phi_actions: (N, M) aggregated actions.
    """
    related_mask, distances = find_locally_related(
        robot_pos, actions_idx, max_neighbors, n_robots
    )
    weights = compute_lia_weights(distances, related_mask, beta)
    phi_obs = aggregate_lia(weights, obs)
    phi_actions = aggregate_lia(weights, actions_onehot)
    return phi_obs, phi_actions
