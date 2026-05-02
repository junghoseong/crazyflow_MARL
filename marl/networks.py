"""Shared Actor and Critic networks for LIA_MADDPG.

Actor: obs_i -> action logits (Gumbel-Softmax for discrete tasks)
Critic: [o_i, a_i, phi(o), phi(a)] -> Q-value

Both use 2 hidden layers (128 units), residual connections, and batch normalization
as described in Section III-A of the paper.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import Array


class ResidualBlock(nn.Module):
    """Fully-connected layer with residual connection and batch normalization."""

    features: int
    use_running_average: bool = False

    @nn.compact
    def __call__(self, x: Array) -> Array:
        residual = x
        y = nn.Dense(self.features)(x)
        y = nn.BatchNorm(use_running_average=self.use_running_average)(y)
        y = nn.relu(y)
        # Project residual if dimensions mismatch
        if residual.shape[-1] != self.features:
            residual = nn.Dense(self.features)(residual)
        return y + residual


class SharedActor(nn.Module):
    """Shared policy network (mu_bar) for all robots.

    Takes local observation o_i and outputs M-dimensional action logits.
    During training, use Gumbel-Softmax for differentiable discrete sampling.
    During evaluation, use argmax.
    """

    n_tasks: int
    hidden_size: int = 128
    use_running_average: bool = False

    @nn.compact
    def __call__(self, obs: Array) -> Array:
        """Forward pass.

        Args:
            obs: (..., obs_dim) observation vector.

        Returns:
            logits: (..., n_tasks) action logits.
        """
        x = nn.Dense(self.hidden_size)(obs)
        x = nn.BatchNorm(use_running_average=self.use_running_average)(x)
        x = nn.relu(x)

        x = ResidualBlock(self.hidden_size, self.use_running_average)(x)
        x = ResidualBlock(self.hidden_size, self.use_running_average)(x)

        logits = nn.Dense(self.n_tasks)(x)
        return logits


class SharedCritic(nn.Module):
    """Shared value network (G_bar) with LIA input.

    Takes concatenation of [o_i, a_i, phi_i(o), phi_i(a)] and outputs scalar Q-value.
    """

    hidden_size: int = 128
    use_running_average: bool = False

    @nn.compact
    def __call__(self, obs: Array, action: Array, phi_obs: Array, phi_action: Array) -> Array:
        """Forward pass.

        Args:
            obs: (..., obs_dim) robot's own observation.
            action: (..., n_tasks) robot's own action (one-hot or soft).
            phi_obs: (..., obs_dim) LIA-aggregated observations.
            phi_action: (..., n_tasks) LIA-aggregated actions.

        Returns:
            q_value: (...,) scalar Q-value.
        """
        x = jnp.concatenate([obs, action, phi_obs, phi_action], axis=-1)

        x = nn.Dense(self.hidden_size)(x)
        x = nn.BatchNorm(use_running_average=self.use_running_average)(x)
        x = nn.relu(x)

        x = ResidualBlock(self.hidden_size, self.use_running_average)(x)
        x = ResidualBlock(self.hidden_size, self.use_running_average)(x)

        q = nn.Dense(1)(x)
        return q.squeeze(-1)


def gumbel_softmax(
    key: Array,
    logits: Array,
    temperature: float = 1.0,
    hard: bool = False,
) -> Array:
    """Gumbel-Softmax sampling for differentiable discrete actions.

    Args:
        key: JAX PRNG key.
        logits: (..., M) unnormalized log-probabilities.
        temperature: softmax temperature (lower = more discrete).
        hard: if True, return hard one-hot but with straight-through gradient.

    Returns:
        action: (..., M) soft or hard one-hot vector.
    """
    gumbel_noise = -jnp.log(-jnp.log(jax.random.uniform(key, logits.shape) + 1e-20) + 1e-20)
    y = (logits + gumbel_noise) / temperature
    y_soft = jax.nn.softmax(y, axis=-1)

    if hard:
        y_hard = jax.nn.one_hot(jnp.argmax(y_soft, axis=-1), logits.shape[-1])
        # Straight-through estimator: forward uses hard, backward uses soft
        y = y_hard - jax.lax.stop_gradient(y_soft) + y_soft
        return y
    return y_soft


def create_actor(n_tasks: int, hidden_size: int = 128) -> SharedActor:
    """Create a SharedActor instance for training (batch norm in training mode)."""
    return SharedActor(n_tasks=n_tasks, hidden_size=hidden_size, use_running_average=False)


def create_actor_eval(n_tasks: int, hidden_size: int = 128) -> SharedActor:
    """Create a SharedActor instance for evaluation (batch norm in eval mode)."""
    return SharedActor(n_tasks=n_tasks, hidden_size=hidden_size, use_running_average=True)


def create_critic(hidden_size: int = 128) -> SharedCritic:
    """Create a SharedCritic instance for training."""
    return SharedCritic(hidden_size=hidden_size, use_running_average=False)


def create_critic_eval(hidden_size: int = 128) -> SharedCritic:
    """Create a SharedCritic instance for evaluation."""
    return SharedCritic(hidden_size=hidden_size, use_running_average=True)
