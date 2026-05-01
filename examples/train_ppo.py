"""PPO (Proximal Policy Optimization) training for Crazyflow drone environments.

Reference: Schulman et al., "Proximal Policy Optimization Algorithms", 2017.
Based on CleanRL and PureJaxRL implementations adapted for Gymnasium VectorEnv with Dict obs.

Usage:
    python examples/train_ppo.py --env_id DroneReachPos-v0
    python examples/train_ppo.py --env_id DroneFigureEightTrajectory-v0
"""

import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from jax import Array

import crazyflow  # noqa: F401, register gymnasium envs


@dataclass
class PPOConfig:
    """PPO hyperparameters."""

    env_id: str = "DroneReachPos-v0"
    num_envs: int = 64
    freq: int = 50
    max_episode_time: float = 5.0

    # PPO hyperparameters
    total_timesteps: int = 2_000_000
    num_steps: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Network
    hidden_sizes: tuple[int, ...] = (256, 256)

    # Figure-8 specific
    n_samples: int = 10
    samples_dt: float = 0.1
    trajectory_time: float = 10.0

    # Logging
    log_interval: int = 1
    seed: int = 42
    save_path: str = ""

    # Derived (computed in __post_init__)
    batch_size: int = field(init=False)
    minibatch_size: int = field(init=False)
    num_updates: int = field(init=False)

    def __post_init__(self):
        self.batch_size = self.num_envs * self.num_steps
        self.minibatch_size = self.batch_size // self.num_minibatches
        self.num_updates = self.total_timesteps // self.batch_size


class Actor(nn.Module):
    """Gaussian policy network for continuous actions."""

    action_dim: int
    hidden_sizes: tuple[int, ...]

    @nn.compact
    def __call__(self, x: Array) -> tuple[Array, Array]:
        for size in self.hidden_sizes:
            x = nn.Dense(size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = nn.tanh(x)
        mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(x)
        log_std = self.param("log_std", constant(-0.5), (self.action_dim,))
        return mean, log_std


class Critic(nn.Module):
    """Value function network."""

    hidden_sizes: tuple[int, ...]

    @nn.compact
    def __call__(self, x: Array) -> Array:
        for size in self.hidden_sizes:
            x = nn.Dense(size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = nn.tanh(x)
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        return value.squeeze(-1)


def flatten_obs(obs: dict[str, np.ndarray | Array]) -> Array:
    """Flatten a Dict observation into a single array."""
    parts = []
    for key in sorted(obs.keys()):
        val = obs[key]
        if isinstance(val, np.ndarray):
            val = jnp.asarray(val)
        parts.append(val.reshape(val.shape[0], -1))
    return jnp.concatenate(parts, axis=-1)


def gaussian_log_prob(mean: Array, log_std: Array, action: Array) -> Array:
    """Compute log probability of action under Gaussian distribution."""
    std = jnp.exp(log_std)
    log_prob = -0.5 * ((action - mean) / std) ** 2 - log_std - 0.5 * jnp.log(2 * jnp.pi)
    return log_prob.sum(axis=-1)


def gaussian_entropy(log_std: Array) -> Array:
    """Compute entropy of Gaussian distribution."""
    return 0.5 * (1.0 + jnp.log(2 * jnp.pi) + 2 * log_std).sum(axis=-1)


def sample_action(key: Array, mean: Array, log_std: Array) -> Array:
    """Sample action from Gaussian distribution."""
    std = jnp.exp(log_std)
    return mean + std * jax.random.normal(key, shape=mean.shape)


def make_env(config: PPOConfig):
    """Create the vectorized environment."""
    import gymnasium

    from crazyflow.envs import NormalizeActions

    env_kwargs: dict = {
        "num_envs": config.num_envs,
        "freq": config.freq,
        "max_episode_time": config.max_episode_time,
    }

    if config.env_id == "DroneFigureEightTrajectory-v0":
        env_kwargs["n_samples"] = config.n_samples
        env_kwargs["samples_dt"] = config.samples_dt
        env_kwargs["trajectory_time"] = config.trajectory_time

    envs = gymnasium.make_vec(config.env_id, **env_kwargs)
    envs = NormalizeActions(envs)
    return envs


def get_obs_dim(envs) -> int:
    """Compute total observation dimension from Dict observation space."""
    total = 0
    for key in sorted(envs.single_observation_space.keys()):
        space = envs.single_observation_space[key]
        total += int(np.prod(space.shape))
    return total


def compute_gae(
    rewards: Array,
    values: Array,
    dones: Array,
    last_value: Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[Array, Array]:
    """Compute Generalized Advantage Estimation."""
    num_steps = rewards.shape[0]
    advantages = jnp.zeros_like(rewards)
    lastgaelam = jnp.zeros(rewards.shape[1])

    def _step(lastgaelam, t):
        idx = num_steps - 1 - t
        next_value = jnp.where(t == 0, last_value, values[idx + 1])
        next_nonterminal = 1.0 - dones[idx]
        delta = rewards[idx] + gamma * next_value * next_nonterminal - values[idx]
        lastgaelam = delta + gamma * gae_lambda * next_nonterminal * lastgaelam
        return lastgaelam, lastgaelam

    _, advantages_reversed = jax.lax.scan(_step, lastgaelam, jnp.arange(num_steps))
    advantages = advantages_reversed[::-1]
    returns = advantages + values
    return advantages, returns


@jax.jit
def get_action_and_value(
    actor_state: TrainState,
    critic_state: TrainState,
    obs: Array,
    key: Array,
) -> tuple[Array, Array, Array]:
    """Sample action and compute value for given observation."""
    mean, log_std = actor_state.apply_fn(actor_state.params, obs)
    action = sample_action(key, mean, log_std)
    log_prob = gaussian_log_prob(mean, log_std, action)
    value = critic_state.apply_fn(critic_state.params, obs)
    return action, log_prob, value


@partial(jax.jit, static_argnames=["clip_eps", "ent_coef", "vf_coef", "num_minibatches"])
def _update_epoch(
    actor_state: TrainState,
    critic_state: TrainState,
    b_obs: Array,
    b_actions: Array,
    b_log_probs: Array,
    b_advantages: Array,
    b_returns: Array,
    key: Array,
    clip_eps: float,
    ent_coef: float,
    vf_coef: float,
    num_minibatches: int,
):
    """Run one epoch of PPO updates over minibatches."""
    batch_size = b_obs.shape[0]
    minibatch_size = batch_size // num_minibatches

    key, perm_key = jax.random.split(key)
    perm = jax.random.permutation(perm_key, batch_size)
    b_obs = b_obs[perm]
    b_actions = b_actions[perm]
    b_log_probs = b_log_probs[perm]
    b_advantages_perm = b_advantages[perm]
    b_returns = b_returns[perm]

    # Normalize advantages
    b_advantages_perm = (b_advantages_perm - b_advantages_perm.mean()) / (
        b_advantages_perm.std() + 1e-8
    )

    def _minibatch_step(carry, start_idx):
        actor_state, critic_state = carry
        mb_obs = jax.lax.dynamic_slice(b_obs, (start_idx, 0), (minibatch_size, b_obs.shape[1]))
        mb_actions = jax.lax.dynamic_slice(
            b_actions, (start_idx, 0), (minibatch_size, b_actions.shape[1])
        )
        mb_log_probs = jax.lax.dynamic_slice(
            b_log_probs, (start_idx,), (minibatch_size,)
        )
        mb_advantages = jax.lax.dynamic_slice(
            b_advantages_perm, (start_idx,), (minibatch_size,)
        )
        mb_returns = jax.lax.dynamic_slice(b_returns, (start_idx,), (minibatch_size,))

        # Actor loss
        def actor_loss_fn(params):
            mean, log_std = actor_state.apply_fn(params, mb_obs)
            new_log_prob = gaussian_log_prob(mean, log_std, mb_actions)
            entropy = gaussian_entropy(log_std).mean()
            ratio = jnp.exp(new_log_prob - mb_log_probs)
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
            pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()
            loss = pg_loss - ent_coef * entropy
            return loss

        # Critic loss
        def critic_loss_fn(params):
            new_value = critic_state.apply_fn(params, mb_obs)
            return vf_coef * ((new_value - mb_returns) ** 2).mean()

        actor_grads = jax.grad(actor_loss_fn)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=actor_grads)

        critic_grads = jax.grad(critic_loss_fn)(critic_state.params)
        critic_state = critic_state.apply_gradients(grads=critic_grads)

        return (actor_state, critic_state), None

    start_indices = jnp.arange(num_minibatches) * minibatch_size
    (actor_state, critic_state), _ = jax.lax.scan(
        _minibatch_step, (actor_state, critic_state), start_indices
    )
    return actor_state, critic_state, key


def train(config: PPOConfig):
    """Main PPO training loop."""
    key = jax.random.key(config.seed)

    # Create environment
    envs = make_env(config)
    obs_dim = get_obs_dim(envs)
    action_dim = int(np.prod(envs.single_action_space.shape))

    print(f"Environment: {config.env_id}")
    print(f"  Observation dim: {obs_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  Num envs: {config.num_envs}")
    print(f"  Total timesteps: {config.total_timesteps}")
    print(f"  Num updates: {config.num_updates}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Minibatch size: {config.minibatch_size}")
    print()

    # Initialize networks
    key, actor_key, critic_key = jax.random.split(key, 3)
    dummy_obs = jnp.zeros((1, obs_dim))

    actor = Actor(action_dim=action_dim, hidden_sizes=config.hidden_sizes)
    actor_params = actor.init(actor_key, dummy_obs)

    critic = Critic(hidden_sizes=config.hidden_sizes)
    critic_params = critic.init(critic_key, dummy_obs)

    # Create optimizers with gradient clipping
    actor_tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.learning_rate, eps=1e-5),
    )
    critic_tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.learning_rate, eps=1e-5),
    )

    actor_state = TrainState.create(apply_fn=actor.apply, params=actor_params, tx=actor_tx)
    critic_state = TrainState.create(apply_fn=critic.apply, params=critic_params, tx=critic_tx)

    # Reset environment
    obs_dict, _ = envs.reset(seed=config.seed)
    obs = flatten_obs(obs_dict)

    # Training loop
    global_step = 0
    start_time = time.time()
    episode_returns = []
    current_returns = np.zeros(config.num_envs)

    for update in range(1, config.num_updates + 1):
        # Storage for this rollout
        all_obs = []
        all_actions = []
        all_log_probs = []
        all_rewards = []
        all_dones = []
        all_values = []

        # Collect trajectories
        for step in range(config.num_steps):
            key, action_key = jax.random.split(key)

            action, log_prob, value = get_action_and_value(
                actor_state, critic_state, obs, action_key
            )

            all_obs.append(obs)
            all_actions.append(action)
            all_log_probs.append(log_prob)
            all_values.append(value)

            # Step environment (convert JAX action to numpy for JaxToNumpy or pass directly)
            action_np = np.asarray(action)
            obs_dict, reward, terminated, truncated, info = envs.step(action_np)
            done = np.logical_or(terminated, truncated)

            obs = flatten_obs(obs_dict)
            all_rewards.append(jnp.asarray(reward))
            all_dones.append(jnp.asarray(done))

            # Track episode returns
            current_returns += np.asarray(reward)
            for i in range(config.num_envs):
                if done[i]:
                    episode_returns.append(current_returns[i])
                    current_returns[i] = 0.0

            global_step += config.num_envs

        # Stack collected data
        all_obs = jnp.stack(all_obs)  # (num_steps, num_envs, obs_dim)
        all_actions = jnp.stack(all_actions)  # (num_steps, num_envs, action_dim)
        all_log_probs = jnp.stack(all_log_probs)  # (num_steps, num_envs)
        all_rewards = jnp.stack(all_rewards)  # (num_steps, num_envs)
        all_dones = jnp.stack(all_dones)  # (num_steps, num_envs)
        all_values = jnp.stack(all_values)  # (num_steps, num_envs)

        # Compute last value for GAE
        _, _, last_value = get_action_and_value(actor_state, critic_state, obs, key)

        # Compute GAE
        advantages, returns = compute_gae(
            all_rewards, all_values, all_dones, last_value, config.gamma, config.gae_lambda
        )

        # Flatten batch
        b_obs = all_obs.reshape(-1, obs_dim)
        b_actions = all_actions.reshape(-1, action_dim)
        b_log_probs = all_log_probs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        # PPO update epochs
        for epoch in range(config.update_epochs):
            key, epoch_key = jax.random.split(key)
            actor_state, critic_state, _ = _update_epoch(
                actor_state,
                critic_state,
                b_obs,
                b_actions,
                b_log_probs,
                b_advantages,
                b_returns,
                epoch_key,
                config.clip_eps,
                config.ent_coef,
                config.vf_coef,
                config.num_minibatches,
            )

        # Logging
        if update % config.log_interval == 0:
            elapsed = time.time() - start_time
            sps = int(global_step / elapsed)
            if episode_returns:
                recent_returns = episode_returns[-100:]
                mean_return = np.mean(recent_returns)
                print(
                    f"Update {update:4d}/{config.num_updates} | "
                    f"Step {global_step:8d} | "
                    f"Mean Return: {mean_return:7.2f} | "
                    f"Episodes: {len(episode_returns):5d} | "
                    f"SPS: {sps:5d}"
                )
            else:
                print(
                    f"Update {update:4d}/{config.num_updates} | "
                    f"Step {global_step:8d} | "
                    f"SPS: {sps:5d} | "
                    f"(collecting episodes...)"
                )

    # Save model
    if config.save_path:
        save_dir = Path(config.save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        # Save as numpy arrays for portability
        actor_params_np = jax.tree.map(np.asarray, actor_state.params)
        critic_params_np = jax.tree.map(np.asarray, critic_state.params)
        np.savez(save_dir / "actor_params.npz", **jax.tree.leaves(actor_params_np))
        np.savez(save_dir / "critic_params.npz", **jax.tree.leaves(critic_params_np))
        print(f"\nModel saved to {save_dir}")

    envs.close()

    print(f"\nTraining complete!")
    print(f"  Total timesteps: {global_step}")
    print(f"  Total episodes: {len(episode_returns)}")
    if episode_returns:
        print(f"  Final mean return (last 100): {np.mean(episode_returns[-100:]):.2f}")
    print(f"  Wall time: {time.time() - start_time:.1f}s")

    return actor_state, critic_state


def main():
    import argparse

    parser = argparse.ArgumentParser(description="PPO Training for Crazyflow Drones")
    parser.add_argument("--env_id", type=str, default="DroneReachPos-v0")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--freq", type=int, default=50)
    parser.add_argument("--max_episode_time", type=float, default=5.0)
    parser.add_argument("--total_timesteps", type=int, default=2_000_000)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--num_minibatches", type=int, default=4)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="")
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--samples_dt", type=float, default=0.1)
    parser.add_argument("--trajectory_time", type=float, default=10.0)
    args = parser.parse_args()

    config = PPOConfig(
        env_id=args.env_id,
        num_envs=args.num_envs,
        freq=args.freq,
        max_episode_time=args.max_episode_time,
        total_timesteps=args.total_timesteps,
        num_steps=args.num_steps,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        num_minibatches=args.num_minibatches,
        update_epochs=args.update_epochs,
        clip_eps=args.clip_eps,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        save_path=args.save_path,
        log_interval=args.log_interval,
        n_samples=args.n_samples,
        samples_dt=args.samples_dt,
        trajectory_time=args.trajectory_time,
    )

    from crazyflow.utils import enable_cache

    enable_cache()
    train(config)


if __name__ == "__main__":
    main()
