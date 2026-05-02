"""Training script for LIA_MADDPG on 2D Task Allocation.

Implements Algorithm 1 (Off-line Centralized Training) from the paper.

Usage:
    python examples/train_lia_maddpg.py --n_robots 30 --n_tasks 5
    python examples/train_lia_maddpg.py --n_robots 60 --n_tasks 10 --num_episodes 5000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from marl.envs.task_allocation import TaskAllocationEnv
from marl.lia import compute_lia
from marl.maddpg import (
    MADDPGConfig,
    ActorTrainState,
    CriticTrainState,
    actor_inference,
    create_train_states,
    soft_update,
    update_actor,
    update_critic,
)
from marl.networks import gumbel_softmax
from marl.replay_buffer import ReplayBuffer, Transition


def get_temperature(step: int, config: MADDPGConfig) -> float:
    """Linearly anneal Gumbel-Softmax temperature."""
    frac = min(step / max(config.temperature_decay_steps, 1), 1.0)
    return config.temperature_start + frac * (config.temperature_end - config.temperature_start)


def collect_episode(
    env: TaskAllocationEnv,
    actor_state: ActorTrainState,
    buffer: ReplayBuffer,
    config: MADDPGConfig,
    key: jax.Array,
    episode_num: int,
) -> tuple[jax.Array, float]:
    """Run one episode, collecting transitions into the replay buffer."""
    key, reset_key = jax.random.split(key)
    state, obs = env.reset(reset_key)
    episode_reward = 0.0
    temperature = get_temperature(episode_num, config)

    for t in range(config.max_steps):
        key, action_key = jax.random.split(key)

        # Actor inference (eval mode — no batch_stats mutation)
        logits = actor_inference(
            actor_state.params, actor_state.batch_stats,
            obs, config.n_tasks, config.hidden_size,
        )

        # Gumbel-Softmax for exploration
        actions_onehot = gumbel_softmax(action_key, logits, temperature=temperature, hard=False)
        actions_idx = jnp.argmax(actions_onehot, axis=-1)

        # Compute LIA aggregations
        phi_obs, phi_actions = compute_lia(
            robot_pos=state.robot_pos,
            actions_idx=actions_idx,
            obs=obs,
            actions_onehot=actions_onehot,
            max_neighbors=config.max_neighbors,
            n_robots=config.n_robots,
            beta=config.beta,
        )

        # Environment step
        result = env.step(state, actions_onehot)

        # Store transition
        transition = Transition(
            obs=np.asarray(obs),
            actions=np.asarray(actions_onehot),
            actions_idx=np.asarray(actions_idx),
            phi_obs=np.asarray(phi_obs),
            phi_actions=np.asarray(phi_actions),
            rewards=np.asarray(result.rewards),
            next_obs=np.asarray(result.obs),
            dones=np.asarray(result.dones.astype(jnp.float32)),
            robot_pos=np.asarray(state.robot_pos),
        )
        buffer.add(transition)

        episode_reward += float(jnp.sum(result.rewards))
        obs = result.obs
        state = result.state

        if result.info["episode_done"]:
            break

    return key, episode_reward


def train_step(
    actor_state: ActorTrainState,
    critic_state: CriticTrainState,
    target_actor_params: dict,
    target_critic_params: dict,
    buffer: ReplayBuffer,
    config: MADDPGConfig,
    key: jax.Array,
    episode_num: int,
    rng: np.random.Generator,
) -> tuple[ActorTrainState, CriticTrainState, dict, dict, jax.Array]:
    """Sample batch and update networks (Algorithm 1 inner loop)."""
    if not buffer.is_ready(config.batch_size):
        return actor_state, critic_state, target_actor_params, target_critic_params, key

    batch, indices, is_weights = buffer.sample(config.batch_size, rng)
    temperature = get_temperature(episode_num, config)

    n_update_robots = min(config.n_robots, 10)
    robot_indices = rng.choice(config.n_robots, size=n_update_robots, replace=False)

    all_td_errors = []
    for robot_idx in robot_indices:
        key, k1 = jax.random.split(key)

        critic_state, td_errors = update_critic(
            critic_state=critic_state,
            target_critic_params=target_critic_params,
            target_actor_params=target_actor_params,
            actor_batch_stats=actor_state.batch_stats,
            critic_batch_stats_for_target=critic_state.batch_stats,
            batch=batch,
            gamma=config.gamma,
            config=config,
            robot_idx=int(robot_idx),
        )

        actor_state = update_actor(
            actor_state=actor_state,
            critic_state=critic_state,
            batch=batch,
            config=config,
            robot_idx=int(robot_idx),
            temperature=temperature,
        )

        all_td_errors.append(np.asarray(td_errors))

    mean_td = np.mean(np.stack(all_td_errors), axis=0)
    buffer.update_priorities(indices, mean_td)

    target_actor_params = soft_update(actor_state.params, target_actor_params, config.tau_soft)
    target_critic_params = soft_update(critic_state.params, target_critic_params, config.tau_soft)

    return actor_state, critic_state, target_actor_params, target_critic_params, key


def train(config: MADDPGConfig):
    """Main training loop for LIA_MADDPG."""
    print("=" * 70)
    print("LIA_MADDPG Training")
    print("=" * 70)
    print(f"  Robots: {config.n_robots}, Tasks: {config.n_tasks}")
    print(f"  Episodes: {config.num_episodes}")
    print(f"  Obs dim: {config.env_config.obs_dim}")
    print(f"  Buffer capacity: {config.buffer_capacity}, Batch size: {config.batch_size}")
    print(f"  Actor LR: {config.actor_lr}, Critic LR: {config.critic_lr}")
    print(f"  LIA beta: {config.beta}")
    print("=" * 70)

    key = jax.random.key(config.seed)
    rng = np.random.default_rng(config.seed)

    env = TaskAllocationEnv(config.env_config)

    key, init_key = jax.random.split(key)
    actor_state, critic_state, target_actor_params, target_critic_params = create_train_states(
        config, init_key
    )

    buffer = ReplayBuffer(
        capacity=config.buffer_capacity,
        n_robots=config.n_robots,
        obs_dim=config.env_config.obs_dim,
        n_tasks=config.n_tasks,
    )

    episode_rewards = []
    start_time = time.time()

    for episode in range(1, config.num_episodes + 1):
        key, ep_reward = collect_episode(
            env=env,
            actor_state=actor_state,
            buffer=buffer,
            config=config,
            key=key,
            episode_num=episode,
        )
        episode_rewards.append(ep_reward)

        actor_state, critic_state, target_actor_params, target_critic_params, key = train_step(
            actor_state=actor_state,
            critic_state=critic_state,
            target_actor_params=target_actor_params,
            target_critic_params=target_critic_params,
            buffer=buffer,
            config=config,
            key=key,
            episode_num=episode,
            rng=rng,
        )

        if episode % config.log_interval == 0:
            recent = episode_rewards[-config.log_interval:]
            mean_reward = np.mean(recent)
            std_reward = np.std(recent)
            elapsed = time.time() - start_time
            temp = get_temperature(episode, config)
            print(
                f"Episode {episode:5d}/{config.num_episodes} | "
                f"Reward: {mean_reward:8.1f} +/- {std_reward:6.1f} | "
                f"Buffer: {buffer.size:5d} | "
                f"Temp: {temp:.3f} | "
                f"Time: {elapsed:6.1f}s"
            )

        if config.save_path and episode % config.save_interval == 0:
            save_checkpoint(actor_state, critic_state, config, episode)

    if config.save_path:
        save_checkpoint(actor_state, critic_state, config, config.num_episodes)

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Final mean reward (last 100): {np.mean(episode_rewards[-100:]):.1f}")

    return actor_state, critic_state, target_actor_params, target_critic_params


def save_checkpoint(
    actor_state: ActorTrainState,
    critic_state: CriticTrainState,
    config: MADDPGConfig,
    episode: int,
):
    """Save model checkpoint."""
    save_dir = Path(config.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    actor_leaves = jax.tree.leaves(actor_state.params)
    critic_leaves = jax.tree.leaves(critic_state.params)

    np.savez(
        save_dir / f"actor_params_ep{episode}.npz",
        *[np.asarray(leaf) for leaf in actor_leaves],
    )
    np.savez(
        save_dir / f"critic_params_ep{episode}.npz",
        *[np.asarray(leaf) for leaf in critic_leaves],
    )

    if actor_state.batch_stats:
        bs_leaves = jax.tree.leaves(actor_state.batch_stats)
        np.savez(
            save_dir / f"actor_batch_stats_ep{episode}.npz",
            *[np.asarray(leaf) for leaf in bs_leaves],
        )

    print(f"  Checkpoint saved at episode {episode} -> {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="LIA_MADDPG Training")

    parser.add_argument("--n_robots", type=int, default=60)
    parser.add_argument("--n_tasks", type=int, default=10)
    parser.add_argument("--area_size", type=float, default=1000.0)
    parser.add_argument("--max_steps", type=int, default=150)
    parser.add_argument("--d_bind", type=float, default=30.0)
    parser.add_argument("--max_neighbors", type=int, default=10)
    parser.add_argument("--num_episodes", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--buffer_capacity", type=int, default=5000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau_soft", type=float, default=0.01)
    parser.add_argument("--actor_lr", type=float, default=0.001)
    parser.add_argument("--critic_lr", type=float, default=0.002)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--temperature_start", type=float, default=1.0)
    parser.add_argument("--temperature_end", type=float, default=0.1)
    parser.add_argument("--temperature_decay_steps", type=int, default=3000)
    parser.add_argument("--beta", type=float, default=-1.0)
    parser.add_argument("--phi_1", type=float, default=10.0)
    parser.add_argument("--phi_2", type=float, default=0.001)
    parser.add_argument("--phi_3", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="")

    args = parser.parse_args()

    config = MADDPGConfig(
        n_robots=args.n_robots,
        n_tasks=args.n_tasks,
        area_size=args.area_size,
        max_steps=args.max_steps,
        d_bind=args.d_bind,
        max_neighbors=args.max_neighbors,
        num_episodes=args.num_episodes,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_capacity,
        gamma=args.gamma,
        tau_soft=args.tau_soft,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        hidden_size=args.hidden_size,
        temperature_start=args.temperature_start,
        temperature_end=args.temperature_end,
        temperature_decay_steps=args.temperature_decay_steps,
        beta=args.beta,
        phi_1=args.phi_1,
        phi_2=args.phi_2,
        phi_3=args.phi_3,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        seed=args.seed,
        save_path=args.save_path,
    )

    train(config)


if __name__ == "__main__":
    main()
