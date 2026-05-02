"""Evaluation script for LIA_MADDPG.

Runs trained policy (with policy improvement) and greedy baseline
over 100 random test scenarios. Reports NATU, NATC, and DR metrics.

Usage:
    python examples/eval_lia_maddpg.py --n_robots 30 --n_tasks 5 --num_scenarios 100
    python examples/eval_lia_maddpg.py --n_robots 60 --n_tasks 10 --use_policy_improvement
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from marl.envs.task_allocation import EnvConfig, TaskAllocationEnv
from marl.lia import compute_lia
from marl.maddpg import MADDPGConfig, actor_inference, create_train_states, greedy_policy, policy_improvement
from marl.networks import SharedActor  # noqa: F401
from marl.utils import aggregate_metrics, evaluate_scenario


def run_episode(
    env: TaskAllocationEnv,
    config: MADDPGConfig,
    actor_state,
    key: jax.Array,
    use_policy_improvement: bool = True,
) -> dict:
    """Run a single evaluation episode with trained policy.

    Returns scenario evaluation dict.
    """
    key, reset_key = jax.random.split(key)
    state, obs = env.reset(reset_key)

    robot_utilities = np.zeros(config.n_robots)
    binding_times = np.full(config.n_robots, -1, dtype=np.int32)
    cumulative_cost = np.zeros(config.n_robots)

    for t in range(config.max_steps):
        key, k1 = jax.random.split(key)

        if use_policy_improvement:
            actions = policy_improvement(
                obs=obs,
                actor_params=actor_state.params,
                actor_batch_stats=actor_state.batch_stats,
                state=state,
                config=config,
                key=k1,
            )
        else:
            logits = actor_inference(
                actor_state.params, actor_state.batch_stats,
                obs, config.n_tasks, config.hidden_size,
            )
            actions = jax.nn.one_hot(jnp.argmax(logits, axis=-1), config.n_tasks)

        result = env.step(state, actions)
        state = result.state
        obs = result.obs
        rewards = np.asarray(result.rewards)

        robot_utilities += rewards
        active = np.asarray(state.robot_status) < 0.5
        newly_bound = (binding_times < 0) & active
        binding_times = np.where(newly_bound, t, binding_times)
        cumulative_cost += np.asarray(state.robot_speed) * config.env_config.tau

        if result.info["episode_done"]:
            break

    return evaluate_scenario(robot_utilities, binding_times, config.max_steps)


def run_greedy_episode(
    env: TaskAllocationEnv,
    config: MADDPGConfig,
    key: jax.Array,
) -> dict:
    """Run a single episode with greedy heuristic baseline."""
    key, reset_key = jax.random.split(key)
    state, obs = env.reset(reset_key)

    robot_utilities = np.zeros(config.n_robots)
    binding_times = np.full(config.n_robots, -1, dtype=np.int32)

    for t in range(config.max_steps):
        actions = greedy_policy(state, config)
        result = env.step(state, actions)
        state = result.state
        obs = result.obs
        rewards = np.asarray(result.rewards)

        robot_utilities += rewards
        active = np.asarray(state.robot_status) < 0.5
        newly_bound = (binding_times < 0) & active
        binding_times = np.where(newly_bound, t, binding_times)

        if result.info["episode_done"]:
            break

    return evaluate_scenario(robot_utilities, binding_times, config.max_steps)


def evaluate(config: MADDPGConfig, num_scenarios: int = 100, use_policy_improvement: bool = True):
    """Run full evaluation comparing trained policy vs greedy baseline."""
    print("=" * 70)
    print("LIA_MADDPG Evaluation")
    print("=" * 70)
    print(f"  Robots: {config.n_robots}, Tasks: {config.n_tasks}")
    print(f"  Scenarios: {num_scenarios}")
    print(f"  Policy improvement: {use_policy_improvement}")
    print("=" * 70)

    key = jax.random.key(config.seed + 1000)
    env = TaskAllocationEnv(config.env_config)

    # Initialize networks (with random weights for now; in practice load from checkpoint)
    key, init_key = jax.random.split(key)
    actor_state, _, _, _ = create_train_states(config, init_key)

    ours_results = []
    greedy_results = []
    start_time = time.time()

    for i in range(num_scenarios):
        key, k1, k2 = jax.random.split(key, 3)

        ours = run_episode(env, config, actor_state, k1, use_policy_improvement)
        greedy = run_greedy_episode(env, config, k2)

        ours_results.append(ours)
        greedy_results.append(greedy)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  Scenario {i + 1}/{num_scenarios} ({elapsed:.1f}s)")

    # Compute max possible utility for normalization
    max_possible = config.phi_1 * config.n_robots

    ours_metrics = aggregate_metrics(
        ours_results, greedy_results,
        max_possible_utility=max_possible,
        max_time=float(config.max_steps),
    )
    greedy_metrics = aggregate_metrics(
        greedy_results,
        max_possible_utility=max_possible,
        max_time=float(config.max_steps),
    )

    elapsed = time.time() - start_time
    print(f"\nEvaluation complete in {elapsed:.1f}s")
    print()

    print("--- LIA_MADDPG ---")
    print(f"  NATU: {ours_metrics['NATU_mean']:.3f} +/- {ours_metrics['NATU_std']:.3f}")
    print(f"  NATC: {ours_metrics['NATC_mean']:.3f} +/- {ours_metrics['NATC_std']:.3f}")
    if "DR" in ours_metrics:
        print(f"  DR:   {ours_metrics['DR']:.1%}")
    print(f"  Avg binding rate: {ours_metrics['avg_binding_rate']:.1%}")

    print()
    print("--- Greedy Baseline ---")
    print(f"  NATU: {greedy_metrics['NATU_mean']:.3f} +/- {greedy_metrics['NATU_std']:.3f}")
    print(f"  NATC: {greedy_metrics['NATC_mean']:.3f} +/- {greedy_metrics['NATC_std']:.3f}")
    print(f"  Avg binding rate: {greedy_metrics['avg_binding_rate']:.1%}")

    return ours_metrics, greedy_metrics


def main():
    parser = argparse.ArgumentParser(description="LIA_MADDPG Evaluation")

    parser.add_argument("--n_robots", type=int, default=60)
    parser.add_argument("--n_tasks", type=int, default=10)
    parser.add_argument("--area_size", type=float, default=1000.0)
    parser.add_argument("--max_steps", type=int, default=150)
    parser.add_argument("--d_bind", type=float, default=30.0)
    parser.add_argument("--max_neighbors", type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--beta", type=float, default=-1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_scenarios", type=int, default=100)
    parser.add_argument("--use_policy_improvement", action="store_true", default=True)
    parser.add_argument("--no_policy_improvement", action="store_false", dest="use_policy_improvement")

    args = parser.parse_args()

    config = MADDPGConfig(
        n_robots=args.n_robots,
        n_tasks=args.n_tasks,
        area_size=args.area_size,
        max_steps=args.max_steps,
        d_bind=args.d_bind,
        max_neighbors=args.max_neighbors,
        hidden_size=args.hidden_size,
        beta=args.beta,
        seed=args.seed,
    )

    evaluate(config, args.num_scenarios, args.use_policy_improvement)


if __name__ == "__main__":
    main()
