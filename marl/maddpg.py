"""LIA_MADDPG: Local Information Aggregation based MADDPG algorithm.

Implements:
- Algorithm 1: Off-line Centralized Training (CTDE)
- Algorithm 2: On-line Distributed Policy Improvement and Execution

Reference: Section III of the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from jax import Array

from marl.envs.task_allocation import EnvConfig, EnvState
from marl.networks import SharedActor, SharedCritic, gumbel_softmax


@dataclass
class MADDPGConfig:
    """LIA_MADDPG hyperparameters."""

    # Environment
    n_robots: int = 60
    n_tasks: int = 10
    area_size: float = 1000.0
    max_steps: int = 150
    d_bind: float = 30.0
    max_neighbors: int = 10

    # Training
    num_episodes: int = 5000
    batch_size: int = 64
    buffer_capacity: int = 5000
    gamma: float = 0.99
    tau_soft: float = 0.01
    actor_lr: float = 0.001
    critic_lr: float = 0.002

    # Network
    hidden_size: int = 128

    # Gumbel-Softmax
    temperature_start: float = 1.0
    temperature_end: float = 0.1
    temperature_decay_steps: int = 3000

    exploration_noise: float = 0.1

    # LIA
    beta: float = -1.0

    # Reward params
    phi_1: float = 10.0
    phi_2: float = 0.001
    phi_3: float = 1.0

    # Logging
    log_interval: int = 50
    save_interval: int = 500
    seed: int = 42
    save_path: str = ""

    @property
    def env_config(self) -> EnvConfig:
        return EnvConfig(
            n_robots=self.n_robots,
            n_tasks=self.n_tasks,
            area_size=self.area_size,
            max_steps=self.max_steps,
            d_bind=self.d_bind,
            max_neighbors=self.max_neighbors,
            phi_1=self.phi_1,
            phi_2=self.phi_2,
            phi_3=self.phi_3,
        )


class ActorTrainState(TrainState):
    """Train state for actor with batch stats."""

    batch_stats: Any = None


class CriticTrainState(TrainState):
    """Train state for critic with batch stats."""

    batch_stats: Any = None


def _make_variables(params: dict, batch_stats: Any) -> dict:
    """Build Flax variables dict from params and batch_stats."""
    v = {"params": params}
    if batch_stats:
        v["batch_stats"] = batch_stats
    return v


def create_train_states(
    config: MADDPGConfig,
    key: Array,
) -> tuple[ActorTrainState, CriticTrainState, dict, dict]:
    """Initialize actor/critic train states and target network params."""
    obs_dim = config.env_config.obs_dim
    n_tasks = config.n_tasks
    k1, k2 = jax.random.split(key)

    # Actor (training mode — BatchNorm updates running stats)
    actor_train = SharedActor(
        n_tasks=n_tasks, hidden_size=config.hidden_size, use_running_average=False
    )
    dummy_obs = jnp.zeros((1, obs_dim))
    actor_variables = actor_train.init(k1, dummy_obs)

    actor_state = ActorTrainState.create(
        apply_fn=actor_train.apply,
        params=actor_variables["params"],
        tx=optax.adam(config.actor_lr),
        batch_stats=actor_variables.get("batch_stats", {}),
    )

    # Critic (training mode)
    critic_train = SharedCritic(hidden_size=config.hidden_size, use_running_average=False)
    dummy_a = jnp.zeros((1, n_tasks))
    dummy_phi_o = jnp.zeros((1, obs_dim))
    dummy_phi_a = jnp.zeros((1, n_tasks))
    critic_variables = critic_train.init(k2, dummy_obs, dummy_a, dummy_phi_o, dummy_phi_a)

    critic_state = CriticTrainState.create(
        apply_fn=critic_train.apply,
        params=critic_variables["params"],
        tx=optax.adam(config.critic_lr),
        batch_stats=critic_variables.get("batch_stats", {}),
    )

    target_actor_params = jax.tree.map(jnp.copy, actor_state.params)
    target_critic_params = jax.tree.map(jnp.copy, critic_state.params)

    return actor_state, critic_state, target_actor_params, target_critic_params


# Eval-mode singletons (use_running_average=True → no mutable needed)
_actor_eval_cache: dict[tuple, SharedActor] = {}
_critic_eval_cache: dict[tuple, SharedCritic] = {}


def _actor_eval(n_tasks: int, hidden_size: int) -> SharedActor:
    key = (n_tasks, hidden_size)
    if key not in _actor_eval_cache:
        _actor_eval_cache[key] = SharedActor(
            n_tasks=n_tasks, hidden_size=hidden_size, use_running_average=True
        )
    return _actor_eval_cache[key]


def _critic_eval(hidden_size: int) -> SharedCritic:
    if hidden_size not in _critic_eval_cache:
        _critic_eval_cache[hidden_size] = SharedCritic(
            hidden_size=hidden_size, use_running_average=True
        )
    return _critic_eval_cache[hidden_size]


def actor_inference(
    params: dict,
    batch_stats: Any,
    obs: Array,
    n_tasks: int,
    hidden_size: int = 128,
) -> Array:
    """Run actor in eval mode (no batch_stats mutation)."""
    model = _actor_eval(n_tasks, hidden_size)
    return model.apply(_make_variables(params, batch_stats), obs)


def critic_inference(
    params: dict,
    batch_stats: Any,
    obs: Array,
    action: Array,
    phi_obs: Array,
    phi_action: Array,
    hidden_size: int = 128,
) -> Array:
    """Run critic in eval mode (no batch_stats mutation)."""
    model = _critic_eval(hidden_size)
    return model.apply(
        _make_variables(params, batch_stats), obs, action, phi_obs, phi_action
    )


def soft_update(params: dict, target_params: dict, tau: float) -> dict:
    """Soft update: target = tau * params + (1 - tau) * target."""
    return jax.tree.map(lambda p, tp: tau * p + (1.0 - tau) * tp, params, target_params)


# ------------------------------------------------------------------
# Critic update (Eq. 19)
# ------------------------------------------------------------------

def update_critic(
    critic_state: CriticTrainState,
    target_critic_params: dict,
    target_actor_params: dict,
    actor_batch_stats: Any,
    critic_batch_stats_for_target: Any,
    batch: dict,
    gamma: float,
    config: MADDPGConfig,
    robot_idx: int,
) -> tuple[CriticTrainState, Array]:
    """Update critic for one robot using TD error (Eq. 19)."""
    obs_i = batch["obs"][:, robot_idx, :]
    action_i = batch["actions"][:, robot_idx, :]
    phi_obs_i = batch["phi_obs"][:, robot_idx, :]
    phi_act_i = batch["phi_actions"][:, robot_idx, :]
    reward_i = batch["rewards"][:, robot_idx]
    next_obs_i = batch["next_obs"][:, robot_idx, :]
    done_i = batch["dones"][:, robot_idx]

    hs = config.hidden_size

    # Target actor → next actions (eval mode)
    next_logits = actor_inference(
        target_actor_params, actor_batch_stats, next_obs_i, config.n_tasks, hs
    )
    next_action = jax.nn.softmax(next_logits, axis=-1)

    # Approximate next phi (reuse current step's phi)
    next_phi_obs = phi_obs_i
    next_phi_act = phi_act_i

    # Target Q (eval mode)
    target_q = critic_inference(
        target_critic_params, critic_batch_stats_for_target,
        next_obs_i, next_action, next_phi_obs, next_phi_act, hs,
    )
    target_y = reward_i + gamma * (1.0 - done_i) * target_q

    # Critic loss with mutable batch_stats
    def critic_loss_fn(params):
        variables = _make_variables(params, critic_state.batch_stats)
        q_val, updates = critic_state.apply_fn(
            variables, obs_i, action_i, phi_obs_i, phi_act_i,
            mutable=["batch_stats"],
        )
        loss = jnp.mean((q_val - jax.lax.stop_gradient(target_y)) ** 2)
        return loss, updates.get("batch_stats", {})

    (loss, new_bs), grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(critic_state.params)
    critic_state = critic_state.apply_gradients(grads=grads)
    if new_bs:
        critic_state = critic_state.replace(batch_stats=new_bs)

    # TD errors for priority replay (eval mode)
    current_q = critic_inference(
        critic_state.params, critic_state.batch_stats,
        obs_i, action_i, phi_obs_i, phi_act_i, hs,
    )
    td_errors = jnp.abs(current_q - target_y)

    return critic_state, td_errors


# ------------------------------------------------------------------
# Actor update (Eq. 20)
# ------------------------------------------------------------------

def update_actor(
    actor_state: ActorTrainState,
    critic_state: CriticTrainState,
    batch: dict,
    config: MADDPGConfig,
    robot_idx: int,
    temperature: float,
) -> ActorTrainState:
    """Update actor for one robot using policy gradient (Eq. 20)."""
    obs_i = batch["obs"][:, robot_idx, :]
    phi_obs_i = batch["phi_obs"][:, robot_idx, :]
    phi_act_i = batch["phi_actions"][:, robot_idx, :]
    hs = config.hidden_size

    def actor_loss_fn(params):
        variables = _make_variables(params, actor_state.batch_stats)
        logits, updates = actor_state.apply_fn(
            variables, obs_i, mutable=["batch_stats"],
        )
        action = jax.nn.softmax(logits / temperature, axis=-1)

        # Q value (eval mode on critic — no grad through critic)
        q_val = critic_inference(
            critic_state.params, critic_state.batch_stats,
            obs_i, action, phi_obs_i, phi_act_i, hs,
        )
        return -jnp.mean(q_val), updates.get("batch_stats", {})

    (loss, new_bs), grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(actor_state.params)
    actor_state = actor_state.apply_gradients(grads=grads)
    if new_bs:
        actor_state = actor_state.replace(batch_stats=new_bs)

    return actor_state


# ------------------------------------------------------------------
# Algorithm 2: Distributed Policy Improvement
# ------------------------------------------------------------------

def policy_improvement(
    obs: Array,
    actor_params: dict,
    actor_batch_stats: Any,
    state: EnvState,
    config: MADDPGConfig,
    key: Array,
) -> Array:
    """On-line policy improvement during execution (Algorithm 2)."""
    N, M = config.n_robots, config.n_tasks

    # Policy output (eval mode)
    logits = actor_inference(actor_params, actor_batch_stats, obs, M, config.hidden_size)
    actions_idx = jnp.argmax(logits, axis=-1)

    # Deviation probability
    h_bar = config.env_config.task_max_demand
    target_h = state.task_bound_count[actions_idx]
    h_remaining = jnp.maximum(h_bar - target_h.astype(jnp.float32), 0.0)

    alpha = config.max_neighbors
    robot_dists = jnp.linalg.norm(
        state.robot_pos[:, None, :] - state.robot_pos[None, :, :], axis=-1
    )
    robot_dists = robot_dists + jnp.eye(N) * 1e9
    _, nb_idx = jax.lax.top_k(-robot_dists, k=alpha)

    nb_targets = actions_idx[nb_idx]
    same_target = nb_targets == actions_idx[:, None]
    beta_i = jnp.sum(same_target, axis=-1).astype(jnp.float32)
    alpha_f = jnp.float32(alpha)

    deviation_prob = jnp.exp(-h_remaining * (alpha_f - beta_i))

    # Greedy fallback
    robot_to_task_dist = jnp.linalg.norm(
        state.robot_pos[:, None, :] - state.task_pos[None, :, :], axis=-1
    )
    greedy_score = config.phi_1 * state.task_rewards[None, :] - robot_to_task_dist
    greedy_targets = jnp.argmax(greedy_score, axis=-1)

    k1, _ = jax.random.split(key)
    rand_vals = jax.random.uniform(k1, (N,))
    deviate = rand_vals < deviation_prob
    final_targets = jnp.where(
        deviate & (state.robot_status > 0.5), greedy_targets, actions_idx
    )
    return jax.nn.one_hot(final_targets, M)


# ------------------------------------------------------------------
# Greedy baseline
# ------------------------------------------------------------------

def greedy_policy(state: EnvState, config: MADDPGConfig) -> Array:
    """Greedy heuristic baseline: maximize (reward - distance) per robot."""
    N, M = config.n_robots, config.n_tasks
    robot_to_task_dist = jnp.linalg.norm(
        state.robot_pos[:, None, :] - state.task_pos[None, :, :], axis=-1
    )
    score = config.phi_1 * state.task_rewards[None, :] - robot_to_task_dist
    targets = jnp.argmax(score, axis=-1)
    return jax.nn.one_hot(targets, M)
