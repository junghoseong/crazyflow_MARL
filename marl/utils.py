"""Evaluation metrics for LIA_MADDPG.

Implements the three key performance metrics from Section IV of the paper:
- NATU: Normalized Average Total Utility
- NATC: Normalized Average Time Cost
- DR:   Dominance Rate
"""

from __future__ import annotations

import numpy as np


def compute_natu(
    total_utilities: np.ndarray,
    max_possible_utility: float,
) -> tuple[float, float]:
    """Compute Normalized Average Total Utility.

    NATU measures how well the algorithm allocates tasks by comparing
    actual total utility to the maximum possible utility.

    Args:
        total_utilities: (n_scenarios,) total utility per test scenario.
        max_possible_utility: theoretical maximum (e.g., phi_1 * sum(task_rewards) * N).

    Returns:
        mean_natu: mean normalized utility across scenarios.
        std_natu: standard deviation.
    """
    if max_possible_utility <= 0:
        return 0.0, 0.0
    normalized = total_utilities / max_possible_utility
    return float(np.mean(normalized)), float(np.std(normalized))


def compute_natc(
    completion_times: np.ndarray,
    max_time: float,
) -> tuple[float, float]:
    """Compute Normalized Average Time Cost.

    NATC measures efficiency: lower is better. Normalized by max episode length.

    Args:
        completion_times: (n_scenarios,) average completion time per scenario.
        max_time: maximum episode time (for normalization).

    Returns:
        mean_natc: mean normalized time cost.
        std_natc: standard deviation.
    """
    if max_time <= 0:
        return 0.0, 0.0
    normalized = completion_times / max_time
    return float(np.mean(normalized)), float(np.std(normalized))


def compute_dr(
    utilities_ours: np.ndarray,
    utilities_baseline: np.ndarray,
) -> float:
    """Compute Dominance Rate.

    DR = fraction of scenarios where our method outperforms the baseline.

    Args:
        utilities_ours: (n_scenarios,) total utility per scenario for our method.
        utilities_baseline: (n_scenarios,) total utility per scenario for baseline.

    Returns:
        dr: dominance rate in [0, 1].
    """
    wins = np.sum(utilities_ours > utilities_baseline)
    return float(wins / len(utilities_ours))


def evaluate_scenario(
    robot_utilities: np.ndarray,
    binding_times: np.ndarray,
    max_steps: int,
) -> dict:
    """Evaluate a single scenario.

    Args:
        robot_utilities: (N,) final utility per robot.
        binding_times: (N,) step at which each robot was bound (-1 if not bound).
        max_steps: max episode length.

    Returns:
        Dictionary with total_utility, avg_time, n_bound, n_robots.
    """
    total_utility = float(np.sum(robot_utilities))

    bound_mask = binding_times >= 0
    n_bound = int(np.sum(bound_mask))
    avg_time = float(np.mean(binding_times[bound_mask])) if n_bound > 0 else float(max_steps)

    return {
        "total_utility": total_utility,
        "avg_completion_time": avg_time,
        "n_bound": n_bound,
        "n_robots": len(robot_utilities),
        "binding_rate": n_bound / len(robot_utilities),
    }


def aggregate_metrics(
    scenario_results: list[dict],
    baseline_results: list[dict] | None = None,
    max_possible_utility: float = 1.0,
    max_time: float = 150.0,
) -> dict:
    """Aggregate metrics across multiple test scenarios.

    Args:
        scenario_results: list of evaluate_scenario outputs.
        baseline_results: optional baseline results for DR computation.
        max_possible_utility: for NATU normalization.
        max_time: for NATC normalization.

    Returns:
        Dictionary with NATU, NATC, DR, and individual statistics.
    """
    utilities = np.array([r["total_utility"] for r in scenario_results])
    times = np.array([r["avg_completion_time"] for r in scenario_results])
    binding_rates = np.array([r["binding_rate"] for r in scenario_results])

    natu_mean, natu_std = compute_natu(utilities, max_possible_utility)
    natc_mean, natc_std = compute_natc(times, max_time)

    metrics = {
        "NATU_mean": natu_mean,
        "NATU_std": natu_std,
        "NATC_mean": natc_mean,
        "NATC_std": natc_std,
        "avg_binding_rate": float(np.mean(binding_rates)),
        "avg_total_utility": float(np.mean(utilities)),
        "std_total_utility": float(np.std(utilities)),
    }

    if baseline_results is not None:
        baseline_utilities = np.array([r["total_utility"] for r in baseline_results])
        metrics["DR"] = compute_dr(utilities, baseline_utilities)

    return metrics
