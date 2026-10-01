from __future__ import annotations

import warp as wp


@wp.kernel
def generalized_advantage_estimation_kernel(
    rewards: wp.array2d(dtype=float),
    dones: wp.array2d(dtype=wp.uint8),
    values: wp.array2d(dtype=float),
    next_done: wp.array(dtype=wp.uint8),
    next_value: wp.array(dtype=float),
    advantages: wp.array2d(dtype=float),
    returns: wp.array2d(dtype=float),
    rollout_steps: int,
    gamma: float,
    gae_lambda: float,
):
    """One GPU thread computes the complete reverse-time GAE scan for one environment."""

    env_id = wp.tid()
    last_advantage = float(0.0)
    for reverse_index in range(rollout_steps):
        step = rollout_steps - 1 - reverse_index
        following_value = next_value[env_id]
        following_done = next_done[env_id]
        if step < rollout_steps - 1:
            following_value = values[step + 1, env_id]
            following_done = dones[step + 1, env_id]
        next_nonterminal = 1.0 - float(following_done)
        delta = rewards[step, env_id] + gamma * following_value * next_nonterminal - values[step, env_id]
        last_advantage = delta + gamma * gae_lambda * next_nonterminal * last_advantage
        advantages[step, env_id] = last_advantage
        returns[step, env_id] = last_advantage + values[step, env_id]
