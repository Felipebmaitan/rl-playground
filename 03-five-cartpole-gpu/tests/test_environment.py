from __future__ import annotations

import numpy as np
import pytest
import torch

from five_cartpole import PhysicsConfig, TaskConfig, FiveCartPoleGPU
from five_cartpole.reference import semi_implicit_step


def available_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def test_environment_shapes_and_auto_reset() -> None:
    task = TaskConfig(task="balance", max_episode_steps=2)
    env = FiveCartPoleGPU(32, task=task, device=available_device(), capture_graph=False)
    assert env.reset().shape == (32, env.observation_size)
    action = torch.zeros((32, 1), device=env.device)
    env.step(action)
    observation, reward, done = env.step(action)
    env.synchronize()
    assert observation.isfinite().all().item()
    assert reward.isfinite().all().item()
    assert done.bool().all().item()
    assert (env.episode_step == 0).all().item()


def test_aggressive_actions_never_emit_unbounded_rewards() -> None:
    task = TaskConfig(task="swingup", max_episode_steps=500)
    env = FiveCartPoleGPU(256, task=task, device=available_device(), capture_graph=False)
    for step in range(500):
        sign = -1.0 if step % 2 else 1.0
        _, reward, _ = env.step(torch.full((256, 1), sign, device=env.device))
        assert reward.isfinite().all().item()
        assert (reward >= task.terminal_penalty).all().item()
        assert (reward <= 1.0).all().item()


def test_reward_requires_all_links_to_be_upright_straight_and_still() -> None:
    physics = PhysicsConfig(gravity=0.0, cart_damping=0.0, joint_damping=0.0, substeps=1)
    task = TaskConfig(task="balance", max_episode_steps=100, terminate_at_track_limit=False)
    env = FiveCartPoleGPU(
        5, physics=physics, task=task, device=available_device(), capture_graph=False
    )
    states = torch.zeros((5, 12), dtype=torch.float32, device=env.device)
    states[1, 1:6] = torch.pi  # Straight, but entirely downward.
    states[2, 5] = torch.pi  # Four links upright, one entirely downward.
    states[3, 7:12] = 8.0  # Upright, but passing through much too quickly.
    states[4, 1:6] = torch.tensor(
        [0.0, 0.6, -0.6, 0.4, -0.4], device=env.device
    )  # Bent chain.
    env.state.copy_(states)

    _, reward, _ = env.step(torch.zeros((5, 1), device=env.device))
    env.synchronize()
    upright, downward, two_up, moving, bent = reward.cpu().tolist()

    assert upright > 0.99
    assert abs(downward) < 1e-5
    assert two_up < 0.13
    assert moving < upright * 0.60
    assert bent < upright


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for kernel/reference comparison")
def test_cuda_step_matches_numpy_reference() -> None:
    physics = PhysicsConfig(substeps=1)
    task = TaskConfig(task="balance", max_episode_steps=100, terminate_at_track_limit=False)
    env = FiveCartPoleGPU(1, physics=physics, task=task, capture_graph=False)
    initial = np.array(
        [0.1, 0.20, -0.15, 0.30, -0.25, 0.10, -0.1, 0.2, -0.3, 0.1, 0.25, -0.2],
        dtype=np.float32,
    )
    env.state[0].copy_(torch.from_numpy(initial).to(env.device))
    action_value = 0.35
    expected = semi_implicit_step(initial, action_value, physics)
    env.step(torch.tensor([[action_value]], device=env.device))
    env.synchronize()
    np.testing.assert_allclose(env.state[0].cpu().numpy(), expected, rtol=2e-4, atol=2e-5)
