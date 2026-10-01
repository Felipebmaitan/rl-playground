from __future__ import annotations

from dataclasses import asdict

import torch
import warp as wp

from .config import PhysicsConfig, TaskConfig
from .kernels import EnvParams, reset_kernel, step_kernel


class TripleCartPoleGPU:
    """Fixed-size, auto-resetting vector environment with persistent GPU buffers."""

    observation_size = 12
    action_size = 1
    state_size = 8

    def __init__(
        self,
        num_envs: int,
        physics: PhysicsConfig | None = None,
        task: TaskConfig | None = None,
        device: str = "cuda:0",
        seed: int = 1,
        capture_graph: bool = True,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.num_envs = num_envs
        self.physics = physics or PhysicsConfig()
        self.task = task or TaskConfig()
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("A CUDA device was requested, but PyTorch cannot access CUDA")

        wp.init()
        self.wp_device = wp.device_from_torch(self.device)
        self.state = torch.empty((num_envs, self.state_size), dtype=torch.float32, device=self.device)
        self.action = torch.zeros((num_envs, self.action_size), dtype=torch.float32, device=self.device)
        self.observation = torch.empty(
            (num_envs, self.observation_size), dtype=torch.float32, device=self.device
        )
        self.reward = torch.empty(num_envs, dtype=torch.float32, device=self.device)
        self.done = torch.empty(num_envs, dtype=torch.uint8, device=self.device)
        self.episode_step = torch.empty(num_envs, dtype=torch.int32, device=self.device)
        self.episode_count = torch.empty(num_envs, dtype=torch.int32, device=self.device)
        self.episode_return = torch.empty(num_envs, dtype=torch.float32, device=self.device)
        self.completed_return = torch.empty(num_envs, dtype=torch.float32, device=self.device)
        self.completed_length = torch.empty(num_envs, dtype=torch.int32, device=self.device)

        self.params = self._make_params(seed)
        self._arrays = [
            wp.from_torch(tensor, requires_grad=False)
            for tensor in (
                self.state,
                self.action,
                self.observation,
                self.reward,
                self.done,
                self.episode_step,
                self.episode_count,
                self.episode_return,
                self.completed_return,
                self.completed_length,
            )
        ]
        if self.device.type == "cuda":
            self.torch_stream = torch.cuda.current_stream(self.device)
            self.warp_stream = wp.stream_from_torch(self.torch_stream)
        else:
            self.torch_stream = None
            self.warp_stream = wp.get_stream(self.wp_device)

        self._graph = None
        self.reset()
        if capture_graph and self.device.type == "cuda":
            self._capture_step_graph()
            self.reset()

    def _make_params(self, seed: int) -> EnvParams:
        physics = self.physics
        task = self.task
        params = EnvParams()
        params.cart_mass = physics.cart_mass
        params.mass1, params.mass2, params.mass3 = physics.link_masses
        params.length1, params.length2, params.length3 = physics.link_lengths
        params.gravity = physics.gravity
        params.cart_damping = physics.cart_damping
        params.joint_damping = physics.joint_damping
        params.max_force = physics.max_force
        params.physics_dt = physics.control_dt / physics.substeps
        params.substeps = physics.substeps
        params.track_limit = physics.track_limit
        params.max_episode_steps = task.max_episode_steps
        params.task_id = 1 if task.task == "swingup" else 0
        params.reset_position_noise = task.reset_position_noise
        params.reset_angle_noise = task.reset_angle_noise
        params.reset_velocity_noise = task.reset_velocity_noise
        params.mean_upright_weight = task.mean_upright_weight
        params.all_upright_weight = task.all_upright_weight
        params.stable_straight_weight = task.stable_straight_weight
        params.precision_bonus_weight = task.precision_bonus_weight
        params.stillness_coefficient = task.stillness_coefficient
        params.precision_cosine = task.precision_cosine
        params.precision_angular_velocity = task.precision_angular_velocity
        params.precision_cart_velocity = task.precision_cart_velocity
        params.precision_position_fraction = task.precision_position_fraction
        params.position_cost = task.position_cost
        params.velocity_cost = task.velocity_cost
        params.action_cost = task.action_cost
        params.terminal_penalty = task.terminal_penalty
        params.terminate_at_track_limit = int(task.terminate_at_track_limit)
        params.seed = seed
        return params

    def _launch_reset(self) -> None:
        wp.launch(
            reset_kernel,
            dim=self.num_envs,
            inputs=[self.params, *self._arrays[:8]],
            device=self.wp_device,
        )
        self.completed_return.zero_()
        self.completed_length.zero_()

    def _launch_step(self) -> None:
        wp.launch(
            step_kernel,
            dim=self.num_envs,
            inputs=[self.params, *self._arrays],
            device=self.wp_device,
        )

    def _capture_step_graph(self) -> None:
        # CUDA does not allow capture on PyTorch's legacy default stream. Record
        # once on a temporary non-default stream; the finished graph can then be
        # replayed on the shared current stream without cross-stream hand-offs.
        capture_torch_stream = torch.cuda.Stream(device=self.device)
        capture_warp_stream = wp.stream_from_torch(capture_torch_stream)
        with wp.ScopedStream(capture_warp_stream):
            with wp.ScopedCapture() as capture:
                self._launch_step()
        self._graph = capture.graph
        wp.synchronize_stream(capture_warp_stream)

    def reset(self) -> torch.Tensor:
        with wp.ScopedStream(self.warp_stream):
            self._launch_reset()
        return self.observation

    @torch.no_grad()
    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if action.shape != self.action.shape:
            raise ValueError(f"expected action shape {tuple(self.action.shape)}, got {tuple(action.shape)}")
        self.action.copy_(action)
        with wp.ScopedStream(self.warp_stream):
            if self._graph is None:
                self._launch_step()
            else:
                wp.capture_launch(self._graph, stream=self.warp_stream)
        return self.observation, self.reward, self.done

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        else:
            wp.synchronize_device(self.wp_device)

    def metadata(self) -> dict[str, object]:
        return {
            "num_envs": self.num_envs,
            "device": str(self.device),
            "physics": asdict(self.physics),
            "task": asdict(self.task),
        }
