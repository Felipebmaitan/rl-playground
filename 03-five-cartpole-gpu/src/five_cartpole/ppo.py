from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import time

import torch
import warp as wp
from torch import nn

from .env import FiveCartPoleGPU
from .policy import ActorCritic
from .training_kernels import generalized_advantage_estimation_kernel


@dataclass(slots=True)
class PPOConfig:
    rollout_steps: int = 32
    total_timesteps: int = 100_000_000
    learning_rate: float = 5e-5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 3
    minibatch_size: int = 131_072
    clip_coefficient: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.003
    target_kl: float | None = 0.012
    max_grad_norm: float = 0.5
    use_bfloat16: bool = True
    capture_policy_graph: bool = True
    anneal_learning_rate: bool = True
    minimum_learning_rate_fraction: float = 0.10


class PPOTrainer:
    def __init__(
        self,
        environment: FiveCartPoleGPU,
        agent: ActorCritic,
        config: PPOConfig,
    ) -> None:
        self.env = environment
        self.agent = agent
        self.config = config
        self.device = environment.device
        self.optimizer = torch.optim.Adam(
            agent.parameters(),
            lr=config.learning_rate,
            eps=1e-5,
            fused=self.device.type == "cuda",
        )
        steps, envs, obs = config.rollout_steps, environment.num_envs, environment.observation_size
        self.observations = torch.empty((steps, envs, obs), device=self.device)
        self.raw_actions = torch.empty((steps, envs, 1), device=self.device)
        self.log_probabilities = torch.empty((steps, envs), device=self.device)
        self.rewards = torch.empty((steps, envs), device=self.device)
        self.dones = torch.empty((steps, envs), dtype=torch.uint8, device=self.device)
        self.values = torch.empty((steps, envs), device=self.device)
        self.advantages = torch.empty((steps, envs), device=self.device)
        self.returns = torch.empty((steps, envs), device=self.device)
        self.next_observation = environment.observation
        self.next_done = torch.zeros(envs, dtype=torch.uint8, device=self.device)
        self.next_value = torch.empty(envs, dtype=torch.float32, device=self.device)
        self._policy_graph: torch.cuda.CUDAGraph | None = None
        self._graph_outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._gae_arrays = [
            wp.from_torch(tensor, requires_grad=False)
            for tensor in (
                self.rewards,
                self.dones,
                self.values,
                self.next_done,
                self.next_value,
                self.advantages,
                self.returns,
            )
        ]

    def _autocast(self):
        enabled = self.config.use_bfloat16 and self.device.type == "cuda"
        return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=enabled)

    def _capture_policy_graph(self) -> None:
        if self.device.type != "cuda" or not self.config.capture_policy_graph:
            return
        # The observation tensor and model parameter storages never move. CUDA
        # Graph replay therefore sees newly simulated states and newly optimized
        # weights without rebuilding the graph.
        current_stream = torch.cuda.current_stream(self.device)
        warmup_stream = torch.cuda.Stream(device=self.device)
        warmup_stream.wait_stream(current_stream)
        with torch.cuda.stream(warmup_stream), torch.no_grad():
            for _ in range(3):
                with self._autocast():
                    self.agent.sample(self.next_observation)
        current_stream.wait_stream(warmup_stream)
        graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(graph):
            with self._autocast():
                outputs = self.agent.sample(self.next_observation)
        self._policy_graph = graph
        self._graph_outputs = outputs

    def _sample_policy(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._policy_graph is None and self.config.capture_policy_graph and self.device.type == "cuda":
            self._capture_policy_graph()
        if self._policy_graph is not None:
            self._policy_graph.replay()
            assert self._graph_outputs is not None
            return self._graph_outputs
        with self._autocast():
            return self.agent.sample(self.next_observation)

    @torch.no_grad()
    def collect_rollout(self) -> None:
        self.env.reset_rollout_statistics()
        for step in range(self.config.rollout_steps):
            self.observations[step].copy_(self.next_observation)
            self.dones[step].copy_(self.next_done)
            action, raw_action, log_probability, value = self._sample_policy()
            self.raw_actions[step].copy_(raw_action)
            self.log_probabilities[step].copy_(log_probability)
            self.values[step].copy_(value)
            self.next_observation, reward, done = self.env.step(action)
            self.rewards[step].copy_(reward)
            self.next_done.copy_(done)

        with self._autocast():
            _, _, next_value = self.agent(self.next_observation)
        self.next_value.copy_(next_value)
        with wp.ScopedStream(self.env.warp_stream):
            wp.launch(
                generalized_advantage_estimation_kernel,
                dim=self.env.num_envs,
                inputs=[
                    *self._gae_arrays,
                    self.config.rollout_steps,
                    self.config.gamma,
                    self.config.gae_lambda,
                ],
                device=self.env.wp_device,
            )

    def update(self) -> dict[str, float]:
        batch_size = self.config.rollout_steps * self.env.num_envs
        observations = self.observations.reshape(batch_size, -1)
        raw_actions = self.raw_actions.reshape(batch_size, -1)
        old_log_probabilities = self.log_probabilities.reshape(batch_size)
        advantages = self.advantages.reshape(batch_size)
        returns = self.returns.reshape(batch_size)
        old_values = self.values.reshape(batch_size)
        advantage_mean = advantages.mean()
        advantage_std = advantages.std(unbiased=False)
        advantages.sub_(advantage_mean).div_(advantage_std + 1e-8)
        minibatch_size = min(self.config.minibatch_size, batch_size)
        num_minibatches = (batch_size + minibatch_size - 1) // minibatch_size

        metrics: dict[str, torch.Tensor] = {}
        stop_for_kl = False
        observed_kl = 0.0
        for _ in range(self.config.update_epochs):
            # Flattening is time-major, so each contiguous block already spans
            # thousands of independent environments. Shuffling block order keeps
            # PPO stochastic without six expensive million-element GPU gathers.
            minibatch_order = torch.randperm(num_minibatches).tolist()
            for minibatch_index in minibatch_order:
                start = minibatch_index * minibatch_size
                batch_indices = slice(start, min(start + minibatch_size, batch_size))
                with self._autocast():
                    new_log_probability, entropy, new_value = self.agent.evaluate_actions(
                        observations[batch_indices], raw_actions[batch_indices]
                    )
                    log_ratio = new_log_probability - old_log_probabilities[batch_indices]
                    # Clipping before exp prevents one extreme sample from
                    # overflowing the complete minibatch update.
                    ratio = log_ratio.clamp(-20.0, 20.0).exp()
                    approximate_kl = ((ratio - 1.0) - log_ratio).mean()
                    observed_kl = approximate_kl.detach().item()
                    if (
                        self.config.target_kl is not None
                        and observed_kl > self.config.target_kl
                    ):
                        stop_for_kl = True
                        break
                    unclipped_policy_loss = -advantages[batch_indices] * ratio
                    clipped_policy_loss = -advantages[batch_indices] * ratio.clamp(
                        1.0 - self.config.clip_coefficient,
                        1.0 + self.config.clip_coefficient,
                    )
                    policy_loss = torch.maximum(unclipped_policy_loss, clipped_policy_loss).mean()
                    value_delta = new_value - old_values[batch_indices]
                    clipped_value = old_values[batch_indices] + value_delta.clamp(
                        -self.config.clip_coefficient, self.config.clip_coefficient
                    )
                    value_loss = 0.5 * torch.maximum(
                        (new_value - returns[batch_indices]).square(),
                        (clipped_value - returns[batch_indices]).square(),
                    ).mean()
                    entropy_loss = entropy.mean()
                    loss = (
                        policy_loss
                        + self.config.value_coefficient * value_loss
                        - self.config.entropy_coefficient * entropy_loss
                    )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.agent.parameters(), self.config.max_grad_norm, error_if_nonfinite=True
                )
                self.optimizer.step()
                with torch.no_grad():
                    self.agent.log_std.clamp_(-5.0, 1.0)
                metrics = {
                    "loss": loss.detach(),
                    "policy_loss": policy_loss.detach(),
                    "value_loss": value_loss.detach(),
                    "entropy": entropy_loss.detach(),
                    "approx_kl": approximate_kl.detach(),
                    "old_value": old_values[batch_indices].mean().detach(),
                }
            if stop_for_kl:
                break
        if not metrics:
            raise RuntimeError("target KL stopped PPO before any minibatch update")
        metrics["kl_stopped"] = torch.tensor(float(stop_for_kl), device=self.device)
        metrics["observed_kl"] = torch.tensor(observed_kl, device=self.device)
        return {name: value.item() for name, value in metrics.items()}

    def _assert_finite_agent(self) -> None:
        for name, parameter in self.agent.named_parameters():
            if not torch.isfinite(parameter).all().item():
                raise FloatingPointError(f"non-finite policy parameter after PPO update: {name}")

    def train(
        self,
        checkpoint_dir: Path | None = None,
        initial_completed_steps: int = 0,
        best_return: float = -math.inf,
    ) -> None:
        batch_size = self.config.rollout_steps * self.env.num_envs
        # A rollout batch cannot be split without changing PPO's sampling
        # geometry, so round upward and never stop short of the requested budget.
        updates = max(1, math.ceil(self.config.total_timesteps / batch_size))
        if initial_completed_steps < 0 or initial_completed_steps % batch_size != 0:
            raise ValueError("resumed completed steps must be a non-negative whole rollout batch")
        starting_update = initial_completed_steps // batch_size
        if starting_update >= updates:
            print(
                f"target already reached: completed_steps={initial_completed_steps:,} "
                f"target_steps={self.config.total_timesteps:,}"
            )
            return
        start_time = time.perf_counter()
        completed_steps = initial_completed_steps
        session_steps = 0
        remaining_updates = updates - starting_update
        for update in range(starting_update + 1, updates + 1):
            if self.config.anneal_learning_rate:
                session_update = update - starting_update - 1
                progress = session_update / max(remaining_updates - 1, 1)
                fraction = 1.0 - progress * (1.0 - self.config.minimum_learning_rate_fraction)
                self.optimizer.param_groups[0]["lr"] = self.config.learning_rate * fraction
            self.collect_rollout()
            metrics = self.update()
            self._assert_finite_agent()
            completed_steps += batch_size
            session_steps += batch_size
            self.env.synchronize()
            elapsed = time.perf_counter() - start_time
            steps_per_second = session_steps / elapsed
            episode_count = self.env.rollout_episode_count.item()
            mean_return = (
                self.env.rollout_return_sum.item() / episode_count if episode_count else float("nan")
            )
            print(
                f"update={update}/{updates} steps={completed_steps:,} sps={steps_per_second:,.0f} "
                f"return={mean_return:.3f} loss={metrics['loss']:.4f} "
                f"kl={metrics['approx_kl']:.5f} observed_kl={metrics['observed_kl']:.5f} "
                f"kl_stop={int(metrics['kl_stopped'])}"
            )
            payload = {
                "model": self.agent.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "model_config": {
                    "observation_size": self.agent.observation_size,
                    "hidden_size": self.agent.hidden_size,
                    "depth": self.agent.depth,
                },
                "environment": self.env.metadata(),
                "completed_steps": completed_steps,
                "mean_return": mean_return,
            }
            if checkpoint_dir is not None and math.isfinite(mean_return) and mean_return > best_return:
                best_return = mean_return
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save(payload, checkpoint_dir / "best.pt")
                print(f"new_best_return={best_return:.3f} checkpoint={checkpoint_dir / 'best.pt'}")
            if checkpoint_dir is not None and (update % 25 == 0 or update == updates):
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save(payload, checkpoint_dir / "latest.pt")
