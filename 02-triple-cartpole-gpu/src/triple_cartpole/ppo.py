from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import torch
from torch import nn

from .env import TripleCartPoleGPU
from .policy import ActorCritic


@dataclass(slots=True)
class PPOConfig:
    rollout_steps: int = 64
    total_timesteps: int = 100_000_000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 4
    minibatch_size: int = 65_536
    clip_coefficient: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.002
    max_grad_norm: float = 0.5
    use_bfloat16: bool = True
    capture_policy_graph: bool = True


class PPOTrainer:
    def __init__(
        self,
        environment: TripleCartPoleGPU,
        agent: ActorCritic,
        config: PPOConfig,
    ) -> None:
        self.env = environment
        self.agent = agent
        self.config = config
        self.device = environment.device
        self.optimizer = torch.optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)
        steps, envs, obs = config.rollout_steps, environment.num_envs, environment.observation_size
        self.observations = torch.empty((steps, envs, obs), device=self.device)
        self.raw_actions = torch.empty((steps, envs, 1), device=self.device)
        self.log_probabilities = torch.empty((steps, envs), device=self.device)
        self.rewards = torch.empty((steps, envs), device=self.device)
        self.dones = torch.empty((steps, envs), dtype=torch.float32, device=self.device)
        self.values = torch.empty((steps, envs), device=self.device)
        self.advantages = torch.empty((steps, envs), device=self.device)
        self.returns = torch.empty((steps, envs), device=self.device)
        self.next_observation = environment.observation
        self.next_done = torch.zeros(envs, dtype=torch.float32, device=self.device)
        self._policy_graph: torch.cuda.CUDAGraph | None = None
        self._graph_outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self.completed_return_sum = torch.zeros((), device=self.device)
        self.completed_episode_count = torch.zeros((), device=self.device)

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
        self.completed_return_sum.zero_()
        self.completed_episode_count.zero_()
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
            self.completed_return_sum.add_(self.env.completed_return.sum())
            self.completed_episode_count.add_(done.sum())

        with self._autocast():
            _, _, next_value = self.agent(self.next_observation)
        last_advantage = torch.zeros(self.env.num_envs, device=self.device)
        for step in reversed(range(self.config.rollout_steps)):
            if step == self.config.rollout_steps - 1:
                next_nonterminal = 1.0 - self.next_done
                following_value = next_value
            else:
                next_nonterminal = 1.0 - self.dones[step + 1]
                following_value = self.values[step + 1]
            delta = (
                self.rewards[step]
                + self.config.gamma * following_value * next_nonterminal
                - self.values[step]
            )
            last_advantage = (
                delta
                + self.config.gamma
                * self.config.gae_lambda
                * next_nonterminal
                * last_advantage
            )
            self.advantages[step].copy_(last_advantage)
        self.returns.copy_(self.advantages + self.values)

    def update(self) -> dict[str, float]:
        batch_size = self.config.rollout_steps * self.env.num_envs
        observations = self.observations.reshape(batch_size, -1)
        raw_actions = self.raw_actions.reshape(batch_size, -1)
        old_log_probabilities = self.log_probabilities.reshape(batch_size)
        advantages = self.advantages.reshape(batch_size)
        returns = self.returns.reshape(batch_size)
        old_values = self.values.reshape(batch_size)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        minibatch_size = min(self.config.minibatch_size, batch_size)

        metrics: dict[str, torch.Tensor] = {}
        for _ in range(self.config.update_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, minibatch_size):
                batch_indices = indices[start : start + minibatch_size]
                with self._autocast():
                    new_log_probability, entropy, new_value = self.agent.evaluate_actions(
                        observations[batch_indices], raw_actions[batch_indices]
                    )
                    log_ratio = new_log_probability - old_log_probabilities[batch_indices]
                    # Clipping before exp prevents one extreme sample from
                    # overflowing the complete minibatch update.
                    ratio = log_ratio.clamp(-20.0, 20.0).exp()
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
                    "approx_kl": ((ratio - 1.0) - log_ratio).mean().detach(),
                    "old_value": old_values[batch_indices].mean().detach(),
                }
        return {name: value.item() for name, value in metrics.items()}

    def _assert_finite_agent(self) -> None:
        for name, parameter in self.agent.named_parameters():
            if not torch.isfinite(parameter).all().item():
                raise FloatingPointError(f"non-finite policy parameter after PPO update: {name}")

    def train(self, checkpoint_dir: Path | None = None) -> None:
        batch_size = self.config.rollout_steps * self.env.num_envs
        updates = max(1, self.config.total_timesteps // batch_size)
        start_time = time.perf_counter()
        completed_steps = 0
        for update in range(1, updates + 1):
            self.collect_rollout()
            metrics = self.update()
            self._assert_finite_agent()
            completed_steps += batch_size
            self.env.synchronize()
            elapsed = time.perf_counter() - start_time
            steps_per_second = completed_steps / elapsed
            episode_count = self.completed_episode_count.item()
            mean_return = self.completed_return_sum.item() / episode_count if episode_count else float("nan")
            print(
                f"update={update}/{updates} steps={completed_steps:,} sps={steps_per_second:,.0f} "
                f"return={mean_return:.3f} loss={metrics['loss']:.4f} kl={metrics['approx_kl']:.5f}"
            )
            if checkpoint_dir is not None and (update % 25 == 0 or update == updates):
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model": self.agent.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "environment": self.env.metadata(),
                        "completed_steps": completed_steps,
                    },
                    checkpoint_dir / "latest.pt",
                )
