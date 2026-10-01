from __future__ import annotations

import math

import torch
from torch import nn


def _layer_init(layer: nn.Linear, std: float = math.sqrt(2.0)) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, observation_size: int, hidden_size: int = 256, depth: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_size = observation_size
        for _ in range(depth):
            layers.extend((_layer_init(nn.Linear(input_size, hidden_size)), nn.Tanh()))
            input_size = hidden_size
        self.trunk = nn.Sequential(*layers)
        self.actor_mean = _layer_init(nn.Linear(hidden_size, 1), std=0.01)
        self.critic = _layer_init(nn.Linear(hidden_size, 1), std=1.0)
        self.log_std = nn.Parameter(torch.tensor([-0.7], dtype=torch.float32))

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.trunk(observation)
        # PPO can otherwise increase variance without bound. Keeping this in a
        # physically useful range also prevents exp/log-probability overflow.
        log_std = self.log_std.float().clamp(-5.0, 1.0)
        return self.actor_mean(hidden).float(), log_std, self.critic(hidden).squeeze(-1).float()

    def sample(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std, value = self(observation)
        raw_action = mean + torch.exp(log_std) * torch.randn_like(mean)
        normalized_action = torch.tanh(raw_action)
        log_probability = self._log_probability(raw_action, normalized_action, mean, log_std)
        return normalized_action, raw_action, log_probability, value

    def evaluate_actions(
        self, observation: torch.Tensor, raw_action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std, value = self(observation)
        normalized_action = torch.tanh(raw_action)
        log_probability = self._log_probability(raw_action, normalized_action, mean, log_std)
        entropy = (log_std + 0.5 * (1.0 + math.log(2.0 * math.pi))).sum(-1)
        return log_probability, entropy, value

    @staticmethod
    def _log_probability(
        raw_action: torch.Tensor,
        normalized_action: torch.Tensor,
        mean: torch.Tensor,
        log_std: torch.Tensor,
    ) -> torch.Tensor:
        inverse_variance_error = (raw_action - mean) / torch.exp(log_std)
        gaussian_log_probability = (
            -0.5 * inverse_variance_error.square()
            - log_std
            - 0.5 * math.log(2.0 * math.pi)
        ).sum(-1)
        squash_correction = torch.log(1.0 - normalized_action.square() + 1e-6).sum(-1)
        return gaussian_log_probability - squash_correction

    def deterministic_action(self, observation: torch.Tensor) -> torch.Tensor:
        mean, _, _ = self(observation)
        return torch.tanh(mean)
