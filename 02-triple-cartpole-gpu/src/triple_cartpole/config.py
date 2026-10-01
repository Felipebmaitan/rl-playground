from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PhysicsConfig:
    """Physical parameters for three uniform rods connected by frictional pins."""

    cart_mass: float = 1.0
    link_masses: tuple[float, float, float] = (0.20, 0.15, 0.10)
    link_lengths: tuple[float, float, float] = (0.60, 0.50, 0.40)
    gravity: float = 9.81
    cart_damping: float = 0.05
    joint_damping: float = 0.002
    max_force: float = 15.0
    control_dt: float = 1.0 / 60.0
    substeps: int = 4
    track_limit: float = 2.4

    def __post_init__(self) -> None:
        if self.cart_mass <= 0.0:
            raise ValueError("cart_mass must be positive")
        if any(value <= 0.0 for value in self.link_masses + self.link_lengths):
            raise ValueError("link masses and lengths must be positive")
        if self.control_dt <= 0.0 or self.substeps < 1:
            raise ValueError("control_dt must be positive and substeps must be >= 1")


@dataclass(frozen=True, slots=True)
class TaskConfig:
    """Reset distribution, episode length, and dense reward weights."""

    task: str = "swingup"
    max_episode_steps: int = 1_200
    reset_position_noise: float = 0.10
    reset_angle_noise: float = 0.12
    reset_velocity_noise: float = 0.05
    mean_upright_weight: float = 0.15
    all_upright_weight: float = 0.25
    stable_straight_weight: float = 0.45
    precision_bonus_weight: float = 0.15
    stillness_coefficient: float = 0.02
    precision_cosine: float = 0.985
    precision_angular_velocity: float = 0.75
    precision_cart_velocity: float = 0.50
    precision_position_fraction: float = 0.15
    position_cost: float = 0.08
    velocity_cost: float = 0.002
    action_cost: float = 0.001
    terminal_penalty: float = -1.0
    terminate_at_track_limit: bool = True

    def __post_init__(self) -> None:
        if self.task not in {"balance", "swingup"}:
            raise ValueError("task must be 'balance' or 'swingup'")
        if self.max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive")
        reward_weights = (
            self.mean_upright_weight,
            self.all_upright_weight,
            self.stable_straight_weight,
            self.precision_bonus_weight,
        )
        if any(weight < 0.0 for weight in reward_weights):
            raise ValueError("reward weights must be non-negative")
        if abs(sum(reward_weights) - 1.0) > 1e-6:
            raise ValueError("positive reward weights must sum to 1.0")
