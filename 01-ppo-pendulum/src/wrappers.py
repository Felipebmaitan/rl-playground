from __future__ import annotations

import gymnasium as gym
import numpy as np


class NoVelocityObservation(gym.ObservationWrapper):
    """Expose Pendulum's angle representation but hide angular velocity.

    Pendulum observations are ``[cos(theta), sin(theta), angular_velocity]``.
    Keeping the first two values preserves the angle while making the current
    observation insufficient to identify the direction and speed of motion.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        source_space = env.observation_space
        if not isinstance(source_space, gym.spaces.Box) or source_space.shape != (3,):
            raise ValueError("NoVelocityObservation expects Pendulum's three-value Box observation.")
        self.observation_space = gym.spaces.Box(
            low=source_space.low[:2],
            high=source_space.high[:2],
            dtype=source_space.dtype,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return observation[:2]
