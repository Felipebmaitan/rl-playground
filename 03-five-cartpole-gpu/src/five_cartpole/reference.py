"""Readable NumPy reference dynamics used to validate the CUDA kernel."""

from __future__ import annotations

import numpy as np

from .config import PhysicsConfig


NUM_LINKS = 5


def dynamics_constants(config: PhysicsConfig) -> tuple[np.ndarray, np.ndarray, float]:
    masses = np.asarray(config.link_masses, dtype=np.float64)
    lengths = np.asarray(config.link_lengths, dtype=np.float64)
    centers = 0.5 * lengths
    inertias = masses * lengths**2 / 12.0

    trailing_mass = np.asarray([masses[index + 1 :].sum() for index in range(NUM_LINKS)])
    h = masses * centers + lengths * trailing_mass
    angular_mass = np.empty((NUM_LINKS, NUM_LINKS), dtype=np.float64)
    for row in range(NUM_LINKS):
        for column in range(NUM_LINKS):
            if row == column:
                angular_mass[row, column] = (
                    masses[row] * centers[row] ** 2
                    + trailing_mass[row] * lengths[row] ** 2
                    + inertias[row]
                )
            elif row < column:
                angular_mass[row, column] = lengths[row] * h[column]
            else:
                angular_mass[row, column] = lengths[column] * h[row]
    return h, angular_mass, config.cart_mass + float(masses.sum())


def acceleration(state: np.ndarray, normalized_action: float, config: PhysicsConfig) -> np.ndarray:
    """Return cart acceleration followed by five absolute angular accelerations."""

    state = np.asarray(state, dtype=np.float64)
    theta = state[1 : NUM_LINKS + 1]
    velocity = state[NUM_LINKS + 1 :]
    angular_velocity = velocity[1:]
    h, angular_mass, total_mass = dynamics_constants(config)

    mass_matrix = np.empty((NUM_LINKS + 1, NUM_LINKS + 1), dtype=np.float64)
    mass_matrix[0, 0] = total_mass
    mass_matrix[0, 1:] = h * np.cos(theta)
    mass_matrix[1:, 0] = mass_matrix[0, 1:]
    for row in range(NUM_LINKS):
        for column in range(NUM_LINKS):
            mass_matrix[row + 1, column + 1] = (
                angular_mass[row, column] * np.cos(theta[row] - theta[column])
            )

    bias = np.empty(NUM_LINKS + 1, dtype=np.float64)
    bias[0] = (
        -np.sum(h * np.sin(theta) * angular_velocity**2)
        + config.cart_damping * velocity[0]
    )
    for row in range(NUM_LINKS):
        bias[row + 1] = (
            np.sum(
                angular_mass[row]
                * np.sin(theta[row] - theta)
                * angular_velocity**2
            )
            - config.gravity * h[row] * np.sin(theta[row])
            + config.joint_damping * angular_velocity[row]
        )

    generalized_force = np.zeros(NUM_LINKS + 1, dtype=np.float64)
    generalized_force[0] = np.clip(normalized_action, -1.0, 1.0) * config.max_force
    return np.linalg.solve(mass_matrix, generalized_force - bias)


def semi_implicit_step(
    state: np.ndarray, normalized_action: float, config: PhysicsConfig
) -> np.ndarray:
    result = np.asarray(state, dtype=np.float64).copy()
    dt = config.control_dt / config.substeps
    for _ in range(config.substeps):
        result[NUM_LINKS + 1 :] += acceleration(result, normalized_action, config) * dt
        result[: NUM_LINKS + 1] += result[NUM_LINKS + 1 :] * dt
        result[1 : NUM_LINKS + 1] = np.arctan2(
            np.sin(result[1 : NUM_LINKS + 1]), np.cos(result[1 : NUM_LINKS + 1])
        )
    return result
