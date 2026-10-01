"""Readable NumPy reference dynamics used to validate the CUDA kernel."""

from __future__ import annotations

import numpy as np

from .config import PhysicsConfig


def dynamics_constants(config: PhysicsConfig) -> tuple[np.ndarray, np.ndarray, float]:
    masses = np.asarray(config.link_masses, dtype=np.float64)
    lengths = np.asarray(config.link_lengths, dtype=np.float64)
    centers = 0.5 * lengths
    inertias = masses * lengths**2 / 12.0

    h = np.empty(3, dtype=np.float64)
    h[0] = masses[0] * centers[0] + (masses[1] + masses[2]) * lengths[0]
    h[1] = masses[1] * centers[1] + masses[2] * lengths[1]
    h[2] = masses[2] * centers[2]

    angular_mass = np.empty((3, 3), dtype=np.float64)
    angular_mass[0, 0] = (
        masses[0] * centers[0] ** 2
        + (masses[1] + masses[2]) * lengths[0] ** 2
        + inertias[0]
    )
    angular_mass[1, 1] = (
        masses[1] * centers[1] ** 2 + masses[2] * lengths[1] ** 2 + inertias[1]
    )
    angular_mass[2, 2] = masses[2] * centers[2] ** 2 + inertias[2]
    angular_mass[0, 1] = angular_mass[1, 0] = (
        masses[1] * lengths[0] * centers[1]
        + masses[2] * lengths[0] * lengths[1]
    )
    angular_mass[0, 2] = angular_mass[2, 0] = masses[2] * lengths[0] * centers[2]
    angular_mass[1, 2] = angular_mass[2, 1] = masses[2] * lengths[1] * centers[2]
    total_mass = config.cart_mass + float(masses.sum())
    return h, angular_mass, total_mass


def acceleration(state: np.ndarray, normalized_action: float, config: PhysicsConfig) -> np.ndarray:
    """Return [cart acceleration, three absolute angular accelerations]."""

    state = np.asarray(state, dtype=np.float64)
    theta = state[1:4]
    velocity = state[4:8]
    angular_velocity = velocity[1:4]
    h, angular_mass, total_mass = dynamics_constants(config)

    mass_matrix = np.empty((4, 4), dtype=np.float64)
    mass_matrix[0, 0] = total_mass
    mass_matrix[0, 1:] = h * np.cos(theta)
    mass_matrix[1:, 0] = mass_matrix[0, 1:]
    for row in range(3):
        for column in range(3):
            mass_matrix[row + 1, column + 1] = (
                angular_mass[row, column] * np.cos(theta[row] - theta[column])
            )

    bias = np.empty(4, dtype=np.float64)
    bias[0] = (
        -np.sum(h * np.sin(theta) * angular_velocity**2)
        + config.cart_damping * velocity[0]
    )
    for row in range(3):
        bias[row + 1] = (
            np.sum(
                angular_mass[row]
                * np.sin(theta[row] - theta)
                * angular_velocity**2
            )
            - config.gravity * h[row] * np.sin(theta[row])
            + config.joint_damping * angular_velocity[row]
        )

    generalized_force = np.array(
        [np.clip(normalized_action, -1.0, 1.0) * config.max_force, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    return np.linalg.solve(mass_matrix, generalized_force - bias)


def semi_implicit_step(
    state: np.ndarray, normalized_action: float, config: PhysicsConfig
) -> np.ndarray:
    result = np.asarray(state, dtype=np.float64).copy()
    dt = config.control_dt / config.substeps
    for _ in range(config.substeps):
        result[4:8] += acceleration(result, normalized_action, config) * dt
        result[0:4] += result[4:8] * dt
        result[1:4] = np.arctan2(np.sin(result[1:4]), np.cos(result[1:4]))
    return result

