from __future__ import annotations

import numpy as np

from triple_cartpole.config import PhysicsConfig
from triple_cartpole.reference import acceleration, dynamics_constants, semi_implicit_step


def test_angular_mass_constants_are_symmetric_positive_definite() -> None:
    _, angular_mass, total_mass = dynamics_constants(PhysicsConfig())
    np.testing.assert_allclose(angular_mass, angular_mass.T)
    assert np.all(np.linalg.eigvalsh(angular_mass) > 0.0)
    assert total_mass > 0.0


def test_downward_equilibrium_has_zero_acceleration() -> None:
    state = np.array([0.0, np.pi, np.pi, np.pi, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(acceleration(state, 0.0, PhysicsConfig()), 0.0, atol=1e-12)


def test_reference_integrator_stays_finite() -> None:
    state = np.array([0.0, 0.2, -0.1, 0.3, 0.0, 0.0, 0.0, 0.0])
    config = PhysicsConfig()
    for _ in range(1_000):
        state = semi_implicit_step(state, 0.15, config)
    assert np.isfinite(state).all()

