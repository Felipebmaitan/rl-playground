from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    """Hyperparameters for one directly comparable Pendulum experiment."""

    name: str
    policy_kind: str
    no_velocity: bool
    total_timesteps: int
    n_envs: int
    learning_rate: float
    n_steps: int
    batch_size: int
    n_epochs: int
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0


# Keep experiment-specific defaults together so changing one condition does not
# silently change the others. Command-line options in train.py can override the
# run length and number of parallel environments for a particular run.
EXPERIMENTS = {
    "mlp_full": ExperimentConfig(
        name="mlp_full",
        policy_kind="mlp",
        no_velocity=False,
        total_timesteps=200_000,
        n_envs=4,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
    ),
    "mlp_no_velocity": ExperimentConfig(
        name="mlp_no_velocity",
        policy_kind="mlp",
        no_velocity=True,
        total_timesteps=200_000,
        n_envs=4,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
    ),
    "mlp_lstm_no_velocity": ExperimentConfig(
        name="mlp_lstm_no_velocity",
        policy_kind="mlp_lstm",
        no_velocity=True,
        total_timesteps=200_000,
        n_envs=4,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
    ),
}


def select_experiment(policy_kind: str, no_velocity: bool) -> ExperimentConfig:
    if policy_kind == "mlp":
        return EXPERIMENTS["mlp_no_velocity" if no_velocity else "mlp_full"]
    if policy_kind == "mlp_lstm" and no_velocity:
        return EXPERIMENTS["mlp_lstm_no_velocity"]
    raise ValueError("--mlp-lstm is only part of this comparison with --no-velocity.")
