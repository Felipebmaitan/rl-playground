from __future__ import annotations

import argparse
from pathlib import Path

from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from experiments import ExperimentConfig, select_experiment
from wrappers import NoVelocityObservation

PROJECT_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_DIR / "runs"
ENV_ID = "Pendulum-v1"
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one Pendulum partial-observability experiment.")
    policy_group = parser.add_mutually_exclusive_group()
    policy_group.add_argument("--mlp", dest="policy_kind", action="store_const", const="mlp", default="mlp", help="Use standard SB3 PPO (the default).")
    policy_group.add_argument("--mlp-lstm", dest="policy_kind", action="store_const", const="mlp_lstm", help="Use SB3-Contrib RecurrentPPO with MlpLstmPolicy.")
    parser.add_argument("--no-velocity", action="store_true", help="Hide Pendulum's angular-velocity observation.")
    parser.add_argument("--total-timesteps", type=int, help="Override the selected experiment's default training budget.")
    parser.add_argument("--n-envs", type=int, help="Override the selected experiment's number of parallel environments.")
    return parser.parse_args()


def model_path(config: ExperimentConfig) -> Path:
    return PROJECT_DIR / "models" / f"ppo_pendulum_{config.name}"


def main() -> None:
    args = parse_args()
    config = select_experiment(args.policy_kind, args.no_velocity)
    total_timesteps = args.total_timesteps if args.total_timesteps is not None else config.total_timesteps
    n_envs = args.n_envs if args.n_envs is not None else config.n_envs
    if total_timesteps <= 0 or n_envs <= 0:
        raise ValueError("--total-timesteps and --n-envs must be positive.")

    MODEL_DIR = PROJECT_DIR / "models"
    MODEL_DIR.mkdir(exist_ok=True)
    RUNS_DIR.mkdir(exist_ok=True)

    env = make_vec_env(
        ENV_ID,
        n_envs=n_envs,
        seed=SEED,
        wrapper_class=NoVelocityObservation if config.no_velocity else None,
    )
    algorithm = RecurrentPPO if config.policy_kind == "mlp_lstm" else PPO
    policy = "MlpLstmPolicy" if config.policy_kind == "mlp_lstm" else "MlpPolicy"
    model = algorithm(
        policy=policy,
        env=env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        tensorboard_log=str(RUNS_DIR),
        verbose=1,
        seed=SEED,
    )
    model.learn(total_timesteps=total_timesteps, tb_log_name=config.name, progress_bar=True)
    saved_model_path = model_path(config)
    model.save(saved_model_path)
    env.close()
    print(f"Saved {config.name} model to {saved_model_path}.zip")


if __name__ == "__main__":
    main()
