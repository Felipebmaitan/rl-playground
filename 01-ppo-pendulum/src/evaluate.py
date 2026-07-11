from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO

from experiments import ExperimentConfig, select_experiment
from wrappers import NoVelocityObservation

PROJECT_DIR = Path(__file__).resolve().parents[1]
ENV_ID = "Pendulum-v1"
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one trained Pendulum experiment.")
    policy_group = parser.add_mutually_exclusive_group()
    policy_group.add_argument("--mlp", dest="policy_kind", action="store_const", const="mlp", default="mlp", help="Evaluate standard PPO (the default).")
    policy_group.add_argument("--mlp-lstm", dest="policy_kind", action="store_const", const="mlp_lstm", help="Evaluate RecurrentPPO.")
    parser.add_argument("--no-velocity", action="store_true", help="Use the observation with angular velocity hidden.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to evaluate.")
    parser.add_argument("--render", action="store_true", help="Render evaluation episodes.")
    parser.add_argument("--action-noise-std", type=float, default=0.0, help="Gaussian action-noise standard deviation.")
    return parser.parse_args()


def model_path(config: ExperimentConfig) -> Path:
    return PROJECT_DIR / "models" / f"ppo_pendulum_{config.name}.zip"


def main() -> None:
    args = parse_args()
    if args.episodes <= 0 or args.action_noise_std < 0:
        raise ValueError("--episodes must be positive and --action-noise-std cannot be negative.")
    config = select_experiment(args.policy_kind, args.no_velocity)
    saved_model_path = model_path(config)
    if not saved_model_path.exists():
        raise FileNotFoundError(f"Model not found at {saved_model_path}. Train the matching flag combination first.")

    env = gym.make(ENV_ID, render_mode="human" if args.render else None)
    if config.no_velocity:
        env = NoVelocityObservation(env)
    algorithm = RecurrentPPO if config.policy_kind == "mlp_lstm" else PPO
    model = algorithm.load(saved_model_path, env=env)

    rewards: list[float] = []
    for episode in range(args.episodes):
        observation, _ = env.reset(seed=SEED + episode)
        terminated = truncated = False
        total_reward = 0.0
        rng = np.random.default_rng(SEED + episode)
        lstm_state = None
        episode_start = np.ones((1,), dtype=bool)
        while not (terminated or truncated):
            action, lstm_state = model.predict(
                observation,
                state=lstm_state,
                episode_start=episode_start,
                deterministic=True,
            )
            if args.action_noise_std:
                action = action + rng.normal(0.0, args.action_noise_std, size=action.shape)
                action = np.clip(action, env.action_space.low, env.action_space.high)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            episode_start = np.array([terminated or truncated], dtype=bool)
        rewards.append(total_reward)

    print(f"{config.name}: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f} over {args.episodes} episodes")
    env.close()


if __name__ == "__main__":
    main()
