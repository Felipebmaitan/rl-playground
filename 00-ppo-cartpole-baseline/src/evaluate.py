from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "models" / "ppo_cartpole.zip"
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the trained PPO CartPole model.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to evaluate.")
    parser.add_argument("--render", action="store_true", help="Render the environment while evaluating.")
    parser.add_argument(
        "--random-action-prob",
        type=float,
        default=0.0,
        help="Probability of replacing the policy action with a random action at each step.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run `python .\\src\\train.py` first."
        )

    render_mode = "human" if args.render else None
    env = gym.make("CartPole-v1", render_mode=render_mode)
    model = PPO.load(MODEL_PATH, env=env)

    rewards = evaluate_with_disturbances(
        model=model,
        env=env,
        episodes=args.episodes,
        random_action_prob=args.random_action_prob,
    )
    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))

    print(f"Mean reward over {args.episodes} episodes: {mean_reward:.2f} +/- {std_reward:.2f}")

    if args.render:
        run_episode(
            model=model,
            env=env,
            rng=np.random.default_rng(SEED + 10_000),
            random_action_prob=args.random_action_prob,
        )

    env.close()


def evaluate_with_disturbances(
    model: PPO,
    env: gym.Env,
    episodes: int,
    random_action_prob: float,
) -> list[float]:
    rewards = []

    for episode in range(episodes):
        reward = run_episode(
            model=model,
            env=env,
            rng=np.random.default_rng(SEED + episode),
            random_action_prob=random_action_prob,
        )
        rewards.append(reward)

    return rewards


def run_episode(
    model: PPO,
    env: gym.Env,
    rng: np.random.Generator,
    random_action_prob: float,
) -> float:
    observation, _ = env.reset()
    terminated = False
    truncated = False
    total_reward = 0.0

    while not (terminated or truncated):
        action, _ = model.predict(observation, deterministic=True)

        if rng.random() < random_action_prob:
            action = env.action_space.sample()

        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)

    return total_reward


if __name__ == "__main__":
    main()
