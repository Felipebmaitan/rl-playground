from __future__ import annotations

from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODEL_DIR / "ppo_cartpole"

ENV_ID = "CartPole-v1"
SEED = 42
TOTAL_TIMESTEPS = 50_000


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)

    env = make_vec_env(
        ENV_ID,
        n_envs=4, # Number of environments to run in parallel
        seed=SEED,
        env_kwargs={"render_mode": None},
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=1024, # Steps collected PER environment before policy update (per rollout), so policy updates every 4096 steps.
        batch_size=64, # How many samples to use for each gradient update.
        n_epochs=10, # Numbers of passes over each rollout data before collecting more. So, if we collect 4096 steps of data per rollout, with a batch size of 64 and 10 epochs, we will perform 640 gradient updates before collecting new data.
        gamma=0.99, # Discount factor. How much we value future rewards compared to immediate rewards. A value of 0 means we only care about immediate rewards, while a value of 1 means we care equally about all future rewards.
        gae_lambda=0.95, # Generalized Advantage Estimation. Controls bias/variance tradeoff. 1.0 uses longer-horizon advantage estimates; 0.0 uses one-step TD estimates.
        clip_range=0.2, # Clipping range. Discourages the policy from changing too much in a single update. The lower, the more conservative the update.
        ent_coef=0.015, # Entropy bonus. Higher values encourage more random/exploratory policies, but can slow or prevent convergence.
        verbose=1,
        seed=SEED,
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        progress_bar=True,
    )
    model.save(MODEL_PATH)
    env.close()

    print(f"Saved model to {MODEL_PATH}.zip")

    sanity_check_saved_model()


def sanity_check_saved_model() -> None:
    env = gym.make(ENV_ID)
    model = PPO.load(f"{MODEL_PATH}.zip", env=env)
    observation, _ = env.reset(seed=SEED)
    action, _ = model.predict(observation, deterministic=True)
    print(f"Sanity-check action for first observation: {action}")
    env.close()


if __name__ == "__main__":
    main()
