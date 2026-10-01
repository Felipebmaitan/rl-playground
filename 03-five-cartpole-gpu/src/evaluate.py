from __future__ import annotations

import argparse
from pathlib import Path

import torch

from five_cartpole import TaskConfig, FiveCartPoleGPU
from five_cartpole.policy import ActorCritic
from five_cartpole.renderer import FiveCartPoleRenderer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate or display the five-pole CartPole")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--task", choices=("balance", "swingup"), default="swingup")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--steps", type=int, default=3_600)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--depth", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    environment = FiveCartPoleGPU(1, task=TaskConfig(task=args.task), seed=args.seed, capture_graph=True)
    agent = None
    if args.model is not None:
        checkpoint = torch.load(args.model, map_location=environment.device, weights_only=False)
        model_config = checkpoint.get("model_config", {}) if isinstance(checkpoint, dict) else {}
        hidden_size = model_config.get("hidden_size", args.hidden_size)
        depth = model_config.get("depth", args.depth)
        agent = ActorCritic(environment.observation_size, hidden_size, depth).to(environment.device)
        agent.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
        agent.eval()

    renderer = FiveCartPoleRenderer(environment.physics) if args.render else None
    total_reward = 0.0
    try:
        for _ in range(args.steps):
            with torch.no_grad():
                if agent is None:
                    action = 2.0 * torch.rand((1, 1), device=environment.device) - 1.0
                else:
                    action = agent.deterministic_action(environment.observation)
                _, reward, done = environment.step(action)
            total_reward += reward.item()
            if renderer is not None:
                if not renderer.process_events():
                    break
                renderer.draw(
                    environment.state[0].tolist(),
                    reward.item(),
                    environment.episode_step[0].item(),
                )
            if done.item():
                print(f"episode return={total_reward:.3f}")
                total_reward = 0.0
    finally:
        if renderer is not None:
            renderer.close()


if __name__ == "__main__":
    main()
