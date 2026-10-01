from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import random

import numpy as np
import torch

from triple_cartpole import TaskConfig, TripleCartPoleGPU
from triple_cartpole.policy import ActorCritic
from triple_cartpole.ppo import PPOConfig, PPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on the GPU-native triple CartPole")
    parser.add_argument("--task", choices=("balance", "swingup"), default="swingup")
    parser.add_argument("--num-envs", type=int, default=16_384)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--total-timesteps", type=int, default=100_000_000)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--minibatch-size", type=int, default=65_536)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-bfloat16", action="store_true")
    parser.add_argument("--no-compile", action="store_true", help="disable torch.compile for the policy")
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument("--no-policy-graph", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="one tiny rollout/update, no checkpoint")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("models"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        # Large enough to exercise the batched GPU path, but still exactly one
        # short PPO update and far below a meaningful training run.
        args.num_envs = min(args.num_envs, 4_096)
        args.rollout_steps = 16
        args.total_timesteps = args.num_envs * args.rollout_steps
        args.minibatch_size = args.total_timesteps
        args.update_epochs = 1

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    task = TaskConfig(task=args.task)
    environment = TripleCartPoleGPU(
        num_envs=args.num_envs,
        task=task,
        device=args.device,
        seed=args.seed,
        capture_graph=not args.no_cuda_graph,
    )
    agent = ActorCritic(environment.observation_size, args.hidden_size, args.depth).to(environment.device)
    if not args.no_compile and importlib.util.find_spec("triton") is not None:
        agent.compile(mode="max-autotune")
    elif not args.no_compile:
        print("torch.compile skipped: this Windows PyTorch install has no Triton backend")
    config = PPOConfig(
        rollout_steps=args.rollout_steps,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        use_bfloat16=not args.no_bfloat16,
        capture_policy_graph=not args.no_policy_graph,
    )
    print(f"device={environment.device} gpu={torch.cuda.get_device_name(environment.device) if environment.device.type == 'cuda' else 'CPU'}")
    print(f"envs={args.num_envs:,} rollout={args.rollout_steps} batch={args.num_envs * args.rollout_steps:,} task={args.task}")
    trainer = PPOTrainer(environment, agent, config)
    trainer.train(checkpoint_dir=None if args.smoke_test else args.checkpoint_dir)
    if args.smoke_test:
        print("Smoke test passed; no checkpoint was written.")


if __name__ == "__main__":
    main()
