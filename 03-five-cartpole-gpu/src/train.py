from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import random

import numpy as np
import torch

from five_cartpole import TaskConfig, FiveCartPoleGPU
from five_cartpole.policy import ActorCritic
from five_cartpole.ppo import PPOConfig, PPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on the GPU-native five-pole CartPole")
    parser.add_argument("--task", choices=("balance", "swingup"), default="swingup")
    parser.add_argument("--num-envs", type=int, default=32_768)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--total-timesteps", type=int, default=100_000_000)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=131_072)
    parser.add_argument("--update-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.003)
    parser.add_argument("--target-kl", type=float, default=0.012)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-bfloat16", action="store_true")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="experiment with torch.compile instead of the faster manual policy CUDA graph",
    )
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument("--no-policy-graph", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="one tiny rollout/update, no checkpoint")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--resume",
        type=Path,
        help="resume model and optimizer state; total-timesteps remains the cumulative target",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test and args.resume is not None:
        raise ValueError("--smoke-test and --resume cannot be used together")
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
    environment = FiveCartPoleGPU(
        num_envs=args.num_envs,
        task=task,
        device=args.device,
        seed=args.seed,
        capture_graph=not args.no_cuda_graph,
    )
    checkpoint = None
    initial_completed_steps = 0
    best_return = float("-inf")
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=environment.device, weights_only=False)
        saved_environment = checkpoint.get("environment", {})
        current_environment = environment.metadata()
        for section in ("physics", "task"):
            if saved_environment.get(section) != current_environment.get(section):
                raise ValueError(
                    f"resume checkpoint {section} does not match the current environment"
                )
        model_config = checkpoint.get("model_config", {})
        args.hidden_size = int(model_config.get("hidden_size", args.hidden_size))
        args.depth = int(model_config.get("depth", args.depth))
        saved_observation_size = int(
            model_config.get("observation_size", environment.observation_size)
        )
        if saved_observation_size != environment.observation_size:
            raise ValueError("resume checkpoint observation size does not match the environment")
        initial_completed_steps = int(checkpoint.get("completed_steps", 0))
        best_path = args.resume.with_name("best.pt")
        if best_path.exists():
            best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
            best_return = float(best_checkpoint.get("mean_return", float("-inf")))
        else:
            best_return = float(checkpoint.get("mean_return", float("-inf")))

    agent = ActorCritic(environment.observation_size, args.hidden_size, args.depth).to(environment.device)
    if checkpoint is not None:
        agent.load_state_dict(checkpoint["model"])
    if args.compile and importlib.util.find_spec("triton") is not None:
        agent.compile(mode="max-autotune")
        args.no_policy_graph = True
    elif args.compile:
        print("torch.compile skipped: this Windows PyTorch install has no Triton backend")
    config = PPOConfig(
        rollout_steps=args.rollout_steps,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        entropy_coefficient=args.entropy_coefficient,
        target_kl=args.target_kl,
        use_bfloat16=not args.no_bfloat16,
        capture_policy_graph=not args.no_policy_graph,
    )
    print(f"device={environment.device} gpu={torch.cuda.get_device_name(environment.device) if environment.device.type == 'cuda' else 'CPU'}")
    print(f"envs={args.num_envs:,} rollout={args.rollout_steps} batch={args.num_envs * args.rollout_steps:,} task={args.task}")
    trainer = PPOTrainer(environment, agent, config)
    if checkpoint is not None:
        trainer.optimizer.load_state_dict(checkpoint["optimizer"])
        for parameter_group in trainer.optimizer.param_groups:
            parameter_group["lr"] = args.learning_rate
        print(
            f"resumed={args.resume} completed_steps={initial_completed_steps:,} "
            f"best_return={best_return:.3f} target_steps={args.total_timesteps:,} "
            f"lr={args.learning_rate:g} epochs={args.update_epochs} "
            f"entropy={args.entropy_coefficient:g} target_kl={args.target_kl:g}"
        )
    trainer.train(
        checkpoint_dir=None if args.smoke_test else args.checkpoint_dir,
        initial_completed_steps=initial_completed_steps,
        best_return=best_return,
    )
    if args.smoke_test:
        print("Smoke test passed; no checkpoint was written.")


if __name__ == "__main__":
    main()
