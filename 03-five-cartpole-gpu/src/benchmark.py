from __future__ import annotations

import argparse
import time

import torch

from five_cartpole import TaskConfig, FiveCartPoleGPU


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the fused GPU simulator")
    parser.add_argument("--num-envs", type=int, default=65_536)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--task", choices=("balance", "swingup"), default="swingup")
    parser.add_argument("--no-cuda-graph", action="store_true")
    args = parser.parse_args()

    environment = FiveCartPoleGPU(
        args.num_envs,
        task=TaskConfig(task=args.task),
        capture_graph=not args.no_cuda_graph,
    )
    action = torch.zeros((args.num_envs, 1), device=environment.device)
    for _ in range(args.warmup):
        environment.step(action)
    environment.synchronize()
    start = time.perf_counter()
    for _ in range(args.steps):
        environment.step(action)
    environment.synchronize()
    elapsed = time.perf_counter() - start
    control_steps = args.num_envs * args.steps
    physics_steps = control_steps * environment.physics.substeps
    print(f"GPU: {torch.cuda.get_device_name(environment.device)}")
    print(f"environments: {args.num_envs:,}; control calls: {args.steps:,}; elapsed: {elapsed:.3f}s")
    print(f"control steps/s: {control_steps / elapsed:,.0f}")
    print(f"physics substeps/s: {physics_steps / elapsed:,.0f}")


if __name__ == "__main__":
    main()

