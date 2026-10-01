# 02 - GPU Triple CartPole

## Setup

Python 3.11 or 3.12 and an NVIDIA GPU are recommended. From this folder:

```powershell
python -m venv .venv --system-site-packages
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`--system-site-packages` reuses the CUDA-enabled PyTorch already installed on this
machine. If that PyTorch build is not available, install a CUDA build of PyTorch
for the local driver before installing the remaining requirements.

## Train

Do a short integration test without saving a model:

```powershell
python .\src\train.py --smoke-test
```

Start the default GPU PPO experiment (only after choosing the task and training
budget):

```powershell
python .\src\train.py --task swingup
```

The default task starts the three links hanging downward and asks the policy to
swing them up while keeping the cart near the center. Use `--task balance` for
the easier upright stabilization task. Run `python .\src\train.py --help` for
the environment count, rollout length, network, and checkpoint options.

## Evaluate

Open the renderer with a random policy to inspect the environment:

```powershell
python .\src\evaluate.py --render --task swingup
```

Evaluate a checkpoint visually:

```powershell
python .\src\evaluate.py --render --task swingup --model .\models\latest.pt
```

Benchmark only the simulator:

```powershell
python .\src\benchmark.py --num-envs 65536 --steps 1000
```

## Motivation

## Experiment diary

## Results

## Failed attempts

## Next steps

## References

- [NVIDIA Warp documentation](https://nvidia.github.io/warp/)
- [Warp and PyTorch interoperability](https://nvidia.github.io/warp/latest/user_guide/interoperability/pytorch.html)
- [PufferLib documentation](https://puffer.ai/docs.html)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)

## Codex Notes

- Added a custom, batched triple-pendulum dynamics kernel. One CUDA thread owns
  one complete environment and performs all physics substeps, reward calculation,
  termination, automatic reset, and observation construction without CPU copies.
- Added a GPU-resident PPO path, a deterministic Pygame viewer, simulator
  benchmarks, and physics/integration tests. No full training run was started.

