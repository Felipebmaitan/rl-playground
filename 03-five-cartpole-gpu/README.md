# 03 - GPU Five-Pole CartPole

## Setup

Python 3.11 or 3.12 and an NVIDIA GPU are recommended. From this folder:

```powershell
python -m venv .venv --system-site-packages
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The environment uses NVIDIA Warp for the fused CUDA physics kernel and PyTorch
for the policy and PPO updates. `--system-site-packages` reuses a compatible
CUDA-enabled PyTorch installation when one is already available.

## Train

Validate the complete rollout and PPO update path without saving a model:

```powershell
python .\src\train.py --smoke-test
```

Start the five-link swing-up experiment:

```powershell
python .\src\train.py --task swingup
```

Use `--task balance` to start near the upright state instead. Run
`python .\src\train.py --help` for environment, policy, rollout, and training
budget options.

## Evaluate

Inspect the five-link physics with random actions:

```powershell
python .\src\evaluate.py --render --task swingup
```

Render a trained deterministic policy:

```powershell
python .\src\evaluate.py --render --task swingup --model .\models\best.pt
```

Benchmark the simulator without policy inference:

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
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)

## Codex Notes

- Reused the GPU-resident CartPole architecture with a five-link analytical
  dynamics model, a register-level 6x6 LDLT solver, equal-link straightness and
  stability rewards, and a five-link Pygame renderer.
- Optimized the end-to-end PPO path with a fused Warp GAE pass, GPU-side episode
  statistics, persistent zero-copy Warp/PyTorch views, fused CUDA Adam, and
  contiguous randomized minibatches. Benchmarks on the RTX 5080 favored 32,768
  environments, 32 rollout steps, and 131,072-sample minibatches. This preserves
  a useful temporal horizon while sustaining roughly 2.2 million training steps
  per second; these values are now the defaults.
- Tested `torch.compile` with Triton on Windows, but the warm compiled path was
  slower than the explicit policy CUDA graph for this small control network, so
  compilation remains an opt-in experiment via `--compile`.
- After the tapered-link baseline, changed the default experiment to five
  identical 0.55 m, 0.20 kg rods, a 50 N actuator, and a +/-5 m track. The
  original 18 N tapered 5B checkpoints remain under `models/five-billion`; the
  equal-link run uses a separate checkpoint directory so the two variants can
  be compared.
- Added cumulative-budget checkpoint resume support. For example,
  `python .\src\train.py --task swingup --total-timesteps 5000000000
  --checkpoint-dir .\models\equal-links-50n-5m-5b --resume
  .\models\equal-links-50n-5m-5b\latest.pt` restores both the policy and Adam
  state, preserves the previous best-return threshold, and finishes only the
  remaining rollout batches.
- Increased the equal-link experiment's boundary from +/-5 m to +/-10 m while
  retaining the 50 N actuator. The completed +/-5 m checkpoints remain under
  `models/equal-links-50n-5m-5b`; the wider-track run starts from scratch in
  `models/equal-links-50n-10m-5b` because position normalization, termination,
  and reward geometry all depend on the configured track limit.
- After the +/-10 m policy collapsed from a best return of 428.32 to single
  digits, restarted from `best.pt` with a 5e-5 learning rate, three PPO epochs,
  a 0.012 target-KL guard, and a modest entropy increase from 0.002 to 0.003.
  Resumed learning-rate annealing now starts from the requested new rate and
  spans only the remaining updates.
