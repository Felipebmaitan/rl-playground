# 01 - PPO Pendulum

i want to play a little bit with recurrent networks. this might not want to be the best testing scenario because the information I'm hiding from the model here (the angular velocity) can be calculated trivially given the angular position metric. anyways it'll probably be cool to test it out.

it seemed kind of weird to me that taking time into consideration is not the standard approach when training control policies (specially for robotics). i mean if the system if fully observable it makes sense, to just use non recurrent networks, but if it is not (which I'd bet to be most robotics systems, since basically all variables are unknown and dependent on the env). i may figure out why it is soon lol (or figure out it is de defaut and i'm just a little slow on the head) 

MAN THIS IS SOOOOOOOOOOOOOOOO SLOW ON MY LAPTOP OMGGGGGGGGGGG

308 it/s for mlp-lstm??? this is going to take forever jesus

## Setup

From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## References

- [Gymnasium Pendulum documentation](https://gymnasium.farama.org/environments/classic_control/pendulum/)
- [Proximal Policy Optimization Algorithms (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation (Schulman et al., 2016)](https://arxiv.org/abs/1506.02438)

## Experiment workflow

The current purpose of this folder is to compare a memoryless policy with a recurrent one when part of the state is hidden. Pendulum normally returns `[cos(theta), sin(theta), angular_velocity]`. `NoVelocityObservation` removes the last value, leaving the agent to infer movement from the sequence of observed angles.

| Configuration | Training command | Observation | Model output |
| --- | --- | --- | --- |
| Full-state MLP baseline | `python .\src\train.py --mlp` | angle and angular velocity | `models/ppo_pendulum_mlp_full.zip` |
| Partial-observation MLP | `python .\src\train.py --mlp --no-velocity` | angle only | `models/ppo_pendulum_mlp_no_velocity.zip` |
| Partial-observation MLP-LSTM | `python .\src\train.py --mlp-lstm --no-velocity` | angle only, with memory across steps | `models/ppo_pendulum_mlp_lstm_no_velocity.zip` |

`--mlp` is the default, so `python .\src\train.py` also runs the full-state MLP baseline. The LSTM configuration is intentionally available only with `--no-velocity`; that keeps the comparison focused on whether recurrence can recover the missing information.

### Hyperparameters and shorter runs

Each configuration has its own named settings in `src/experiments.py`. Change the matching entry there to tune its learning rate, rollout length, minibatch size, epochs, or default training budget without changing the other experiments.

For a shorter run, override the shared run settings at the command line:

```powershell
python .\src\train.py --mlp --no-velocity --total-timesteps 50000 --n-envs 4
python .\src\train.py --mlp-lstm --no-velocity --total-timesteps 50000 --n-envs 4
```

TensorBoard logs are grouped under `runs/` by configuration:

```powershell
tensorboard --logdir runs
```

### Evaluation

Use exactly the same policy and observation flags that were used for training:

```powershell
python .\src\evaluate.py --mlp
python .\src\evaluate.py --mlp --no-velocity
python .\src\evaluate.py --mlp-lstm --no-velocity
```

The evaluator uses deterministic actions. For a fair comparison, train each configuration with several seeds and compare the mean episode return across the same evaluation seeds. Pendulum rewards are negative costs, so values closer to zero are better. The LSTM evaluator keeps its hidden state across steps and resets it when an episode ends.
