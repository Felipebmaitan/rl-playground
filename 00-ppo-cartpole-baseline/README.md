# 00 - PPO CartPole Baseline

This was honestly a little bit boring. The code was all ready, so I did not need to do much in here. I mostly used it to better understand the hyperparameters we set when training policies and how we choose them. Some of them were kind of familiar, like number of epochs and learning rate from past experiences training yolo, but the parameters specific to RL are new.

rollout != epoch. 

rollout is the data collection, epoch is the training pass.

I asked Codex to add disturbance to the code since there was none in the reference code from Gymnasium. Turns out the model is very decent considering the extremely low amount of data we trained it on.

## Setup

From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train

```powershell
python .\src\train.py
```


## Evaluate

Run evaluation without rendering:

```powershell
python .\src\evaluate.py
```

Run evaluation with rendering:

```powershell
python .\src\evaluate.py --render
```

## Codex Notes