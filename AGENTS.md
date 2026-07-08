# AGENTS.md

This repository is an RL playground: a collection of small, self-contained reinforcement learning projects used for study, experimentation, and implementation practice.

The goal is not to make one polished RL library. The goal is to keep many focused experiments organized enough that they can be revisited, compared, rewritten, and learned from.

## Repository Shape

- Each top-level experiment should live in its own subfolder.
- Each experiment folder should be treated like a small project with its own `README.md`.
- The root `README.md` explains the overall playground, motivation, and project index.
- Experiment READMEs should explain the journey: what was tried, what worked, what failed, what papers or references were used, and what changed between runs.

Example structure:

```text
RL-Playground/
  AGENTS.md
  README.md
  ppo-double-pendulum/
    README.md
    src/
    configs/
    runs/
  ppo-lunar-lander/
    README.md
    src/
    configs/
    runs/
```

## Project Philosophy

- Prefer clarity.
- Prefer readable implementations.
- Keep experiments reproducible where practical: seed values, environment versions, hyperparameters, and commands matter.
- Document failed attempts with the user.
- Make it easy to compare experiments without forcing every project into the same shape.
- When an implementation follows a paper, tutorial, or reference repo, cite it in the project README.
- When Codex or another AI tool contributes substantially, mention that openly where useful.

## Experiment Folder Guidelines

Each experiment folder should ideally include:

- `README.md`: diary, motivation, references, results, notes, and next steps.
- Source code in a predictable folder such as `src/`.
- Configuration files for hyperparameters and environment settings, when useful.
- A clear training command and evaluation command.
- Notes on dependencies and environment setup.
- Saved artifacts only when they are small and meaningful.

Avoid committing large generated outputs unless they are intentionally part of the experiment record. Prefer ignoring bulky logs, checkpoints, videos, and run directories by default.

## README Style

Experiment READMEs are allowed to be informal and personal. They should capture the actual learning process, not just the final polished result. READMEs will be mostly personal and written by the user so don't touch it unless you're clearly prompted to do so or if it the intention is to add sources (references/citations) like papers, blogs used, etc.

## Coding Preferences

- Keep code local to each experiment unless there is a strong reason to share it.
- Use simple names and explicit control flow, especially for core RL logic.
- Add comments for non-obvious math, algorithm steps, tensor shapes, or environment quirks.
- Keep notebooks optional. If notebooks are used, prefer also having script equivalents for repeatable runs.

## Dependency Preferences

This repo may use different tools across experiments, including but not limited to:

- Python
- PyTorch
- Gymnasium
- Stable-Baselines3
- CleanRL-style single-file baselines
- Jupyter notebooks
- Weights & Biases, TensorBoard, or local logging

Do not assume every subproject uses the same stack. Check the local experiment folder first.

## Working With Codex

Codex is allowed to help with structure, implementation, debugging, refactors, documentation, and experiment setup.

When making changes:

- Read the relevant experiment README before editing that experiment.
- Preserve the diary-like nature of the notes, if allowed to do so by the user.
- Do not erase failed attempts or learning notes unless explicitly asked.
- Keep edits scoped to the requested experiment or root documentation.
- Prefer adding clear next steps over pretending an experiment is finished.

Codex is allowed to write ONLY (unless prompted specifically by the user otherwise) to write in the end of the README files, explaining what it has done in the repository as requested by the user. Use the readme of the first project "00-ppo-cartpole-baseline" as the base of what the general structure should be.