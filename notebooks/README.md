# Tutorials

Five runnable notebooks that take you from "capture a session" to "benchmark a language
world model" — all offline, on synthetic data, no coding agent or GPU required. Every cell
executes end-to-end against the real `episodic` package (no stubs).

| # | Notebook | You learn |
| - | -------- | --------- |
| 01 | [Quickstart](01_quickstart.ipynb) | capture → `CodingEpisode` → summary → PR notes |
| 02 | [Datasets](02_datasets.ipynb) | export SFT · DPO · Reward · RLDS · WM and read the rows |
| 03 | [Reward & fidelity](03_reward_and_fidelity.ipynb) | the reward vector and content-type-aware observation scoring |
| 04 | [Replay & loop](04_replay_and_loop.ipynb) | snapshot a task, replay it on a real local git repo, run the RL loop |
| 05 | [World models (AgentWorld)](05_world_model_agentworld.ipynb) | turn expansion, OOD splits, `worldbench`, the Turing test |

## Setup

```bash
pip install -e ".[tutorials]"      # jupyter, nbformat, nbconvert, matplotlib, pandas
```

## Run them

In Jupyter:

```bash
jupyter lab notebooks/            # or: jupyter notebook
```

Headless (also how they are verified in CI):

```bash
python -m nbconvert --to notebook --execute --inplace notebooks/05_world_model_agentworld.ipynb
```

## Regenerate

The notebooks are generated from a single script so they stay reproducible and reviewable as
plain Python:

```bash
python notebooks/build.py          # rebuild all five
python notebooks/build.py 05       # rebuild one (filter by filename substring)
```

Edit `notebooks/build.py`, regenerate, then execute headless to confirm — the same
test → fix → re-test loop the rest of the project uses.
