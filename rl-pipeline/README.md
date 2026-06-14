# rl-pipeline

Offline RL / dataset pipeline for Episodic. Transforms stored coding episodes into training artefacts for SFT, preference learning, and offline RL.

## Pipeline

```
Episodes → quality filter → SFT warmup dataset → preference pairs → reward model → offline RL batches → replay evaluation
```

Each stage is a pure function in `stages.py`. `pipeline.py` wires them into a CLI.

## Commands

```bash
# Run every stage in sequence
python3 pipeline.py all --out /tmp/rlout

# Individual stages
python3 pipeline.py quality-filter --min-composite 0.5 --out /tmp/rlout
python3 pipeline.py sft            --out /tmp/rlout
python3 pipeline.py pref-pairs     --out /tmp/rlout
python3 pipeline.py reward-model   --out /tmp/rlout
python3 pipeline.py rl-batches     --out /tmp/rlout
python3 pipeline.py eval           --out /tmp/rlout
```

Common flags:
- `--home PATH` — override the Episodic store directory (default: auto-detected)
- `--out PATH` — output directory (default: `rl-pipeline/out/`)
- `--min-composite FLOAT` — quality filter threshold (default: 0.5)

## Output files

| File | Contents |
|---|---|
| `quality.jsonl` | Episodes that passed the quality filter |
| `sft.jsonl` | SFT warmup rows: `{messages, meta}` |
| `pref_pairs.jsonl` | Preference pairs: `{prompt, chosen, rejected, meta}` |
| `reward_model.json` | Linear reward model: `{weights, features, r2}` |
| `rl_batches.jsonl` | Offline RL transitions (see below) |
| `eval.json` | Reward model evaluation: `{n, pearson, mean_abs_err}` |

## Offline RL tuple

Each row in `rl_batches.jsonl` represents one environment transition:

```json
{
  "state":       {"intent": "...", "step_index": 0, "prev_observation": ""},
  "action":      {"type": "tool_call", "tool": "edit_file", "input": {...}},
  "observation": "file saved",
  "reward":      0.0,
  "terminal":    false,
  "discount":    1.0
}
```

The final transition in each episode has `terminal: true` and `reward` equal to the episode's composite reward scalar. All intermediate transitions carry `reward: 0.0` (reward-at-terminal convention). `discount` is always `1.0`.

This maps directly to the TRL `RewardTrainer` input and d3rlpy dataset format. Load with:

```python
# TRL / datasets
from datasets import load_dataset
ds = load_dataset("json", data_files="rl_batches.jsonl")

# d3rlpy
import json, d3rlpy
transitions = [json.loads(l) for l in open("rl_batches.jsonl")]
```

## Reward model

A lightweight linear model fitted via least-squares on 7 features:

| Feature | Description |
|---|---|
| `test_pass` | Fraction of tests passing |
| `edit_focus` | edits / (edits + reads + shells) |
| `file_edits_norm` | min(file_edits / 10, 1) |
| `tests_run_norm` | min(tests_run / 5, 1) |
| `has_feedback` | 1 if any human feedback present |
| `file_reads_norm` | min(file_reads / 20, 1) |
| `bias` | constant 1.0 |

`numpy` is used when available (`np.linalg.lstsq`); falls back to pure-Python Gaussian elimination otherwise. The model never crashes on `ImportError`.

## Dependencies

`numpy>=1.24` is the only runtime dependency and is optional — all stages have pure-Python fallbacks.

Optional training backends (not required to run the pipeline):
- `trl` / `datasets` — preference and SFT training
- `d3rlpy` — offline RL training
- `torch` — required by the above
