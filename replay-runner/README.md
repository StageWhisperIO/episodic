# replay-runner

Replay past agent episodes against a fresh workspace to benchmark model performance.

## Usage

### Create a replay task from an episode

```
episodic replay-task create --episode <episode_id>
```

This snapshots the episode into a replay manifest under `.episodic/replays/<replay_id>/`:

- `manifest.json` — base commit, remote URL, repo, branch, test command, lockfile hashes, expected outcome, and scoring rules
- `prompt.txt` — the original intent/prompt sent to the agent
- `expected.diff` — concatenated unified diffs from the episode

### Run a replay

```
episodic replay-task run --replay <replay_id> --model qwen-coder
```

The runner:
1. Clones `remote_url` and checks out `base_commit` (or copies a local repo if remote is unavailable)
2. Executes `EPISODIC_REPLAY_CMD` with the new model
3. Runs `test_command` in the workspace
4. Computes scores against the expected outcome

## EPISODIC_REPLAY_CMD

Set this environment variable to the command template used to drive the agent:

```
export EPISODIC_REPLAY_CMD="my-agent --model {model} --prompt {prompt_file} --workspace {workspace}"
```

Template variables:

| Variable | Value |
|---|---|
| `{model}` | Model name passed to `--model` |
| `{prompt_file}` | Path to `prompt.txt` containing the original intent |
| `{workspace}` | Path to the cloned/copied workspace directory |

If `EPISODIC_REPLAY_CMD` is not set, `run_replay` operates in **dry-run mode**: it returns a result dict describing what would run without executing anything.

## What gets snapshotted

| Field | Description |
|---|---|
| `base_commit` | Git commit SHA the agent started from |
| `remote_url` | Git remote used to clone a fresh workspace |
| `branch` | Branch the episode was recorded on |
| `lockfiles` | SHA-256 hashes of dependency lock files present at record time |
| `test_command` | Detected or recorded test command (e.g. `pytest -q`) |
| `prompt.txt` | Verbatim intent/prompt from the episode |
| `expected.diff` | Unified diffs of all files the original agent changed |

## Scoring

Scores are computed after the replay run:

- **tests_pass** (weight 0.6): fraction of tests that passed in the workspace after the agent ran
- **diff_overlap** (weight 0.4): Jaccard similarity between files the replay agent changed and files the original agent changed

`total = 0.6 * tests_pass + 0.4 * diff_overlap`

Scoring is **dry-run-safe**: if no workspace is created (missing remote, no `EPISODIC_REPLAY_CMD`), `run_replay` returns `scores: null` and a `note` describing what would have run. It never raises an exception.
