# Episodic

Turn coding-agent sessions into structured, outcome-labeled **episodes** — instant
session summaries and PR notes, SFT / DPO / reward / RLDS datasets, replayable tasks, a
reward model, and offline-RL transition batches. You bring the trainer (TRL is the default,
any backend plugs in); Episodic produces everything it eats.

Episodic runs as a **Claude Code plugin** (and a Codex integration). It captures your
session invisibly through hooks, normalizes it into a single `CodingEpisode` object, and
gives you immediate value — *what changed, why, tests run, risky edits, a suggested PR title
and description* — before any ML exists. Underneath, every session quietly becomes training
data labeled by real outcomes (tests, PR merge, review, revert, one-click feedback).

> Stored locally by default. Nothing leaves your machine unless you link a PR or run an exporter.

---

## Why it exists

| Phase | Goal | What you get |
| ----- | ---- | ------------ |
| 1 | Plugin entry point | `/trace` commands inside Claude Code / Codex |
| 2 | Invisible session capture | initial prompt, repo state, tool calls, edits, commands, tests, approvals, final diff |
| 3 | Episode model | every session becomes one `CodingEpisode` (the central abstraction) |
| 4 | Outcome linking | episode ↔ PR / CI / review / merge / revert |
| 5 | Dataset export | SFT · DPO · Reward · RLDS · Parquet · JSONL |
| 6 | Replay harness | re-run a task at its base commit with another model |
| 7 | Offline RL | quality filter → SFT → preference pairs → reward model → RL batches → replay eval |

---

## Quickstart

### 1. Install the CLI

```bash
pip install -e .          # provides the `episodic` command
# or run without installing:  PYTHONPATH=src python -m episodic.cli --help
```

The base install is **stdlib-only** (fast hooks, no heavy deps). Optional extras:

```bash
pip install -e ".[datasets]"   # pyarrow + datasets  → real Parquet export
pip install -e ".[rl]"         # numpy               → reward-model fit
pip install -e ".[trl]"        # torch + trl         → SFT / DPO fine-tuning
pip install -e ".[unsloth]"    # unsloth (CUDA GPU)  → fast 4-bit LoRA SFT / DPO
```

### 2. Add the Claude Code plugin

This repo is a plugin marketplace. From Claude Code:

```
/plugin marketplace add /path/to/episodic
/plugin install episodic
```

That wires the capture hooks (`SessionStart`, `UserPromptSubmit`, `PreToolUse`,
`PostToolUse`, `Stop`, `SessionEnd` → `episodic ingest`) and the `/trace` command.
The `episodic` CLI must be on your `PATH`.

### 3. Use it

```bash
/trace start Add retry logic to the http client   # name the session
# ... work normally; capture is automatic ...
/trace summary                                     # what changed, tests, risks, PR notes
/trace mark useful                                 # one-click feedback (becomes a reward signal)
/trace create-pr-notes                             # ready-to-paste PR title + description
/trace export-episode --format dpo                 # export training data
/trace replay-task create                          # snapshot a replayable task
```

Everything is also available directly: `episodic summary`, `episodic list`,
`episodic link --auto`, `episodic dashboard`, …

### 4. Train (pluggable)

Datasets are JSONL, so training is just another filter on the pipe. Backends are
interchangeable — **TRL** (default, runs on a MacBook via MPS), **Unsloth** (`unsloth-sft` /
`unsloth-dpo`, fast 4-bit LoRA on a CUDA GPU), the `command` trainer that shells out to any
executable, or your own via an `episodic.trainers` entry point.

```bash
episodic train --list                                          # show backends
episodic export-episode --all --format sft --out - | \
  episodic train --trainer trl-sft --model HuggingFaceTB/SmolLM2-135M-Instruct --out runs/sft
episodic train runs/dpo.jsonl --trainer command \
  --config '{"command": "my-trainer --data {dataset} --out {out}"}'
```

Each run writes `manifest.json` (dataset sha256, episode ids, base commit, config, metrics)
for provenance. Without `[trl]` installed, `episodic train` prints the ready dataset path
instead of failing — separation of mechanism (Episodic) from policy (your trainer).

> **What Episodic does not do:** run the gradient updates for you beyond invoking a backend.
> It produces datasets, a (linear) reward model, and RL transition batches; the LLM training
> step is owned by the trainer plugin.

### 5. Keep the outcome signal honest

Outcomes evolve after you stop typing — CI finishes, the PR merges, and sometimes a change
causes a bug weeks later. Two commands keep the reward label tracking reality:

```bash
episodic link --refresh-all                 # re-pull every in-flight PR (run from cron / a watch loop)
episodic regression <bugfix-or-revert-sha>  # git-blame the fix back to the episodes that caused it
episodic regression <sha> --apply           # mark them caused_regression and recompute reward
```

`regression` parses the fix's diff, blames the pre-image lines, and maps culprit commits to
episodes (exact commit match; `--fuzzy` also penalizes file-overlap). A confirmed regression
scores like a revert and is excluded from SFT / treated as `rejected` in DPO.

---

## Architecture

```
Claude Code / Codex  ──hooks──►  episodic ingest
                                      │            (optional) OTel/OTLP token+cost  ──►  collector/otel
                                      ▼
                          episode store  (.episodic/: append-only events.jsonl per session)
                                      │  normalize
                                      ▼
                              CodingEpisode  ── the central object, validated by schemas/episode.schema.json
                                      │
   ┌──────────────┬──────────────────┼───────────────────┬─────────────────┐
   ▼              ▼                   ▼                   ▼                 ▼
 summaries     github-linker     dataset-exporters    replay-runner     rl-pipeline
 + PR notes    (PR/CI/merge)     (sft/dpo/reward/      (snapshot +       (filter→sft→pref→
 (no ML)       → outcome label    rlds/parquet/jsonl)   re-run model)     reward→rl→eval)
                                                                  dashboard (browse + label)
```

## The CodingEpisode

The normalized unit of everything downstream (full JSON Schema in
[`schemas/episode.schema.json`](schemas/episode.schema.json)):

```ts
type CodingEpisode = {
  intent: string
  repo_state: { root, repo, remote_url, branch, base_commit, dirty }
  steps: AgentStep[]        // ordered tool calls / edits / commands / prompts
  diffs: Patch[]            // final unified diff, per file, +/- counts
  commands: CommandRun[]
  tests: TestRun[]          // framework, passed/failed, ok
  human_feedback: Feedback[]
  outcome: Outcome          // open | accepted | merged | failed | reverted | abandoned
  reward_vector: RewardVector  // test_pass, human_label, outcome, cost_efficiency, edit_focus, composite
  stats: EpisodeStats          // tool calls, edits, tokens, cost, duration
  labels: string[]
}
```

## CLI reference

| Command | Does |
| ------- | ---- |
| `episodic start <intent>` | name the active session |
| `episodic summary [--episode ID] [--json]` | what changed, why, tests, missing tests, risky edits, PR notes, follow-ups |
| `episodic mark <label>` | feedback: `useful`, `wrong`, `too_broad`, `too_slow`, `needed_human_rescue`, `accepted_as_is`, `accepted_after_edits` |
| `episodic create-pr-notes` | suggested PR title + description |
| `episodic export-episode --format <fmt> [--all]` | `sft` · `dpo` · `reward` · `rlds` · `parquet` · `jsonl` |
| `episodic link [--pr URL \| --auto]` | attach PR / CI / merge / review outcome (uses `gh`) |
| `episodic replay-task create \| run --replay ID --model M` | snapshot / re-run a task |
| `episodic list` / `show ID [--validate]` | browse episodes |
| `episodic dashboard [--port N]` | local web UI: browse + one-click labels |
| `episodic schema dump` | regenerate `schemas/episode.schema.json` |

## Dataset formats

`export-episode` writes JSONL (and Parquet when `pyarrow` is installed) into `.episodic/exports/`:

- **SFT** — `intent → good trajectory` (only episodes that passed the quality bar)
- **DPO** — `chosen > rejected` preference pairs grouped by intent
- **Reward** — `trajectory → reward_vector` (+ scalar composite)
- **RLDS** — per-episode `observation / action / reward / is_terminal / discount` steps
- **Parquet** — flattened analytics rows (falls back to JSONL without `pyarrow`)
- **JSONL** — full episodes, one per line

## Offline RL pipeline

```bash
python rl-pipeline/pipeline.py all          # runs the whole chain on your episode store
```

`Episodes → quality filter → SFT warmup → preference pairs → reward model → RL batches → replay eval`.
Pure-Python by default; `numpy` optional; batches are TRL / d3rlpy-ready. See
[`rl-pipeline/README.md`](rl-pipeline/README.md).

---

## Repo layout

Maps the conceptual components (initial prompt §11) to the Python package:

| Component | Location |
| --------- | -------- |
| schemas | [`schemas/`](schemas/) (canonical JSON Schema) + `src/episodic/schema.py` (source of truth) |
| episode-store | [`episode-store/`](episode-store/) → `src/episodic/store.py`, `paths.py` |
| collector | [`collector/`](collector/) → `src/episodic/collector/` (hook + OTel adapter) |
| core normalize / summary / reward | `src/episodic/core/` |
| plugin-claude-code | [`plugin-claude-code/`](plugin-claude-code/) (manifest, `/trace`, hooks) |
| plugin-codex | [`plugin-codex/`](plugin-codex/) (notify + rollout import) |
| github-linker | [`github-linker/`](github-linker/) → `src/episodic/github/` |
| dataset-exporters | [`dataset-exporters/`](dataset-exporters/) → `src/episodic/exporters/` |
| replay-runner | [`replay-runner/`](replay-runner/) → `src/episodic/replay/` |
| dashboard | [`dashboard/`](dashboard/) → `src/episodic/dashboard/` |
| rl-pipeline | [`rl-pipeline/`](rl-pipeline/) |

## Privacy

The store lives in `./.episodic/` (per repo, git-ignored) or `$EPISODIC_HOME`. Capture is
local-first. `link` talks to GitHub only when you ask; exporters write files only where you point them.

## Development

```bash
PYTHONPATH=src python -m pytest tests -q     # 27 tests
python plugin-codex/test_codex.py            # codex mapping
python rl-pipeline/test_pipeline.py          # pipeline stages
```

## Status

Phases 1–7 are implemented end-to-end as a working foundation. Heuristic summaries and the
reward vector are intentionally ML-free so the tool is useful on day one; the exporters and
RL pipeline produce real, schema-validated data ready for training backends.

MIT licensed.
