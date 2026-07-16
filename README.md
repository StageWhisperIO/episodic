# Episodic

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/StageWhisperIO/episodic)

Episodic records your coding-agent sessions and turns each one into a structured,
outcome-labeled *episode*. From those episodes it produces session summaries and PR notes,
training datasets (SFT, DPO, reward, RLDS), replayable tasks, a reward model, and offline-RL
transition batches. It doesn't train models itself; you point it at a trainer (TRL by
default, or any other backend).

It runs as a Claude Code plugin, and there's a Codex integration too. Capture happens in the
background through hooks, and each session is normalized into one `CodingEpisode` object. You
get something useful right away, before any ML is involved: what changed, why, which tests
ran, which edits look risky, and a suggested PR title and description. The same episodes
double as training data, labeled by what actually happened — tests passing, a PR merging, a
review, a revert, or a one-click rating.

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

The base install uses only the standard library, so the hooks stay fast and there's nothing
heavy to pull in. Optional extras:

```bash
pip install -e ".[datasets]"   # pyarrow + datasets  → real Parquet export
pip install -e ".[rl]"         # numpy               → reward-model fit
pip install -e ".[trl]"        # torch + trl         → SFT / DPO fine-tuning
pip install -e ".[unsloth]"    # unsloth (NVIDIA/AMD/Intel GPU, Linux/Windows) → fast 4-bit LoRA
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

Datasets are JSONL, so training is just another step in the pipe, and you can swap the
backend. The options: TRL (the default, runs on a MacBook via MPS and includes `trl-sao`, a
local single-rollout RL trainer); Tinker (`tinker-sft` / `tinker-sao`, opt-in LoRA runs on
Thinking Machines' GPUs from any machine); Unsloth (`unsloth-sft` / `unsloth-dpo`, fast
4-bit LoRA on an NVIDIA/AMD/Intel GPU under Linux or Windows, though not Apple Silicon yet);
the `command` trainer, which shells out to any executable; or your own, registered through an
`episodic.trainers` entry point.

Both SAO trainers implement Single-rollout Asynchronous Optimization (arXiv:2607.07508): one
rollout per prompt, a DIS token-level trust region, and a running-mean reward baseline by
default. Setting `critic_model` in the train config upgrades the baseline to a local value
model (small HF model + value head, frozen attention, 2 critic updates per policy step) that
can be pretrained from `export-episode --format reward` via `critic_pretrain`. That lets you
run the policy on Tinker and the critic on a MacBook.

```bash
episodic train --list                                          # show backends
episodic export-episode --all --format sft --out - | \
  episodic train --trainer trl-sft --model HuggingFaceTB/SmolLM2-135M-Instruct --out runs/sft
episodic train runs/dpo.jsonl --trainer command \
  --config '{"command": "my-trainer --data {dataset} --out {out}"}'
```

Each run writes `manifest.json` (dataset sha256, episode ids, base commit, config, metrics)
so you can trace what produced what. If `[trl]` isn't installed, `episodic train` prints the
path to the ready dataset instead of failing. Episodic handles the data; your trainer handles
the training.

> **What Episodic doesn't do:** the gradient updates. It produces datasets, a (linear) reward
> model, and RL transition batches; the actual LLM training step belongs to the trainer plugin.

### 5. Keep the outcome label current

An outcome isn't settled when you stop typing. CI finishes later, the PR merges later, and
sometimes a change turns out to cause a bug weeks down the line. Two commands keep the reward
label in sync with what actually happened:

```bash
episodic link --refresh-all                 # re-pull every in-flight PR (run from cron / a watch loop)
episodic regression <bugfix-or-revert-sha>  # git-blame the fix back to the episodes that caused it
episodic regression <sha> --apply           # mark them caused_regression and recompute reward
```

`regression` parses the fix's diff, blames the pre-image lines, and maps culprit commits to
episodes (exact commit match, squash-merge aware; `--fuzzy` also penalizes file-overlap). A
confirmed regression scores like a revert and is excluded from SFT / treated as `rejected` in DPO.

### 6. Close the loop — `episodic loop`

This is where producing datasets becomes improving a model: quality-filter, train, replay-eval
on held-out tasks, compare the reward against the base model, and promote if it wins. It reuses
the same pieces as everything else — the exporters, the trainer registry, and the replay harness.

```bash
episodic loop --config loop.json                 # plan only (no code is executed)
episodic loop --config loop.json --execute       # run replay-eval and decide promotion
```

```jsonc
// loop.json
{
  "trainer": "trl-grpo",        // any registered backend; grpo is reward-model-driven RL
  "format": "sft",
  "train_config": {"model": "Qwen/Qwen2.5-0.5B-Instruct", "reward_model": "runs/rm"},
  "base_model": "runs/base",
  "holdout_frac": 0.2, "seed": 0,
  "replay_cmd": "my-agent --model {model} --task {prompt_file} --repo {workspace}",
  "promote_margin": 0.02, "eval_concurrency": 4, "execute": true
}
```

The RL chain is three trainers you can swap out: `trl-sft` (warmup), then `trl-reward` (a reward
model from preference pairs), then `trl-grpo` (policy RL driven by that reward model). The loop
decides whether to promote based on real replay-eval reward — it runs the model on held-out tasks
and scores the tests plus diff overlap — rather than a proxy.

> **Security:** without `--execute` / `"execute": true`, the loop only filters episodes and writes the
> dataset file — it never invokes the trainer, clones a repo, or runs a test command; it just prints a
> plan (trainer, dataset path/row count, config, replay-eval plan). `--execute` is what actually trains
> and runs replay-eval. Untrusted episode-derived test commands run without a shell; only your own
> `replay_cmd` template runs via a shell.
>
> **Running the tuned model back in the agent** is the open-model lane: Codex `--oss` or a
> custom agent pointed at the tuned model dir, not hosted Claude Code.

### 7. Verify & explore — testing tools + tutorials

A self-contained testing layer lets you check the install, generate realistic episodes
without running a coding agent, and measure how well a model predicts environment
observations. It all runs offline, with no GPU.

```bash
episodic doctor                              # end-to-end self-check (synthetic store, no network)
episodic worldbench --predictor prefix --turing   # next-observation fidelity + double-blind judge
pip install -e ".[tutorials]" && jupyter lab notebooks/   # five runnable tutorials
```

- **`episodic doctor`** runs every subsystem against a throwaway store and prints one verdict.
- **`episodic.testing`** is a deterministic, schema-valid episode factory (`make_episode`,
  `make_population`, `populate_store`) — the basis for tests and notebooks.
- **`episodic.fidelity` + `episodic worldbench`** implement the AgentWorld content-type-aware
  observation rubric, OOD source splits, and a Turing-test judge.
- **[notebooks/](notebooks/)** — five tutorials, generated reproducibly from
  `notebooks/build.py` and run headless in CI.

Full guide: [`docs/TESTING.md`](docs/TESTING.md) · tutorial index: [`notebooks/README.md`](notebooks/README.md).

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
 (no ML)       → outcome label    rlds/wm/parquet/      re-run model)     reward→rl→eval)
                                  jsonl)
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
| `episodic export-episode --format <fmt> [--all]` | `sft` · `dpo` · `reward` · `rlds` · `wm` · `parquet` · `jsonl` |
| `episodic link [--pr URL \| --auto]` | attach PR / CI / merge / review outcome (uses `gh`) |
| `episodic replay-task create \| run --replay ID --model M [--execute]` | snapshot / re-run a task (`run` only plans unless `--execute` clones + runs) |
| `episodic worldbench [--predictor P] [--source-holdout] [--turing]` | benchmark next-observation prediction (world-model fidelity) |
| `episodic doctor [--json]` | end-to-end self-check on the install (synthetic store, no network) |
| `episodic list` / `show ID [--validate]` | browse episodes |
| `episodic dashboard [--port N]` | local web UI: browse + one-click labels |
| `episodic train [dataset] --trainer T [--config]` | train on an exported dataset via a pluggable backend |
| `episodic loop [--config] [--execute]` | filter → train → replay-eval → promote (plan-only unless `--execute`) |
| `episodic schema dump` | regenerate `schemas/episode.schema.json` |

## Dataset formats

`export-episode` writes JSONL (and Parquet when `pyarrow` is installed) into `.episodic/exports/`:

- **SFT** — `intent → good trajectory` (only episodes that passed the quality bar)
- **DPO** — `chosen > rejected` preference pairs grouped by intent
- **Reward** — `trajectory → reward_vector` (+ scalar composite)
- **RLDS** — per-episode `observation / action / reward / is_terminal / discount` steps
- **WM** — language-world-model samples: `history + action → next observation` as SFT messages
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
| world model | `src/episodic/worldmodel/`, `fidelity/`, `worldbench/` (AgentWorld next-observation prediction) |
| testing tools | `src/episodic/testing/` (episode factory), `selfcheck/` (`episodic doctor`) |
| tutorials | [`notebooks/`](notebooks/) + [`docs/TESTING.md`](docs/TESTING.md) |

## Privacy

The store lives in `./.episodic/` (per repo, git-ignored) or `$EPISODIC_HOME`. Capture is
local-first. `link` talks to GitHub only when you ask; exporters write files only where you point them.

## Development

```bash
PYTHONPATH=src python -m pytest tests -q     # 285 tests
python plugin-codex/test_codex.py            # codex mapping
python rl-pipeline/test_pipeline.py          # pipeline stages
episodic doctor                              # end-to-end install self-check
python notebooks/build.py                    # regenerate the tutorial notebooks
```

## Status

Phases 1 through 7 are implemented end to end. The summaries and the reward vector are
deliberately ML-free, so the tool is useful on day one; the exporters and RL pipeline produce
real, schema-validated data that a training backend can use as-is.

MIT licensed.
