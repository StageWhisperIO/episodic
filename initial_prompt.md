| Phase | Goal               | Main UX                                   |
| ----- | ------------------ | ----------------------------------------- |
| 1     | Plugin entry point | Works inside Claude Code / Codex          |
| 2     | Session capture    | User gets useful summaries immediately    |
| 3     | Episode model      | Every session becomes structured data     |
| 4     | Outcome linking    | Session connects to PR, CI, review, merge |
| 5     | Dataset export     | SFT / DPO / reward / RL formats           |
| 6     | Replay harness     | Re-run task with another model            |
| 7     | Offline RL         | Train from real coding trajectories       |

### 1. Build the plugin first

Create a Claude Code / Codex plugin or wrapper.

User-facing commands:

```bash
/trace start
/trace summary
/trace mark useful
/trace mark wrong
/trace create-pr-notes
/trace export-episode
/trace replay-task
```

The plugin should feel like a productivity tool, not telemetry software.

### 2. Capture session data invisibly

Capture:

```txt
initial prompt
repo + branch + base commit
tool calls
file reads
file edits
shell commands
test results
approval/denial events
final diff
token/cost/time stats
```

Store locally by default.

### 3. Use OTel as plumbing

Claude Code and Codex already expose OTel-style telemetry. Use OTel for ingestion, but hide it from the user.

Architecture:

```txt
Claude Code / Codex Plugin
        ↓
OTel Collector / Local Adapter
        ↓
Episode Store
        ↓
Dashboard + Exporters + Replay
```

### 4. Create the core object: Coding Episode

Normalize each session into:

```ts
type CodingEpisode = {
  intent: string
  repo_state: RepoState
  steps: AgentStep[]
  diffs: Patch[]
  commands: CommandRun[]
  tests: TestRun[]
  human_feedback: Feedback[]
  outcome: Outcome
  reward_vector: RewardVector
}
```

This is the central abstraction.

### 5. Add immediate user value

After each session, generate:

```txt
What changed
Why it changed
Files touched
Tests run
Tests missing
Risky edits
Suggested PR title
Suggested PR description
Follow-up TODOs
```

This makes the tool useful before any ML exists.

### 6. Link to GitHub / GitLab

Connect episode to:

```txt
commit
branch
PR
CI run
review comments
merge status
revert status
manual edits after agent
```

This turns coding telemetry into outcome-labeled data.

### 7. Add lightweight feedback

Use one-click labels:

```txt
Useful
Wrong
Too broad
Too slow
Needed human rescue
Accepted as-is
Accepted after edits
```

This creates preference data for DPO and reward modeling.

### 8. Build dataset exporters

Export the same episode store into multiple formats:

```txt
SFT:      intent → good trajectory
DPO:      accepted trajectory > rejected trajectory
Reward:   trajectory → score vector
RLDS:     state/action/observation/reward/terminal
Parquet:  analytics and research
JSONL:    simple open format
```

### 9. Build replayable tasks

For selected episodes, snapshot:

```txt
base commit
repo setup
dependency lockfiles
test command
initial prompt
expected outcome
scoring rules
```

Command:

```bash
agenttrace replay create --episode abc123
agenttrace replay run --model qwen-coder
```

### 10. Offline RL pipeline

Offline RL input:

```txt
state: repo context + previous steps
action: tool call / edit / command / response
observation: command output / file content / test result
reward: test pass + human label + PR outcome + cost score
terminal: accepted / merged / failed / reverted
```

Pipeline:

```txt
Episodes
  ↓
Quality filter
  ↓
SFT warmup dataset
  ↓
Preference pairs
  ↓
Reward model
  ↓
Offline RL training batches
  ↓
Replay evaluation
```

### 11. Package as OSS

Repo structure:

```txt
  plugin-claude-code/
  plugin-codex/
  collector/
  episode-store/
  dashboard/
  github-linker/
  dataset-exporters/
  replay-runner/
  rl-pipeline/
  schemas/
```
