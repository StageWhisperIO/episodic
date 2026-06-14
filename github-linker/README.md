# github-linker

Links Episodic episodes to GitHub pull requests, enriching them with real outcome data for use as labeled training samples.

## Requirements

- `gh` CLI authenticated (`gh auth login`) **or** `GITHUB_TOKEN` set in the environment.
- Python 3.10+, stdlib only.

## Usage

Link the PR for the current branch automatically:

```
episodic link --auto
```

Link a specific PR by URL or number:

```
episodic link --pr https://github.com/owner/repo/pull/42
episodic link --pr 42
```

## Outcome fields

| Field | Description |
|-------|-------------|
| `status` | `open`, `accepted`, `merged`, `failed`, `reverted`, or `abandoned` |
| `commit` | HEAD commit SHA of the PR branch |
| `branch` | Branch name |
| `pr_url` | Full GitHub PR URL |
| `pr_number` | PR number |
| `pr_state` | Raw GitHub state (`OPEN`, `MERGED`, `CLOSED`) |
| `ci_status` | Aggregated CI result: `success`, `failure`, or `pending` |
| `review_decision` | GitHub review decision (`APPROVED`, `CHANGES_REQUESTED`, etc.) |
| `merged` | `true` if the PR was merged |
| `reverted` | `true` if a revert commit was detected after the episode's base commit |
| `manual_edits_after_agent` | `true` if commits were added after the agent's base commit |
| `linked_at` | ISO timestamp of when the link was performed |

## Training data

Running `episodic link` turns raw episodes into outcome-labeled records. The populated `outcome` block gives downstream reward models a ground-truth signal: whether the agent's work was merged, CI-clean, human-reviewed, or reverted, enabling supervised fine-tuning and reinforcement learning from real deployment outcomes.
