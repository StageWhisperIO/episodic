---
description: Episodic controls for the current coding session
argument-hint: start <intent> | summary | mark <label> | create-pr-notes | export-episode --format <fmt> | replay-task create
allowed-tools: Bash(episodic:*)
---

The user ran Episodic with arguments: `$ARGUMENTS`

Subcommands:
- `start <intent>` — begin tracing and record what this session is meant to do
- `summary` — what changed, why, tests, risks, suggested PR title/description, follow-ups
- `mark <label>` — one of: useful, wrong, too_broad, too_slow, needed_human_rescue, accepted_as_is, accepted_after_edits
- `create-pr-notes` — print a ready-to-paste PR title and description
- `export-episode --format <sft|dpo|reward|rlds|jsonl|parquet>` — export training data
- `replay-task create` — snapshot this episode as a replayable task

Run it and present the output to the user verbatim:

!`episodic ${ARGUMENTS:-summary}`
