# Harbor interoperability

[Harbor](https://github.com/laude-institute/harbor) (from the Terminal-Bench team) is an
eval harness: you author tasks that ship their own verifier (an instruction + a container
environment + a test script), run agents against them in parallel cloud sandboxes, and get
back trials — each a rollout that produces a reward. Its reward is clean because the verifier
is authored up front.

Episodic sits at the opposite end: it observes real coding sessions and has to *infer* reward
from captured signals. The two are complementary, so Episodic ships a one-way bridge that turns
its observational data into Harbor's clean-verifier currency.

## What was implemented

A `harbor` exporter (`episodic export-episode --format harbor --out DIR`, registered alongside
`sft`/`dpo`/`reward`/…) mints one Harbor task package per qualifying episode:

```
DIR/
  dataset.toml          # the minted dataset (a Harbor Dataset = collection of tasks)
  manifest.json         # minted ids + skipped ids with reasons
  README.md             # how to run the dataset with `harbor run`
  tasks/<episode_id>/
    task.toml           # instruction + environment + verifier + provenance
    Dockerfile          # clones repo_state.remote_url @ base_commit (or a mount note)
    tests/run-tests.sh  # the CAPTURED test command — the verifier
    solution.patch      # the recorded unified diff — the gold reference
    metadata.json       # full provenance (reward_vector, outcome, verifier, feedback)
```

### The gate (verifier-as-artifact)

An episode is only minted when its verifier is real, not a guess. It must be:

- trusted (`validity.trust != "low"`),
- not a bad outcome (`is_bad`), and
- backed by a **captured** test command that actually passed — a `tests[]` entry with `ok: true`,
  or a `commands[]` entry with `is_test: true` and `exit_code == 0`.

Episodes without a real passed verifier are skipped and listed in `manifest.json` with a reason
(`no_verifier`, `bad_outcome`, `low_trust`, `unsafe_id`). This is the point of the whole exercise:
a minted task carries ground-truth reward, not the 0.5 neutral prior.

## What is automatic

`episodic loop` mints Harbor tasks from its training partition into `<out>/harbor` on every run
(dry-run and execute), and records `harbor: {tasks, skipped, out_dir}` in `loop.json`. Because the
loop's training set is already the trusted, good, verifiable episodes, minting reuses that filter
and adds only the verifier check. Disable with `mint_harbor: false` in the loop config.

## Lessons → where each landed

- **Verifier-as-shipped-artifact** → `tests/run-tests.sh` holds the captured command; the gate
  requires it to have passed.
- **Task minting (Episode → Harbor Task)** → the `harbor` exporter.
- **Trial = rollout that produces a reward** → each task's `metadata.json`/`task.toml` carries the
  reward the recorded rollout produced.
- **Job / Trial / Dataset layering** → `dataset.toml` + `manifest.json` describe the minted dataset.
- **Container-env per task** → the `Dockerfile` makes each task runnable by Harbor's environment
  layer (Daytona/Modal/E2B/…); Episodic does not build live cloud-runtime adapters.
- **Distribution via hub** → the output is `harbor run`/publish-ready; pushing to the Harbor hub
  still needs the Harbor CLI and its auth.

## Reuse

Verifier resolution mirrors `replay.infer_test_command` / `create_replay` (which already clone
`remote_url` at `base_commit` and run the recorded test command). Task ids pass through
`paths.safe_id`. `remote_url` is validated against a git-remote allowlist and shell-quoted before
being embedded in the generated `Dockerfile`, so a crafted `remote_url` cannot inject build steps.
