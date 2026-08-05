# Molt interoperability

[Molt](https://github.com/NVIDIA-NeMo/labs-molt) is NVIDIA NeMo Labs' agentic-first RL framework
(Ray · vLLM · AutoModel/FSDP2). Its training contract is one prompt dataset plus one agent file:
`--data.prompt_dataset prompts.jsonl` (with `--data.input_key input`) and `--train.agent_path`
pointing at a Python file whose `Env.step(state) -> Result(reward=...)` grades a rollout.

The `molt` exporter (`episodic export-episode --format molt --out DIR`) mints that bundle from
verified episodes. Unlike the [Harbor](harbor-interop.md) exporter — which writes one declarative
task directory per episode — Molt is dataset + agent, so the bundle is flat:

```
DIR/
  prompts.jsonl          # one row per verified episode: input + label + episode_id + composite_reward
  agents/episodic_env.py # a generated Env whose reward replays the captured verifier
  README.md              # the exact molt.cli.train_rl_ray command
  manifest.json          # minted ids + skipped ids with reasons
```

## How reward works

Each `prompts.jsonl` row carries the verifier as a JSON `label`
(`remote_url`, `base_commit`, `test_command`, `framework`). At rollout time the generated
`EpisodicEnv.step`:

1. clones `remote_url` at `base_commit` into a fresh workspace,
2. extracts the model's unified diff from a ```` ```diff ```` block in its response,
3. `git apply`s it, and
4. runs the captured `test_command` under `bash -c` with `set -o pipefail` — reward is `1.0` on
   pass, else `0.0`. (`pipefail` matters: real captured commands are often shell one-liners like
   `pytest -q | tail`, and without it the pipe would mask a test failure and report a false pass.)

This is the coding-RL analog of the Harbor Dockerfile + `run-tests.sh`: the captured test is the
verifier, so the reward is ground truth rather than a proxy. The agent file is a single static
module; all per-episode variation lives in `prompts.jsonl`.

## The gate

An episode is minted only when the reward can be reconstructed in a fresh rollout env. It must be
trusted, not a bad outcome, have a captured passing verifier, **and** carry a clonable `remote_url`
(the extra requirement over Harbor, which could fall back to a mount note). Others are skipped and
listed in `manifest.json` with a reason (`no_remote`, `no_verifier`, `bad_outcome`, `low_trust`,
`unsafe_id`).

## Portability

Real captured test commands frequently embed absolute host paths (e.g.
`cargo test --manifest-path /Users/you/repo/Cargo.toml`) that will not exist in a fresh clone. The
exporter first **relativizes** any path under `repo_state.root`
(`normalize.relativize_command`) — `/Users/you/repo/sub/Cargo.toml` becomes `./sub/Cargo.toml` — so
the command runs in the cloned `/workspace`. This is the same normalization now applied at capture
time in `build_episode`, so newly captured episodes store relative commands from the start.

A command that still references an absolute path *outside* the repo root after that pass is flagged
`portable: false` in `prompts.jsonl`, counted under `non_portable` in `manifest.json`, and noted in
the README. Minting is not blocked. On the real StageWhisper corpus this lifted portable minted
tasks from 4/17 to 17/17.

## Safety

`remote_url` and `base_commit` reach `git` through `subprocess` list form (no shell) and are
rejected if they begin with `-` (option injection); `base_commit` is additionally validated against
a ref allowlist at export time and dropped if unsafe. The `test_command` runs under `bash -c` — the
same trust posture as the Harbor `run-tests.sh` — because it is the recorded command from a
trusted, gated episode. Episode ids pass through `paths.safe_id`.

## Reuse

The verifier detection and trust/quality gate (`_captured_verifier`, `is_trusted`, `is_bad`) are
shared with the Harbor exporter. The clone-checkout-run flow mirrors `replay.create_replay` /
`run_replay`.
