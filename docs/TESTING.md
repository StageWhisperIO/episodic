# Testing tools

Episodic ships a self-contained testing layer so you can validate an install, generate
realistic episodes without a coding agent, and benchmark how faithfully a model predicts
environment observations — all offline, no network, no model required.

Everything here is exercised by the test suite and the [tutorial notebooks](../notebooks/),
and is reachable from Python or the CLI.

---

## `episodic doctor` — end-to-end self-check

Runs every subsystem against a synthetic store in a throwaway `EPISODIC_HOME` and reports a
single health verdict. Use it after install, after upgrading, or in CI.

```bash
episodic doctor            # human-readable checklist, exit 0 when healthy
episodic doctor --json     # machine-readable report
```

```
[ok  ] imports            14 modules import
[ok  ] schema_valid       synthetic episode is schema-valid
[ok  ] store_roundtrip    saved/loaded 6 episodes
[ok  ] exporters          all 7 formats exported: {'sft': 6, 'dpo': 0, 'reward': 8, ...}
[ok  ] reward_finite      reward composites finite and in [0,1]
[ok  ] worldmodel         turn-expansion + OOD split ok (train=3, holdout=17)
[ok  ] fidelity           exact-match and runtime-mask scoring ok
[ok  ] worldbench         worldbench oracle=1.0 empty=0.3
[ok  ] replay_plan        replay returns a plan without executing
[ok  ] loop_dry_run       loop dry-run decision=dry_run

12/12 checks ok — install is healthy
```

Required checks must pass for exit 0; optional checks (`schema_file_sync`, `optional_deps`)
report `skip`/info and never fail the run. The backend is `episodic.selfcheck.run_checks()`,
which returns `{ok, passed, total, failed, checks}`.

---

## Synthetic episode factory — `episodic.testing`

Deterministic, schema-valid episodes built from `sha256`-seeded content (no `random`, no
clock), so the same call always yields the same data. This is what powers the notebooks and
most tests.

```python
from episodic import testing

ep = testing.make_episode("ep_demo", intent="add retry to http client",
                          outcome="merged", passed=3, failed=0)

pop = testing.make_population(20, seed=0)        # mixed outcomes + sources, OOD-ready
testing.populate_store(20, seed=0)               # save a population into the active store
```

Key entry points:

| Function | Returns |
| -------- | ------- |
| `make_step(index, ...)` | one schema-valid `AgentStep` (action → observation) |
| `terminal_observation(cmd, output)` / `make_test_observation(passed, failed)` | realistic observation strings |
| `make_trajectory(id, intent, turns, ...)` | an episode from explicit steps |
| `make_episode(id, ...)` | a complete episode with a computed `reward_vector` |
| `make_population(n, seed=, sources=)` | `n` episodes cycling outcomes and source repos |
| `populate_store(n, ...)` | saves a population and returns it |

`make_population` cycles outcomes `merged · merged · accepted · abandoned · reverted` and
sources `repo-alpha · repo-beta · repo-gamma` so reward, DPO pairing, and OOD splits all have
signal. Pass `sources=[...]` for finer control over the OOD partition.

---

## World-model fidelity — `episodic.fidelity`

Scores a *predicted* observation against the *ground-truth* observation along five
dimensions, following the AgentWorld content-type-aware rubric.

```python
from episodic import fidelity

fidelity.score_observation("x", "x")["composite"]                       # 1.0 (exact)
fidelity.score_observation("done at 2026-06-14T10:00:00Z",
                          "done at 2026-06-14T11:22:33Z")["factuality"]  # 1.0 (runtime masked)
```

- **Dimensions:** `format`, `factuality` (recall), `consistency` (precision), `realism`
  (success/error/empty class match), `quality` (length agreement), plus a weighted
  `composite`.
- **Runtime masking:** timestamps, hex, UUIDs, durations and pids are normalized before token
  comparison, so a faithful prediction is not penalized for a different wall-clock time.
- **Content classification:** `classify_content` splits text into deterministic vs
  runtime-metadata tokens and flags JSON / error / empty responses.

---

## Turn expansion & OOD splits — `episodic.worldmodel`

Turns trajectories into next-observation prediction samples and partitions them so the
held-out set comes from *different source repos* than training.

```python
from episodic import worldmodel

samples = worldmodel.wm_samples(pop, one_per_trajectory=True, seed=0)   # Echo-Trap-safe pool
train, holdout, mapping = worldmodel.ood_split(pop, holdout_frac=0.3, seed=0)
msgs = worldmodel.to_messages(samples[0])                               # SFT messages
```

`one_per_trajectory=True` samples a single turn per episode (deterministically) to avoid the
Echo-Trap leak where adjacent turns from one trajectory land in both train and eval.
`ood_split` partitions at the *source* level — `train` and `holdout` sources are disjoint.

The `wm` export format writes these as SFT messages (assistant = the observation):

```bash
episodic export-episode --all --format wm
```

---

## `episodic worldbench` — next-observation benchmark

Scores a predictor over the world-model pool and breaks the result down by domain and source.

```bash
episodic worldbench --predictor prefix              # baseline: echo previous observation
episodic worldbench --predictor oracle              # upper bound (composite 1.0)
episodic worldbench --predictor empty               # lower bound
episodic worldbench --source-holdout                # evaluate OOD only
episodic worldbench --turing                         # add the double-blind discriminator test
episodic worldbench --cmd 'my-model < {prompt_file}' --execute   # your own model
```

Named predictors: `oracle` (returns the target), `empty` (`""`), `echo` (the action),
`prefix` (the previous observation). `--cmd` runs a shell template per turn and requires
`--execute` — the history is written to a temp file substituted as `{prompt_file}`.

The report includes `overall`, `by_domain`, `by_source`, and — with `--turing` — a
double-blind judge that pairs the real observation against the prediction and reports
`discriminator_accuracy` and `indistinguishability` (1.0 = a judge can't tell them apart).
Pass a `judge=` callable to `run_bench` to blend a rubric score with the rule score
(`"hybrid": true`).

---

## Running the suite

```bash
PYTHONPATH=src python -m pytest tests -q     # full unit suite
python plugin-codex/test_codex.py            # codex rollout mapping
python rl-pipeline/test_pipeline.py          # pipeline stages
episodic doctor                              # end-to-end install check
```

The notebooks double as integration tests — see
[`notebooks/README.md`](../notebooks/README.md) for how to execute them headless.
