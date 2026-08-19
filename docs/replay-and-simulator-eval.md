# Replay gate + learned simulator: evaluation report

Synthesis of 2 implementation-verification passes, 1 independent cross-validation run against the real
StageWhisper corpus, and 3 independent adversarial evaluations of the uncommitted work on top of clean
HEAD `664f830`: a model-driven replay-eval gate (`src/episodic/replay/modelrun.py`,
`episodic loop --eval-backend {stub,mlx,serving,tinker}`) and a validated learned simulator
(`src/episodic/worldmodel/validate.py`, `episodic wm-validate`), plus a history-budget O(n²)→O(n) fix in
`worldmodel/__init__.py:render_history`. This report does not modify that work; it re-derives the load-bearing
numbers from source, from a fresh full-suite run, and from the real corpus where the evaluators' own re-runs
disagreed with the documented claims.

Both milestones are real code with real test coverage — not facades. Both also fall short, in specific and
now-documented ways, of the claims the in-flight `docs/roadmap.md` update makes for them. The rest of this
report is about exactly where that line falls.

## 1. Executive summary

- **Both gates are real, uncommitted, opt-in.** `git diff --stat` against HEAD `664f830`: 10 modified files
  (+503/-45) plus 5 new untracked test files and the 2 new source files (`replay/modelrun.py`,
  `worldmodel/validate.py`). At HEAD, neither `episodic loop` nor `episodic wm-validate` had any way to make a
  model produce a diff, or to validate a trained world model, at all. Nothing here is committed or pushed, per
  the task instructions.
- **Milestone 1 (replay gate) mechanism is genuinely proven, its one measured "quality" number is not.** Real
  `git clone` (verified via `git remote -v`), real `git apply` of a unified diff, and a real
  `pytest`/`cargo` subprocess whose pass/fail genuinely flips the promote/keep_base decision — independently
  reproduced by all three evaluators. But the only "candidate vs. base" score anyone has actually measured used
  an **oracle-diff-apply stub** (the episode's own captured diff, not a model-generated one) standing in for
  "trained model," and it landed slightly *below* a no-op baseline (0.2571 vs 0.3, n=14 paired) — diagnosed as
  build-timeout noise on Cargo/Rust holdouts, not a quality signal. No real distilled-vs-frontier comparison has
  ever been run through this path, and no diff-generating policy model exists anywhere in the repo to be the
  "candidate" side of one.
- **Milestone 2 (simulator) fidelity ordering is the most solid result in this report, reproduced four
  independent times** at different splits/seeds/iteration counts: `oracle (1.0) > trained (0.40–0.45) >
  prefix ≈ empty (0.09–0.10)`. That is a real, non-trivial, repeatedly-confirmed discriminative gap from a real
  local `mlx-sft` LoRA run (no Tinker, no network) against the real 104-episode StageWhisper corpus.
- **The sim↔replay correlation — the one number that would actually validate "simulator as a cheap proxy for
  real replay" — does not hold up, and not just because it's small-n.** Four different measurement attempts
  produced four different, inconsistent results: the roadmap's own curated n=5 run (Pearson −0.08, Spearman
  +0.29), an unfiltered run of the actual shipped `--replay-correlate` command at n=16 (Pearson −0.23, Spearman
  −0.12, because the shipped code silently blends non-verifier episodes' near-tautological scores into the
  correlation), an independently-run n=6 variant (Pearson −0.24, Spearman −0.41), and a sign flip to **+0.41**
  at the same n=5 just from swapping the sim predictor to the degenerate `prefix` baseline. None of these are
  the same measurement, and none of them supports a directional claim in either direction.
- **Two concrete defects found, source-verified, neither exploited live:** (1) `episodic loop
  --eval-backend serving` fails open to the real public `https://api.openai.com` with no API key and no CLI way
  to configure `eval_backend_config` for that backend — the same fail-open pattern already found and fixed once
  in `serving/router.py`, not generalized to this new call site. (2) `worldmodel/validate.py:offline_replay_scores`
  → `replay.create_replay` has no disk guard, no per-episode workspace cleanup, and no holdout cap (unlike
  `loop`'s default `max_holdout=50`) — reproduced crashing the whole `wm-validate --replay-correlate` run with
  an uncaught `OSError: No space left on device` after 12 full monorepo clones, discarding every already-computed
  score.
- **Positioning gap:** `docs/roadmap.md` Phase 3 documents Milestone 2 (the simulator) in detail. Milestone 1
  (the replay gate itself — `modelrun.py`, `--eval-backend`, the `test_cwd`/shell-command relativization fix) is
  mentioned nowhere in `docs/roadmap.md`, `README.md`, or `docs/flywheel.md`, and `--eval-backend serving`'s
  fail-open default is undocumented anywhere a user would see it before running it.
- **Net:** the flywheel now has the *mechanism* for a model-vs-model replay gate and a trained world-model
  simulator, both real and independently reproducible against real data. Neither one yet produces a trustworthy
  quality number — the replay gate has never scored a real model, and the simulator's fidelity (real) has not
  been shown to predict real replay outcomes (unmeasured, not disproven). Getting to a distilled-vs-frontier
  cost/quality number is several concrete, listed steps away (§5), not a data-analysis exercise on what's
  already been run.

## 2. Full test-suite result

```
python -m pytest -q
585 passed, 1 skipped in 40.31s
```

Reproduced fresh for this report, twice, matching every implementation summary and adversarial evaluation once
the diff under review had fully accumulated (the "replay" implementation summary's 559/1 count was against an
earlier slice of the same diff, before the simulator work landed in the tree). The 1 skip is the pre-existing
`trl`-unavailable branch in `tests/test_trainers.py`, unrelated to this work.

One of my own runs hit 10 transient failures in `tests/test_serving_server.py` (`Connection refused` against a
loopback `ThreadingHTTPServer`) that disappeared both on an isolated re-run of that file and on a clean full-suite
re-run — a local port-binding flake in an already-committed, untouched-by-this-diff test module, not a
regression. All four other evaluation passes (2 implementation, 1 cross-validation, 3 adversarial) independently
reported clean 585/1 (or an earlier equivalent) runs.

No test in the reviewed diff hits real network or a real LLM API. HTTP-backed tests
(`tests/test_replay_modelrun.py`'s `serving`-backend cases) inject a fake opener; `mlx`/`tinker` code paths are
exercised only via monkeypatched loaders or `ValueError`/`TrainerUnavailable` pre-flight checks in tests. The
one real trainer exercised anywhere in this work is local `mlx-sft` against the already-HF-cached
`HuggingFaceTB/SmolLM2-135M-Instruct` — confirmed offline (`HF_HUB_OFFLINE=1` reproductions), no Tinker calls,
no billing.

## 3. Milestone 1 — model-driven replay-eval gate

**What it is:** `replay/modelrun.py` (new) adds a `generate()` abstraction with four backends
(`stub`, `mlx`, `serving`, `tinker`) that `episodic loop --eval-backend ...` and `_eval_one` use to produce a
candidate diff, `git apply` it into a real clone of the episode's `repo_state.root`, run the episode's own
captured test command, and score `0.6 * tests_pass + 0.4 * diff_overlap` for both a "candidate" and a "base" run
before deciding promote vs. keep_base.

**Status: mechanism proven real; quality claim unproven.**

**Runnable funnel on the real corpus.** Of the real StageWhisper store (104–108 episodes across different
snapshots taken during this evaluation window, as live capture continued in parallel), only episodes with both
a captured test-verifier *and* a still-git-available `repo_state.root` can drive a real replay. The
cross-validation run measured this directly: **18/104 runnable (17%)**, consistent with the previously-known
~94→17 verifier funnel and with a separate `export-episode --format molt --all` run on the same store (19/104
minted, 18%, 84 skipped for `no_verifier`/`low_trust`).

**Paired candidate/base scores (cross-validation run, 3 workers, 18 runnable episodes):**

| | n | mean score |
|---|---|---|
| candidate (oracle-diff-apply stub) | 14 paired | 0.2571 |
| base (no-op stub) | 14 paired | 0.3000 |

4/18 base runs were excluded for transient `git clone` failures (rc=128) under concurrent I/O load against the
89 GB monorepo — a real, reproducible resource constraint on this host, not a scoring bug.

Candidate scoring *below* base is not a regression signal. Only 4 of the 14 paired episodes had any non-empty
captured diff text at all, and for those 4, applying the diff produced scores identical to the empty-diff stub
— most of the affected test commands are `cargo test`/`cargo build` invocations that hit the harness's fixed
120s subprocess timeout regardless of whether the diff was applied, so the observed gap is build-timeout noise,
not a measured quality difference. **This run validates that the loop→replay plumbing executes correctly
against real repos and real tests. It does not validate model quality — nothing "trained" was ever compared to
anything else here.**

**What is real:**
- Real `git clone` (verified via `git remote -v`, not a filesystem copy) of the episode's actual local repo.
- Real `git apply` of a unified diff (either the episode's own captured diff, used as an "oracle" stand-in for a
  trained model, or a `stub`-supplied fixed breaking/fixing diff in tests).
- Real `pytest`/`cargo` subprocess execution whose pass/fail genuinely drives `tests_pass` — `test_run_loop_eval_backend_stub_{promotes,keeps_base}_*`
  demonstrably flip the promote/keep_base decision based on whether the injected diff breaks or fixes a real
  test run, not a hardcoded equality.
- `--eval-backend mlx` independently confirmed (by one evaluator) to call real `mlx_lm.load`/`generate` against
  the cached `HuggingFaceTB/SmolLM2-135M-Instruct` weights offline — not a stub — when explicitly configured.
- `tests/test_replay_real_corpus.py` (new, ships in the suite) runs for real, not skipped, against a live copy
  of the real 105-episode store: 4/4 pass, including a full `loop._eval_one` execution for a real captured
  episode. The `_resolve_test_command`/`test_cwd` relativization fix was confirmed against real captured
  subdirectory commands (`stagewhisper-mobile/src-tauri`, `web/backend`,
  `integrations/hermes-stagewhisper-plugin`).

**What is stubbed or missing:**
- Every "candidate" score measured anywhere in this report used either a `stub` backend fed the episode's own
  recorded diff (an oracle proxy, not a trained policy's output) or a fixture diff in unit tests. **No real
  trained/distilled model has ever generated the diff on either side of a comparison through this path**, and
  there is currently no diff-generating policy model anywhere in the codebase to serve as one — the world model
  trained in Milestone 2 predicts tool *observations*, not diffs.
- `episodic loop --execute` with no `--eval-backend` and no `EPISODIC_REPLAY_CMD` still resolves to `dry_run`
  (no clone, no test, `scores=null`) — the promote/keep_base gate is structurally inert unless a backend is
  explicitly configured. This is opt-in by design, but worth naming: the default quickstart path does not
  exercise this gate at all.
- The standalone `episodic replay-task run` CLI command was not updated to use the new `eval_backend`
  abstraction; it still only supports the older `EPISODIC_REPLAY_CMD` external-process template.
- **Defect (unexploited, source-verified):** `--eval-backend serving` fails open. `serving.build('openai', {})`
  returns a backend with `base_url='https://api.openai.com'` and `api_key=None`; `HTTPBackend`'s auth-header
  logic omits the `Authorization` header when the key is falsy rather than refusing to send the request, and
  the CLI exposes `--eval-backend {stub,mlx,tinker,serving}` with no flag to set `eval_backend_config`
  (`base_url`/`api_key`) for the `serving` choice — only `--eval-model-dir`/`--eval-sampler-path`, which apply to
  `mlx`/`tinker`. Run exactly as the flag's help text describes, `episodic loop --execute --eval-backend serving`
  would issue a real, unauthenticated HTTPS request to the real OpenAI API before any auth failure surfaces.
  This is the same class of bug already found and fixed once at a different call site
  (`serving/router.py:backend_for`, which now raises instead of defaulting) but not generalized here.
  `tests/test_replay_modelrun.py`'s `serving`-backend tests both supply an explicit `base_url` + fake opener, so
  the unconfigured default path is untested. Not triggered by any test or manual run in this evaluation — only
  matters if a user opts into `--eval-backend serving` without also configuring a base URL, which the CLI gives
  them no way to do anyway.

## 4. Milestone 2 — validated learned simulator

**What it is:** `worldmodel/validate.py` (new) + `episodic wm-validate` split real episodes via
`loop.split_episodes`, train (or load) a world-model predictor, score the holdout with
`worldbench.rollout_bench`/`rollout_turing_test` against `oracle`/`prefix`/`empty` baselines, and — with
`--replay-correlate` — attempt to correlate the simulator's rollout fidelity against a real, offline replay-eval
score for the same holdout episodes.

**Status: fidelity result solid and reproduced; correlation result not trustworthy at any of the n's measured.**

### 4.1 Trained-WM fidelity vs. prefix / empty / oracle

The ordering `oracle > trained > prefix ≈ empty` was independently reproduced **four times**, at different
holdout sizes, seeds, and LoRA iteration counts, all against real copies of the real StageWhisper corpus:

| Source | split | LoRA iters | oracle | trained | prefix | empty |
|---|---|---|---|---|---|---|
| `docs/roadmap.md` (documented) | 48 train / 16 holdout | 60 | 1.0 | 0.44 | 0.09 | 0.09 |
| Cross-validation run | 12 train / 6 holdout | 30 | 1.0 | 0.3964 | 0.0972 | 0.0972 |
| Adversarial eval #1 (own reproduction) | 16-episode holdout | — | 1.0 | ~0.45 | ~0.093 | ~0.093 |
| Adversarial eval #3 (own reproduction) | 48 train / 16 holdout, seed 0 | 5 (vs. roadmap's 60) | 1.0 | 0.4203 | 0.0938 | 0.0938 |

This is the most solid result in this report: a real, local `mlx-sft` LoRA adapter over
`HuggingFaceTB/SmolLM2-135M-Instruct` (no Tinker, no network, HF-cached weights) genuinely discriminates a
trained world-model from degenerate baselines on real held-out captured episodes, consistently, even at 5
LoRA iterations instead of 60. Turing indistinguishability shows the same ordering (oracle 1.0, trained
0.625–0.75, prefix/empty 0.625).

The bounded-context fix underneath this (`render_history`'s O(n²)→O(history_budget) rewrite, needed because the
naive version OOM'd exporting the 20 real episodes with ≥200 steps — the bucket holding 15/16 of the corpus's
reward≥0.5 "good" episodes) was independently differential-tested by one evaluator (500 randomized trials,
naive-vs-budgeted, 500/500 identical) — a real confirmation of the roadmap's "verified byte-identical" claim,
though that differential test itself is not checked into the repo; the committed coverage
(`test_render_history_is_bounded_by_history_budget`) is real but narrower than the doc's claim.

### 4.2 Sim↔replay correlation — the headline trustworthiness number

This is the number that would actually support "the simulator is a cheap, valid proxy for real replay
outcomes" — the core "validated simulation" claim for the WMO bar. **It does not hold up, across every
independent attempt to measure it:**

| Source | n | filter | Pearson | Spearman |
|---|---|---|---|---|
| `docs/roadmap.md` (documented, curated) | 5 | manually pre-filtered to verifier-confirmed + git-available | −0.08 | +0.29 |
| Adversarial eval #2 — actual shipped command, unfiltered, same 16-episode holdout | 16 | none (as ships) | −0.23 | −0.12 |
| Cross-validation run | 6 | Milestone-1 candidate score as ground truth | −0.2422 | −0.4140 |
| Adversarial eval #2 — same n=5, `prefix` predictor swapped in | 5 | same manual pre-filter | **+0.41** | — |

Four measurement attempts, four different numbers, spanning both signs. None of this is a small-n technicality
alone — it reflects three compounding, source-verified problems:

1. **The "real" ground truth is itself a stub.** Every one of these correlations pairs the simulator's fidelity
   against the same oracle-diff-apply-stub replay score from §3, not a real model's replay outcome. For
   Cargo/Rust holdout episodes that score is bimodal (0.6 if a from-scratch, no-cache build finishes and passes
   within the 120s timeout, 0.0 if it doesn't) — a build-speed artifact, not a trajectory-fidelity signal.
2. **The shipped, undocumented default conflates verifier and non-verifier episodes.**
   `offline_replay_scores` always clones and applies the episode's own diff regardless of whether a real test
   command was captured; `run_replay` internally knows this (`note = 'no test command captured...'`) but
   `offline_replay_scores` discards `note`/`test_command` and keeps only the numeric total, so 10/16 episodes
   in the unfiltered n=16 run get a near-tautological "diff-overlap-only" score of 0.4, indistinguishable in the
   output from a genuine 0.6/0.0 test-verified score. The documented n=5 number required a manual, unshipped
   pre-filter script to avoid this — running the actual `episodic wm-validate --replay-correlate` command as
   documented does not reproduce the documented number.
3. **At the n's achievable on the real corpus today, the sign itself is not stable.** Swapping only the sim
   predictor (trained mlx model → degenerate `prefix` baseline) at the same n=5 flips Pearson from −0.08 to
   +0.41. No version of this measurement should be read as evidence for or against the simulator's predictive
   value.

**Robustness defect on top of all of this:** `worldmodel/validate.py:offline_replay_scores` → `replay.create_replay`
has no disk-space guard around its `mkdir`, no per-episode workspace cleanup, and no `max_holdout` cap (`loop`'s
default is 50; `wm-validate` has none). One evaluator reproduced `wm-validate --replay-correlate` crashing
outright with an uncaught `OSError: No space left on device` after 12 full local clones of the 89 GB monorepo
consumed 11 GB, discarding every already-computed score in the run. The cross-validation run hit the same
underlying disk-pressure issue and required manual mid-run cleanup to complete.

**Conclusion for §4.2:** the simulator's value as a cheap proxy for real replay outcomes is **neither confirmed
nor disproven** by any run in this evaluation — it is unmeasured, because every attempt to measure it inherited
either a stubbed ground truth, a scoring conflation bug, or too few clean data points to have a stable sign.
Do not cite any single one of the four numbers above as "the" correlation.

## 5. Where this leaves Episodic vs. the WMO bar, and the next concrete step

**What's earned:** a real model-vs-model replay gate mechanism (clone → apply → test → score → promote/keep_base,
mechanically proven against real repos) and a real, repeatedly-reproduced trained-world-model fidelity gap
(oracle > trained > degenerate baselines) on the real StageWhisper corpus, both with genuine test coverage and
zero Tinker billing or real-network calls anywhere in the test suite.

**What's not earned yet:** the two specific claims that would move this from "mechanism exists" to "validated
against the WMO bar" — a **distilled-vs-frontier cost/quality delta** (nothing has ever generated a real diff on
either side of a replay comparison) and a **validated simulator** (fidelity is real; predictive value for real
replay outcomes is unmeasured, per §4.2).

**Exact next steps, in order, to reach a trustworthy distilled-vs-frontier number:**

1. Close the `--eval-backend serving` fail-open gap before pointing it at anything real: require an explicit
   `base_url`/`api_key` (or a `BackendUnavailable`-style raise on an unconfigured default), matching the fix
   already applied in `serving/router.py:backend_for`, and add a CLI flag to set `eval_backend_config` for the
   `serving` choice.
2. Get a real diff-generating candidate into the loop — either fine-tune a small local model (`mlx-sft`) to
   produce diffs from episode context (there is currently no such model in the repo; the world model predicts
   tool observations, not diffs), or point `candidate` at a real frontier endpoint via the now-fixed `serving`
   backend and compare it against a real local/distilled `base`.
3. Add cost/token/latency capture to the replay-eval path itself — `_eval_one`'s output row today is
   `{episode_id, candidate, base, reason}`, quality-only; the `cost_usd`/`cost_efficiency` fields elsewhere in
   the schema are captured from the *original* session, not measured during replay, so even a real run today
   produces a quality delta with no accompanying $/token figure.
4. Fix `wm-validate`'s robustness before relying on any correlation number from it: guard the `mkdir` in
   `offline_replay_scores`/`create_replay` against disk exhaustion, clean up each replay workspace after
   scoring, and add a `max_holdout` cap analogous to `loop`'s default of 50.
5. Fix the score-conflation bug: propagate a `has_verifier`/`note` signal out of `offline_replay_scores` so
   `--replay-correlate` can default to filtering to genuinely test-verified episodes, instead of requiring a
   manual, unshipped pre-filter script to reproduce the documented n=5 number.
6. Scale n: only 5–18 real episodes (of 104) currently have a usable, fast, non-timeout-bound verifier. Either
   grow the real corpus's verifier coverage or add synthetic/factory-generated episodes with reliable fast
   verifiers so the correlation in §4.2 can be measured at a sample size where a sign is meaningful.
7. Only then run the fixed gate for real — `candidate` = a real trained/distilled small model, `base` = a real
   frontier endpoint — through replay-eval with cost instrumentation attached, to produce the actual
   $-cost-vs-quality number this milestone is aiming for.

**Documentation gap to close regardless:** Milestone 1 (the replay gate itself) is not mentioned anywhere in
`docs/roadmap.md`, `README.md`, or `docs/flywheel.md`, and `episodic wm-validate` is a fully working, registered
subcommand absent from all three as well; `tests/test_docs_cli_parity.py` only guards against stale/removed
command references, not missing new ones, so this drift is not caught by CI.
