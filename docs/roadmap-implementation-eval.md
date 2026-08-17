# Roadmap implementation evaluation

Synthesis of 6 implementation-verification passes and 3 independent adversarial evaluations of the
uncommitted Phase 1-3 work described in [`docs/roadmap.md`](roadmap.md). This report itself does not
modify the roadmap work; it only reads the working tree and, where a claim needed grounding, re-derives
it from source or a live pytest run.

## 1. Executive summary

- The wiring is real, not a facade: `episodic loop --execute` -> promote -> `episodic serve` ->
  OTLP ingest -> new episode genuinely closes, verified independently three times against the real
  StageWhisper store, including outside the pytest harness.
- Full suite: **513 passed, 1 skipped**, ~29s, reproduced fresh for this report. No test hits real
  network or a real LLM API.
- The roadmap's central P1.2 claim — "agent-as-a-judge lifts signal on no-verifier episodes" — is the
  one place evaluators genuinely disagree, and the disagreement resolves in favor of the more
  adversarial finding once you look at the code: every "lift" number in the implementation summaries
  came from a **stub** judge that always returns a fixed good score. The one evaluation that ran the
  **real** judge on real episodes found scores went *down*, 8/8, because `JUDGE_TRAJECTORY_LIMIT = 4000`
  truncates each trajectory from the **start**, so the judge never sees how long real sessions end.
  Source-verified in this report (`src/episodic/core/rubric.py:120`).
- Two additional bugs, source-verified in this report, that no evaluator's "working" verdict accounted
  for: (a) `episodic serve` silently falls through to a live `https://api.openai.com` call for any tier
  without an explicit backend config (fail-open, not fail-closed); (b) `ensure_reward` re-judges the
  *entire* historical episode store on every single `episodic loop` invocation with no caching, which is
  why one evaluator measured a 9m40s "quickstart" against the real 102-episode store against a
  documented "60-second quickstart" claim in `README.md`.
- Phase 3's `worldbench`/`wm`-export path is O(n²) on trajectory length with a heavy constant
  (pre-existing code, not introduced by this diff) and one evaluator could not finish a full-corpus real
  run; it does not affect the loop's actual training path, which is O(n) and fast on the same data.
- Positioning docs (README, `docs/flywheel.md`) are internally consistent and parity-tested against the
  real CLI, but the "60-second quickstart" claim does not hold once judged against the real corpus at
  realistic scale — same root cause as the judge-caching bug above.
- Net: the flywheel *mechanism* is sound and does close. The specific claims "judge lifts real
  no-verifier signal," "60-second quickstart," and "serve is safe to run with a partial backend config"
  do not hold as stated once tested against the real store with real (non-stub) components.

## 2. Full test-suite result

```
python -m pytest -q
513 passed, 1 skipped in 29.53s
```

Reproduced fresh for this report (2026-08-17), matching all three independent adversarial evaluations
(which separately reported 513/1 across their own sessions as the diff accumulated review batches;
earlier implementation-summary runs reported lower counts — 369, 416, 418, 460, 511 — because they ran
against earlier, smaller slices of the same uncommitted diff before later tracks were reviewed and left
in the tree).

The 1 skip is `tests/test_trainers.py:74`, a "trl unavailable" branch that isn't exercised because `trl`
*is* installed in this environment — unrelated to this roadmap, and reproduces identically on a stash of
plain `main` per two of the three adversarial evaluations.

No test reaches real network or a real LLM: HTTP-backed tests inject a `FakeOpener`/injected opener
against `*.local` hosts or a real local `ThreadingHTTPServer` bound to `127.0.0.1`; judge/labeler tests
use `sh -c "printf 'SCORE: ...'"` stand-ins; `mlx`/`tinker` SDKs are monkeypatched with fakes. One
evaluator noted that running the *entire* suite with `EPISODIC_HOME` exported globally (rather than
scoped per-test) trips an unrelated pre-existing failure in `plugin-codex/test_codex.py`, confirmed to
also fail identically on plain `main` — a pre-existing test-isolation quirk, not a regression.

## 3. Per-track status

| Track | Status | Evidence | Gaps / bugs |
|---|---|---|---|
| **A — `episodic serve` proxy + router** (P1.1, Phase 2 A2) | **partial** | Real `ThreadingHTTPServer`-backed proxy, backend registry (`openai`/`ollama`/`vllm`/`tinker`) mirroring `trainers/__init__.py`. Verified live by two evaluators as real subprocesses against real localhost sockets: `GET /v1/models`, streaming + non-streaming `/v1/chat/completions`, tier-forcing, and a real trained `router_model.json` (`serving/difficulty.py:learn_router`, hand-rolled logistic regression, trained in 1.7-13ms against 61 real reward-labeled episodes, `positive_rate=0.77`) all round-tripped correctly. `loop/__init__.py` writes `served_ref` into `promoted.json`, consumed correctly by `router.resolve_served_ref`. | **Confirmed by source inspection in this report**: `serving/router.py`'s `_tier_config` does `tier_config.setdefault("backend", "openai")` for *any* tier lacking explicit config, and `serving/openai.py`'s `OpenAIBackend.default_base_url = "https://api.openai.com"`. A server started with only `--distilled-*` configured (the CLI's own minimal example) will silently proxy a request routed to `"frontier"` — or, given the router's own measured 0.77 real positive-rate, *most* real traffic once `--router-model` is enabled — to the live public OpenAI API. One adversarial evaluator reproduced this live (a real `HTTP Error 401: Unauthorized` from `api.openai.com` in 0.31s). This is fail-open, not fail-closed, and leaks prompt content by default rather than erroring with "frontier backend not configured." Roadmap itself already flags the router as "not yet tuned" for accuracy; this sharpens it into a concrete safety/privacy default, not just a tuning gap. |
| **B — default judge, epoch loop, co-evolving evaluator** (P1.2, Phase 2 B2/B3) | **partial** | CLI defaults judge on (`cli.py:493`, `config.setdefault("judge", not args.no_judge)`). `rubric.default_judge`/`safe_judge` wraps the existing labeler path and degrades a criterion to not-applicable rather than crashing on any judge failure — respects the known unauthenticated-labeler gotcha. Epoch loop (`_run_epoch`/`run_loop`) is byte-identical to the pre-epoch manifest at `epochs=1` and correctly chains `base_model` across epochs at `epochs>1`, tested with concrete numeric assertions. `local_critic`/`trl_reward` evaluators use real tiny HF `transformers` backbones (real forward/backward, not mocked math) for `local_critic`; `trl_reward` is unit-tested with a monkeypatched trainer/loader. | Four bugs, all source-verified in this report: **(1)** `core/rubric.py:120`, `trajectory_text(episode)[:JUDGE_TRAJECTORY_LIMIT]` with `JUDGE_TRAJECTORY_LIMIT = 4000` — truncates from the **start**, so on real multi-step sessions the judge sees only early exploration and never the fix/diff/explanation at the end. **(2)** One adversarial evaluator ran the real (non-stub) judge against 8 real substantive no-verifier StageWhisper episodes and got rubric-score deltas negative on 8/8 (mean -0.0765) — the opposite of "lift." **(3)** `loop/__init__.py:36`, `_resolve_judge` reads `config.get("judge")` with no default — the "on by default" behavior lives only in `cli.py`'s argparse handling. Source-verified here: `tests/test_flywheel_e2e.py`'s `loop_config` (line 122) has no `"judge"` key, so the flagship end-to-end proof of the closed flywheel runs with the judge **off** and never exercises P1.2. **(4)** `loop/__init__.py:28-32` and `:231`, `ensure_reward(list(store.iter_episodes(start)), judge)` re-judges every historical episode on every single `episodic loop` invocation with no caching — source-consistent with one evaluator's measured 9m40s real-store run (see §5). |
| **C — learned simulator / world-model env** (Phase 3) | **partial** | `worldmodel/env.py:WorldModelEnv`/`rollout`, `worldmodel/inference.py` (mlx/tinker predictors via injected fakes), `fidelity.trajectory_score`, `worldbench.rollout_bench`/`rollout_turing_test`, and the `episodic doctor` `worldmodel_env` check are all real and unit-tested with concrete numeric assertions (oracle composite=1.0/drift=0.0; degenerate predictors score measurably lower — a real discriminative signal, not a rubber stamp). `episodic loop --sim-prefilter` correctly caps which holdout episodes get the expensive real-replay budget; code-reviewed and confirmed the actual promote/keep_base decision is always driven by real replay-eval, never the sim score. | One evaluator measured `worldmodel.expand_turns`/`render_history` (the function `episodic worldbench`'s default path and `export-episode --format wm` call per turn) recomputing the entire trajectory prefix from scratch on every turn index — O(n²) with a heavy re-serialization constant. 25s for one real 3,943-step StageWhisper episode; a full `episodic worldbench --predictor prefix --turing` run over the real 102-episode store did not finish in several minutes and grew to 6GB RSS before being killed. Confirmed pre-existing on plain `main` (not introduced by this diff) and does **not** affect the loop's actual training path — `exporters.segment_episode` (what `sft`-format training/`--sim-prefilter` actually uses) accumulates history incrementally and processes the same worst-case episode in 0.45s. The two other adversarial evaluators ran smaller or capped real-scale checks (`max_holdout` forced small, or the `prefix` predictor only) and did not hit this — not a contradiction, just a difference in what got exercised. |
| **D — positioning** (README/docs/flywheel.md reframe) | **working**, with one caveat | README opens with the capture -> reward -> distill -> serve -> co-evolve framing and a "60-second quickstart"; `docs/flywheel.md` is the deep dive with an honest Episodic-vs-WMO table and caveats section. `tests/test_docs_cli_parity.py` mechanically fails the build if README's CLI reference table drifts from the real argparse surface, and it passes. All cross-checked function/flag references (`rubric.default_judge`, `serving/router.py:resolve_served_ref`, `serving/difficulty.py:learn_router`, `loop/evaluator.py:refresh`, every documented CLI flag) exist and match described behavior. README's status language avoids claiming an unproven lift number, matching the roadmap's own closing line. | The "60-second quickstart" claim (`README.md:44`) does not hold at real-corpus scale: with judge on (the documented default) it took one evaluator 580.72s (9m40s), verified end-to-end against the real 102-episode store, even in dry-run with a stub trainer — a direct consequence of Track B bug (4) above (no reward caching, O(store size) judging every run). `docs/flywheel.md` caveats that judging has "a real cost per episode" but doesn't say that cost is unbounded/uncached or quantify it. |

## 4. Does the flywheel close on real StageWhisper data?

Yes, mechanically — this is the strongest, least disputed finding across all six verification passes
and all three adversarial evaluations. `loop.run_loop(execute=True)` -> `promoted.json` with a real
`served_ref` -> `episodic serve` -> a real `/v1/chat/completions` round trip (mocked upstream only,
never a real network target) -> OTLP ingest -> `service.finalize_session` -> a new episode file, all
run against real captured StageWhisper episodes and reproduced independently outside pytest by more
than one evaluator, with the real store's `git status`/file count confirmed unchanged before and after.

But "closes" needs three qualifiers that the roadmap's "done" language doesn't carry:

1. **It closes with judge off.** `tests/test_flywheel_e2e.py`, the one test cited as flagship proof of
   the closed loop, does not set `"judge"` in its config, and `loop.run_loop` has no such default
   (Track B, bug 3 above) — so the shipped proof of "the flywheel closes" and the shipped proof of
   "the judge is on by default" are two different, non-overlapping claims, not one.
2. **It closes at filtered scale.** The e2e test (and most manual real-store runs across the
   verification passes) filter or cap to episodes under ~200 steps, covering 83/102 real episodes by
   count but excluding the long tail that trips the Track C performance bug.
3. **It closes cheaply only with a stub trainer/judge.** Every "fast" real-store timing reported (2-6s,
   ~3.5s, 1.7s for router training) used a no-op `command` trainer stub and either no judge or a fixed-
   score judge stub. The one run that used the real judge against the real store measured minutes, not
   seconds (§5).

So: the wiring is sound and the loop genuinely closes end to end without touching real network or
mutating the real store. Whether it closes *as advertised* — safely, in 60 seconds, with the judge
actually on — does not hold once tested with the real (non-stub) components against the full real
corpus.

## 5. Does the default judge lift signal on no-verifier episodes?

This is the one place the evidence genuinely conflicts, and it matters because it's the roadmap's
headline P1.2 claim ("this is the highest-leverage fix for the reward-crux").

**What the "lift" evidence actually measured.** Every implementation-summary and adversarial-evaluation
number showing a lift (composite rising for 94-100 of 100 real episodes; train/holdout admission
roughly doubling at realistic thresholds; the unit test
`test_loop_default_judge_lifts_a_no_verifier_episode_into_the_training_pool`) used a **synthetic judge
command that deterministically returns a fixed high score**, e.g. `sh -c "printf 'SCORE: 0.9 clear and
correct'"`. That is a legitimate way to test that the *wiring* is correct — a good score really does
flow through the rubric's 0.20 weight into the composite and really can cross a promotion threshold —
but it says nothing about what the real judge does when it reads real trajectory content.

**What the real judge actually did.** One adversarial evaluator, with a genuinely authenticated labeler
in that session, ran `rubric.score_episode` with `judge=None` vs. the real judge on 8 real, substantive
(>=10-step) no-verifier StageWhisper episodes: **8 of 8 deltas were negative** (mean rubric delta
-0.0765) — turning the judge on made the reward signal worse on that sample, not better. This report
independently confirmed the root cause by reading the source: `core/rubric.py:120` slices
`trajectory_text(episode)` to the first `JUDGE_TRAJECTORY_LIMIT = 4000` characters — a **head**
truncation. On real multi-step coding sessions, the first 4000 characters are early exploration; the
fix, diff, or explanation the judge would need to see to score the episode well lives near the end and
is cut off. The same evaluator additionally found the roadmap's own stated premise — "today `judge=None`
makes `rubric` a neutral 0.5" — does not hold on real data either: 0 of 102 real episodes ever hit that
fallback, and the mean unjudged rubric score across the real store was 0.29, not 0.5.

**Verdict.** The judge *mechanism* is sound (weighting, degrade-on-failure, threshold-crossing all work
as designed). The specific empirical claim that the real judge, as shipped, lifts signal on the real
StageWhisper no-verifier corpus is **not supported** by the one real-judge measurement available, and is
directly contradicted by it, with a concrete, source-verified root cause (head-truncation of trajectory
context). No evaluator ran the real (non-stub) judge across the full real corpus, so the population-level
effect size beyond n=8 is unknown — this is a real gap in the evidence, not just a disagreement to wave
away. Before this can honestly be called "done": fix the truncation to keep the trajectory's tail (or
summarize instead of hard-cutting), rerun the real-judge comparison at full real-corpus scale, and add
`judge=True` to `test_flywheel_e2e.py` (or a companion test) so the flagship proof actually exercises the
feature it is cited as evidence for.

## 6. What is stubbed or faked (honest inventory)

- **Every "judge lifts real signal" number in the implementation summaries used a fixed-score judge
  stub**, not the real LLM judge — see §5. This is the single biggest gap between "the tests pass" and
  "the feature does what the roadmap claims" in this whole review.
- **All automated tests avoid real network/LLM calls by design**, per this task's own constraint: fake
  HTTP openers, a local `ThreadingHTTPServer` standing in for real upstreams, `sh -c "printf ..."`
  judge/labeler stand-ins, monkeypatched `mlx`/`tinker` SDK modules. This is correct and was verified
  independently by all three adversarial evaluations grepping every backend test for real hostnames. The
  necessary trade-off: none of the shipped automated tests would have caught the truncation-driven sign
  flip (§5) or the fail-open OpenAI leak (Track A) — both were found only by evaluators manually running
  real commands (a real judge, a real unconfigured `serve` tier) outside the test suite.
- **`trl_reward` evaluator is unit-tested with `trainers.train`/`load_trl_reward_model` monkeypatched**,
  not a real DPO/reward-model training run — reasonable given cost, but means the "co-evolving evaluator"
  claim for that specific backend is unverified past the mock boundary. `local_critic`, by contrast, does
  use real tiny `transformers` models with real forward/backward passes.
- **`mlx`/`tinker` predictors for `--sim-backend`** were verified only via injected fake SDK modules and
  `episodic doctor`'s import-availability check, not real model weights or a real inference call.
- **Side effects left uncleaned in the real StageWhisper store, flagged for the user, not fixed here**:
  one verification pass's `episodic loop` run against the real store's default `--out` wrote a new
  `exports/loop/` directory under `stagewhisper/.episodic/` that the permission system blocked from being
  removed (outside the designated scratch area); a separate adversarial evaluation observed two new
  episode files (`ep_real1.json`, `ep_4c4fc929b978.json`) appear in the real store mid-session with
  timestamps matching that session's first `EPISODIC_HOME`-pointed command, attributed to ambient
  Episodic capture hooks in the surrounding harness rather than anything in the code paths exercised
  (`run_loop`, `doctor`, reward/rubric calls) — plausible but not conclusively ruled out. Neither was
  cleaned up by this report; both are pre-existing state in `../stagewhisper/.episodic/` for the user to
  inspect and decide on.
- **Router "not yet tuned"** is the roadmap's own honest caveat for accuracy; this report's Track A
  finding sharpens it into a concrete safety/privacy default (fail-open to a live third-party API), not
  merely an accuracy gap.

No files in the roadmap diff were modified to produce this report; only this file was written, and only
this repository's working tree was touched (no commits, no pushes, no writes to
`../stagewhisper/.episodic/`).

## 7. Post-review fixes

The four decision-grade findings above were fixed before this work was committed. The sections above are
left intact as the pre-fix record; this section states what changed.

1. **Judge head-truncation (§5, Track B bug 1).** `core/rubric.py` now clips the trajectory to a
   head+tail window (`clip_trajectory`) instead of `[:4000]`, so the judge sees how a session ends — the
   fix, diff, and explanation. Verified on the real store's longest episode (`ep_edb8ff40807b`, 3,943
   steps / 1.5M chars): the old head-only slice dropped the final 200 chars; the new clip keeps them.
2. **`serve` fail-open to public OpenAI (Track A).** `serving/router.py:backend_for` now raises
   `BackendUnavailable` (→ HTTP 503) for any tier that resolves to the `openai` backend without an
   explicit `base_url` or `api_key`, instead of silently proxying to `https://api.openai.com`. Covered by
   new tests in `test_serving_router.py` (unconfigured tier, escalation to an unconfigured frontier,
   distilled `served_ref` without an upstream) and `test_serving_server.py` (503 response).
3. **Re-judging the whole store every run (Track B bug 4, §3 D caveat).** `loop.ensure_reward` now reads
   and writes a persistent judge-reward cache keyed by episode content + a judge signature (command +
   rubric fingerprint + trajectory limit), so unchanged episodes are judged once and reused across runs
   and epochs. Verified on 20 real episodes: run 1 made 40 judge calls, run 2 made 0; the real store was
   not written to. This makes the repeated flywheel runs cheap; the first judged pass over a large store
   is still O(store), so the README claim was relabelled from "60-second quickstart" to "3-command
   quickstart".
4. **Flagship e2e ran with the judge off (Track B bug 3).** `test_flywheel_e2e.py` now sets `judge` on in
   its loop config and asserts the judge-reward cache is written, so the end-to-end flywheel proof
   actually exercises P1.2.

Full suite after the fixes: **523 passed, 1 skipped** (the same unrelated `trl`-availability skip).
