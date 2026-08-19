# Roadmap: best-in-class OSS for self-improving coding agents

## Context

Two external references set the target. **Experiential Labs / World Model Optimizer** (YC: "frontier-quality,
50%+ cheaper endpoints by continuously improving smaller models on simulations built from production agent
traces") shows the path to business viability: traces → simulation → router → distilled OSS model → an
OpenAI-compatible endpoint that improves over time. **The Red Queen Gödel Machine** (arXiv:2606.26294) is the
algorithmic continuation: recursive self-improvement under *non-stationary* utilities — make evaluation part of
the loop and co-evolve the evaluator with the agent (controlled utility evolution over epochs; a cheap
agent-as-a-judge code-review signal lifts pass rate over static-verifier SOTA).

Episodic is already the front half of WMO and shares its stack: it ingests OpenTelemetry traces
(`collector/otel.py`, port 4318), distills via Tinker (`trainers/tinker.py`), and runs a
partition→export→train→replay-eval→promote loop (`loop/__init__.py`). The missing pieces are a **serving spine**
and a **co-evolving evaluator** — not the learning core. This roadmap closes that gap.

## Decisions

- **Phase 1 anchor:** close the flywheel — serve the promoted model + turn the agent-as-a-judge on by default.
- **Serving architecture:** thin, backend-agnostic proxy/router. Episodic hosts the OpenAI-compatible surface
  and routing only; generation runs on external backends (vLLM / Ollama / Tinker / any OpenAI-compatible
  `base_url`). Keeps the stdlib-core ethos and avoids coupling to one runtime.

## Phase 1 — Close the flywheel

**Status: done.** `episodic loop --execute` → promote → `episodic serve` → OTLP ingest → next
`episodic loop` is wired end to end and covered by `tests/test_flywheel_e2e.py` against real
StageWhisper episodes (skipped, not failed, when that store isn't present). See
[`docs/flywheel.md`](flywheel.md) for the walkthrough.

Goal: `episodic loop` → promote → `episodic serve` → the endpoint's own traffic is captured back via OTLP
ingest → next `episodic loop`. A self-contained, continuously-improving system, mostly wiring existing parts.

### P1.1 — `episodic serve`: thin OpenAI-compatible proxy/router — done
- New package `src/episodic/serving/`. Reuse the `ThreadingHTTPServer` pattern from `dashboard/server.py`.
- Endpoints: `POST /v1/chat/completions` (streaming + non-streaming), `GET /v1/models`.
- Backend adapters via a small registry mirroring `trainers/__init__.py` (`register()`/`get()`/`available()`):
  `openai` (any base_url), `ollama`, `vllm`, `tinker`. The server forwards; it never holds weights.
- Reads the latest promotion pointer to know which model is the "improved" tier. Requires promotion to record a
  **servable reference**, so extend the loop's `promoted.json` (`loop/__init__.py:207-212`) with a
  `served_ref` (upstream model id / path an external backend can load), not just a local `model_dir`.
- Minimal router: two tiers — distilled (promoted) vs frontier fallback — default to distilled, escalate on a
  cheap signal (leave the learned cost-aware router to Phase 2).

### P1.2 — Agent-as-a-judge on by default (the RQGM cheap win) — done
- Wire `reward_vector(episode, judge=...)` (`core/reward.py:152`) with a default judge from
  `rubric.openrubrics_judge(generate)` (`core/rubric.py:135`); today `judge=None` makes `rubric` a neutral 0.5.
- Provide the `generate` adapter reusing the labeler LLM-call path (`core/feedback.py`); respect the labeler
  auth gotcha (API key / token, not the unauthenticated `claude -p`).
- Enable it in the default loop config so `rubric` (weight 0.20) carries a real code-review signal. This is the
  highest-leverage fix for the reward-crux and the 94→17 verifier funnel — it gives signal on the 77/94
  episodes that have no runnable test (`no_verifier`), complementary to tests, not a replacement.

### P1.3 — Flywheel wiring + quickstart — done
- Document and wire the closed loop end to end; ship a 3-command quickstart at parity with `wmo`:
  `episodic loop --execute` → `episodic serve` → point your agent at the endpoint.
  (README's "3-command quickstart" + `docs/flywheel.md`.)

### Verification (Phase 1)
- Unit: proxy request/response against a mock upstream; backend-adapter registry; router escalation;
  `served_ref` read from `promoted.json`; default-judge wiring (mock `generate` → `rubric` becomes non-neutral).
- E2E on **real StageWhisper episodes** (`../stagewhisper/.episodic`, per project practice — not just
  fixtures): run `loop` with a `command`-trainer stub, promote, `serve`, hit `/v1/chat/completions` through the
  proxy to a mock/Ollama upstream, confirm a trace is ingested and a new episode appears — proving the flywheel
  closes at small scale (not 1T).

## Phase 2 — Router + co-evolving evaluator (RQGM)

**Status: wired, not yet tuned.** All three pieces exist and are unit-tested (`tests/test_loop.py`,
`tests/test_loop_evaluator.py`, `tests/test_serving_difficulty.py`); none has been validated
against a large real workload yet, so treat the defaults as a first pass.

- **Cost-aware router (A2) — done, opt-in:** `serving/difficulty.py:learn_router` fits a small logistic model
  from reward/validity difficulty signals; `episodic loop --router` trains it and writes `router_model.json`,
  `episodic serve --router-model` consumes it. `serve`'s default routing is still the plain length heuristic
  (`serving/router.py:default_escalate`) — this is WMO's 50%-cheaper lever, not yet WMO's tuning.
- **Epoch-structured loop (B2) — done:** `run_loop` (`loop/__init__.py`) takes `epochs` and drives
  `_run_epoch` per iteration; the evaluator is fixed within an epoch and refreshed only at epoch boundaries
  (`loop/evaluator.py:refresh`), preserving per-epoch promotion guarantees while the objective evolves (RQGM
  controlled utility evolution). `epochs=1` (the default) is byte-for-byte the pre-epoch manifest shape.
- **Learned evaluator (B3) — done:** `loop/evaluator.py:build` selects `rubric_judge` (default), `local_critic`
  (`trainers/critic.py:LocalCritic`, retrained in-process from `export(..., "reward")` rows), or `trl_reward`
  (a `trl-reward`-trained sequence-classification model, loaded via `trainers/critic.py:load_trl_reward_model`),
  refreshed at epoch boundaries — the co-evolution that resists reward-hacking and stagnation.

## Phase 3 — Learned simulator (WMO step 1)

**Status: done, opt-in, and validated against a real trained WM on the real StageWhisper corpus.** Exercised by
`tests/test_worldmodel_env.py`, `tests/test_worldmodel_inference.py`, `tests/test_worldmodel_validate.py`,
`tests/test_wm_validate_cli.py`, and the rollout/turing additions to `tests/test_worldbench.py` /
`tests/test_fidelity.py`; a required `episodic doctor` check exercises the env with the synthetic `prefix`
predictor (no mlx/tinker dependency).

- `worldmodel/env.py:WorldModelEnv` + `rollout()` give the dataset-shaper a real `reset/step` API, feeding the
  model's own predicted observations back into context (closed loop) instead of always teacher-forcing on
  ground truth, bounded by the same history-budget truncation `exporters.segment_episode` uses (the budget now
  lives once in `worldmodel/__init__.py:HISTORY_BUDGET`; `env.py` imports it instead of redefining it).
  `worldmodel/inference.py:mlx_predictor` / `tinker_predictor` provide real model inference for it.
- `episodic loop --sim-prefilter` uses a `WorldModelEnv` rollout + `fidelity.trajectory_score` pass to rank
  which holdout episodes are worth the fixed, expensive real-`replay` budget; the actual promote/keep_base
  decision is still driven entirely by real replay-eval results, never the sim.
- `fidelity.trajectory_score` aggregates per-turn `score_observation` into a trajectory-level composite/drift
  figure; `worldbench.rollout_bench` / `rollout_turing_test` are the trajectory-horizon siblings of
  `run_bench` / `turing_test`, reusing the same discriminator unchanged.
- **Bugfix — unbounded history growth:** `worldmodel/__init__.py:render_history` (and its callers
  `expand_turns`/`wm_samples`) previously rebuilt an ever-growing, untruncated context string per turn — O(n²)
  in an episode's step count with no budget, unlike `exporters.segment_episode`/`WorldModelEnv._context`.
  Reproduced live: exporting the real 104-episode StageWhisper corpus to `wm` format OOM'd (SIGKILL) on the 20
  episodes with ≥200 steps — the exact bucket holding 15/16 of the corpus's reward≥0.5 "good" episodes. Fixed
  by giving `render_history` a `history_budget` param (default 4000, matching `HISTORY_BUDGET`) and rewriting it
  to walk backward from the tail and stop once enough characters are collected, so the cost per call is
  `O(history_budget)` instead of `O(step_count)` — verified byte-identical to the naive "build full context then
  slice" reference via randomized differential testing. Exporting all 48 real train episodes (including 15 with
  200–3943 steps) to `wm` format went from 86s (post-memory-fix, pre-algorithmic-fix) to 1.4s.
- **CLI wiring for a trained WM:** `MLXSFTTrainer.train()` always emits a LoRA adapter dir, but neither
  `episodic worldbench --backend mlx` nor `episodic loop --sim-backend mlx` exposed a way to load one —
  `--adapter-path` / `--sim-adapter-path` close that gap.
- **`episodic wm-validate` (new):** partitions real episodes via `loop.split_episodes` (per-episode hash
  holdout — the real corpus is single-source, so `worldmodel.ood_split`/`--source-holdout` would be
  structurally meaningless here), trains a WM locally via `mlx-sft` (`--execute`, no Tinker, no network billing)
  or loads an existing adapter (`--adapter-path`), scores the holdout with `worldbench.rollout_bench` /
  `rollout_turing_test` against `oracle`/`prefix`/`empty`, and optionally (`--replay-correlate`, needs
  `--execute`) correlates the sim rollout composite against a real, offline replay-eval score via the new
  `worldmodel/validate.py` (`sim_trajectory_score`/`sim_scores`, `offline_replay_scores` — clones the episode's
  own local `repo_state.root` instead of dialing `remote_url`, then git-applies the episode's own recorded
  diff as the "oracle" candidate and runs the captured test command — and stdlib-only `pearson`/`spearman`
  in `correlate`).
  - **Real-corpus fidelity run** (48 train / 16 holdout episodes from the real StageWhisper store, `mlx-sft` on
    `HuggingFaceTB/SmolLM2-135M-Instruct`, 60 LoRA iters over 23,226 turn rows, `max_turns=5`): mean rollout
    composite — oracle 1.0, trained 0.44, prefix/empty 0.09 (identical); Turing indistinguishability — oracle
    1.0, trained 0.75, prefix/empty 0.625. Ordering (`oracle > trained > prefix == empty`) holds on both fidelity
    and indistinguishability, on a real 16-episode holdout that now includes the previously-OOM-ing big
    episodes.
  - **Sim↔real correlation (honest negative result):** of the 16 holdout episodes, only 5 have a genuinely
    captured, correctly-scoped test command with `git_available(root)` true (`exporters._captured_verifier`
    with the `total is not None` guard) — all 5 are Cargo/Rust builds with no build cache. Real replay-eval score
    (oracle-diff applied + recorded test command, 120s timeout) was bimodal (0.6 when the build finished and
    passed, 0.0 when it didn't — i.e. mostly a build-timeout artifact, not a code-quality signal) against a
    narrow-range sim score (0.37–0.49). Pearson r = **-0.08**, Spearman ρ = **0.29**, n = 5 — statistically
    meaningless at this sample size, and the "real" axis here is confounded by Cargo compile time far more than
    by trajectory fidelity. This is reported as a real, currently-weak finding, not a validated lift: the sim
    rollout score and this real-replay-eval score are answering genuinely different questions (does the WM's
    predicted trajectory look like the recorded one, vs. does an oracle diff cleanly rebuild in 120s), and n=5
    real-repo Rust builds is nowhere near enough to draw a general conclusion either way.

## Positioning (throughout) — done

Reframe README/docs around the flywheel: capture → reward → distill → serve → co-evolve. README opens with
this framing and a 3-command quickstart; `docs/flywheel.md` is the deep dive, including an honest
Episodic-vs-WMO comparison. Publishing a benchmark that shows the flywheel lifting a small model on real
captured coding sessions is still open — nothing here claims a lift number yet.
