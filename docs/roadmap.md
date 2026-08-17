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

**Status: done, opt-in.** Exercised by `tests/test_worldmodel_env.py`, `tests/test_worldmodel_inference.py`,
and the rollout/turing additions to `tests/test_worldbench.py` / `tests/test_fidelity.py`; a required
`episodic doctor` check exercises the env with the synthetic `prefix` predictor (no mlx/tinker dependency).

- `worldmodel/env.py:WorldModelEnv` + `rollout()` give the dataset-shaper a real `reset/step` API, feeding the
  model's own predicted observations back into context (closed loop) instead of always teacher-forcing on
  ground truth, bounded by the same history-budget truncation `exporters.segment_episode` uses.
  `worldmodel/inference.py:mlx_predictor` / `tinker_predictor` provide real model inference for it.
- `episodic loop --sim-prefilter` uses a `WorldModelEnv` rollout + `fidelity.trajectory_score` pass to rank
  which holdout episodes are worth the fixed, expensive real-`replay` budget; the actual promote/keep_base
  decision is still driven entirely by real replay-eval results, never the sim.
- `fidelity.trajectory_score` aggregates per-turn `score_observation` into a trajectory-level composite/drift
  figure; `worldbench.rollout_bench` / `rollout_turing_test` are the trajectory-horizon siblings of
  `run_bench` / `turing_test`, reusing the same discriminator unchanged.

## Positioning (throughout) — done

Reframe README/docs around the flywheel: capture → reward → distill → serve → co-evolve. README opens with
this framing and a 3-command quickstart; `docs/flywheel.md` is the deep dive, including an honest
Episodic-vs-WMO comparison. Publishing a benchmark that shows the flywheel lifting a small model on real
captured coding sessions is still open — nothing here claims a lift number yet.
