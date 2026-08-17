# The flywheel

Two external references set the shape of this: **World Model Optimizer** (traces → simulation →
router → distilled model → an OpenAI-compatible endpoint that improves over time) shows the path
to something a team would actually run in production, and the **Red Queen Gödel Machine**
(arXiv:2606.26294) is the algorithmic continuation — recursive self-improvement under
*non-stationary* utilities, where the evaluator co-evolves with the agent instead of staying a
fixed static check.

Episodic already had the front half of that loop: it ingests OpenTelemetry traces
(`collector/otel.py`), distills via a pluggable trainer, and runs a
partition → export → train → replay-eval → promote loop (`loop/__init__.py`). What was missing
was a serving spine and a co-evolving evaluator. This page documents what closes that gap.

## The five stages

**Capture** — `collector/otel.py` and the Claude Code / Codex hooks record every session
invisibly: prompt, repo state, tool calls, edits, commands, tests, approvals, final diff. Nothing
here is new to the flywheel; it's the same capture path every other Episodic feature builds on.

**Reward** — `core/reward.py:reward_vector` turns an episode into `test_pass`, `human_label`,
`outcome`, `cost_efficiency`, `edit_focus`, and `rubric` components plus a `composite`. The
`rubric` component can be scored by an agent-as-a-judge callable
(`core/rubric.py:openrubrics_judge`); `episodic loop` builds one by default
(`core/rubric.py:default_judge`, wired through `loop/__init__.py:_resolve_judge`) so episodes with
no runnable test still get a real code-review signal instead of a neutral 0.5. `--no-judge` turns
it off; `--judge-cmd` points it at any command that reads a prompt on stdin and returns a score
(needs `ANTHROPIC_API_KEY` / `CLAUDE_CODE_OAUTH_TOKEN`, or point it at something else entirely).

**Distill** — `episodic loop` (`loop/__init__.py:run_loop`) filters episodes above a composite
threshold, exports a training set, trains a candidate through the trainer registry
(`trainers/__init__.py`, e.g. `trl-sft`, `trl-sao`, `mlx`, `tinker-sao`, `command`), and
replay-evaluates the candidate against the base model on held-out episodes. It promotes only if
the candidate's real replay-eval reward beats the base model's by `promote_margin`. A promoted run
writes `<out>/promoted.json` with a `served_ref` — the model id or path an external backend can
actually load.

**Serve** — `src/episodic/serving/` is a thin, backend-agnostic OpenAI-compatible proxy/router
(`episodic serve`), the same `ThreadingHTTPServer` shape as `episodic dashboard`. It resolves
`served_ref` from the latest `promoted.json` (`serving/router.py:resolve_served_ref`), routes
`POST /v1/chat/completions` and `GET /v1/models` to it by default, and escalates to a configured
"frontier" tier on a cheap signal. It never holds weights itself; the actual backend (`openai`,
`ollama`, `vllm`, `tinker`) is whatever is running your promoted checkpoint.

**Co-evolve** — the judge that scores the `rubric` component doesn't have to stay static.
`episodic loop --epochs N` restructures the run into epochs: the evaluator (judge or critic) is
fixed within an epoch and only refreshed at epoch boundaries (`loop/evaluator.py:refresh`),
following RQGM's controlled-utility-evolution shape rather than reward hacking against a moving
target every step. `--evaluator local_critic` or `--evaluator trl_reward` swap the plain
LLM-judge for a small trained reward model (`trainers/critic.py:LocalCritic`, or a
`trl-reward`-trained sequence-classification model) that retrains on each epoch's training split.
`episodic loop --router` is the adjacent cost-aware piece: it learns which requests are worth
escalating to the frontier tier from accumulated reward/validity difficulty signals
(`serving/difficulty.py:learn_router`), instead of the plain length heuristic `serve` falls back
to by default.

## Worked example

```bash
episodic loop --config loop.json --execute
```

Inspect what got promoted:

```bash
cat "$(python3 -c "from episodic import paths; print(paths.exports_dir() / 'loop' / 'promoted.json')")"
# {"model_dir": "...", "served_ref": "...", "candidate_mean": 0.71, "base_mean": 0.58, ...}
```

Serve it and point an agent at it:

```bash
episodic serve --port 8000
# any OpenAI-compatible client -> http://localhost:8000/v1
curl http://localhost:8000/v1/chat/completions \
  -d '{"messages": [{"role": "user", "content": "fix the failing test"}]}'
```

If that agent has OTel telemetry pointed at Episodic's collector, its own traffic gets captured
back through the normal hook path, and `episodic list` shows the new episode once that session
finalizes — the same episodes the next `episodic loop --execute` trains and replay-evaluates on.

`tests/test_flywheel_e2e.py` runs exactly this sequence end to end — `loop.run_loop` (with a
`command` trainer stub so no real training happens) promoting a candidate against real episodes
from a sibling project's store, `serving.server.build_server` serving it through a mocked
upstream, `collector.otel.build_otel_server` ingesting a synthetic usage payload, and
`service.finalize_session` producing a new episode with that usage folded in. No real network or
LLM call happens anywhere in the test; every upstream is injected. It's skipped automatically when
that real store isn't present on disk — `tests/test_loop.py`, `tests/test_serving_*.py`, and
`tests/test_otel.py` cover the same wiring against synthetic fixtures so the suite stays green
everywhere else.

## Episodic vs World Model Optimizer

| | Episodic | WMO |
| --- | --- | --- |
| Loop shape | capture → reward → distill → serve → co-evolve | traces → simulation → router → distilled model → endpoint |
| Distribution | open source, runs fully local or against any backend you point it at | hosted product |
| Serving | backend-agnostic thin proxy (`openai`/`ollama`/`vllm`/`tinker` adapters); Episodic never holds weights | runs the endpoint for you |
| Router | fixed two-tier distilled/frontier switch, with an optional learned escalation model (`--router`) | a learned, cost-aware router is the core lever |
| Evaluator | agent-as-a-judge by default, optional epoch-refreshed local critic / trained reward model | not public |

The shapes match because Episodic is deliberately built as the front half of the same idea. The
gap that's left is depth, not structure: WMO's router and simulator have presumably seen a lot
more production traffic than anything exercised here so far.

## Caveats

- `serve` forwards to a real backend; it doesn't train or host weights. You still need an
  `openai`/`ollama`/`vllm`-compatible server or a Tinker sampler actually running the promoted
  checkpoint.
- The default judge shells out to an LLM per scored episode. It's opt-out (`--no-judge`), and a
  failed or unauthenticated judge call degrades that one criterion to "not applicable" rather than
  failing the run (`core/rubric.py:safe_judge`) — but it's still a real cost per episode.
- `--router` and `--evaluator local_critic/trl_reward` are new and haven't been validated against
  a large real workload; treat them as a first pass, not a tuned default.

## Reuse

Serving reuses the trainer-registry pattern (`register()`/`get()`/`available()`) from
`trainers/__init__.py` for its backend adapters, and the `ThreadingHTTPServer` handler shape from
`dashboard/server.py`. The judge reuses the exact labeler LLM-call path from `core/feedback.py`
(`command_generate`), so it inherits the same auth behavior as `episodic label`. The epoch loop
and evaluator refresh reuse the existing exporters (`exporters.export(..., "reward" | "dpo", ...)`)
and `trainers/critic.py` pretraining helpers rather than adding new dataset shapes.
