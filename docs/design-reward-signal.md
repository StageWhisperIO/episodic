# Design: closing Episodic's reward-signal gap

Status: implemented (2026-07-08), uncommitted. All four workstreams shipped and validated with a real
Haiku pass on real sessions: `core/feedback.py` (LLM labeler), `core/deploydetect.py` +
`deployments[]`, `core/segment.py` + `episodic segment`, and `--store` self-capture on `episodic
label`. Operational caveat: the default `claude -p` labeler is not authenticated as a headless
subprocess — set `ANTHROPIC_API_KEY`/`CLAUDE_CODE_OAUTH_TOKEN` or point `--cmd`/`$EPISODIC_LABELER_CMD`
at an authenticated command. The sections below are the design of record.

## Problem

Episodic faithfully records the **state/action** side of a trajectory (prompts → tools → edits →
diffs → commands, now with real exit codes and failed-command reconstruction) but captures almost
none of the **reward** side. The learning signal — did it work, did it ship, did the user accept it,
did it need rework — is either absent or defaulted to a neutral prior.

Evidence (live session `ed925780`, this project's own dev session): 2354 events, 42 user prompts,
resumed 26×, a dozen unrelated tasks fused into one episode with one reward vector. Every user
correction (*"wait, I'm confused"*, *"report what went wrong"*) and acceptance (*"yes"*, *"push"*,
*"commit to main"*) is captured as raw `user_prompt` text and then dropped: `human_feedback: []`,
`labels: []`, `outcome: open`, composite ≈ 0.43 from neutral priors.

Where the neutrality comes from (`core/reward.py`):
- `human_label` → `0.5` whenever `human_feedback` is empty (it is always empty; only the explicit
  `episodic mark` CLI ever populates it).
- `outcome` → `OUTCOME_SCORES["open"] = 0.0` → normalized `0.5` whenever a PR/deploy isn't linked.
- Net: `test_pass`, `outcome`, `human_label`, `cost_efficiency` all fall back to `0.5`; only `rubric`
  (0.2 weight) and `edit_focus` carry real information. A deploy-broke-then-fixed episode is reward-
  indistinguishable from a clean one-shot success.

Goal: convert already-captured signal into real reward, and start capturing the outcome signal we
don't yet see. Four workstreams, sequenced by leverage.

---

## Running the labeler (auth)

The labeler shells out to a configurable command (`--cmd` or `$EPISODIC_LABELER_CMD`), defaulting to
`claude -p --model claude-haiku-4-5-20251001`. That default is **not authenticated as a headless
subprocess** — the interactive Claude Code OAuth is not inherited by child processes, so a bare
`claude -p` returns "Not logged in" and the pass yields zero labels. To run it, provide credentials one
of these ways:

- `ANTHROPIC_API_KEY=sk-... episodic label --all --save` (the CLI is picked up by `claude -p`), or
- `CLAUDE_CODE_OAUTH_TOKEN=... episodic label ...`, or
- point at any authenticated stdin→stdout command: `--cmd "codex exec -q"`, `EPISODIC_LABELER_CMD="ollama run llama3.1" episodic label ...`.

`episodic label --raw` prints the raw model output for one session to diagnose auth/parse problems.
Everything runs out-of-band with `EPISODIC_DISABLE=1` set, so the labeler's own model calls are not
re-captured.

## 1. Implicit feedback mining → `human_feedback` + `labels` (start here)

The signal already exists as text; this is post-processing, no new capture plumbing.

**Detection.** Over the ordered `user_prompt` events, classify each turn's *relationship to the
preceding agent action* into the existing `FEEDBACK_LABELS`:
- Negative — correction/dissatisfaction/distrust: *"still broken", "why is it …", "are you sure?",
  "that didn't work", "no,", "report what went wrong", "revert"* → `wrong` / `needed_human_rescue`.
- Friction — scope/speed complaints: *"too much", "why so slow", "that's overkill"* → `too_broad` /
  `too_slow`.
- Positive — acceptance/green-light: *"yes", "lgtm", "push", "commit", "ship it", "perfect"* →
  `accepted_as_is`; acceptance that follows a requested change → `accepted_after_edits`.
- Neutral — new task / question / info → no label.

**Who labels: the in-loop LLM, not a regex.** Episodic runs inside a coding agent, so the labeler is
the same model, invoked out-of-band. This reuses the existing optional-judge pattern
(`reward_vector(episode, judge=None)`, `rubric.score_episode(episode, judge=judge)`) — feedback
labeling is a judge that emits a label instead of a score, so it fits the architecture rather than
introducing a new one. A lexical pass is demoted to an optional cheap pre-filter/fallback, not the
primary path; classifying *"are you 100% sure?"* or *"on production the gap is much larger"* as
feedback-on-my-last-action is exactly what regex is bad at and an LLM is good at.

Constraints on the labeler:
- **Separate invocation, not inline self-assessment.** The working agent must not label its own work
  in its own turn: (a) self-serving bias — a model is reluctant to stamp its own output `wrong`; (b)
  inline meta-work pollutes context and makes Episodic visible, breaking its invisibility premise. Use
  a fresh headless invocation (same model, or cheaper e.g. Haiku) with a neutral/adversarial grader
  prompt.
- **Batch at finalize** (rides the existing `SessionEnd`/`Stop` finalize hook in `collector/hook.py`;
  transcript already parsed by `transcript.py`). One call per session sees the whole arc, so it can
  distinguish *complained → fixed → accepted* (`accepted_after_edits`) from a bare `wrong` — which a
  per-turn `Stop` labeler cannot. Per-turn labeling stays available for freshness if we want it.
- **Cache the label into the episode JSON** so it is stable for RL — non-determinism exists only at
  label time, exactly like the frozen rubric-judge score.

Each emitted feedback item keeps provenance: `{ts, label, note, source: "mined", confidence,
evidence_step_index, model}`.

**Schema.** `FEEDBACK_LABELS` already covers the classes. Extend the `human_feedback` item with
optional `source` (`"explicit" | "mined"`), `confidence`, and `evidence_step_index` (all additive;
`_validate` ignores unknown-but-declared fields — same pattern as the `reconstructed` marker). Add a
label like `mined_feedback` to `labels` so mined vs explicit episodes are filterable.

**Reward impact.** `_human_label` starts firing. To stay safe: weight mined items by confidence, and
consider a separate `has_mined_feedback` flag so we can down-weight mined vs explicit signal in
`WEIGHTS` if needed. Nothing else in the vector changes.

**Migration.** Runs as a re-derivation over stored `events.jsonl` → rebuild episodes; retroactively
labels the entire back-catalog (incl. this session). Reversible: mined items are tagged `source:
"mined"` and can be stripped.

**Risk.** (a) Self-serving bias if the working agent grades itself → separate adversarial invocation.
(b) Hallucinated/over-eager labels → require the labeler to cite the `evidence_step_index` and a short
rationale, weight by returned `confidence`, and keep every item `source: "mined"` so a human can
audit/veto. (c) Cost/latency → one cheap batched call per session at finalize, cached.

---

## 2. Outcome & deployment tracking (env → prod)

Turn `outcome: open` into ground truth. PR/merge linking partially exists (`github-linker/`); the gap
is **deploys and their success/failure**, especially promotion up the chain.

**Detection.**
- Commits/PRs/CI: extend the existing linker to stamp `outcome.commit/pr_*/ci_status/merged/reverted`.
- Deploys: recognize deploy actions (`wrangler pages deploy`, `vercel`, `netlify deploy`, `gh
  workflow run`, `kubectl apply`, env-promotion scripts) from `commands[]` and CI events. Record a new
  `deployments[]`: `{ts, target_env (dev|staging|prod), method, ref, status, verified}`.
- Verification-of-success: correlate post-deploy checks (a curl/browser hit against the prod URL, a
  smoke test) and — critically — later user turns (workstream 1) that say the deploy *didn't* fix the
  issue → `deployments[].verified = false`, feeding a negative outcome.
- Cross-session / backgrounded deploys depend on the
  [background-command capture gap](../.episodic) fix — deploys are frequently backgrounded or in CI,
  so their real result never lands today.

**Schema.** New top-level `deployments[]`. Extend `outcome` with `deployed: bool`,
`deploy_env: str|null`, `deploy_verified: bool|null`. Add `OUTCOME_STATUSES` member(s) if we want a
`deployed_failed` terminal state, or express it purely via `deploy_verified=false` + `caused_regression`.

**Reward impact.** New `deploy` component (or fold into `outcome`): verified prod deploy → strong
positive; deployed-then-reverted or deploy-verified-false → strong negative. This is the highest-value
reward once available; give it real weight in `WEIGHTS` (rebalance so the six existing components make
room).

**Migration.** Linker already backfills asynchronously (`linked_at`); deploy detection re-derives from
stored commands. Prod-verification-via-user-feedback needs workstream 1.

---

## 3. Sub-trajectory segmentation

Stop fusing multi-task / multi-resume sessions into one episode+reward.

**Detection.** Segment the event stream at boundaries: a new top-level `user_prompt` that starts an
unrelated task (topic shift), a terminal outcome (commit/push/deploy) followed by a new ask, or a
`session_start` after a long gap. Heuristic first (prompt-similarity + terminal-action markers),
optional judge for boundary confirmation. This session's 42 prompts / 26 resumes is the canonical
stress case.

**Schema.** Introduce sub-episodes: either child records referencing a parent `session_id`, or a
`segments[]` index (`{start_step, end_step, intent, outcome, reward_vector}`) on the episode. Prefer
child episodes reusing `EPISODE_SCHEMA` so exporters/trainers work unchanged; the session becomes a
container. `episode_id_from_session` grows a segment suffix.

**Reward impact.** Reward attaches per attempt, not per fused session — so attempt-1 (broken deploy)
and attempt-2 (fix) get distinct scores. Enables the segmented-SFT / per-attempt-advantage RL work
already prototyped (see the RL-loop notes).

**Migration.** Re-derivation splits existing sessions; keep the fused episode id stable as the parent
so nothing dangles. Depends on 1–2 to make per-segment reward meaningful.

---

## 4. Self-capture of Episodic's own dev sessions

Mostly falls out of 1–3. This repo is already captured (`.episodic/` here, plugin enabled globally);
it just needs the reward layer to be meaningful. Value: Episodic dogfoods its own RL signal, and
sessions like this one (rich, explicit user feedback about the product) become first-class training
data instead of being missed.

**Action.** Once 1–3 land, add Episodic's own store to the dataset export set and confirm mined
feedback + outcomes populate on its episodes. No new mechanism.

---

## Sequencing & principles

1 → 2 → 3, with 4 riding on all three. Rationale: 1 is cheap, retroactive, zero new capture, and
immediately converts the back-catalog (including this conversation) into reward-bearing data; 2 adds
the strongest ground-truth reward but needs new capture + the background-command fix; 3 restructures
episodes and wants 1–2's signal to be worth attaching per-segment.

Cross-cutting principles: keep detection **deterministic-first, judge-optional** (stdlib-core);
**provenance-tag** every derived signal (`source`, `confidence`, evidence pointer) so it's auditable
and reversible; make all schema additions **additive**; and implement everything as **re-derivation
over stored `events.jsonl`** so the whole history upgrades and nothing is destructive.
