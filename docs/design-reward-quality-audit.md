# Design: reward-quality audit — separating signal from noise in Episodic's dataset

Status: proposal (2026-07-09). Applies OpenAI's "Separating signal from noise in coding evaluations"
(SWE-Bench Pro audit) to Episodic's reward/dataset layer.

## What the article found

OpenAI audited SWE-Bench Pro and estimated **~30% of tasks are broken** — the task's pass/fail does
not reflect real model capability. Their QA pipeline:

1. **Automated filter** over three inputs — the *prompt/instructions*, the *model attempts*, and the
   *grading tests* — flags likely-broken tasks (flagged 286).
2. **Human-supervised agent review** — Codex investigator agents with repo/env access run tests,
   inspect files, and study failure modes to separate *reasonable ambiguity* (resolvable from nearby
   code/conventions) from *true* defects; several independent repeats → a researcher's final judgment.
3. **Human annotation** — 5 engineers per task, forming an **independent judgment first** (from
   problem statement + tests + gold patch) *before* seeing the pipeline output, then a label +
   **severity**, escalating disagreements.

Result: pipeline 200 (27.4%) broken, humans 249 (34.1%), 74% category overlap; humans marked more and
often multiple categories (conservative labeling; biggest gap = low-coverage).

**Failure taxonomy (4):** overly-strict tests (enforce unspecified impl detail → invalidate correct
work), underspecified prompts (hidden tests enforce uninferable requirements), low-coverage tests
(under-check → incomplete fixes pass), misleading prompt (points at wrong behavior). Thesis:
**successes must reflect complete valid solutions and failures must reflect real limitations — not
test/prompt artifacts — and agents scale this data-quality check.**

## Why this is directly about Episodic

Episodic's `reward_vector` *is the grade* for the offline RL / SFT / GRPO loop. Training on
noisy-reward episodes is garbage-in. The article's failure modes map onto Episodic's reward noise:

- **overly-strict / env-blocked / low-coverage tests → noisy `test_pass`** (`reward.terminal_test_signal`
  trusts parsed counts; `blocked_on_env` already catches the env case, nothing catches low-coverage or
  over-strict).
- **underspecified / misleading `intent` → the SFT pair teaches the wrong task** (the user's real
  requirement wasn't the stated one).

**Episodic's unique advantage over a static benchmark:** it now captures *independent* ground-truth
signals a benchmark lacks — mined user feedback, `outcome_hint`, deploy verification, human labels
(shipped in `feedback.py`/`deploydetect.py`). Those can **cross-check the automated (test/rubric)
reward**: when the test-based grade and the human/outcome grade disagree, that episode's reward is
noise. This is exactly "signal vs noise," and Episodic has the human signal the benchmark had to pay 5
engineers to produce.

## Plan: an episode-validity / reward-trust layer (mirrors flag → investigate → label)

### 1. Deterministic contradiction flagger — `core/validity.py` (cheap, stdlib, no LLM)

Per episode, emit `flags[]` from signals already captured, each mapped to the article's taxonomy:

- **`test_reward_false_positive`** — `test_pass` high / tests green, but mined feedback is
  `wrong`/`needed_human_rescue` or `outcome_hint.success == "no"`. → *low-coverage / overly-lenient*.
- **`test_reward_false_negative`** — tests red, but user `accepted_as_is` / merged. → *overly-strict or
  env-blocked*.
- **`outcome_contradiction`** — `outcome` merged/deploy-verified-true but negative feedback; or
  reverted/deploy-verified-false but positive. → reward inputs disagree.
- **`unverified_success`** — high composite but **no** tests AND **no** verified deploy AND **no**
  positive human signal → success is unbacked (analogue of low-coverage).
- **`env_blocked`** — reuse `terminal_test_signal(...)[2]` (already a label) → *tests don't measure
  capability*.
- **`underspecified_intent`** — short/vague `intent` plus an early correction/rescue feedback turn
  (user had to re-specify) → *underspecified/misleading*.
- **`low_coverage_proxy`** — diff edits source files with no accompanying test-file edits and no
  executed tests, yet composite is high → under-checked.

Reduce flags to a **`trust ∈ {high, medium, low}`** tier by severity (contradictions = low).

### 2. Agent-assisted validator (reuse the labeler seam `generate`)

For flagged episodes only (cost scales with the flagged subset, like the article), an investigator
prompt classifies into an Episodic-adapted taxonomy `{overly_strict_tests, underspecified_intent,
low_coverage_tests, misleading_intent, reward_contradiction}` with **severity + evidence**. Borrow the
article's rigor: form judgment from steps + tests + diff + feedback **before** anchoring on the
existing reward (independent-first); run **N independent passes and take a conservative majority**
(the "5 reviewers" / adversarial-verify pattern, which Episodic's workflow tooling already supports).
Provenance-tag every verdict (`source: "mined"`, model, confidence) so a human can veto — the
escalation path.

### 3. Schema (additive)

`episode["validity"] = {trust, flags: [...], category, severity, rationale, source}`. Optional
top-level, defaults `null`; `_validate` already tolerates additive fields.

### 4. The payoff — gate the training set

- `exporters.is_good` / dataset export / the RL `loop` filter **exclude or down-weight `trust: low`
  episodes** — the "~30% broken → filter them out" move, applied to Episodic's own SFT/GRPO inputs.
- Optionally a hard gate: `trust: low` → ineligible for the SFT positive set and GRPO groups; log what
  was dropped (no silent truncation).

### 5. CLI + fits the existing audit ritual

`episodic audit [--validate] [--store]` — flag (and optionally LLM-validate) episodes, print a
signal/noise report with per-category counts and a "% of dataset low-trust" headline (Episodic's own
"~30% broken" number). This complements the existing **daily capture-fidelity audit** with a
**reward-fidelity audit**: capture-fidelity asks "does the episode match what happened?"; this asks
"does the reward match whether it was actually good?"

## Sequencing

1 (deterministic contradiction flags — cheapest, immediate, reuses signals already captured) → 4
(gate the training set — the highest-leverage payoff) → 2 (LLM validator taxonomy for the flagged
subset) → 5 (CLI + report, wire into the daily audit).

## Principles carried from the article

- **Cross-check, don't trust one grader** — the test/rubric reward is validated against the
  independent human/outcome signal; disagreement = noise.
- **Independent-first judgment** — the validator must not anchor on the existing reward.
- **Severity + conservative multi-pass labeling**, taxonomy-driven.
- **Agents for scale, humans for escalation** — provenance-tagged verdicts a human can veto.
- **No silent truncation** — always log how many episodes were filtered and why.
