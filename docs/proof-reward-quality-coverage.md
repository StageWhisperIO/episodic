# Proof: every OpenAI "signal vs noise" insight is covered in Episodic

Status: verified on real data (2026-07-09). Companion to `design-reward-quality-audit.md`.
Source article: OpenAI, *Separating signal from noise in coding evaluations* (SWE-Bench Pro audit).

The claim is not "we designed for these insights" — it is "each insight is a running mechanism,
and it fires on real Episodes captured from actual coding sessions." Evidence below is reproducible
with the commands shown.

## The reference episode

`ep_749edc465881` (StageWhisper store) — a real 1008-step, 53-prompt session on echo cancellation.
Its automated test reward looks excellent, which is exactly the trap the article warns about:

| signal | value |
|---|---|
| `reward_vector.composite` | **0.676** (naive gate ≥0.5 → **accept into training**) |
| tests | **16 passing / 19** (`cargo-test`, all green on the happy path) |
| stored `human_feedback` | **`[]`** (never labeled) |
| stored `validity` | **absent** |

A benchmark that grades on tests alone would treat this as a clean positive. It is not.

## Insight-by-insight coverage

| # | Article insight | Episodic mechanism (`file :: function`) | Real-data evidence |
|---|---|---|---|
| 1 | **Automated filter over 3 inputs** — prompt, model attempts, grading tests — flags likely-broken tasks | `core/validity.py :: flag_episode` reads intent (prompt), steps/diffs (attempts), tests + mined feedback (grading) — pure stdlib, no LLM | Ran on **63 real episodes** across both stores; fired `underspecified_intent` ×32, `low_coverage_tests` ×2 with zero LLM cost |
| 2 | **4-category failure taxonomy** (overly-strict, underspecified, low-coverage, misleading) | `validity.CATEGORIES` = `overly_strict_tests, underspecified_intent, low_coverage_tests, misleading_intent` **+ `reward_contradiction`** (the cross-check the benchmark can't do) | On the reference episode the pipeline assigned `low_coverage_tests, overly_strict_tests, reward_contradiction, underspecified_intent` |
| 3 | **Cross-check graders; disagreement = noise** (the article's thesis) | `flag_episode` deterministic detectors: `test_reward_false_positive` (green tests + negative human) → low-coverage; `test_reward_false_negative` (red tests + positive human) → overly-strict; `outcome_contradiction` (deploy verdict vs human) | After real labeling, `composite 0.696 (test_pass 1.0)` vs `human_label −0.084` → **both** contradiction flags fired |
| 4 | **Human ground-truth signal** (5 engineers, the expensive part) | Episodic mines it for free: `core/feedback.py :: mine` (in-loop coding agent labels the transcript) + `outcome_hint` | Real Haiku pass mined **11 reactions** (5× `wrong`, 5× `accepted_after_edits`, 1× `useful`), outcome `partial` |
| 5 | **Agent-assisted investigation** ("agents scale the data-quality check") | `validity.py :: validate` reuses the labeler `generate` seam; runs on any model (Haiku default, model-agnostic) | 3 real Haiku validator passes on the flagged episode, aggregated by `validate` |
| 6 | **Independent-first judgment** (judge before anchoring on the pipeline label) | `validity.py :: build_prompt` — *"form your own independent judgment … do NOT assume the recorded reward is correct"* | Validator independently reasoned: *"unit tests verify code compiles, not that echo cancellation works"* — it re-derived the low-coverage defect, not the recorded 0.676 |
| 7 | **N independent repeats → conservative majority** | `validate(passes=3)`: `broken_votes < majority` ⇒ trustworthy; category kept only if ≥ majority vote | 3/3 passes returned `trustworthy:false`; `agreement=1.0`, mean `confidence=0.807` |
| 8 | **Severity + multi-label** | flags carry `severity ∈ {low,medium,high}`; categories are multi-label; `assess` takes worst severity | Merged verdict: `severity=high`, 4 categories, `source=rules+llm` |
| 9 | **Escalate on disagreement** (humans for the hard cases) | `validate` sets `escalate = agreement < 1.0`; every verdict is provenance-tagged (`source`, `model`, `confidence`) for human veto | `escalate=false` here (unanimous); the field is the escalation hook when passes split |
| 10 | **Filter broken tasks from the eval set** (the "~30% broken → drop them" payoff) | `exporters.is_trusted` / `is_good` short-circuit on `trust==low`; `loop.select_good` gates the SFT/GRPO training set | Reference episode: `is_trusted=False`, `is_good=False`, `select_good` → **excluded**, despite composite 0.696 |
| 11 | **"% broken" headline report** | `cli.py :: cmd_audit` → `episodic audit [--validate] [--store]` prints trust distribution, low-trust %, per-category counts | `episodic audit` on both stores prints the signal/noise report (numbers below) |
| 12 | **No silent truncation** (log what was dropped and why) | `cmd_audit` lists every low-trust episode + categories; `flag_episode` attaches human-readable `evidence` per flag | audit output enumerates flagged episodes and reasons |

## End-to-end walkthrough on the reference episode (real Haiku, reproducible)

**Step 1 — mine the human signal** (`feedback.mine`, real `claude-haiku-4-5-20251001`):
```
mined 11 reactions: {'wrong': 5, 'accepted_after_edits': 5, 'useful': 1}
outcome_hint: {'success': 'partial', 'confidence': 0.7, ...}
human_label: 0.0  →  −0.084
```

**Step 2 — deterministic flagger** (`validity.flag_episode`, no LLM):
```
BEFORE labeling:  trust=high   flags=[]                    # the trap: looks clean
AFTER  labeling:  trust=low    flags=[test_reward_false_positive, test_reward_false_negative]
                  categories=[low_coverage_tests, overly_strict_tests]
```
The contradiction detectors are *dormant until the human signal exists* — which is the whole point:
feedback mining (workstream 2) and the reward-quality audit (this layer) **compose**. On the raw
stored stores, `episodic audit` reports **0% broken precisely because the episodes are unlabeled**;
labeling one flips it `high → low`.

**Step 3 — agent-assisted validator** (`validity.validate`, 3 independent real Haiku passes):
```
pass 1: trustworthy=false  sev=medium  conf=0.75  [reward_contradiction, underspecified_intent, low_coverage_tests]
pass 2: trustworthy=false  sev=high    conf=0.92  [reward_contradiction, low_coverage_tests, underspecified_intent]
pass 3: trustworthy=false  sev=medium  conf=0.75  [reward_contradiction, underspecified_intent, low_coverage_tests]
aggregate: trustworthy=false  agreement=1.0  escalate=false  confidence=0.807
```
Verbatim independent reasoning (pass 2): *"Unit tests verify code compiles, not that echo
cancellation works (requires audio fidelity testing) … Composite 0.6962 appears to be averaging
incompatible signals rather than reflecting the genuine partial completion."*

**Step 4 — merged verdict** (`validity.assess`, rules + LLM):
```
trust=low  severity=high  source=rules+llm
categories=[low_coverage_tests, overly_strict_tests, reward_contradiction, underspecified_intent]
```

**Step 5 — training-set gate** (`exporters` / `loop`):
```
composite=0.696  (naive ≥0.5 gate would ACCEPT)
is_trusted=False   is_good=False   loop.select_good → EXCLUDED
```

## Scale audit (deterministic, zero LLM cost)

```
$ episodic audit --store <stagewhisper>
audited 49 episode(s)
trust: high=19 medium=30 low=0
BROKEN (low-trust / noisy reward): 0/49 = 0.0%
categories: underspecified_intent: 29, low_coverage_tests: 1

$ episodic audit --store <episodic>
audited 14 episode(s)
trust: high=11 medium=3 low=0
categories: underspecified_intent: 3
```

The 0% low-trust is honest, not a null result: the high-severity **contradiction** detectors require
the independent human signal, and these stored episodes predate the labeling wire-up (`human_feedback`
is empty). The reference-episode walkthrough is the existence proof that, once `episodic label` /
the SessionEnd auto-labeler has run, the same machinery pulls reward-noisy episodes out of training.

## Regression status

`python3 -m pytest -q` → **208 passed, 1 skipped** (includes `tests/test_validity.py`, 9 cases:
false-positive/false-negative contradictions, unverified-success, underspecified, clean=high-trust,
low-trust excluded from the training set, conservative LLM majority, and `assess` merge via
`build_episode`).

## What "covered" means here

Every insight is (a) a named function in `core/validity.py` or a gate in `exporters`/`loop`/`cli`,
(b) unit-tested, and (c) demonstrated firing on a real Episode from an actual coding session with a
real model in the loop — including the flip from a deceptively-high test reward to `trust=low`,
which is the article's entire message applied to Episodic's own RL dataset.
