# Evaluation: ai-trains-ai–inspired reward changes on real Episodes

Status: evaluated 2026-07-14, **not committed**. Question posed: *build the five ideas mined from
[ai-trains-ai](https://github.com/Danau5tin/ai-trains-ai), test end to end on real Episodes, and settle
whether they are actual improvements — especially #1.*

**Decision applied (drop anything without a demonstrated absolute improvement over baseline):** only **#2**
survives — it both filters reward-noise correctly *and* fixed a real correctness bug. **#1 dropped**
(measured regression), **#3 and #5 dropped** (no measured improvement — unexercised on this corpus). The
only change left in the tree is #2 (`loop.ensure_validity` + `preflight_dropped`) plus this record.

Method: implemented all changes; built a ground-truth eval set by labeling **8 real episodes** with real
Haiku (`claude-haiku-4-5-20251001`) to obtain independent `human_label`; then ran an **adversarial
evaluation workflow** (7 subagents: 4 independent facet analysts that *recomputed* every number from the
raw data, an advocate + a skeptic, and a synthesis judge). Every headline figure below was reproduced
independently at least twice.

## What was built (all changes additive; suite 302 passed, 1 skipped)

| # | Idea | Implementation | Verdict | Kept? |
|---|---|---|---|---|
| 1 | Baseline-uplift reward | `reward.baseline_uplift` + `composite_v2`/`WEIGHTS_V2` (additive A/B) | regression on real data | **Dropped** |
| 2 | Pre-flight validity gate | `loop.ensure_validity` recomputes trust before training; `preflight_dropped` in manifest | correct + fixed a real bug | **Kept** |
| 3 | Cluster/repo holdout | `loop._holdout_key` + `split_episodes(holdout_key=…)` | sound but unexercised here | **Dropped** |
| 5 | Retro exporter | `exporters/retro.py` + `retro` format | no bearing on reward quality | **Dropped** |

(#4 anti-coasting/efficiency was folded into #1's coasting=0.5 — dropped with #1.) The three dropped changes
were reverted after this evaluation; #1/#3/#5 can be resurrected from git history if the follow-ups below
(fixed hidden eval, adequate corpus) are ever done.

## Verdict on #1 — NOT an improvement on real data → keep as diagnostic, do not make default

Ground truth = mined `human_label`. Adding `baseline_uplift` **degrades** reward↔human alignment:

| vs `human_label` (n=8) | `composite` | `composite_v2` |
|---|---|---|
| Spearman | **0.976** | **0.643** |
| Kendall τ-b | 0.929 | 0.643 |
| Concordant human-ordered pairs | **27/28** | 23/28 |

`baseline_uplift` **on its own is anti-correlated** with human quality (Spearman −0.41). Three reasons:

1. **Near-zero coverage.** Across both real stores (97 episodes), a baseline-uplift is *definable* for only
   **8/97 (8.2%)** and **non-neutral for 1/97 (~1%)** — most coding sessions run tests 0–1 times, so there
   is no pre/post arc. 0 regressions, 0 failed prod deploys (both fallback arms never fired). ai-trains-ai
   gets uplift for free from a *fixed hidden eval scored every episode*; captured coding sessions have no
   such structure.
2. **Degenerate on the majority.** 7/8 episodes are coasting green→green → uplift is a constant 0.5 (identical
   to the "no uplift" fallback), contributing zero discriminating signal; it only reweights other terms.
3. **The one firing points backwards.** `ep_8f9a957edf43` (tests 0.45→1.0, uplift 0.77 — the highest) is the
   **2nd-worst** episode by human ground truth (`human_label` −0.67; 9/12 feedback "wrong"). `composite`
   ranks it correctly near the bottom; `composite_v2` elevates it to 2nd-best. This is the classic
   **test-gaming false positive** — tests driven green while humans judged the work wrong.

**Counterfactual (smoking gun):** neutralize *only* that one episode's uplift (0.77→0.5) → Spearman(v2,human)
jumps 0.643 → **1.0**. The entire regression is attributable to the `baseline_uplift` term acting on a
single point, not to the reweighting.

**Honest caveats:** (a) n=8, only one non-coasting episode — this rejects the *current implementation as a
default reward*, not the idea in its intended regime. (b) `human_label` is a 0.15-weight component of both
composites; a de-circularization check (strip it from both) collapses the correlations to 0.43 vs 0.12 — the
absolute 0.976 is largely a circularity artifact, but the **direction (v1 > v2) survives**. (c) The real
defect is the **measurement surface**: uplift reads the agent's *own, gameable* in-session tests instead of a
fixed hidden eval.

## Verdicts on the supporting changes

- **#2 pre-flight gate — keep, but blunt.** Correctly drops all 8/8 reward-noisy (trust=low) episodes.
  **A real bug was found and fixed during end-to-end testing:** `ensure_validity` trusted a *stale*
  capture-time `validity: {trust: high}` on disk (written before any feedback was mined), so the gate
  passed all 8 noisy episodes through (`dropped=0`). Fixed to always recompute the deterministic assessment
  while preserving any expensive LLM verdict; regression-tested. Post-fix: `dropped=8`. **Caveat:** the gate
  is high-precision / low-recall — on this all-coasting corpus it also drops the 4 human-*good* episodes,
  leaving an empty training set. Follow-up: per-signal validity / segment mega-sessions rather than
  all-or-nothing trust.
- **#3 cluster holdout — keep.** Sound leakage-prevention; unit-tested (same-intent/repo never split across
  train/holdout). Not exercised by this dataset otherwise.
- **#5 retro exporter — keep, out of scope.** Orthogonal tooling; no bearing on the reward-quality question;
  demonstrated on the 8 real labeled episodes.

## Recommended next steps

1. **Keep `composite` as the default reward.** Ship `baseline_uplift`/`composite_v2` as a **diagnostic**
   (a red→green trajectory flag + test-gaming detector when uplift disagrees with `human_label`/`outcome`).
2. **Fix the measurement surface before re-litigating #1** — score uplift against a fixed, hidden, ungameable
   eval, not in-session tests.
3. **Re-test on an adequate corpus** with real red→green gains, regressions, and failed deploys.
4. **Pair the #2 gate with a recall path** and grow the labeled corpus beyond 8 all-coasting episodes.

## Provenance

Adversarial evaluation workflow `ai-trains-ai-eval` (7 subagents, 0 errors, ~221k tokens). Facet analysts
independently globbed the stores and recomputed Spearman/Kendall/concordance/coverage from
`metrics.json`; advocate returned `context-dependent`, skeptic returned `regression`, synthesis judge
returned **`keep-as-diagnostic-only`**. This mirrors the ai-trains-ai discipline (independent judgment,
N repeats, escalate on disagreement) — the same pattern shipped in `core/validity.py`.
