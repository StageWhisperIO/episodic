# Flywheel lift on the trusted gate: the provable-business report

The thesis Episodic has to prove is narrow and falsifiable: **captured coding episodes, turned into
training data, make a model measurably better at a task — and the "better" is scored by a gate the
candidate cannot game.** Traces in, a trained policy out, and a trusted discriminator in between that
says the lift is real. That is the same loop a distillation business (WMO / Experiential Labs) sells:
traces → distilled model → served endpoint. This report is the first end-to-end measurement of that loop
on Episodic, and the harness that makes the measurement reproducible in-tree rather than a one-off.

## 1. Executive summary

- **The loop closes with a positive, honest lift.** SFT on 66 Episodic-captured gold patches lifts a
  capable 4B base from **9/16 to 13/16** solved on a held-out set, scored entirely through the
  harness-controlled replay-eval gate. **+4, zero regressions** — every class the base already solved, it
  still solved.
- **Trained on Tinker (remote GPUs), not locally.** `Qwen/Qwen3.5-4B` LoRA rank 32, 3 epochs / 51 steps,
  285s, loss 0.44→0.35. Both the trained sampler and an untrained-base sampler were created on Tinker and
  every generated diff was sampled remotely; the laptop only orchestrated and ran the small `pytest`
  verifiers. (Local 4B mlx training had crashed the machine — this is the offloaded rerun.) All checkpoints
  were deleted afterward; zero storage residue.
- **The lift lands exactly where a strong base fails.** `bound` 0→1, `ds` 0→2, `dedup` 0→1 — the classes
  where the base cannot emit a correct unified diff. `exc / logic / operator / str` held at base level. One
  hard OOD singleton (`inverted-logic`) unsolved by both.
- **This number is smaller than the toy number, and that is the point.** An earlier 0→11 came from a 0.5B
  model that could not emit *any* valid diff, so nearly anything read as lift. +4 on a base that already
  solves 9/16 is the credible measurement: the training teaches diff-shaped fixes precisely in the failure
  classes, on top of a baseline that is already good.
- **The measurement is now first-class and CI-guarded.** `src/episodic/eval/` (`redgreen`, `gate`,
  `flywheel`) + `episodic eval-flywheel` + `tests/test_eval.py`. The gate-discrimination and
  lift-measurement mechanics run deterministically in CI with no model, no network, no GPU. The real
  base-vs-trained magnitude is one command away (`--backend tinker`), no longer a job-tmp script.

## 2. What was measured

Each task is a real git repository committed in a **buggy (red)** state, with a covering `pytest` verifier
and a **gold fix** recorded as the episode's unified diff. `build_task` asserts the bug is genuinely red at
base and green after the fix before it is admitted, so every task is test-necessary by construction. The
corpus spans bug classes — operator, boundary, data-structure, exception, boolean-logic, string — plus
singleton out-of-distribution classes that appear only in the held set. The prompt the model sees carries
only a **visible** subset of the assertions; the committed, executed verifier also holds **hidden**
assertions with different inputs, so a candidate that hardcodes the visible answer still fails.

**The gate.** A candidate diff is scored by cloning the repo at the base commit, `git apply`-ing the
candidate, **reverting any candidate edits to verifier files**, and running the captured test command.
Score = `0.6 · tests_pass + 0.4 · diff_overlap`, with a strict pass (skipped ≠ passed, must have >0 passed
and 0 failed/errored). The verifier-revert step is what makes the gate *trusted*: a candidate cannot pass
by editing the test, deleting it, injecting a skip, or hardcoding the visible answer — those SWE-bench-style
reward hacks are neutralized before scoring (the last one by hidden held-out assertions the candidate never
sees).

**The flywheel.** Episodes split stratified by bug class into train / held. The training set becomes SFT
rows (`intent + diff-instruction` → fenced gold diff). A base sampler (untrained) and a trained sampler
generate a diff per held task; both are scored through the same gate. Lift = trained-solved − base-solved.

## 3. The result

```
model      Qwen/Qwen3.5-4B (Tinker, LoRA rank 32, 3 epochs / 51 steps, 285s, loss 0.44→0.35)
corpus     82 clean tasks → 66 train / 16 held (stratified)
base       9 / 16 solved
trained    13 / 16 solved
lift       +4, zero regressions
```

| class          | held | base | trained |
|----------------|------|------|---------|
| bound          | 2    | 0    | **2**   |
| ds             | 2    | 0    | **2**   |
| dedup          | 1    | 0    | **1**   |
| exc            | 2    | 2    | 2       |
| logic          | 2    | 2    | 2       |
| operator       | 2    | 2    | 2       |
| str            | 2    | 1    | 1       |
| off-by-op      | 1    | 1    | 1       |
| off-by-one     | 1    | 1    | 1       |
| inverted-logic | 1    | 0    | 0       |

The +4 is entirely in `bound`, `ds`, `dedup` — where the base scores 0. Nothing the base solved regressed.
That shape (lift concentrated in the base's blind spots, no collateral loss) is the signature of a real,
targeted improvement rather than a decoding-luck artifact.

## 4. What is proven vs. what is not

**Proven.**
- The gate discriminates: on every task, the oracle (gold diff) passes green while an empty diff and a
  diff-then-break candidate both fail red. This is asserted for all 12 canonical tasks in CI
  (`test_gate_discriminates_on_every_task`).
- The gate resists verifier tampering: a candidate that rewrites the test to `assert True` still scores
  not-solved, because the edit is reverted before scoring (`test_gate_reverts_verifier_tampering`).
- The gate resists hardcoded-constant stubs: the model sees only the visible assertion, while the executed
  verifier also runs hidden assertions with different inputs. A stub that returns the visible answer passes
  the visible check but fails the hidden one, and the hidden assertions never appear in the prompt
  (`test_gate_rejects_hardcoded_visible_constant`, `test_hidden_assertions_are_not_leaked_to_the_prompt`).
- The flywheel harness measures lift correctly: with a base that solves 0 and a trained oracle that solves
  all, measured lift equals the held count (`test_stub_flywheel_measures_full_lift`).
- On a real trained model (Tinker, 4B), the loop produces a positive lift with no regressions — §3.

**Not proven / honest limits.**
- **Synthetic corpus.** These are constructed red→green tasks, not captured production episodes. They prove
  the *loop* and the *gate*; they are not evidence about real-world coding-task difficulty. The captured
  StageWhisper corpus still lacks clonable, runnable verifiers at scale (the 94→17→4 funnel), which is why
  the lift is measured on constructed tasks the gate can actually run.
- **In-tree corpus is the 12-task canonical set; the +4 came from an 82-task run.** The committed generator
  emits 12 diverse, leak-free tasks — enough to CI-guard the mechanism. The magnitude number in §3 used a
  larger job-tmp corpus of the same construction. `episodic eval-flywheel --backend tinker --variants K`
  reproduces a larger corpus and the real measurement on demand; the exact 82-task run is not baked into the
  repo.
- **Serving side of the loop is not exercised here.** This report measures traces→train→gate. The
  train→serve→endpoint half (`episodic serve`) is separate and out of scope for this measurement.

## 5. The real corpus today: the trusted-task funnel

Measured on the 104 captured StageWhisper episodes (read-only, via `EPISODIC_HOME`), the funnel is:

```
104  captured
102  clonable (local git repo present — no network, no auth)
102  + base_commit
 20  + captured a test command / verifier      (82 sessions ran no capturable test)
  5  + also has a usable diff                   (15 have no diff)
  0  certifiably test-necessary                 (fails red without the diff, passes green with it)
```

`episodic eval-flywheel --certify` runs the last step: clone at base, run the captured test **without** the
diff (must fail red) and **with** it (must pass green). **Zero of 104** pass — 28 because the captured diff
does not make the test pass in the clone (the diff and the test are from different scopes, or a heavy
Rust/TS build does not come up), 3 because the test passes with *or* without the diff (a broad suite that
does not depend on the change). Clone/env is **not** the bottleneck; the captured `(diff, test)` pairs are
simply not matched red→green units.

This is the honest ceiling on "provable lift on real work" from *this* corpus: the trusted gate has
**nothing real to grade** from a Tauri desktop/mobile monorepo whose tests are whole-workspace cargo
builds. Closing it needs change-scoped, certified units from test-rich repos.

**Resolution — `episodic mine-history`.** The miner harvests exactly those units from a repo's git
history: for each commit that changes both a test and source, it injects the test at the parent commit
(which must fail **red** without the change) and applies the source change (which must pass **green**),
keeping only test-necessary units. It is change-scoped capture sourced from history, and every unit is
certified by construction.

Run against Episodic's own history (a test-rich Python repo), 120 commits scanned:

```
StageWhisper monorepo   →   0 certified tasks
Episodic (mine-history) →  59 certified red→green tasks   (249s, 54 MB of hardlinked clones)
```

Each mined task is a real, multi-file, change-scoped unit (across `loop`, `replay`, `exporters`,
`core/episode`, `examples/`, …) with a covering test that fails without the change and passes with it —
the same shape the synthetic corpus fakes, now from real code. This is the corpus the flywheel needs:
`episodic eval-flywheel` (no `--generate`) picks up the mined `swe`-labelled episodes and runs the gate +
lift over them. `--certify` is the gate that measures the trusted-task count; the miner is what moves it
from 0 to 59.

## 6. Lift on real mined tasks: three levers

The synthetic +4 is on easy single-function tasks. The 59 mined Episodic tasks are real, multi-file, and
much harder. Three follow-up experiments, all on `Qwen/Qwen3.5-4B` via Tinker, scored through the gate on
real clones:

| run | corpus / split | held | base | trained | lift |
|-----|----------------|------|------|---------|------|
| single-shot SFT, id-split | 27 tractable mined, held = every 4th | 7 | 0 | 0 | 0 |
| single-shot SFT, hold-out-small | 27 tractable, held = 8 smallest | 8 | 1 | 2 | **+1** |
| agentic 2-turn SFT, hold-out-small | same split | 8 | 1 | 1 | 0 |
| STaR / RL-on-gate, hold-out-small | same split | 8 | 1 | 1 | 0 |

Findings, stated honestly:

- **Difficulty and split dominate.** With an id-ordered split the held set was all hard multi-file
  internals (`loop`, `replay`, `store`) — base and trained both solve **0/7**. Holding out the *smallest*
  tasks instead, the base already solves the one trivial one (`examples/clamp.py`) and a trained model
  reaches **2/8** — a **+1** that is real but marginal.
- **±1 at held=8 is inside training noise.** The +1 came from one task (`core/testdetect.py`); a second
  training run (the agentic row) did not reproduce it and landed at 0. The signal is not yet separable from
  run-to-run variance at this corpus size.
- **Agentic feedback did not help these tasks.** A 2-turn generate→apply→test→retry loop solved no more
  than single-shot. The unsolved tractable tasks are real multi-file internal changes (`ids`+`paths`,
  `service`, `rewards`, `normalize`); a plain test-failure message is not enough signal for a 4B model to
  locate the fix. Agentic pays off when the model is close (a syntax slip it can read off the error), not
  when the change is out of reach.
- **RL-on-gate is starved by a weak base.** STaR sampled 3 rollouts per train task and kept only
  gate-passers: the base solved just **2 of 19** train tasks, so the expert-iteration SFT trained on 2 of
  the model's own diffs — far too little to move the held score (1/8 → 1/8). RL/STaR amplifies a policy
  that already succeeds sometimes; here the base almost never does, so there is nothing to bootstrap.

**Honest read.** The infrastructure — mine → certify → gate → train (SFT / agentic / STaR) → score — runs
end-to-end on real tasks, and on the easy real subset it produces a small positive lift. But meaningful,
noise-separable lift on *hard* real repo-internal tasks is not there at 4B with single-shot SFT, 2-turn
agentic, or RL-on-gate. All three converge on the same bottleneck: **a 4B base solves ~1/8 held and ~2/19
train, so there is almost no signal for any method to amplify.**

### 6.1 Bigger model, tested: `Qwen3-30B-A3B` does not move it

The obvious next hypothesis — "the 4B is just too weak; a bigger model breaks the wall" — was tested
directly. Same mined corpus, same hold-out-small split (held = 8), single-shot SFT, but
`Qwen/Qwen3-30B-A3B-Instruct-2507` (a 30B mixture-of-experts) instead of the 4B:

| model | held | base | trained | lift |
|-------|------|------|---------|------|
| Qwen3.5-4B | 8 | 1 | 2 | +1 |
| Qwen3-30B-A3B | 8 | 1 | 1 | **0** |

The 30B solves exactly the same single trivial task (`examples/clamp.py`) as the 4B and **zero** of the
seven hard multi-file internals (`rewards`, `ids`+`paths`, `service`, `testdetect`, `store`, `normalize`).
This **refutes the "just use a bigger model" lever for this corpus**: the held tasks are dense-coupling
repo internals where the missing ingredient is not raw model capacity applied to a single prompt but
(a) *task shape* — self-contained, single-responsibility changes a model can actually get in one shot —
and (b) *method* — multi-step read→edit→run tool use rather than one blind diff. (One caveat worth noting:
the enriched SFT rows had to be diff-size-filtered to fit the 30B's 32768-token training limit, which the
4B did not enforce; the training set was therefore slightly smaller. But the base — untrained — score being
identically 1/8 shows the ceiling is the task/prompt, not the training set.)

The corrected read: the wall is **task shape + single-shot method + corpus size**, not base capacity. The
levers that remain are therefore **more, easier, more diverse mined tasks** (to give a held set with
tractable units that show separable lift, and to give RL enough base successes to bootstrap) and a **real
multi-step tool-using agent** (not a single diff) — which is what actually closes multi-file bugs. Bigger
model alone is spent as a lever here.

### 6.2 More diverse tasks: mining external test-rich repos

To grow the corpus beyond Episodic's own history, the miner was pointed at the `maddox` family of
production repos. Most are not provisionable on a laptop: `data_lib` pulls torch/torchvision from a
CUDA-only index and `pymaddox` from a private authenticated index, with tests that download pretrained
weights over the network; `webapi` needs a live MongoDB + Azure. The one tractable target is
**`ask_ai/python_api`** — all-public light deps (`fastapi`/`httpx`/`openai`/`tinydb`/`pydantic-settings`),
a flat module layout, and a hermetic 223-pass / 1.5s suite once a single required secret is stubbed
(`OPENROUTER_API_KEY=dummy`). `mine-history` grew `--python-bin` (a provisioned venv), `--import-root`
(a repo subdir scoped onto `PYTHONPATH`), and `--env KEY=VALUE` (stubbed secrets) to mine it. This is the
"more diverse tasks" lever in progress; the point is that external corpora bring a *provisioning* cost that
Episodic's own stdlib-only history did not. The 24 ask_ai tasks are certified by construction and score
correctly through the provisioned gate (oracle→green, empty→red), but the repo's commits bundle prompt/
context `.md` and JSON data with code, so only ~5 are clean single-responsibility code changes — external
corpus fit is not automatic.

### 6.3 Tool-using agent, tested: a 4B policy cannot drive it yet

The single-diff runner patches blind — it never sees the code it must change. `agentic.build_tool_agent`
is a harness-controlled multi-step loop (`READ` / `LS` / `TEST` / `PATCH` against the checkout, sandboxed
to the workspace) so the policy can read across files before editing. It is unit-tested and validated
end-to-end on a real external clone: with an *oracle* policy (READ the file, then emit the gold patch) it
drives the ask_ai clone to green in two steps. But the lever question is whether a real policy exploits it.
Isolating the lever — same base `Qwen3.5-4B`, no training, single sampler, on the 10 smallest code-only
mined tasks — single-shot vs 6-step tool agent:

| scorer | tasks | solved |
|--------|-------|--------|
| single-shot diff | 10 | 1 |
| 6-step tool agent | 10 | **0** |

The tool agent solved *fewer*: it even lost the one trivial task (`clamp.py`) single-shot got. A 4B policy
spends its steps on malformed actions and leaves the checkout in a worse state than one clean diff would.
The capability is real and sandbox-safe, but **the multi-step protocol needs a stronger policy to pay off**;
its overhead dominates when the model is weak. This is the same conclusion as §6.1 from the other side:
the binding constraint is policy capability on hard, multi-file tasks — not the harness, the gate, the
corpus plumbing, or the tool interface, all of which now work.

### 6.4 Lessons from an outside RL pipeline (Narreddi, *Training AI to Paint with Code*)

A useful outside data point: Surya Narreddi & Cameron Franz RL-trained Qwen-3.5-35B (GRPO) to write
p5.brush JS that renders watercolours, judged by a model. They hit a ceiling — reward plateaued at 0.65
with every rollout identical (a flat clip-art flower); *reward climbed but capability did not*. Their
diagnosis, from decomposing the sub-rewards in isolation, and the fixes that broke through, map directly
onto our verifiable-reward setting even though their reward is subjective:

- **Their ceiling was a starved, redundant, saturated reward.** Nine sub-signals; four quality judges +
  prompt-adherence correlated 0.85–0.95 (measuring the same thing five times); a code-length term worth ~⅓
  of the reward had saturated by step 30 (zero gradient after); the one signal with real variance (HPSv3)
  was weighted 0.10. **Our analogue:** the gate blended `0.6·tests_pass + 0.4·diff_overlap`, but
  `diff_overlap` (Jaccard of changed filenames vs the gold patch) rewards *file-targeting, not
  correctness* — a correlated, gameable term — and the eval layer then collapsed everything to a *binary*
  `ok`, discarding the partial-credit `tests_pass` the replay already computes. On hard tasks the binary
  reward is compressed to all-zero → no gradient, which is exactly why STaR had only 2/19 wins to learn from.
- **Their fix #1: pairwise judging for dynamic range** (absolute 0–10 scores compressed near zero; "which
  of these two is better?" opened the range). **Our implementation:** `gate.graded_score` surfaces a
  partial-credit `pass_fraction = passed/(passed+failed+errors)` and `rewards.graded_gate_reward` exposes it
  as the RL reward (feeds SAO's group-relative `running_baseline`, which turns a graded reward into non-zero
  advantage *even when no rollout fully passes*). `measure_lift` now reports `base/trained_pass_fraction` and
  `fraction_lift` alongside binary solves — partial progress the binary metric hides.
- **Their fix #2: collapse the rubric to orthogonal components; drop the correlated/saturated ones.** **Our
  implementation:** `graded_score`/`graded_gate_reward` are pure-test (they exclude `diff_overlap`);
  `gate.reward_components_report` prints each component's mean/variance and the `pass_fraction`↔`diff_overlap`
  correlation, so a dead or redundant signal is visible the way theirs was.
- **Their reference-pool + curriculum insight (anchor to achievable examples; a reward that is all-0 or
  all-1 teaches nothing).** **Our implementation:** `flywheel.learnable_band` samples *n* rollouts per task
  and keeps only those the base solves on 0<k<n — the band where a gradient exists — so training and
  eval focus on the learnable frontier instead of all-hard (all-zero) or trivial (all-one) tasks.
- **Their system-prompt finding: a 400-line API reference made the model hallucinate APIs; a short
  allowlist beat the full spec.** **Our analogue:** we had been *enriching* prompts with up to 12 KB of base
  source (the same bloat). The tool agent (§6.3) is the structural fix — the policy `READ`s only what it
  needs instead of being handed the whole file — and is the direction to keep over source-dumping.

Net: the article confirms our own conclusion from a different domain — once the plumbing works, *reward
shape and curriculum are the levers*, not more rubric terms. All four mechanisms above ship in-tree with
tests; the graded reward + learnable-band are now what the stronger-policy runs use.

### 6.5 Applying the lessons: what the graded reward revealed (and a trap it exposed)

**Stronger dense policy, single-shot, graded.** Ran the 10 smallest code-only mined tasks single-shot on
dense `Qwen3.5-4B` and `Qwen3.5-9B`, scored with `graded_score`:

| model | solved (binary) | mean pass_fraction |
|-------|-----------------|--------------------|
| Qwen3.5-4B | 1/10 | 0.853 |
| Qwen3.5-9B | 0/10 | 0.753 |

At first read the fractions look encouraging — the 4B is "85% of the way there" where binary said 1/10. But
the per-task fractions were **identical between 4B and 9B on 9 of 10 tasks** (they differ only on
`clamp.py`: 4B 1.0, 9B 0.0). That is a red flag, and decomposing it (the article's own method) shows why:
computing the empty-baseline (no-change) fraction locally, the models' fractions *equal the empty baseline
exactly* on those 9 tasks. **The high fraction was almost entirely a fixed offset — the pre-existing
always-passing tests in each injected test file — not model progress.** The informative range
(oracle − empty) on the whole-file command was only:

```
clamp 1.00 | normalize .20 | store .07 | service .25 | stages .09 | (…) — median ~0.15
```

So the model's true *advantage over doing nothing* was 0.0 on 9/10 tasks; only `clamp` moved. This is
exactly the "starved signal swamped by a saturated one" failure the article describes, reproduced in our
gate. Two fixes, both shipped:

1. **Report advantage, not absolute fraction.** `gate.graded_advantage` returns
   `pass_fraction − empty_baseline_fraction`; for RL, SAO's group-relative baseline cancels the offset
   automatically, so `graded_gate_reward` was already correct — but eval reporting must subtract it or it
   lies.
2. **Score only the fix-relevant tests (the root fix).** The miner now extracts the test functions the
   commit *adds* (`_added_test_selectors`) and scopes the stored command with `pytest -k "test_a or …"`.
   Re-mining ask_ai with this, the informative range jumps from ~0.1 to **0.50–1.00** (three tasks are a
   clean 0.00→1.00). The graded reward now has real dynamic range on *model-driven* progress: a candidate
   that fixes one of two target assertions scores 0.5, not 0.90.

**Second conclusion, same as §6.1/6.3:** the dense 9B did **not** beat the 4B on these tasks (identical
except it lost `clamp`), so a bigger *dense* model is no more a lever than the 30B MoE was. What the lessons
bought is not a lift number today but the two things the article says actually matter once plumbing works:
a reward with **dynamic range on the fix-relevant signal** (`-k`-scoped `pass_fraction` + advantage) and a
**learnable-band** curriculum to spend RL only where that signal is non-zero. Those are the inputs the next
SAO-on-gate run needs; wiring `graded_gate_reward` into the SAO training loop is the remaining step.

## 7. Reproduce it

```bash
# Deterministic, no model — the CI gate (gate discrimination + lift plumbing):
episodic eval-flywheel --generate --json          # exit 0 iff every task's gate is clean
python -m pytest tests/test_eval.py -q

# Real base-vs-trained magnitude on Tinker (needs TINKER_API_KEY; deletes its checkpoints after):
episodic eval-flywheel --generate --variants 6 --backend tinker --model Qwen/Qwen3.5-4B

# Certify how many captured episodes in the current store are trusted, test-necessary tasks:
episodic eval-flywheel --certify --json

# Harvest certified red->green tasks from a test-rich repo's git history into the store:
episodic mine-history /path/to/python-repo --max-commits 120

# Score with a multi-turn agentic runner (generate -> apply -> run test -> feed failure back -> retry):
episodic eval-flywheel --backend tinker --model Qwen/Qwen3.5-4B --agentic-turns 2
```

`src/episodic/eval/redgreen.py` builds the tasks, `gate.py` is the trusted discriminator, `flywheel.py`
runs the split→train→score→lift measurement (mlx or tinker). The stub backend (oracle-vs-empty) is the
deterministic self-check; it is what CI runs and what proves the harness itself is not lying about lift.
