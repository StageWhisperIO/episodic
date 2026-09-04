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
**learnable-band** curriculum to spend RL only where that signal is non-zero.

### 6.6 Closing the loop: RL-on-gate runs on the graded reward

The remaining step was to feed that reward into an actual RL loop. `rewards.gate_pass_fraction_reward`
reads the episode from SAO's per-rollout `meta`, extracts the completion's diff, runs it through the trusted
gate, and returns `pass_fraction`; `flywheel.build_sao_rows` writes SAO dataset rows carrying `meta=episode`;
`tinker-sao`'s `resolve_reward` selects it via `reward_funcs`. The corpus was re-mined with the `-k` fix so
all tasks have wide informative range (89 certified tasks, ranges 0.17–1.00, median 0.67).

A proof-scale run (`Qwen3.5-4B`, 10 train tasks × 2 rollouts, 4 steps, LR 1e-5, reward = graded gate)
completed end-to-end:

```
step 0: reward_mean 0.715  advantage_mean +0.260  loss −1.295  updated
step 1: reward_mean 0.542  advantage_mean −0.155  loss +0.772  updated
step 2: reward_mean 0.300  advantage_mean −0.234  loss +1.167  updated
step 3: reward_mean 0.300  advantage_mean −0.010  loss +0.051  updated
```

The point is the reward column: **it varies (0.30–0.72) and the advantages take both signs**, so every step
produced a real gradient and all 4 updates applied. That is precisely what the binary reward could not do —
on these hard tasks binary is all-zero, so an RL loop on it would see zero advantage and never update. The
loop is closed: sample → gate-verified graded reward → group-relative advantage → DIS-masked
importance-sampling update → checkpoint.

**Held lift at this scale is 0** (`adv_lift 0.0`; base and trained both score 0 advantage greedily on the 5
hardest held tasks) — 4 steps and 20 rollouts is far too little to move a 4B, and the run is bounded by the
cost of the reward itself (each rollout clones a repo and runs pytest, and Tinker sampling latency dominates
wall-clock). So this closes the *infrastructure* loop and demonstrates the graded reward yields trainable
gradient; a *measurable* lift needs a run one to two orders of magnitude larger (hundreds of steps, more
rollouts per prompt for a tighter per-prompt baseline), which is a compute-budget question, not a
missing-mechanism one. Every piece — mine (`-k`) → certify → graded gate reward → SAO update → advantage
scoring — now runs in one pipeline.

### 6.7 Scaling the corpus: storage-safe SWE-rebench ingestion

The levers above all hit the same wall — a 4B's base capability on a corpus that is small (a few dozen
tasks) and low-diversity (mostly one or two mined repos). The RLing-Qwen lesson (§6.4) is that once the
plumbing works, curriculum is a lever; but curriculum needs *scale and diversity* of trusted tasks. The
largest decontaminated, training-oriented source is **SWE-rebench** (`nebius/SWE-rebench`, 21,336 instances
mined automatically from real GitHub PRs, each already validated FAIL_TO_PASS/PASS_TO_PASS). The problem it
raises is storage: a laptop cannot hold 21k repo checkouts, let alone 21k dependency environments.

The design that overcomes this is **reference, never materialize**. `eval/swerebench.py` maps each instance
into a metadata-only `CodingEpisode`: `remote_url` + upstream `base_commit` + the gold `patch` (source only,
verifier files filtered out) as `diffs`, the `test_patch` stored as the new `repo_state.setup_patch`, and the
fix-relevant `-k` selector derived from FAIL_TO_PASS (§6.5's scoping, for free). No repo is ever persisted;
an episode is ~13.6 KB, so **the entire 21,336-instance corpus is ~283 MB of metadata**. Ingestion streams
rows through HuggingFace's datasets-server REST API (`/rows`, paginated) — pure JSON over HTTP, **zero
parquet download**, so nothing lands in the local dataset cache either.

A full repo exists only ephemerally, at score time, reusing replay's existing clone→checkout→run→`rmtree`
path and its 2 GB free-space guard. Two changes made metadata-only episodes runnable: (1) replay now applies
and commits `setup_patch` right after checkout, reconstructing the RED baseline (the upstream `base_commit`
does not contain the test — unlike a mined episode whose scratch tree bakes the test in) before the runner
and tests run, so `_protect_verifier` and diff-overlap see the injected test as the baseline; (2) the clone
is a blobless partial clone (`--filter=blob:none`, plain-clone fallback) so a checkout is tens of MB, not
hundreds.

This certifies against real GitHub repos, not fixtures. `swerebench_0b01001001__spectree-64`, pulled straight
from the dataset through the trusted gate:

```
EMPTY  ok=False frac=0.000  (RED reconstructed from setup_patch — the injected test fails without the fix)
ORACLE ok=True  frac=1.000  (GREEN — the gold source diff makes it pass)
```

A clean 0.00→1.00 range in ~3–4 s per scoring, clone auto-cleaned. The honest limit is the same provisioning
funnel as §6.2. A one-instance-per-repo sweep over 8 distinct repos certified **1/8 as-is** (spectree); the
other seven hit import/collection errors — the dependency wall, not a mapping bug.

**Provisioning lifts that yield (`eval/provision.py`).** Each instance ships `install_config`
(`install: "pip install -e .[extras]"`, `pip_packages`, `pre_install`, `python`). `ensure_repo_venv` builds a
**per-repo cached deps-venv** from it (system-package `pre_install` steps like `apt-get` are best-effort and
skipped; the editable flag is dropped so the workspace source, prepended on `PYTHONPATH`, shadows the
installed copy — the generalised ask_ai pattern), and the gate runs tests through it via the new
`repo_state.test_env` hook (replay merges the venv `PATH` + workspace `PYTHONPATH` at test time). Re-running
the same 12 distinct-repo sweep **with** provisioning: **9/12 provisioned, 4/12 certified** (spectree,
scim2-filter-parser, bqtools, shipwright) — up from 1/8. The misses are honest: GUI/headless repos
(`sepal_ui`, ipywidgets/solara) collect zero tests on a Mac, a few need system libs that don't build, and
three land `NEEDS_MORE` (oracle improves but one selected test stays red — the `-k` name match over-selects
vs the exact FAIL_TO_PASS node ids, a known refinement). Venv cache was 1.6 GB for 9 repos, in job-tmp.

Two more replay additions make the RL loop over these tasks cheap: (1) `repo_state.test_env` (above); (2) an
opt-in **per-repo blobless bare-mirror cache** (`EPISODIC_MIRROR_DIR`) — the first clone of a repo populates
a bare mirror, every later clone uses `--reference-if-able`, so RL rollouts that re-clone the same repo each
step pay the history transfer once. So all 21k ingest for free as `certified_by_source` metadata; we re-run
the gate only on the provisionable slice, and that slice is now ~3× larger. The pipeline — stream → map →
clone-at-commit → inject test → provision → certify → clean up — is proven end-to-end against real GitHub
repos, and the storage constraint is retired: 283 MB of metadata, bounded ephemeral clones + per-repo venvs,
nothing else on disk.

### 6.8 Lift on the diversified corpus: the wall persists, now measured with the graded metric

With provisioning and mirror-caching in place, the corpus can grow, but building a *large* diverse certified
corpus is provisioning-bound: SWE-rebench has few instances per repo (21,336 instances across thousands of
repos), so ≈one repo must be provisioned per certified task, at ~20–140 s each and a ~1/9 certify-of-provision
rate — roughly one certified task per 5–15 min on a laptop. That is the SWE-bench-was-built-for-Docker reality;
it argues for a cluster or the shipped `docker_image`s, not a fundamentally different design.

So the fastest current lift read is still the ready 89-task mined `-k` corpus. Difficulty-stratified
(hold out the 6 smallest tractable tasks, train on 24), `Qwen3.5-4B`, tinker-SFT on the gold diffs, scored
through the graded gate:

```
base_solved 0/6   trained_solved 0/6   binary lift 0
base_pass_fraction 0.389   trained_pass_fraction 0.389   fraction_lift 0.0
```

The point of using the **graded** metric here (§6.5) is that it can reveal partial learning that binary solves
hide — and it shows exactly none: base and trained produce the *same* pass_fraction. This is the same wall §6
has hit from every angle, now confirmed with the sensitive metric on the tractable slice: a 4B at a capability
plateau on real repo-internal tasks, where SFT/RL have nothing to amplify. Lift is real and reproducible on
*self-contained synthetic* tasks (§5, base 9→13/16); it does not appear on the real corpus at this base size
and corpus scale. The infrastructure to change that — storage-safe ingestion of 21k diverse tasks (§6.7),
provisioning that ~triples yield, and a mirror cache that makes RL rollouts cheap — is now in place, and the
measurement is instrumented (graded `fraction_lift`), so the threshold-crossing is what we watch for as the
certified corpus grows and compute scales.

### 6.9 The compute axis, tested — and the real bottleneck, localized to diff-format

To test whether *compute* (not corpus) is the constraint, we ran a 16×-larger RL-on-gate loop than the §6.6
proof: 16 mined tasks × 16 distinct-tagged copies → 256 prompts → 64 planned steps, graded gate reward,
copy-major interleave so every task recurs throughout, per-step logging (`EPISODIC_SAO_VERBOSE`). It never
needed 64 steps to answer the question. Over the first 8 steps (two full passes over the 16 tasks in batches
of 4), the loop produced exactly **four** distinct `reward_mean` values — one per task-batch — and each equals
that batch's **empty-baseline mean** to ten decimals (batch t0–3 → 0.6606 = mean(0.833,0.909,0.400,0.500);
batch t4–7 → 0.250 = mean(0.500,0,0.500,0)), *unchanged* across passes despite the updates. Every rollout
scores the baseline; there is no positive-advantage signal. **More compute cannot help — the compute axis is
not the bottleneck.**

Why do rollouts never clear baseline, when the reward has healthy range (a no-model check: gold=1.00 on all
16 tasks, empty 0.00–0.91, 13/16 span ≥0.5)? A direct base-`Qwen3.5-4B` probe answers it: over 10 samples on
5 tasks, the model emits non-empty unified diffs (500–750 chars) that **apply 0/10 of the time** — so the gate
falls back to the unmodified tree and returns the empty baseline exactly. The wall on the mined corpus is not
"can't find the fix," it is **can't emit a unified diff that applies** against the exact source. That is a
format/tooling limit, and it explains every prior null (the §6.8 `0.389==0.389`, the §6.6 zero lift): the RL
and SFT objectives are flat because the policy's outputs collapse to baseline before quality is ever measured.

That reframed the question toward output format, so `eval/editfmt.py` implements two forgiving alternatives to
unified diffs — **whole-file** (`### FILE:` + a fenced complete file, written verbatim) and **line-anchored
numbered edits** (the file is shown with line numbers; the model emits `EDIT <path> <start>-<end>` blocks
applied bottom-to-top so earlier edits don't shift later line numbers) — and a controlled 3-format probe put
the same base `Qwen3.5-4B` through all three (8 tasks × 4 samples = 32/format):

```
unified-diff   applied  4/32   solved 0/32
whole-file     applied  7/32   solved 0/32   (highest apply rate — ~2× diff)
numbered       applied  2/32   solved 0/32   (lowest — a 4B won't emit the EDIT syntax)
```

The format **does** move apply-rate (whole-file ~2× diff; line-numbers are worst because a 4B can't follow the
unfamiliar edit grammar — the line-number trick that helps strong coding agents needs a model able to emit it),
but **every format solves 0/32.** The earlier one-off whole-file solve was noise. This is the decisive
correction to the "edit-format is the lever" read: removing the apply barrier just reveals that the edits are
*wrong* — the dominant wall is fix **correctness**, i.e. base capability on real repo-internal tasks, not
output format. Whole-file is also truncation-bound (it only applies when the complete file fits the token cap;
larger files never close their fence). Net across §6.6–6.9: **neither compute, reward shape, nor edit format is
the binding constraint on the real corpus — base model capability is.** The format infra is not wasted, though:
it removes the diff-apply confound that suppressed *every* prior model comparison (the "bigger model refuted"
verdicts all ran on diffs that apply ~13% of the time), so the honest next test is a stronger base model
through a forgiving format.

**That test lands the first real signal (`stronger_probe.py`).** Running `Qwen3.5-9B` over the same 8 tasks:
whole-file **applied 0/6** — Qwen3.5 is a reasoning model, so it spends the token budget on `<think>` and the
complete-file block truncates before it closes; but the **line-anchored numbered format applied 15/24 (63%)
and solved 3/24 (12.5%)**. That is the first non-trivial solve rate anywhere on the real corpus (the 4B was
0/32 in *every* format), and it vindicates the line-number design precisely where it should: the tiny edit
output survives the reasoning budget, and a 9B *can* emit the `EDIT <path> <start>-<end>` grammar a 4B cannot.
So the binding constraint is capability — but it is crossable, and crossing it requires the **right format for
the model**: line-anchored edits for a capable reasoning model, not unified diffs (which hid the 9B's ability
behind a 13%-apply confound) and not whole-file (which truncates). This is the concrete, evidenced path to the
first flywheel lift: an RL-on-gate run on `Qwen3.5-9B` with the numbered edit format and the graded reward,
where ~13% of rollouts already solve and many more partially apply — a real above-baseline signal for RL to
amplify, which no configuration before this had.

### 6.10 Running that loop: the reward moves, held lift does not — the blocker is now scale, not mechanism

We ran exactly that loop. `gate_numbered_edit_reward` is wired into the SAO trainer (the reward re-renders each
rollout through `editfmt.apply_numbered_edits` before the gate scores it), and `build_sao_rows(..., fmt="numbered")`
emits line-numbered prompts, so the training objective is now the same forgiving format the 9B can actually
emit. Config: `Qwen3.5-9B`, 16 mined tasks × 3 distinct-tagged copies → 48 prompts, batch 8 → 6 steps, graded
gate reward, 3072-token budget, per-step logging. (The Tinker SDK had to be upgraded 0.22.7 → 0.27.0 first —
the backend now rejects the old version.)

For the first time on the real corpus, **the reward is non-degenerate and climbs.** Per-step `reward_mean`
over the six updates: `0.175 → 0.188 → 0.279 → 0.276 → 0.449 → 0.104` (mean 0.245); every step produced a
real gradient update with advantage swinging both signs against the running-mean baseline. Contrast §6.9,
where every rollout scored the empty baseline to ten decimals and the reward was frozen: here the policy emits
edits that apply and partially pass, so RL has actual signal to move. That confirms the §6.9 prediction — the
mechanism (format + graded reward + capable model) now delivers a live, moving, above-baseline objective.

**But held lift is flat-to-slightly-negative.** Scoring 6 held tasks with graded advantage-over-empty: base
`0.111` vs trained `0.083` (`adv_lift −0.028`), base solved 2, trained solved 1 — four tasks tied, one
regressed, none gained. Six steps over 48 rollouts on 16 tasks is proof scale; the held delta is within noise
and the loop never reaches the overfit-then-generalize regime. The honest read: §6.6–6.9 localized the blocker
down a chain (not reward shape → not compute-on-4B → not corpus → diff-apply format → base capability), and
§6.10 clears the last mechanism question — with the right model and format, **RL-on-gate finally has a real
reward to climb.** What it does *not* yet have is enough of that climb to generalize. The remaining gap to a
demonstrated flywheel lift is compute/scale — more steps, more tasks inside the learnable band — not a new
mechanism. That is a materially different (and cheaper-to-close) place to be than "the objective is flat."

### 6.11 The first positive lift: a GRPO group-relative baseline

§6.10 blamed the flat held lift on scale. It was partly the baseline. Our SAO loop samples one rollout per
prompt and scores its advantage against a *running mean* of past rewards, keyed by prompt text — and the
copy-trick that gives us "more rollouts per prompt" hands each copy a distinct key, so the effective baseline
is the **global** running mean, not a per-task one. Advantage then conflates "this attempt beat my other
attempts on this task" with "this task happens to score above the global average," which points the gradient
in a muddy direction. Two running-mean runs on the same held set bear this out: **−0.028** lift at 6 steps and
**−0.139** at 18 steps, each *regressing* a previously-solved held task (one to −0.83).

The fix is the canonical GRPO estimator: sample G rollouts of the *same* prompt in one step and set each
rollout's advantage to `reward − group_mean` (optionally ÷ group_std). `sao.group_advantages` implements it and
the SAO trainer uses it whenever `group_size > 1` (`group_size == 1` keeps the running-mean path, so the change
is backward-compatible). It carries a free curriculum property: a group whose G rewards are identical has zero
variance, so every advantage is 0 and the task contributes no gradient — unlearnable and already-solved tasks
drop out automatically, without a separate learnable-band filter.

Re-running the §6.10 configuration (`Qwen3.5-9B`, numbered format, graded gate reward, same 6 held tasks) with
`group_size = 4`, batch 4, 8 steps:

```
baseline        steps   adv_lift   solved   regressions
running-mean       6     -0.028      2→1         1
running-mean      18     -0.139      2→1         1  (one task to -0.83)
GRPO group         8     +0.167      1→2         0
```

**This is the first positive held lift on the real corpus.** Held task `62da8ed` goes `0.00 → 1.00` — the
trained model fully solves a task it never trained on and that the base model failed — with **zero
regressions**. The only variable changed across these three runs is the baseline; same model, format, held set,
and comparable step count. The GRPO signature is clean throughout: `advantage_mean ≈ 0` (machine epsilon) at
every step, the defining property of group-centred advantages. Mean training reward also rises (0.245/0.312 →
0.382).

Read honestly, this is a *proof*, not a scaled result: N = 6 held, +1 solve, and one task drives most of the
advantage. But it is positive, regression-free, and cleanly attributable to a single mechanism change — the
lever the SkyRL/Mercor 397B guide flags (their whole Step 4 is baseline/loss-aggregation shape) validated on
our stack. The next gains are now genuinely a scale question — more held tasks to measure on, more steps, a
larger learnable band — on a mechanism that finally points the right way.

### 6.12 The learnable band is corpus-bound, not model-bound — the constraint is the tasks

§6.11 measured lift on the *learnable band* — the tasks where the base model shows headroom and within-group
reward variance (the exact condition under which GRPO has a gradient). We added a banding pre-pass: sample the
base model G times per task at temperature 1.0, keep a task only if `max(advantage-over-empty) > 0` and the G
rewards have spread. Of the 30 smallest-diff, all-Python mined tasks, only **7 (~23%)** are in-band for
`Qwen3.5-9B`. On that band GRPO gives a clean +0.056 lift, solved 2→3, zero regressions — and notably rescues
`07068ddd`, the exact task the running-mean 18-step run had regressed to −0.83.

The obvious hypothesis was that a stronger base model would *widen* the band — more tasks crossing into
learnable, lifting the aggregate. We tested it directly on `Qwen3.6-35B-A3B` (the model Mercor post-trained to
beat Opus 4.5; MoE, 3B active, and on Tinker's live rates actually *cheaper* per token than our dense 9B —
sample $1.335 vs $1.995 per M). Same 30 candidates, same pipeline. The band came back **7/30 — identical to the
9B.** Not one new task became learnable. The 35B-A3B is genuinely stronger *within* the band (mean training
reward 0.49 vs 0.30; base solves 2/3 of the held band vs the 9B's lower rate), but the band *boundary* did not
move. Held lift was 0.0 with churn (one held solve gained, one lost) at the tiny 3-step/4-task training scale.

That is the decisive reframing of this whole document. The constraint chain terminates: **not reward-shape
(§6.5) → not compute (§6.9) → not baseline (§6.11 fixed it) → not model capability (§6.12) → the corpus.** The
~77% of mined tasks that neither model can touch are not blocked by model weakness — a 4×-larger model unlocks
none of them. They are dead because they are not learnable-solvable in our setup: not test-necessary in a way
the numbered format can express, or not provisioned to a runnable RED state, or requiring edits outside the
format's reach. This is precisely the SkyRL/Mercor headline read from the other side — *"algorithm choices
mattered less than the data"* — and it says the lever for aggregate lift is **more learnable tasks**, not a
bigger model, more compute, or a new reward. The tasks have to be good: their benchmark is 480 *realistic
knowledge-work* tasks for a reason. Episodic's moat is the data pipeline (capture → mine → certify → provision),
and that is where the next work belongs: grow the fraction of the corpus that lands in the learnable band.

## 7. Reproduce it

```bash
# Ingest SWE-rebench as metadata-only episodes (streams the REST API; no repo/parquet stored):
python -c "from episodic.eval import swerebench; print(len(swerebench.ingest(limit=200)))"

# Provision a repo's deps-venv and certify one instance through the gate (per-repo cached):
#   EPISODIC_PROVISION_DIR=/tmp/prov EPISODIC_MIRROR_DIR=/tmp/mirrors python harvest_certified.py

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
