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

**Honest read.** The infrastructure — mine → certify → gate → train (SFT / agentic / STaR) → score — runs
end-to-end on real tasks, and on the easy real subset it produces a small positive lift. But meaningful,
noise-separable lift on *hard* real repo-internal tasks is not there at 4B with single-model SFT or 2-turn
agentic. The levers that remain are a bigger model (now justified, since real tasks leave headroom), more
mined tasks from more repos (to shrink the noise band), and a real multi-step agent (tool use, not a single
diff) — which is what actually closes multi-file bugs.

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
