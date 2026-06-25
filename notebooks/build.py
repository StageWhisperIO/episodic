import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

HERE = Path(__file__).resolve().parent


def md(text):
    return ("md", text.strip("\n"))


def code(text):
    return ("code", text.strip("\n"))


def build(name, cells):
    nb = new_notebook(metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    })
    for kind, text in cells:
        nb.cells.append(new_markdown_cell(text) if kind == "md" else new_code_cell(text))
    path = HERE / name
    nbformat.write(nb, str(path))
    return path


NB01 = [
    md("""
# 01 · Quickstart — capture, episode, summary, PR notes

Episodic turns a coding-agent session into one structured **CodingEpisode**. This notebook uses the
built-in synthetic factory (`episodic.testing`) so it runs with **no agent, no network, no models**.
"""),
    code("""
from episodic.testing import make_episode
from episodic.schema import validate_episode

ep = make_episode("ep_demo_1", intent="add retry to the http client",
                  outcome="merged", feedback=["useful"], passed=3, failed=0)
print("schema errors:", validate_episode(ep))
print("intent:", ep["intent"])
print("steps:", [s["type"] for s in ep["steps"]])
print("tests ok:", ep["tests"][0]["ok"], "| composite reward:", ep["reward_vector"]["composite"])
"""),
    md("Each step is an `(action -> observation)` pair — the raw material for everything downstream."),
    code("""
for s in ep["steps"]:
    print(f"[{s['type']:<14}] {str(s['input'])[:50]:<52} -> {s['observation'][:40]!r}")
"""),
    md("Heuristic summary + suggested PR notes — useful **before any ML exists**."),
    code("""
from episodic.core import summary as summary_mod
report = summary_mod.summarize(ep)
print(summary_mod.render_markdown(report))
"""),
    md("Feedback is a reward signal. A bad outcome scores lower than a merged one:"),
    code("""
good = make_episode("ep_good", outcome="merged", feedback=["useful"], passed=3, failed=0)
bad = make_episode("ep_bad", outcome="reverted", feedback=["wrong"], passed=1, failed=2)
print("merged+useful :", good["reward_vector"]["composite"])
print("reverted+wrong:", bad["reward_vector"]["composite"])
assert good["reward_vector"]["composite"] > bad["reward_vector"]["composite"]
print("OK")
"""),
]

NB02 = [
    md("""
# 02 · Datasets — SFT · DPO · Reward · RLDS · WM

Every export is JSONL, so training is just another filter on the pipe. We generate a synthetic
population and export every format.
"""),
    code("""
import json, tempfile, os
from episodic import exporters
from episodic.testing import make_episode, make_population

# A population with repeated intents so DPO can form chosen/rejected pairs.
pop = []
for i in range(6):
    pop.append(make_episode(f"ep_good_{i}", intent="add retry to http client",
                            outcome="merged", feedback=["useful"], passed=3, failed=0))
    pop.append(make_episode(f"ep_bad_{i}", intent="add retry to http client",
                            outcome="reverted", feedback=["wrong"], passed=1, failed=2))
pop += make_population(10, seed=1)
print("episodes:", len(pop), "| formats:", exporters.FORMATS)
"""),
    code("""
out = tempfile.mkdtemp()
counts = {}
for fmt in exporters.FORMATS:
    res = exporters.export(pop, fmt, os.path.join(out, fmt))
    counts[fmt] = res.get("count")
import pandas as pd
pd.DataFrame(sorted(counts.items()), columns=["format", "rows"])
"""),
    md("**SFT** = `intent -> good trajectory` (only episodes that passed the quality bar):"),
    code("""
sft = [json.loads(l) for l in open(os.path.join(out, "sft", "sft.jsonl"))]
print("rows:", len(sft))
print(json.dumps(sft[0]["messages"], indent=2)[:600])
"""),
    md("**DPO** = `chosen > rejected` preference pairs grouped by intent:"),
    code("""
dpo = [json.loads(l) for l in open(os.path.join(out, "dpo", "dpo.jsonl"))]
print("pairs:", len(dpo))
if dpo:
    print("chosen reward:", dpo[0]["meta"]["chosen_reward"], "| rejected:", dpo[0]["meta"]["rejected_reward"])
"""),
    md("**RLDS** = per-step `(observation, action, next_observation, reward, is_terminal, discount)` transitions:"),
    code("""
rlds = [json.loads(l) for l in open(os.path.join(out, "rlds", "rlds.jsonl"))]
step = rlds[0]["steps"][1]
print("action:", step["action"]["tool"], "| reward:", step["reward"], "| terminal:", step["is_terminal"], "| discount:", step["discount"])
"""),
    md("**WM** (world-model) = `history + action -> next observation`, as SFT messages whose assistant turn IS the observation. `trl-sft` can train a coding world model on this directly."),
    code("""
wm = [json.loads(l) for l in open(os.path.join(out, "wm", "wm.jsonl"))]
print("turn-level samples:", len(wm))
print("user (history+action):\\n", wm[0]["messages"][1]["content"][:300])
print("\\nassistant (target observation):\\n", wm[0]["messages"][2]["content"][:150])
"""),
]

NB03 = [
    md("""
# 03 · Reward vector & content-type fidelity

Two scorers: the **episode reward vector** (outcome-labeled quality) and the **content-type-aware
fidelity scorer** from the AgentWorld paper, used to judge how faithful a *predicted* observation is
to the real one.
"""),
    code("""
from episodic.core import reward
from episodic.testing import make_episode
rv = reward.reward_vector(make_episode("ep_r", outcome="merged", feedback=["useful"]))
import json
print(json.dumps(rv["components"], indent=2))
print("composite:", rv["composite"])
"""),
    md("""
## Content-type-aware fidelity (AgentWorld §4.2)

Not all observation content should be matched exactly. The scorer classifies content and masks
**runtime metadata** (timestamps, PIDs, hashes) so non-reproducible noise isn't penalized —
*"a PID of 42731 is as acceptable as the real 18204, provided both are valid."*
"""),
    code("""
from episodic import fidelity

cases = {
    "exact":        ("Build succeeded in 12 steps", "Build succeeded in 12 steps"),
    "runtime-noise":("done pid 42731 at 2026-06-14T11:22:33Z", "done pid 18204 at 2026-06-14T10:00:00Z"),
    "fabrication":  ("file written and database dropped", "file written"),
    "wrong-type":   ("Error: build failed", "Build succeeded"),
}
import pandas as pd
rows = []
for label, (pred, gt) in cases.items():
    s = fidelity.score_observation(pred, gt)
    rows.append({"case": label, **{d: s[d] for d in fidelity.DIMENSIONS}, "composite": s["composite"]})
df = pd.DataFrame(rows).set_index("case")
df
"""),
    md("The five dimensions (Format · Factuality · Consistency · Realism · Quality) per case:"),
    code("""
%matplotlib inline
import matplotlib.pyplot as plt
ax = df[list(fidelity.DIMENSIONS)].plot(kind="bar", figsize=(9, 4))
ax.set_ylim(0, 1.05); ax.set_ylabel("score"); ax.set_title("Fidelity dimensions by case")
plt.tight_layout(); plt.show()
"""),
    md("`classify_content` exposes the content-type split the scorer uses:"),
    code("""
info = fidelity.classify_content('{"pid": 42731, "ts": "2026-06-14T10:00:00Z", "ok": true}')
print(info)
"""),
]

NB04 = [
    md("""
# 04 · Replay harness & the RL loop

The replay harness re-runs a task at its base commit with another model and scores the result
(tests + diff overlap). This notebook builds a **real local git repo** and runs a **real replay** —
no stubs. Then it shows the closed RL loop in plan-only mode.
"""),
    code("""
import subprocess, tempfile, os
from pathlib import Path

work = Path(tempfile.mkdtemp())
os.environ["EPISODIC_HOME"] = str(work / ".episodic")
origin = work / "origin"; origin.mkdir()
(origin / "f.py").write_text("def f():\\n    return 1\\n")
(origin / "test_f.py").write_text("from f import f\\n\\ndef test_f():\\n    assert f() == 1\\n")

def git(*a):
    subprocess.run(["git", *a], cwd=origin, check=True, capture_output=True)
git("init", "-q"); git("config", "user.email", "t@t.dev"); git("config", "user.name", "t")
git("add", "-A"); git("commit", "-q", "-m", "base")
sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=origin, capture_output=True, text=True).stdout.strip()
print("origin:", origin, "| base commit:", sha[:10])
"""),
    code("""
from episodic import replay
from episodic.testing import make_episode

ep = make_episode("ep_replay_demo", intent="edit f.py",
                  remote_url=str(origin), repo_root=str(origin), base_commit=sha, files=("f.py",))
replay.create_replay(ep)
rid = replay.replay_id_for(ep)

# Without --execute the harness only returns a PLAN (no clone, no code runs):
plan = replay.run_replay(rid, "candidate-model")
print("executed:", plan["executed"], "| note:", plan["note"][:60])
"""),
    md("With `execute=True` it clones the repo, runs the recorded test command, and scores. Our toy 'model' is a runner script that edits `f.py`:"),
    code("""
runner = work / "runner.py"
runner.write_text("import os, sys\\nws = sys.argv[2]\\n"
                  "open(os.path.join(ws, 'f.py'), 'a').write('# touched by candidate\\\\n')\\n")
res = replay.run_replay(rid, "candidate", execute=True,
                        runner_cmd=f"python3 {runner} {{model}} {{workspace}} {{prompt_file}}")
print("ran:", res["ran"], "| produced:", res["produced_files"])
print("scores:", res["scores"])
"""),
    md("""
## The closed loop — `episodic loop`

quality-filter → train → replay-eval on held-out tasks → compare reward vs base → promote.
We run it **plan-only** (no `--execute`) so it trains the candidate and reports the plan without
cloning per-task. Pass `execute=True` to actually run replay-eval and decide promotion.
"""),
    code("""
from episodic import loop, store
from episodic.testing import make_episode

for i in range(8):
    e = make_episode(f"ep_loop_{i}", intent=f"task {i}",
                     remote_url=str(origin), repo_root=str(origin), base_commit=sha,
                     outcome="merged" if i % 2 == 0 else "accepted", files=("f.py",))
    store.save_episode(e)

manifest = loop.run_loop({
    "trainer": "command", "format": "sft", "min_composite": 0.0,
    "holdout_frac": 0.4, "seed": 0, "train_config": {"command": "true"},
    "out": str(work / "loopout"),
})
print("decision:", manifest["decision"], "| executed:", manifest["executed"])
print("train episodes:", len(manifest["train_ids"]), "| holdout:", len(manifest["holdout_ids"]))
"""),
]

NB05 = [
    md("""
# 05 · Language World Models for coding agents (AgentWorld)

This is the centerpiece. From *Qwen-AgentWorld: Language World Models for General Agents* (2026): a
**world model** predicts the next **observation** given history + action. Episodic already captures
exactly those `(action -> observation)` pairs, so it is a data factory for a *coding* world model.

We implement and demonstrate, model-free:
1. trajectory → turn expansion
2. the Echo-Trap one-turn-per-trajectory RL pool
3. OOD split by data-source
4. the `wm` SFT export (assistant = observation)
5. **EpisodicWorldBench** with content-type fidelity
6. double-blind **Turing-test** judge calibration
7. the **hybrid** rule + rubric judge
"""),
    code("""
from episodic import worldmodel
from episodic.testing import make_episode

ep = make_episode("ep_wm", intent="add retry to http client", files=("src/http.py",))
samples = worldmodel.expand_turns(ep)
print("turns in episode:", len(ep["steps"]), "-> prediction samples:", len(samples))
s = samples[-1]
print("\\n--- one world-model sample (history+action -> observation) ---")
print(s["history"][:300])
print("TARGET OBSERVATION:", repr(s["target_observation"][:80]))
"""),
    md("**Echo-Trap fix:** expanding every turn yields long shared prefixes that collapse RL reward variance. The RL pool keeps exactly one turn per trajectory:"),
    code("""
from episodic.testing import make_population
pop = make_population(40, seed=1, sources=[f"repo-{i}" for i in range(8)])
all_turns = worldmodel.wm_samples(pop, one_per_trajectory=False)
rl_pool   = worldmodel.wm_samples(pop, one_per_trajectory=True, seed=0)
print("all turns:", len(all_turns), "| one-per-trajectory RL pool:", len(rl_pool), "(== #episodes:", len(pop), ")")
"""),
    md("**OOD split by data-source** (AgentWorld principle iv): partition by repo so the benchmark probes generalization, not memorization. Train/holdout sources are disjoint:"),
    code("""
train, holdout, mapping = worldmodel.ood_split(pop, holdout_frac=0.4, seed=7)
train_src = {worldmodel.source_key(e) for e in train}
hold_src = {worldmodel.source_key(e) for e in holdout}
print("train sources:", sorted(train_src))
print("holdout sources:", sorted(hold_src))
print("disjoint:", train_src.isdisjoint(hold_src))
"""),
    md("""
## EpisodicWorldBench

Score a predictor's next-observation prediction with content-type fidelity. We compare baselines:
`oracle` (returns truth), `prefix` (copies the previous observation), `empty`. A real model plugs in
via `episodic worldbench --cmd '<your-model> {prompt_file}' --execute`.
"""),
    code("""
from episodic import worldbench
import pandas as pd
rows = []
for name in ["oracle", "prefix", "echo", "empty"]:
    overall = worldbench.run_bench(pop, name)["overall"]
    rows.append({"predictor": name, **{d: overall[d] for d in ["factuality", "realism", "composite"]}})
pd.DataFrame(rows).set_index("predictor")
"""),
    code("""
%matplotlib inline
import matplotlib.pyplot as plt
df = pd.DataFrame(rows).set_index("predictor")
ax = df[["composite"]].plot(kind="bar", legend=False, figsize=(7, 3.5), color="#4c72b0")
ax.set_ylim(0, 1.05); ax.set_ylabel("composite fidelity"); ax.set_title("EpisodicWorldBench by predictor")
plt.tight_layout(); plt.show()
"""),
    md("**Turing-test calibration** (AgentWorld §4.2): a discriminator tries to tell real observations from predicted ones. `indistinguishability ≈ 1` means it cannot — a perfect world model. The oracle is indistinguishable; `empty` is obvious:"),
    code("""
for name in ["oracle", "prefix", "empty"]:
    t = worldbench.turing_test(pop, name)
    print(f"{name:<8} discriminator_accuracy={t['discriminator_accuracy']:.2f}  indistinguishability={t['indistinguishability']:.2f}")
"""),
    md("**Hybrid rule + rubric reward** (AgentWorld §3.4.1): blend the rule-based fidelity with an optional LLM-judge. Here a stand-in judge; in production pass an Anthropic/OpenAI judge that returns per-dimension scores."),
    code("""
def stub_rubric_judge(predicted, target):
    # A real judge returns {Factuality, Consistency, ...} in [0,1]; here a length-aware stand-in.
    ratio = min(len(predicted), len(target)) / max(1, len(target))
    return {"factuality": round(ratio, 3), "quality": round(ratio, 3)}

rule = worldbench.run_bench(pop, "prefix")["overall"]["composite"]
hybrid = worldbench.run_bench(pop, "prefix", judge=stub_rubric_judge, judge_weight=0.5)["overall"]["composite"]
print("rule-only composite :", rule)
print("hybrid   composite :", hybrid)
"""),
    md("""
## Where this goes

- Export `wm` and train a coding world model with `trl-sft` (the assistant turn is the observation).
- Use it as a **decoupled simulator**: `episodic worldbench --cmd` points the bench at your model;
  the same predictor interface can drive *simulated* replay-eval — cheaper than cloning + running.
- **Agent–LWM co-evolution**: captured sessions → train WM → WM generates harder scenarios → agent
  improves → more captures. Episodic closes that loop.
"""),
]


NOTEBOOKS = {
    "01_quickstart.ipynb": NB01,
    "02_datasets.ipynb": NB02,
    "03_reward_and_fidelity.ipynb": NB03,
    "04_replay_and_loop.ipynb": NB04,
    "05_world_model_agentworld.ipynb": NB05,
}


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for name, cells in NOTEBOOKS.items():
        if only and only not in name:
            continue
        path = build(name, cells)
        print("wrote", path)
