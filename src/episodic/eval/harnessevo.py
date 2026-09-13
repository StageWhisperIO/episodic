import random

from . import editfmt, gate

DEFAULT_HARNESS = {
    "intro": "Fix the source so the failing tests pass. Files with line numbers:",
    "instructions": ("\nReply with one or more edits. Each replaces an INCLUSIVE 1-indexed line range with "
                      "new code (raw lines, no line-number prefixes), in EXACTLY this format:\n"
                      "EDIT <path> <start>-<end>\n<new lines>\nENDEDIT"),
}


def render_prompt(harness, task, workspace, files):
    parts = [task, "", harness["intro"]]
    for path in files:
        parts.append(f"\n### FILE: {path}\n{editfmt.number_lines(editfmt._read(workspace, path))}")
    parts.append(harness["instructions"])
    return "\n".join(parts)


def build_runner(harness, generate, files):
    def runner(model, workspace, prompt_text):
        prompt = render_prompt(harness, prompt_text, workspace, files)
        text = generate(model, [{"role": "user", "content": prompt}])
        applied, log = editfmt.apply_numbered_edits(text, workspace, files)
        return log, 0 if applied else 1

    return runner


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _score_rows(harness, episodes, generate):
    rows = []
    for episode in episodes:
        files = editfmt._files_of(episode)
        runner = build_runner(harness, generate, files)
        result = gate.graded_score(episode, runner)
        rows.append({"episode": episode, "pass_fraction": result["pass_fraction"], "ok": result["ok"]})
    return rows


def score_harness(harness, episodes, generate):
    rows = _score_rows(harness, episodes, generate)
    return {"mean_pass_fraction": _mean(row["pass_fraction"] for row in rows),
            "solved": sum(1 for row in rows if row["ok"]), "n": len(rows)}


def evolve(episodes, generate, propose, rounds=5, minibatch=4, seed=0):
    order = list(episodes)
    random.Random(seed).shuffle(order)
    n = len(order)
    current = DEFAULT_HARNESS
    pool = [current]
    history = []
    for round_index in range(rounds):
        if n == 0:
            break
        start = (round_index * minibatch) % n
        batch = [order[(start + offset) % n] for offset in range(min(minibatch, n))]
        baseline_rows = _score_rows(current, batch, generate)
        baseline_mean = _mean(row["pass_fraction"] for row in baseline_rows)
        failures = [row["episode"] for row in baseline_rows if row["pass_fraction"] < 1]
        candidate = propose(current, failures)
        candidate_rows = _score_rows(candidate, batch, generate)
        candidate_mean = _mean(row["pass_fraction"] for row in candidate_rows)
        accepted = candidate_mean > baseline_mean
        history.append({"round": round_index, "baseline_mean_pass_fraction": baseline_mean,
                         "candidate_mean_pass_fraction": candidate_mean, "accepted": accepted,
                         "harness": candidate, "failures": [episode["id"] for episode in failures]})
        if accepted:
            current = candidate
            pool.append(candidate)
    validation = [(score_harness(harness, episodes, generate)["mean_pass_fraction"], harness) for harness in pool]
    best = max(validation, key=lambda item: item[0])[1] if validation else current
    return {"best": best, "history": history, "pool": pool}
