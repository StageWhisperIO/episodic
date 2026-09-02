import pathlib

import pytest

from episodic import replay
from episodic.eval import agentic, flywheel, gate, redgreen
from episodic.trainers import rewards
from episodic.worldmodel import validate as wm


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    base = tmp_path_factory.mktemp("eval")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("EPISODIC_HOME", str(base / "home"))
        yield redgreen.generate_corpus(str(base / "repos"), variants=1, save=False)


def test_corpus_is_red_at_base_and_green_after_fix(corpus):
    assert len(corpus) == 12
    classes = {flywheel.bug_class(ep) for ep in corpus}
    assert classes == {"operator", "bound", "ds", "exc", "logic", "str"}
    for ep in corpus:
        assert ep["diffs"][0]["unified"]
        assert "swe" in ep["labels"]


def test_gate_discriminates_on_every_task(corpus):
    report = gate.gate_report(corpus)
    assert report["all_clean"], [row["id"] for row in report["rows"] if not row["clean"]]
    for row in report["rows"]:
        assert row["oracle"]["ok"]
        assert not row["empty"]["ok"]
        assert not row["broken"]["ok"]


def test_stub_flywheel_measures_full_lift(corpus):
    train, held = flywheel.stratified_split(corpus, per_class_held=1)
    assert train and held
    lift = flywheel.oracle_vs_empty_lift(held)
    assert lift["base_solved"] == 0
    assert lift["trained_solved"] == len(held)
    assert lift["lift"] == len(held)


def test_gate_reverts_verifier_tampering(corpus):
    def tamper_runner(model, workspace, prompt_text):
        path = pathlib.Path(workspace) / "test_solution.py"
        path.write_text("def test_solve():\n    assert True\n")
        return "tampered", 0

    assert flywheel.solved(corpus[0], tamper_runner) is False


def test_hidden_assertions_are_not_leaked_to_the_prompt(corpus):
    target = next(ep for ep in corpus if "solve(2, 3) == 5" in ep["intent"])
    committed = (pathlib.Path(target["repo_state"]["root"]) / "test_solution.py").read_text()
    assert "solve(3, 5) == 8" in committed
    assert "solve(3, 5) == 8" not in target["intent"]


def test_gate_rejects_hardcoded_visible_constant(corpus):
    target = next(ep for ep in corpus if "solve(2, 3) == 5" in ep["intent"])

    def stub_runner(model, workspace, prompt_text):
        (pathlib.Path(workspace) / "solution.py").write_text("def solve(a, b):\n    return 5\n")
        return "stub", 0

    assert flywheel.solved(target, stub_runner) is False


def test_certify_accepts_test_necessary_tasks(corpus):
    report = gate.certify_corpus(corpus)
    assert report["certified"] == len(corpus), [row for row in report["rows"] if not row["test_necessary"]]


def test_certify_rejects_episode_without_a_diff():
    from episodic.schema import new_episode

    episode = new_episode(id="ep_no_diff", intent="nothing to grade")
    episode["diffs"] = []
    result = gate.certify_episode(episode)
    assert result["test_necessary"] is False
    assert result["reason"] == "no diff"


def test_agentic_runner_solves_after_test_feedback(corpus):
    episode = next(ep for ep in corpus if flywheel.bug_class(ep) == "operator")
    gold = wm._unified_diff(episode)
    calls = {"n": 0}

    def generate(model, messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return "```diff\nnonsense that will not apply\n```"
        return "```diff\n" + gold + "```"

    test_command, test_cwd = replay._resolve_test_command(episode, episode["repo_state"]["root"])
    runner = agentic.build_agentic_runner(generate, test_command, max_turns=3, test_cwd=test_cwd)
    assert flywheel.solved(episode, runner) is True
    assert calls["n"] >= 2


def test_tool_agent_reads_source_then_patches_to_green(corpus):
    episode = next(ep for ep in corpus if flywheel.bug_class(ep) == "operator")
    gold = wm._unified_diff(episode)
    src_path = next(line[len("+++ b/"):].strip() for line in gold.splitlines()
                    if line.startswith("+++ b/"))
    calls = {"n": 0, "second_prompt": ""}

    def generate(model, messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return f"READ {src_path}"
        calls["second_prompt"] = messages[0]["content"]
        return "```diff\n" + gold + "```"

    test_command, test_cwd = replay._resolve_test_command(episode, episode["repo_state"]["root"])
    runner = agentic.build_tool_agent(generate, test_command, max_steps=4, test_cwd=test_cwd)
    assert flywheel.solved(episode, runner) is True
    assert calls["n"] >= 2
    assert "not a file" not in calls["second_prompt"]
    assert "1\t" in calls["second_prompt"]


def test_tool_agent_read_is_sandboxed(corpus):
    episode = corpus[0]
    calls = {"prompt": ""}

    def generate(model, messages):
        calls["prompt"] = messages[0]["content"]
        return "READ ../../../etc/passwd"

    test_command, test_cwd = replay._resolve_test_command(episode, episode["repo_state"]["root"])
    runner = agentic.build_tool_agent(generate, test_command, max_steps=1, test_cwd=test_cwd)
    assert flywheel.solved(episode, runner) is False


def test_graded_score_opens_dynamic_range(corpus):
    episode = next(ep for ep in corpus if flywheel.bug_class(ep) == "operator")
    gold = wm._unified_diff(episode)
    oracle = gate.graded_score(episode, wm._oracle_diff_runner(gold))
    empty = gate.graded_score(episode, gate.empty_runner)
    assert oracle["ok"] is True and oracle["pass_fraction"] == 1.0
    assert empty["ok"] is False and empty["pass_fraction"] < 1.0
    assert oracle["pass_fraction"] > empty["pass_fraction"]


def test_graded_advantage_subtracts_the_empty_baseline(corpus):
    episode = next(ep for ep in corpus if flywheel.bug_class(ep) == "operator")
    gold = wm._unified_diff(episode)
    oracle = gate.graded_advantage(episode, wm._oracle_diff_runner(gold))
    assert oracle["advantage"] == oracle["pass_fraction"] - oracle["baseline_fraction"]
    assert oracle["advantage"] > 0
    empty = gate.graded_advantage(episode, gate.empty_runner)
    assert empty["advantage"] == 0.0


def test_reward_components_report_has_variance_and_correlation(corpus):
    report = gate.reward_components_report(corpus[:4], gate.empty_runner)
    assert report["n"] == 4
    assert set(report["pass_fraction"]) == {"mean", "var"}
    assert -1.0 <= report["corr_fraction_overlap"] <= 1.0
    assert len(report["rows"]) == 4


def test_learnable_band_keeps_only_the_gradient_band(corpus):
    episode = next(ep for ep in corpus if flywheel.bug_class(ep) == "operator")
    gold = wm._unified_diff(episode)
    calls = {"n": 0}

    def alternating(model, messages):
        calls["n"] += 1
        return "```diff\n" + gold + "```" if calls["n"] % 2 else "```diff\ngarbage\n```"

    def always_gold(model, messages):
        return "```diff\n" + gold + "```"

    banded = flywheel.learnable_band([episode], alternating, n=4)
    assert len(banded["banded"]) == 1
    assert 0 < banded["stats"][0]["solves"] < 4
    solved_out = flywheel.learnable_band([episode], always_gold, n=4)
    assert solved_out["banded"] == [] and solved_out["stats"][0]["solves"] == 4


def test_graded_gate_reward_returns_pass_fraction(corpus):
    episode = next(ep for ep in corpus if flywheel.bug_class(ep) == "operator")
    gold = wm._unified_diff(episode)
    reward = rewards.graded_gate_reward([episode])
    scores = reward(completions=["```diff\n" + gold + "```", "no diff here"],
                    episode_id=[episode["id"], episode["id"]])
    assert scores[0] == 1.0
    assert scores[1] < 1.0


def test_gate_pass_fraction_reward_reads_episode_from_meta(corpus):
    episode = next(ep for ep in corpus if flywheel.bug_class(ep) == "operator")
    gold = wm._unified_diff(episode)
    scores = rewards.gate_pass_fraction_reward(
        completions=["```diff\n" + gold + "```", "garbage", "```diff\n" + gold + "```"],
        meta=[episode, episode, None])
    assert scores[0] == 1.0
    assert scores[1] < 1.0
    assert scores[2] == 0.0


def test_build_sao_rows_carries_episode_meta(corpus, tmp_path):
    path = flywheel.build_sao_rows(corpus[:2], str(tmp_path / "sao.jsonl"))
    lines = [line for line in open(path).read().splitlines() if line]
    assert len(lines) == 2
    import json as _json
    row = _json.loads(lines[0])
    assert row["messages"][0]["role"] == "user"
    assert row["meta"]["id"] == corpus[0]["id"]


def _numbered_edit_solution(episode):
    import os
    import subprocess
    import tempfile

    path = episode["diffs"][0]["file"]
    original = (pathlib.Path(episode["repo_state"]["root"]) / path).read_text()
    with tempfile.TemporaryDirectory() as work:
        target = pathlib.Path(work) / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(original)
        subprocess.run(["git", "init", "-q", work], check=True)
        subprocess.run(["git", "-C", work, "apply", "--recount"],
                       input=wm._unified_diff(episode), text=True, capture_output=True, check=True)
        corrected = target.read_text()
    return f"EDIT {path} 1-100000\n" + corrected.rstrip("\n") + "\nENDEDIT"


def test_gate_numbered_edit_reward_scores_applied_edits(corpus):
    episode = next(ep for ep in corpus if flywheel.bug_class(ep) == "operator")
    scores = rewards.gate_numbered_edit_reward(
        completions=[_numbered_edit_solution(episode), "no edit here", "x"],
        meta=[episode, episode, None])
    assert scores[0] == 1.0
    assert scores[1] < 1.0
    assert scores[2] == 0.0


def test_build_sao_rows_numbered_embeds_line_numbers(corpus, tmp_path):
    import json as _json

    path = flywheel.build_sao_rows(corpus[:1], str(tmp_path / "n.jsonl"), fmt="numbered")
    row = _json.loads(open(path).read().splitlines()[0])
    content = row["messages"][0]["content"]
    assert "EDIT" in content and "1\t" in content
    assert row["meta"]["id"] == corpus[0]["id"]


def test_rollout_and_filter_keeps_only_gate_passing(corpus):
    solvable = next(ep for ep in corpus if flywheel.bug_class(ep) == "operator")
    unsolvable = next(ep for ep in corpus if flywheel.bug_class(ep) == "str")
    gold = wm._unified_diff(solvable)

    def generate(model, messages):
        if solvable["intent"] in messages[0]["content"]:
            return "```diff\n" + gold + "```"
        return "```diff\ngarbage that will not apply\n```"

    result = flywheel.rollout_and_filter([solvable, unsolvable], generate, k=1)
    assert result["solved"] == 1
    assert len(result["rows"]) == 1
    assert gold in result["rows"][0]["messages"][1]["content"]
