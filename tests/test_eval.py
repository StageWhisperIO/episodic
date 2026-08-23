import pathlib

import pytest

from episodic import replay
from episodic.eval import agentic, flywheel, gate, redgreen
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
