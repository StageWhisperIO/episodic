import pathlib

import pytest

from episodic.eval import flywheel, gate, redgreen


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
