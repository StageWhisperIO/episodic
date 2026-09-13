import os
import subprocess
import tempfile

import pytest

from episodic.eval import editfmt, flywheel, gate, harnessevo, redgreen
from episodic.worldmodel import validate as wm


def test_render_prompt_matches_numbered_edit_prompt(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\ny = 2\n")
    task = "fix the thing"
    files = ["a.py"]
    prompt = harnessevo.render_prompt(harnessevo.DEFAULT_HARNESS, task, str(tmp_path), files)
    assert prompt == editfmt.numbered_edit_prompt(task, str(tmp_path), files)
    assert "1\tx = 1" in prompt
    assert "2\ty = 2" in prompt
    assert harnessevo.DEFAULT_HARNESS["intro"] in prompt
    assert harnessevo.DEFAULT_HARNESS["instructions"] in prompt


def test_render_prompt_uses_the_given_harness_instructions(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    harness = {"intro": "CUSTOM INTRO", "instructions": "CUSTOM INSTRUCTIONS"}
    prompt = harnessevo.render_prompt(harness, "task", str(tmp_path), ["a.py"])
    assert "CUSTOM INTRO" in prompt
    assert "CUSTOM INSTRUCTIONS" in prompt
    assert "1\tx = 1" in prompt


def test_build_runner_applies_a_known_edit_block(tmp_path):
    (tmp_path / "m.py").write_text("L1\nL2\n")

    def generate(model, messages):
        return "EDIT m.py 1-2\nNEW1\nNEW2\nENDEDIT"

    runner = harnessevo.build_runner(harnessevo.DEFAULT_HARNESS, generate, ["m.py"])
    log, code = runner("candidate", str(tmp_path), "fix it")
    assert code == 0
    assert (tmp_path / "m.py").read_text() == "NEW1\nNEW2\n"


def test_build_runner_reports_failure_when_nothing_applies(tmp_path):
    (tmp_path / "m.py").write_text("L1\n")

    def generate(model, messages):
        return "no edit here"

    runner = harnessevo.build_runner(harnessevo.DEFAULT_HARNESS, generate, ["m.py"])
    _, code = runner("candidate", str(tmp_path), "fix it")
    assert code == 1


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    base = tmp_path_factory.mktemp("harnessevo")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("EPISODIC_HOME", str(base / "home"))
        yield redgreen.generate_corpus(str(base / "repos"), variants=1, save=False)


def _corrected_source(episode, path):
    root = episode["repo_state"]["root"]
    original = open(f"{root}/{path}").read()
    with tempfile.TemporaryDirectory() as work:
        target = f"{work}/{path}"
        os.makedirs(os.path.dirname(target), exist_ok=True) if os.path.dirname(path) else None
        open(target, "w").write(original)
        subprocess.run(["git", "init", "-q", work], check=True)
        proc = subprocess.run(["git", "-C", work, "apply", "--recount"],
                              input=wm._unified_diff(episode), text=True, capture_output=True)
        assert proc.returncode == 0, proc.stderr
        return open(target).read()


def _numbered_solution(episode, path):
    corrected = _corrected_source(episode, path)
    return f"EDIT {path} 1-100000\n" + corrected.rstrip("\n") + "\nENDEDIT"


def test_evolve_accepts_improving_harness_and_rejects_regressing_one(corpus):
    episode = next(ep for ep in corpus if flywheel.bug_class(ep) == "operator")
    path = episode["diffs"][0]["file"]
    solution = _numbered_solution(episode, path)
    marker = "USE THE MARKER TO SOLVE THIS"

    def generate(model, messages):
        content = messages[0]["content"]
        if marker in content:
            return solution
        return "not a real edit"

    calls = {"n": 0}
    improved_harness = {"intro": harnessevo.DEFAULT_HARNESS["intro"],
                        "instructions": harnessevo.DEFAULT_HARNESS["instructions"] + "\n" + marker}
    regressed_harness = {"intro": harnessevo.DEFAULT_HARNESS["intro"],
                         "instructions": harnessevo.DEFAULT_HARNESS["instructions"]}

    def propose(current_harness, failures):
        calls["n"] += 1
        if calls["n"] == 1:
            assert failures and failures[0]["id"] == episode["id"]
            return improved_harness
        return regressed_harness

    result = harnessevo.evolve([episode], generate, propose, rounds=2, minibatch=1, seed=0)

    assert calls["n"] == 2
    assert [round_info["accepted"] for round_info in result["history"]] == [True, False]
    assert result["best"] == improved_harness
    assert result["pool"] == [harnessevo.DEFAULT_HARNESS, improved_harness]


def test_evolve_is_deterministic_under_a_fixed_seed(corpus):
    episode = next(ep for ep in corpus if flywheel.bug_class(ep) == "operator")
    path = episode["diffs"][0]["file"]
    solution = _numbered_solution(episode, path)
    marker = "USE THE MARKER TO SOLVE THIS"

    def generate(model, messages):
        return solution if marker in messages[0]["content"] else "not a real edit"

    def propose(current_harness, failures):
        return {"intro": harnessevo.DEFAULT_HARNESS["intro"],
                "instructions": harnessevo.DEFAULT_HARNESS["instructions"] + "\n" + marker}

    first = harnessevo.evolve([episode], generate, propose, rounds=1, minibatch=1, seed=7)
    second = harnessevo.evolve([episode], generate, propose, rounds=1, minibatch=1, seed=7)
    assert first["best"] == second["best"]
    assert [r["accepted"] for r in first["history"]] == [r["accepted"] for r in second["history"]]


def test_score_harness_reports_mean_pass_fraction_and_solved_count(corpus):
    episode = next(ep for ep in corpus if flywheel.bug_class(ep) == "operator")
    path = episode["diffs"][0]["file"]
    solution = _numbered_solution(episode, path)

    def generate(model, messages):
        return solution

    result = harnessevo.score_harness(harnessevo.DEFAULT_HARNESS, [episode], generate)
    assert result == {"mean_pass_fraction": 1.0, "solved": 1, "n": 1}
