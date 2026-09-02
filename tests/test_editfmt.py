import pytest

from episodic.eval import editfmt, flywheel
from episodic.worldmodel import validate as wm


def test_number_lines_is_one_indexed_tab_separated():
    assert editfmt.number_lines("a\nb") == "1\ta\n2\tb"


def test_apply_whole_file_labeled(tmp_path):
    (tmp_path / "solution.py").write_text("def add(a, b):\n    return a - b\n")
    text = "### FILE: solution.py\n```python\ndef add(a, b):\n    return a + b\n```"
    applied, _ = editfmt.apply_whole_file(text, str(tmp_path), ["solution.py"])
    assert applied
    assert (tmp_path / "solution.py").read_text() == "def add(a, b):\n    return a + b\n"


def test_apply_whole_file_single_block_fallback(tmp_path):
    (tmp_path / "s.py").write_text("x = 1\n")
    applied, _ = editfmt.apply_whole_file("```python\nx = 2\n```", str(tmp_path), ["s.py"])
    assert applied and (tmp_path / "s.py").read_text() == "x = 2\n"


def test_apply_whole_file_rejects_unlabeled_multi_file(tmp_path):
    (tmp_path / "a.py").write_text("1\n")
    applied, _ = editfmt.apply_whole_file("```python\n2\n```", str(tmp_path), ["a.py", "b.py"])
    assert not applied


def test_apply_numbered_edits_applies_bottom_up(tmp_path):
    (tmp_path / "m.py").write_text("L1\nL2\nL3\nL4\nL5\n")
    text = ("EDIT m.py 2-2\nTWO\nENDEDIT\n"
            "EDIT m.py 4-5\nFOUR\nFIVE_SIX\nENDEDIT")
    applied, _ = editfmt.apply_numbered_edits(text, str(tmp_path), ["m.py"])
    assert applied
    assert (tmp_path / "m.py").read_text() == "L1\nTWO\nL3\nFOUR\nFIVE_SIX\n"


def test_apply_numbered_edits_clamps_out_of_range(tmp_path):
    (tmp_path / "m.py").write_text("only\n")
    applied, _ = editfmt.apply_numbered_edits("EDIT m.py 5-9\nNEW\nENDEDIT", str(tmp_path), ["m.py"])
    assert applied and (tmp_path / "m.py").read_text() == "only\nNEW\n"


def test_apply_numbered_edits_ignores_unknown_file(tmp_path):
    (tmp_path / "m.py").write_text("a\n")
    applied, _ = editfmt.apply_numbered_edits("EDIT other.py 1-1\nx\nENDEDIT", str(tmp_path), ["m.py"])
    assert not applied


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    from episodic.eval import redgreen

    base = tmp_path_factory.mktemp("editfmt")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("EPISODIC_HOME", str(base / "home"))
        yield redgreen.generate_corpus(str(base / "repos"), variants=1, save=False)


def test_wholefile_runner_reaches_green(corpus):
    episode = next(ep for ep in corpus if flywheel.bug_class(ep) == "operator")
    path = episode["diffs"][0]["file"]
    fixed = wm._oracle_diff_runner

    def generate(model, messages):
        source = messages[0]["content"]
        assert "### FILE:" in source and path in source
        return _corrected_via_oracle(episode, path)

    runner = flywheel.edit_runner_for(episode, generate, fmt="wholefile")
    assert flywheel.solved(episode, runner) is True


def test_numbered_runner_reaches_green(corpus):
    episode = next(ep for ep in corpus if flywheel.bug_class(ep) == "operator")
    path = episode["diffs"][0]["file"]

    def generate(model, messages):
        source = messages[0]["content"]
        assert "\t" in source
        return _numbered_edit_via_oracle(episode, path)

    runner = flywheel.edit_runner_for(episode, generate, fmt="numbered")
    assert flywheel.solved(episode, runner) is True


def _apply_gold_to_string(episode, path):
    import subprocess
    import tempfile

    root = episode["repo_state"]["root"]
    original = open(f"{root}/{path}").read()
    with tempfile.TemporaryDirectory() as work:
        target = f"{work}/{path}"
        import os

        os.makedirs(os.path.dirname(target), exist_ok=True) if os.path.dirname(path) else None
        open(target, "w").write(original)
        subprocess.run(["git", "init", "-q", work], check=True)
        proc = subprocess.run(["git", "-C", work, "apply", "--recount"],
                              input=wm._unified_diff(episode), text=True, capture_output=True)
        assert proc.returncode == 0, proc.stderr
        return open(target).read()


def _corrected_via_oracle(episode, path):
    return f"### FILE: {path}\n```python\n{_apply_gold_to_string(episode, path)}```"


def _numbered_edit_via_oracle(episode, path):
    corrected = _apply_gold_to_string(episode, path)
    lines = corrected.splitlines()
    return f"EDIT {path} 1-100000\n" + "\n".join(lines) + "\nENDEDIT"
