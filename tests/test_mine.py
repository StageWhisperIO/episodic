import subprocess

import pytest

from episodic.eval import gate, mine


def _run(cwd, *args):
    subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True)


@pytest.fixture(scope="module")
def mined(tmp_path_factory):
    base = tmp_path_factory.mktemp("mine")
    repo = base / "lib"
    repo.mkdir()
    (repo / "solution.py").write_text("def f():\n    return 1\n")
    _run(repo, "git", "init", "-q")
    _run(repo, "git", "config", "user.email", "t@t.dev")
    _run(repo, "git", "config", "user.name", "t")
    _run(repo, "git", "add", "-A")
    _run(repo, "git", "commit", "-q", "-m", "initial")
    (repo / "solution.py").write_text("def f():\n    return 2\n")
    (repo / "test_solution.py").write_text("from solution import f\n\n\ndef test_f():\n    assert f() == 2\n")
    _run(repo, "git", "add", "-A")
    _run(repo, "git", "commit", "-q", "-m", "fix f and add covering test")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("EPISODIC_HOME", str(base / "home"))
        yield mine.mine_repo(str(repo), str(base / "scratch"), max_commits=10, save=False)


def test_mines_a_red_green_unit(mined):
    assert len(mined) == 1
    episode = mined[0]
    assert "swe" in episode["labels"] and "mined" in episode["labels"]
    assert [d["file"] for d in episode["diffs"]] == ["solution.py"]
    assert episode["tests"][0]["command"].endswith("test_solution.py")


def test_mined_unit_is_test_necessary(mined):
    result = gate.certify_episode(mined[0])
    assert result["test_necessary"] is True
