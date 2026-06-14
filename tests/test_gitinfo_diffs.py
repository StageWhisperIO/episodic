import subprocess

from episodic.core import gitinfo, diffparse


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def test_working_diff_includes_untracked_and_tracked(tmp_path):
    repo = str(tmp_path)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t.dev")
    _git(repo, "config", "user.name", "t")
    (tmp_path / "a.py").write_text("x = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    base = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()

    (tmp_path / "a.py").write_text("x = 2\ny = 3\n")
    (tmp_path / "new.py").write_text("def f():\n    return 1\n")

    patch = gitinfo.working_diff(repo, base)
    parsed = diffparse.parse_unified_diff(patch)
    files = {entry["file"]: entry for entry in parsed}

    assert "new.py" in files
    assert files["new.py"]["status"] == "added"
    assert files["new.py"]["additions"] >= 2
    assert files["new.py"]["unified"]

    assert "a.py" in files
    assert files["a.py"]["additions"] >= 1
