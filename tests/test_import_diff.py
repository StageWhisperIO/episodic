import subprocess

from episodic.core.episode import _build_diffs


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def test_import_never_uses_live_working_tree(tmp_path):
    repo = str(tmp_path)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t.dev")
    _git(repo, "config", "user.name", "t")
    (tmp_path / "a.py").write_text("x = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    base = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()

    (tmp_path / "unrelated.py").write_text("junk = 1\n")

    repo_state = {"root": repo, "base_commit": base}
    events = [{"type": "file_write", "data": {"file_path": str(tmp_path / "created.py")}}]
    diffs, source = _build_diffs(repo_state, repo, events, live=False)

    assert source == "events-untrusted"
    assert not any(d["file"].endswith("unrelated.py") for d in diffs)
    assert any(d["file"].endswith("created.py") for d in diffs)
