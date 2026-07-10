import os
import subprocess

from episodic.core.episode import _build_diffs, _touched_files


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def test_diff_trusted_when_head_matches_base(tmp_path):
    repo = str(tmp_path)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t.dev")
    _git(repo, "config", "user.name", "t")
    (tmp_path / "a.py").write_text("x = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    base = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    (tmp_path / "a.py").write_text("x = 2\n")

    repo_state = {"root": repo, "base_commit": base}
    events = [{"type": "file_edit", "data": {"file_path": str(tmp_path / "a.py")}}]
    diffs, source = _build_diffs(repo_state, repo, events)
    assert source == "git-working-tree"
    assert any(d["file"] == "a.py" and d["unified"] for d in diffs)


def test_working_tree_diff_scoped_to_touched_files(tmp_path):
    repo = str(tmp_path)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t.dev")
    _git(repo, "config", "user.name", "t")
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "b.py").write_text("y = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    base = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    (tmp_path / "a.py").write_text("x = 2\n")
    (tmp_path / "b.py").write_text("y = 2\n")

    repo_state = {"root": repo, "base_commit": base}
    events = [{"type": "file_edit", "data": {"file_path": str(tmp_path / "a.py")}}]
    diffs, source = _build_diffs(repo_state, repo, events)
    assert source == "git-working-tree"
    files = {d["file"] for d in diffs}
    assert files == {"a.py"}


def test_working_tree_diff_falls_back_when_touched_files_are_clean(tmp_path):
    repo = str(tmp_path)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t.dev")
    _git(repo, "config", "user.name", "t")
    (tmp_path / "a.py").write_text("x = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    base = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    (tmp_path / "unrelated.py").write_text("dirty = 1\n")

    repo_state = {"root": repo, "base_commit": base}
    events = [{"type": "file_edit", "data": {"file_path": str(tmp_path / "a.py")}}]
    diffs, source = _build_diffs(repo_state, repo, events)
    assert source == "events"
    files = {d["file"] for d in diffs}
    assert files == {"a.py"}


def test_diff_untrusted_when_head_advanced(tmp_path):
    repo = str(tmp_path)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t.dev")
    _git(repo, "config", "user.name", "t")
    (tmp_path / "a.py").write_text("x = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    base = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    (tmp_path / "a.py").write_text("x = 2\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "later unrelated commit")

    repo_state = {"root": repo, "base_commit": base}
    events = [{"type": "file_edit", "data": {"file_path": str(tmp_path / "a.py")}}]
    diffs, source = _build_diffs(repo_state, repo, events)
    assert source == "events-untrusted"
    assert any(d["file"] == "a.py" for d in diffs)


def test_bash_only_dirty_tree_captures_full_diff(tmp_path):
    repo = str(tmp_path)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t.dev")
    _git(repo, "config", "user.name", "t")
    (tmp_path / "a.py").write_text("x = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    base = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    (tmp_path / "a.py").write_text("x = 2\n")

    repo_state = {"root": repo, "base_commit": base}
    events = [{"type": "shell_command", "data": {"command": "sed -i '' 's/1/2/' a.py"}}]
    diffs, source = _build_diffs(repo_state, repo, events)
    assert source == "git-working-tree"
    files = {d["file"] for d in diffs}
    assert files == {"a.py"}


def test_touched_files_resolves_symlinked_prefix(tmp_path):
    real_root = tmp_path / "real"
    real_root.mkdir()
    link_root = tmp_path / "link"
    os.symlink(real_root, link_root)

    repo_state = {"root": str(real_root)}
    events = [{"type": "file_edit", "data": {"file_path": str(link_root / "src" / "a.py")}}]
    touched = _touched_files(repo_state, str(real_root), events)
    assert touched == {"src/a.py": "modified"}


def test_diff_scoping_resolves_symlinked_event_paths(tmp_path):
    real_repo = tmp_path / "real"
    real_repo.mkdir()
    repo = str(real_repo)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t.dev")
    _git(repo, "config", "user.name", "t")
    (real_repo / "a.py").write_text("x = 1\n")
    (real_repo / "b.py").write_text("y = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    base = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    (real_repo / "a.py").write_text("x = 2\n")
    (real_repo / "b.py").write_text("y = 2\n")

    link_repo = tmp_path / "link"
    os.symlink(real_repo, link_repo)

    repo_state = {"root": repo, "base_commit": base}
    events = [{"type": "file_edit", "data": {"file_path": str(link_repo / "a.py")}}]
    diffs, source = _build_diffs(repo_state, repo, events)
    assert source == "git-working-tree"
    files = {d["file"] for d in diffs}
    assert files == {"a.py"}
