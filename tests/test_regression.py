import subprocess

from episodic.github import regression


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _rev(repo, ref="HEAD"):
    return subprocess.run(
        ["git", "rev-parse", ref], cwd=repo, capture_output=True, text=True
    ).stdout.strip()


def _repo_with_regression(tmp_path):
    repo = str(tmp_path)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t.dev")
    _git(repo, "config", "user.name", "t")
    (tmp_path / "f.py").write_text("line1\nbug\nline3\n")
    (tmp_path / "other.py").write_text("untouched\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "introduce bug")
    bug_commit = _rev(repo)

    (tmp_path / "f.py").write_text("line1\nfixed\nline3\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "fix the bug")
    fix_commit = _rev(repo)
    return repo, bug_commit, fix_commit


def test_culprit_commits_blames_the_bug(tmp_path):
    repo, bug_commit, fix_commit = _repo_with_regression(tmp_path)
    culprits, files = regression.culprit_commits(fix_commit, repo)
    assert bug_commit in culprits
    assert files == {"f.py"}


def test_map_to_episodes_commit_and_file(tmp_path):
    repo, bug_commit, fix_commit = _repo_with_regression(tmp_path)
    episodes = [
        {"id": "ep_exact", "outcome": {"commit": bug_commit}, "diffs": [{"file": "f.py"}]},
        {"id": "ep_file", "outcome": {"commit": "deadbeef"}, "diffs": [{"file": "f.py"}]},
        {"id": "ep_other", "outcome": {"commit": "cafef00d"}, "diffs": [{"file": "other.py"}]},
    ]
    report = regression.regression_report(fix_commit, repo, episodes)
    by_id = {imp["episode_id"]: imp for imp in report["implicated"]}

    assert by_id["ep_exact"]["via"] == "commit"
    assert by_id["ep_exact"]["blamed_lines"] >= 1
    assert by_id["ep_file"]["via"] == "file"
    assert "ep_other" not in by_id
