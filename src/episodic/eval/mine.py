import os
import subprocess

from .. import store
from ..core import diffparse
from ..replay import _is_verifier_file
from ..schema import new_episode

_TS = "2026-08-23T10:00:00+00:00"
_BASE_BRANCH = "episodic-mined-base"


def _git(repo, *args, check=True):
    return subprocess.run(["git", "-C", repo, *args], check=check, capture_output=True, text=True)


def _pytest(repo, test_paths):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run(["python", "-m", "pytest", "-q", *test_paths],
                          cwd=repo, capture_output=True, text=True, env=env)


def _apply(repo, diff_text):
    if not diff_text.strip():
        return True
    if not diff_text.endswith("\n"):
        diff_text += "\n"
    for extra in (["--recount"], ["--recount", "--3way"]):
        result = subprocess.run(["git", "apply", "--whitespace=nowarn", *extra, "-"],
                                input=diff_text, cwd=repo, capture_output=True, text=True)
        if result.returncode == 0:
            return True
    return False


def _candidate_commits(repo, max_commits):
    log = _git(repo, "log", f"-n{max_commits}", "--no-merges", "--format=%H %P").stdout
    commits = []
    for line in log.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            commits.append((parts[0], parts[1]))
    return commits


def _split_commit(repo, parent, sha):
    full = _git(repo, "diff", parent, sha).stdout
    blocks = [b for b in diffparse.parse_unified_diff(full) if b["unified"]]
    tests = [b for b in blocks if _is_verifier_file(b["file"])]
    source = [b for b in blocks if not _is_verifier_file(b["file"])]
    return tests, source


def _build_from_commit(repo, out_dir, name, sha, parent, test_blocks, source_blocks):
    scratch = os.path.join(out_dir, f"{name}_{sha[:12]}")
    subprocess.run(["rm", "-rf", scratch], check=True)
    subprocess.run(["git", "clone", "-q", repo, scratch], check=True, capture_output=True)
    _git(scratch, "config", "user.email", "mine@episodic.dev")
    _git(scratch, "config", "user.name", "episodic-mine")
    _git(scratch, "checkout", "-q", "-b", _BASE_BRANCH, parent)

    test_diff = diffparse.join_unified(b["unified"] for b in test_blocks)
    source_diff = diffparse.join_unified(b["unified"] for b in source_blocks)
    test_paths = [b["file"] for b in test_blocks if not b["file"].endswith("conftest.py")]
    if not test_paths or not _apply(scratch, test_diff):
        subprocess.run(["rm", "-rf", scratch], check=True)
        return None
    _git(scratch, "add", "-A")
    _git(scratch, "commit", "-q", "-m", "mined base (test injected, red)")
    base_commit = _git(scratch, "rev-parse", "HEAD").stdout.strip()

    red = _pytest(scratch, test_paths)
    if red.returncode == 0 or not _apply(scratch, source_diff):
        subprocess.run(["rm", "-rf", scratch], check=True)
        return None
    green = _pytest(scratch, test_paths)
    if green.returncode != 0:
        subprocess.run(["rm", "-rf", scratch], check=True)
        return None
    _git(scratch, "checkout", "--", ".")

    test_command = "python -m pytest -q " + " ".join(test_paths)
    intent = (f"The tests in {', '.join(test_paths)} fail at the current commit:\n\n"
              f"```\n{red.stdout[-1200:].strip()}\n```\n\nFix the source so the tests pass.")
    episode = new_episode(id=f"mine_{name}_{sha[:12]}", intent=intent)
    episode["repo_state"].update({"root": scratch, "remote_url": None,
                                  "base_commit": base_commit, "branch": _BASE_BRANCH})
    episode["labels"] = ["swe", "mined"]
    episode["diffs"] = [{"file": b["file"], "status": "modified", "additions": b["additions"],
                         "deletions": b["deletions"], "unified": b["unified"]} for b in source_blocks]
    episode["commands"] = [{"ts": _TS, "command": test_command, "cwd": scratch, "exit_code": 0,
                            "output_excerpt": "", "is_test": True}]
    episode["tests"] = [{"ts": _TS, "framework": "pytest", "command": test_command,
                         "passed": None, "failed": 0, "skipped": 0, "total": None, "ok": True}]
    episode["outcome"]["status"] = "merged"
    episode["outcome"]["merged"] = True
    return episode


def mine_repo(repo, out_dir, max_commits=200, limit=None, save=True):
    repo = os.path.abspath(repo)
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.basename(repo.rstrip("/")) or "repo"
    episodes = []
    for sha, parent in _candidate_commits(repo, max_commits):
        test_blocks, source_blocks = _split_commit(repo, parent, sha)
        if not test_blocks or not source_blocks:
            continue
        try:
            episode = _build_from_commit(repo, out_dir, name, sha, parent, test_blocks, source_blocks)
        except (subprocess.SubprocessError, OSError):
            episode = None
        if episode is None:
            continue
        if save:
            store.save_episode(episode)
        episodes.append(episode)
        if limit and len(episodes) >= limit:
            break
    return episodes
