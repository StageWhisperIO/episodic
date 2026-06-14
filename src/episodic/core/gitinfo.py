import subprocess
from pathlib import Path


def _run(args, cwd):
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_available(cwd):
    return _run(["git", "rev-parse", "--is-inside-work-tree"], cwd) == "true"


def repo_root(cwd):
    return _run(["git", "rev-parse", "--show-toplevel"], cwd)


def repo_state(cwd):
    state = {
        "root": None,
        "repo": None,
        "remote_url": None,
        "branch": None,
        "base_commit": None,
        "dirty": False,
    }
    root = repo_root(cwd)
    if not root:
        return state
    state["root"] = root
    state["repo"] = Path(root).name
    state["remote_url"] = _run(["git", "remote", "get-url", "origin"], cwd)
    state["branch"] = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd)
    state["base_commit"] = _run(["git", "rev-parse", "HEAD"], cwd)
    status = _run(["git", "status", "--porcelain"], cwd)
    state["dirty"] = bool(status)
    return state


def working_diff(cwd, base_commit=None):
    if base_commit:
        diff = _run(["git", "diff", base_commit, "--no-color"], cwd)
        if diff is not None:
            return diff
    return _run(["git", "diff", "HEAD", "--no-color"], cwd) or ""


def head_commit(cwd):
    return _run(["git", "rev-parse", "HEAD"], cwd)


def name_status(cwd, base_commit):
    if not base_commit:
        return _run(["git", "diff", "HEAD", "--name-status", "--no-color"], cwd) or ""
    return _run(["git", "diff", base_commit, "--name-status", "--no-color"], cwd) or ""
