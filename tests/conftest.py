import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from episodic import new_episode
from episodic.core.reward import reward_vector


def _good_episode():
    episode = new_episode(
        id="ep_test_good",
        agent="claude-code",
        intent="Add a retry helper to the http client",
    )
    episode["repo_state"].update({
        "root": "/tmp/repo",
        "repo": "demo",
        "branch": "feature/retry",
        "base_commit": "abc123",
        "remote_url": "https://github.com/acme/demo.git",
    })
    episode["steps"] = [
        {"index": 0, "ts": "2026-06-14T10:00:00+00:00", "type": "user_prompt", "tool": None,
         "intent": "Add a retry helper", "input": {"prompt": "Add a retry helper to the http client"},
         "observation": "", "approved": None, "duration_ms": None},
        {"index": 1, "ts": "2026-06-14T10:01:00+00:00", "type": "file_edit", "tool": "Edit",
         "intent": "edit src/http.py", "input": {"file_path": "src/http.py"},
         "observation": "applied", "approved": True, "duration_ms": None},
        {"index": 2, "ts": "2026-06-14T10:02:00+00:00", "type": "shell_command", "tool": "Bash",
         "intent": "pytest -q", "input": {"command": "pytest -q"},
         "observation": "3 passed", "approved": True, "duration_ms": None},
    ]
    episode["diffs"] = [{
        "file": "src/http.py", "status": "modified", "additions": 20, "deletions": 2,
        "unified": "diff --git a/src/http.py b/src/http.py\n@@ -1 +1 @@\n-old\n+new\n",
    }]
    episode["commands"] = [{
        "ts": "2026-06-14T10:02:00+00:00", "command": "pytest -q", "cwd": "/tmp/repo",
        "exit_code": 0, "output_excerpt": "3 passed", "is_test": True,
    }]
    episode["tests"] = [{
        "ts": "2026-06-14T10:02:00+00:00", "framework": "pytest", "command": "pytest -q",
        "passed": 3, "failed": 0, "skipped": 0, "total": 3, "ok": True,
    }]
    episode["human_feedback"] = [{"ts": "2026-06-14T10:03:00+00:00", "label": "useful", "note": None}]
    episode["outcome"].update({
        "status": "merged", "merged": True,
        "pr_url": "https://github.com/acme/demo/pull/7", "pr_number": 7,
    })
    episode["stats"].update({"file_edits": 1, "file_reads": 2, "shell_commands": 1, "tests_run": 1})
    episode["labels"] = ["useful"]
    episode["reward_vector"] = reward_vector(episode)
    return episode


def _bad_episode():
    episode = new_episode(
        id="ep_test_bad",
        agent="claude-code",
        intent="Add a retry helper to the http client",
    )
    episode["diffs"] = [{
        "file": "src/http.py", "status": "modified", "additions": 400, "deletions": 5, "unified": None,
    }]
    episode["tests"] = [{
        "ts": "2026-06-14T11:00:00+00:00", "framework": "pytest", "command": "pytest -q",
        "passed": 1, "failed": 2, "skipped": 0, "total": 3, "ok": False,
    }]
    episode["human_feedback"] = [{"ts": "2026-06-14T11:05:00+00:00", "label": "wrong", "note": None}]
    episode["outcome"].update({"status": "reverted", "reverted": True})
    episode["stats"].update({"file_edits": 1, "file_reads": 9, "shell_commands": 2, "tests_run": 1})
    episode["labels"] = ["wrong"]
    episode["reward_vector"] = reward_vector(episode)
    return episode


@pytest.fixture
def sample_episode():
    return _good_episode()


@pytest.fixture
def episodes():
    return [_good_episode(), _bad_episode()]
