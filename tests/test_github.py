import shutil
import pytest

from episodic.github import outcome_from_pr_json, gh_available


def _episode():
    return {
        "id": "ep_test_github",
        "repo_state": {
            "root": "/tmp/repo",
            "repo": "demo",
            "remote_url": "https://github.com/acme/demo.git",
            "branch": "feature/test",
            "base_commit": "abc123",
            "dirty": False,
        },
    }


def test_merged_pr_all_checks_success():
    pr_json = {
        "state": "MERGED",
        "mergedAt": "2026-06-14T10:00:00Z",
        "merged": True,
        "number": 42,
        "url": "https://github.com/acme/demo/pull/42",
        "headRefName": "feature/test",
        "headRefOid": "def456",
        "statusCheckRollup": [
            {"state": "SUCCESS", "conclusion": "SUCCESS"},
            {"state": "SUCCESS", "conclusion": "SUCCESS"},
        ],
        "reviewDecision": "APPROVED",
    }
    outcome = outcome_from_pr_json(pr_json, _episode())
    assert outcome["status"] == "merged"
    assert outcome["merged"] is True
    assert outcome["ci_status"] == "success"
    assert outcome["pr_number"] == 42
    assert outcome["pr_url"] == "https://github.com/acme/demo/pull/42"
    assert outcome["review_decision"] == "APPROVED"
    assert outcome["linked_at"] is not None


def test_open_pr_with_failure_check():
    pr_json = {
        "state": "OPEN",
        "mergedAt": None,
        "merged": False,
        "number": 7,
        "url": "https://github.com/acme/demo/pull/7",
        "headRefName": "feature/test",
        "headRefOid": "ghi789",
        "statusCheckRollup": [
            {"state": "SUCCESS", "conclusion": "SUCCESS"},
            {"state": "FAILURE", "conclusion": "FAILURE"},
        ],
        "reviewDecision": None,
    }
    outcome = outcome_from_pr_json(pr_json, _episode())
    assert outcome["status"] == "open"
    assert outcome["merged"] is False
    assert outcome["ci_status"] == "failure"
    assert outcome["pr_number"] == 7


@pytest.mark.skipif(not shutil.which("gh"), reason="gh not available")
def test_gh_available_when_installed():
    assert gh_available() is True
