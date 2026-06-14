from episodic import github
from episodic.schema import new_episode, validate_episode
from episodic.core import reward


def test_should_refresh_predicate():
    assert github.should_refresh({"pr_number": 1, "status": "open"})
    assert github.should_refresh({"pr_url": "u", "status": "open", "ci_status": "pending"})
    assert not github.should_refresh({"status": "open"})
    assert not github.should_refresh({"pr_number": 1, "status": "abandoned"})
    assert not github.should_refresh(
        {"pr_number": 1, "status": "merged", "pr_state": "MERGED", "ci_status": "success"}
    )


def test_refresh_outcome_carries_regression_flags(monkeypatch):
    episode = new_episode(id="ep1")
    episode["outcome"].update({
        "pr_number": 7,
        "pr_url": "https://x/pr/7",
        "status": "open",
        "caused_regression": True,
        "reverted": True,
        "regression_commits": ["abc"],
    })

    monkeypatch.setattr(github, "gh_available", lambda: True)
    monkeypatch.setattr(github, "fetch_pr", lambda ref, cwd: {
        "merged": True, "state": "MERGED", "number": 7, "url": "https://x/pr/7",
        "statusCheckRollup": [{"conclusion": "SUCCESS"}], "reviewDecision": "APPROVED",
    })

    new_outcome = github.refresh_outcome(episode)
    assert new_outcome["status"] == "merged"
    assert new_outcome["ci_status"] == "success"
    assert new_outcome["caused_regression"] is True
    assert new_outcome["reverted"] is True
    assert new_outcome["regression_commits"] == ["abc"]


def test_regression_penalizes_reward():
    episode = new_episode(id="ep1")
    episode["outcome"]["status"] = "merged"
    clean = reward.reward_vector(episode)["composite"]

    episode["outcome"]["caused_regression"] = True
    penalized = reward.reward_vector(episode)["composite"]

    assert penalized < clean
    assert validate_episode(episode) == []
