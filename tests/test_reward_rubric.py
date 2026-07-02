from episodic.core import reward


def _episode(tests=None, diffs=None):
    return {
        "steps": [],
        "tests": tests or [],
        "diffs": diffs or [],
        "commands": [],
        "human_feedback": [],
        "outcome": {"status": "open"},
        "stats": {"file_edits": 1, "file_reads": 1, "shell_commands": 1, "denials": 0, "cost_usd": 0.0},
    }


def _test(passed=0, failed=0, errors=0, ok=None):
    total = passed + failed + errors
    if ok is None:
        ok = total > 0 and failed == 0 and errors == 0
    return {"passed": passed, "failed": failed, "errors": errors, "total": total, "ok": ok}


def test_weights_sum_to_one_and_include_rubric():
    assert "rubric" in reward.WEIGHTS
    assert round(sum(reward.WEIGHTS.values()), 6) == 1.0


def test_reward_vector_exposes_rubric_and_composite_in_range():
    rv = reward.reward_vector(_episode(tests=[_test(passed=3)], diffs=[{"file": "a.py"}]))
    assert 0.0 <= rv["rubric"] <= 1.0
    assert 0.0 <= rv["composite"] <= 1.0
    assert rv["components"]["rubric"]["hard_pass"] is True
    assert "rubric" in rv["components"]["normalized"]


def test_env_blocked_run_lowers_composite_via_rubric():
    green = reward.reward_vector(_episode(tests=[_test(passed=3)], diffs=[{"file": "a.py"}]))
    blocked = reward.reward_vector(_episode(tests=[_test(errors=6)], diffs=[{"file": "a.py"}]))
    assert blocked["rubric"] < green["rubric"]
    assert blocked["composite"] < green["composite"]


def test_judge_passthrough_changes_composite():
    from episodic.core import rubric

    episode = _episode(tests=[_test(passed=3)], diffs=[{"file": "a.py"}])
    base = reward.reward_vector(episode)["composite"]
    judged = reward.reward_vector(
        episode, judge=rubric.openrubrics_judge(lambda prompt: "SCORE: 0.0"))["composite"]
    assert judged < base
