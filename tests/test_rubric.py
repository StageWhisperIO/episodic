from episodic.core import rubric


def _episode(steps=None, tests=None, diffs=None, denials=0):
    return {
        "steps": steps or [],
        "tests": tests or [],
        "diffs": diffs or [],
        "stats": {"denials": denials},
    }


def _test(passed=0, failed=0, errors=0, ok=None):
    total = passed + failed + errors
    if ok is None:
        ok = total > 0 and failed == 0 and errors == 0
    return {"passed": passed, "failed": failed, "errors": errors, "total": total, "ok": ok}


def test_no_tests_makes_test_criteria_not_applicable():
    result = rubric.score_episode(_episode())
    by_id = {row["id"]: row for row in result["criteria"]}
    assert by_id["has_test_evidence"]["satisfied"] == 0.0
    assert by_id["tests_end_green"]["applicable"] is False


def test_judge_criteria_are_flagged_and_excluded():
    result = rubric.score_episode(_episode(tests=[_test(passed=3)]))
    judged = [row for row in result["criteria"] if row["id"] == "explanation_quality"]
    assert judged and judged[0]["applicable"] is False and judged[0]["satisfied"] is None


def test_hard_violation_halves_the_score():
    green = _episode(tests=[_test(passed=5)], diffs=[{"file": "a.py"}])
    blocked = _episode(tests=[_test(errors=6)], diffs=[{"file": "a.py"}])
    assert rubric.score_episode(green)["hard_pass"] is True
    result = rubric.score_episode(blocked)
    assert result["hard_pass"] is False
    assert result["score"] == round(result["base"] * rubric.HARD_PENALTY, 4)


def test_reproduce_before_fix_rewards_test_first_ordering():
    test_first = _episode(steps=[
        {"index": 0, "type": "shell_command", "input": {"command": "pytest -q"}},
        {"index": 1, "type": "file_edit", "input": {"file_path": "a.py"}},
    ])
    edit_first = _episode(steps=[
        {"index": 0, "type": "file_edit", "input": {"file_path": "a.py"}},
        {"index": 1, "type": "shell_command", "input": {"command": "pytest -q"}},
    ])
    by = lambda ep: {r["id"]: r["satisfied"] for r in rubric.score_episode(ep)["criteria"]}
    assert by(test_first)["reproduce_before_fix"] == 1.0
    assert by(edit_first)["reproduce_before_fix"] == 0.2


def test_scoped_change_penalizes_broad_diffs():
    tight = _episode(diffs=[{"file": f"f{i}.py"} for i in range(2)])
    broad = _episode(diffs=[{"file": f"f{i}.py"} for i in range(12)])
    by = lambda ep: {r["id"]: r["satisfied"] for r in rubric.score_episode(ep)["criteria"]}
    assert by(tight)["change_is_scoped"] == 1.0
    assert by(broad)["change_is_scoped"] == 0.3


def test_empty_episode_scores_none():
    assert rubric.score_episode(_episode())["score"] is not None
    assert rubric.rubric_reward({"steps": [], "tests": [], "diffs": [], "stats": {}}) is not None


def test_judge_scores_the_judge_only_criteria():
    episode = _episode(tests=[_test(passed=3)], diffs=[{"file": "a.py"}])
    judge = rubric.openrubrics_judge(lambda prompt: "SCORE: 0.8\nclear explanation")
    result = rubric.score_episode(episode, judge=judge)
    judged = {row["id"]: row for row in result["criteria"] if row["kind"] == "principle"}
    assert judged["explanation_quality"]["applicable"] is True
    assert judged["explanation_quality"]["satisfied"] == 0.8
    assert judged["correct_beyond_tests"]["satisfied"] == 0.8


def test_openrubrics_judge_parses_score_and_defaults_to_zero():
    assert rubric._parse_verdict("SCORE: 1.0 great")[0] == 1.0
    assert rubric._parse_verdict("SCORE: 0.35 ok")[0] == 0.35
    assert rubric._parse_verdict("no score here")[0] == 0.0


def test_judge_criteria_participate_in_score():
    episode = _episode(tests=[_test(passed=3)], diffs=[{"file": "a.py"}])
    without = rubric.score_episode(episode)["score"]
    with_low_judge = rubric.score_episode(
        episode, judge=rubric.openrubrics_judge(lambda prompt: "SCORE: 0.0"))["score"]
    assert with_low_judge < without
