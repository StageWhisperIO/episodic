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


def test_clip_trajectory_is_a_noop_within_the_limit():
    text = "short trajectory that ends with the fix"
    assert rubric.clip_trajectory(text) == text


def test_clip_trajectory_keeps_head_and_tail_of_a_long_session():
    text = "START-INTENT-" + "a" * 6000 + "-END-FIX-AND-EXPLANATION"
    clipped = rubric.clip_trajectory(text)
    assert len(clipped) <= rubric.JUDGE_TRAJECTORY_LIMIT
    assert clipped.startswith("START-INTENT-")
    assert clipped.endswith("-END-FIX-AND-EXPLANATION")
    assert rubric.JUDGE_TRUNCATION_MARKER in clipped


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


def test_safe_judge_degrades_to_not_applicable_on_exception():
    def broken(episode, criterion):
        raise ValueError("boom")

    judge = rubric.safe_judge(broken)
    satisfied, reason = judge({}, {"desc": "x"})
    assert satisfied is None
    assert "boom" in reason


def test_safe_judge_passes_through_a_working_judge():
    judge = rubric.safe_judge(rubric.openrubrics_judge(lambda prompt: "SCORE: 0.6 fine"))
    satisfied, reason = judge({}, {"desc": "x"})
    assert satisfied == 0.6


def test_default_judge_scores_rubric_via_command_generate():
    episode = _episode(tests=[_test(passed=3)], diffs=[{"file": "a.py"}])
    judge = rubric.default_judge(command="sh -c \"printf 'SCORE: 0.8 clear explanation'\"", timeout=5)
    result = rubric.score_episode(episode, judge=judge)
    judged = {row["id"]: row for row in result["criteria"] if row["kind"] == "principle"}
    assert judged["explanation_quality"]["applicable"] is True
    assert judged["explanation_quality"]["satisfied"] == 0.8
    assert judged["correct_beyond_tests"]["satisfied"] == 0.8


def test_default_judge_degrades_gracefully_on_unauthenticated_labeler():
    episode = _episode(tests=[_test(passed=3)], diffs=[{"file": "a.py"}])
    judge = rubric.default_judge(
        command="sh -c \"printf 'Not logged in' 1>&2; exit 1\"", timeout=5)
    result = rubric.score_episode(episode, judge=judge)
    judged = {row["id"]: row for row in result["criteria"] if row["id"] == "explanation_quality"}
    assert judged["explanation_quality"]["applicable"] is False
    assert judged["explanation_quality"]["satisfied"] is None
    assert result["score"] is not None


class _FakeCritic:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def value(self, texts):
        self.calls.append(list(texts))
        return [self.scores.get(text, 0.0) for text in texts]


def test_critic_judge_scores_via_critic_value():
    critic = _FakeCritic({"rendered": 0.7})
    judge = rubric.critic_judge(critic, render=lambda episode: "rendered")
    satisfied, reason = judge({"id": "ep_a"}, {"desc": "x"})
    assert satisfied == 0.7
    assert "critic score" in reason
    assert critic.calls == [["rendered"]]


def test_critic_judge_clamps_out_of_range_scores():
    critic = _FakeCritic({"rendered": 5.0})
    judge = rubric.critic_judge(critic, render=lambda episode: "rendered")
    satisfied, _ = judge({"id": "ep_b"}, {"desc": "x"})
    assert satisfied == 1.0

    critic_low = _FakeCritic({"rendered": -3.0})
    judge_low = rubric.critic_judge(critic_low, render=lambda episode: "rendered")
    satisfied_low, _ = judge_low({"id": "ep_c"}, {"desc": "x"})
    assert satisfied_low == 0.0


def test_critic_judge_memoizes_per_episode_id():
    critic = _FakeCritic({"rendered": 0.4})
    judge = rubric.critic_judge(critic, render=lambda episode: "rendered")
    judge({"id": "ep_a"}, {"desc": "explanation_quality"})
    judge({"id": "ep_a"}, {"desc": "correct_beyond_tests"})
    assert len(critic.calls) == 1


def test_critic_judge_defaults_render_to_trajectory_text():
    from episodic.core import reward
    from episodic.schema import new_episode

    episode = new_episode(id="ep_render")
    critic = _FakeCritic({})

    def value(texts):
        critic.calls.append(list(texts))
        return [0.5 for _ in texts]

    critic.value = value
    judge = rubric.critic_judge(critic)
    satisfied, _ = judge(episode, {"desc": "x"})
    assert satisfied == 0.5
    assert critic.calls[0][0]


def test_reward_vector_rubric_becomes_non_neutral_with_a_fake_judge():
    from episodic.core import reward
    from episodic.schema import new_episode

    episode = new_episode(id="ep_neutral")
    baseline = reward.reward_vector(episode)

    judge = rubric.openrubrics_judge(lambda prompt: "SCORE: 0.8 solid")
    judged = reward.reward_vector(episode, judge=judge)

    assert judged["rubric"] > baseline["rubric"]
    assert judged["composite"] > baseline["composite"]
    assert judged["components"]["has_rubric"] is True
