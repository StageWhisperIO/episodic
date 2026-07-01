HUMAN_LABEL_SCORES = {
    "useful": 1.0,
    "accepted_as_is": 1.0,
    "accepted_after_edits": 0.5,
    "too_broad": -0.3,
    "too_slow": -0.3,
    "needed_human_rescue": -0.6,
    "wrong": -1.0,
}

OUTCOME_SCORES = {
    "merged": 1.0,
    "accepted": 0.7,
    "open": 0.0,
    "abandoned": -0.5,
    "failed": -1.0,
    "reverted": -1.0,
}

WEIGHTS = {
    "test_pass": 0.30,
    "outcome": 0.30,
    "human_label": 0.20,
    "edit_focus": 0.10,
    "cost_efficiency": 0.10,
}


def _clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def _run_rate(test):
    denom = test.get("passed", 0) + test.get("failed", 0) + test.get("errors", 0)
    if denom == 0:
        return 1.0 if test.get("ok") else 0.0
    return test.get("passed", 0) / denom


def terminal_test_signal(tests):
    runs = [test for test in tests if test.get("total", 0) > 0]
    if not runs:
        return 0.0, False, False
    final = runs[-1]
    ever_green = any(test.get("ok") for test in runs)
    blocked_on_env = (
        not final.get("ok")
        and final.get("errors", 0) > 0
        and final.get("passed", 0) == 0
        and final.get("failed", 0) == 0
    )
    if final.get("ok"):
        score = 1.0
    else:
        score = _run_rate(final)
        if ever_green:
            score *= 0.5
    return round(score, 4), True, blocked_on_env


def _test_pass(episode):
    score, has_tests, _ = terminal_test_signal(episode["tests"])
    return score, has_tests


def _human_label(episode):
    feedback = episode["human_feedback"]
    if not feedback:
        return 0.0, False
    scores = [HUMAN_LABEL_SCORES.get(item["label"], 0.0) for item in feedback]
    return sum(scores) / len(scores), True


def _outcome(episode):
    score = OUTCOME_SCORES.get(episode["outcome"]["status"], 0.0)
    if episode["outcome"].get("caused_regression"):
        score = min(score, -1.0)
    return score


def _cost_efficiency(episode):
    cost = episode["stats"].get("cost_usd", 0.0)
    if cost <= 0:
        return 0.0, False
    edits = max(1, episode["stats"].get("file_edits", 0))
    value = edits / (edits + cost * 10.0)
    return _clamp(value), True


def _edit_focus(episode):
    stats = episode["stats"]
    edits = stats.get("file_edits", 0)
    reads = stats.get("file_reads", 0)
    shells = stats.get("shell_commands", 0)
    denominator = edits + reads + shells
    if denominator == 0:
        return 0.0
    return _clamp(edits / denominator)


def reward_vector(episode):
    test_pass, has_tests = _test_pass(episode)
    human_label, has_feedback = _human_label(episode)
    outcome = _outcome(episode)
    cost_efficiency, has_cost = _cost_efficiency(episode)
    edit_focus = _edit_focus(episode)

    normalized = {
        "test_pass": test_pass if has_tests else 0.5,
        "outcome": (outcome + 1) / 2,
        "human_label": (human_label + 1) / 2 if has_feedback else 0.5,
        "edit_focus": edit_focus,
        "cost_efficiency": cost_efficiency if has_cost else 0.5,
    }
    composite = sum(WEIGHTS[key] * normalized[key] for key in WEIGHTS)

    return {
        "test_pass": round(test_pass, 4),
        "human_label": round(human_label, 4),
        "outcome": round(outcome, 4),
        "cost_efficiency": round(cost_efficiency, 4),
        "edit_focus": round(edit_focus, 4),
        "composite": round(composite, 4),
        "components": {
            "normalized": {key: round(value, 4) for key, value in normalized.items()},
            "weights": WEIGHTS,
            "has_tests": has_tests,
            "has_feedback": has_feedback,
            "has_cost": has_cost,
        },
    }
