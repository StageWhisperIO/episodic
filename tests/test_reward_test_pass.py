from episodic.core import reward


def _run(passed=0, failed=0, errors=0, skipped=0, ok=None):
    total = passed + failed + skipped + errors
    if ok is None:
        ok = total > 0 and failed == 0 and errors == 0
    return {"passed": passed, "failed": failed, "skipped": skipped,
            "errors": errors, "total": total, "ok": ok}


def test_ended_green_is_full_credit_regardless_of_iteration():
    episode = {"tests": [_run(failed=3), _run(failed=1), _run(passed=10)]}
    score, has, blocked = reward.terminal_test_signal(episode["tests"])
    assert has is True and blocked is False
    assert score == 1.0


def test_regression_at_end_is_penalized():
    episode = {"tests": [_run(passed=10), _run(passed=5, failed=5)]}
    score, _, _ = reward.terminal_test_signal(episode["tests"])
    assert score == 0.25  # final rate 0.5, halved because it regressed from green


def test_never_green_gets_progression_credit():
    episode = {"tests": [_run(passed=0, failed=10), _run(passed=8, failed=2)]}
    score, _, _ = reward.terminal_test_signal(episode["tests"])
    assert score == 0.8  # final rate, no regression penalty (never was green)


def test_compile_only_runs_are_no_signal():
    episode = {"tests": [_run(), _run()]}
    score, has, _ = reward.terminal_test_signal(episode["tests"])
    assert has is False


def test_blocked_on_env_when_final_run_is_all_errors():
    episode = {"tests": [_run(errors=6)]}
    score, has, blocked = reward.terminal_test_signal(episode["tests"])
    assert has is True and blocked is True
    assert score == 0.0
