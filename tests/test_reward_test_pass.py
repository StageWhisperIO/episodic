from episodic.core import reward


def _entry(passed, failed, total, ok):
    return {"passed": passed, "failed": failed, "skipped": 0, "total": total, "ok": ok}


def test_test_pass_ignores_zero_total_runs():
    episode = {"tests": [
        _entry(5, 0, 5, True),
        _entry(3, 0, 3, True),
        _entry(0, 1, 1, False),
        _entry(0, 0, 0, False),
        _entry(0, 0, 0, False),
    ]}
    score, has_tests = reward._test_pass(episode)
    assert has_tests is True
    assert score == 2 / 3


def test_test_pass_all_zero_total_is_no_signal():
    episode = {"tests": [_entry(0, 0, 0, False), _entry(0, 0, 0, False)]}
    score, has_tests = reward._test_pass(episode)
    assert has_tests is False
