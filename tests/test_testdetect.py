from episodic.core.testdetect import detect_test_run


def test_passing_no_exit_code():
    result = detect_test_run("python3 -m pytest -q", "5 passed in 0.1s", "ts")
    assert result["ok"] is True and result["passed"] == 5


def test_nonzero_exit_is_failure():
    result = detect_test_run("pytest -q", "command not found: pytest", "ts", exit_code=127)
    assert result["ok"] is False


def test_unparseable_output_without_exit_is_not_ok():
    result = detect_test_run("pytest -q", "ERROR: file or directory not found", "ts", exit_code=None)
    assert result["ok"] is False and result["total"] == 0


def test_reported_failures_are_failure():
    result = detect_test_run("pytest -q", "1 failed, 2 passed", "ts", exit_code=1)
    assert result["ok"] is False and result["failed"] == 1


def test_exit_zero_with_passes_is_ok():
    result = detect_test_run("python3 -m pytest -q", "3 passed", "ts", exit_code=0)
    assert result["ok"] is True and result["passed"] == 3


def test_non_test_command_ignored():
    assert detect_test_run("ls -la", "files", "ts") is None
