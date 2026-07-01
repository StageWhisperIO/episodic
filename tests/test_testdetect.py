from episodic.core.testdetect import classify_command, detect_test_run


def test_grep_pattern_mentioning_pytest_is_not_a_test_run():
    command = (
        'cd web/backend 2>/dev/null; sed -n \'1,80p\' tests/api/routes/test_pricing.py '
        '| grep -nE "pytest|test|def test|assert" | head -40; '
        'grep -nE "pytest|test|sqlite" Makefile 2>/dev/null | head; ls Makefile pyproject.toml'
    )
    assert classify_command(command) is None
    assert detect_test_run(command, "", "ts") is None


def test_reading_pytest_config_is_not_a_test_run():
    assert classify_command("cat pytest.ini") is None
    assert classify_command("ls tests/") is None
    assert classify_command('echo "running pytest"') is None


def test_real_pytest_invocations_still_classify():
    assert classify_command("pytest -q") == "pytest"
    assert classify_command("python3 -m pytest -q") == "pytest"
    assert classify_command("PYTHONPATH=. pytest tests/") == "pytest"
    assert classify_command("uv run --index-url x pytest -q") == "pytest"
    assert classify_command("cargo test") == "cargo-test"
    assert classify_command("npm run test:unit") == "npm-test"
    assert classify_command("cd web && make test") == "make-test"


def test_real_test_after_pipe_or_exploration_still_classifies():
    assert classify_command('echo "starting"; pytest -q') == "pytest"
    assert classify_command("cat input.txt | pytest -") == "pytest"


def test_compile_only_runs_are_not_test_runs():
    assert classify_command("cargo test --no-run --manifest-path x/Cargo.toml") is None
    assert classify_command("pytest --collect-only tests/") is None
    assert detect_test_run("cargo test --no-run", "Finished test [unoptimized]", "ts", exit_code=0) is None


def test_pytest_summary_parsed_from_tail_not_stray_numbers():
    output = "collected 3 items\nsome log line mentioning 5438 failed elsewhere\n" + ("x\n" * 30) + "===== 3 passed in 0.4s ====="
    result = detect_test_run("pytest -q", output, "ts", exit_code=0)
    assert result["passed"] == 3 and result["failed"] == 0 and result["ok"] is True


def test_output_excerpt_is_retained():
    result = detect_test_run("pytest -q", "lots of output\n3 passed in 0.1s", "ts")
    assert result["output_excerpt"].endswith("3 passed in 0.1s")


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
