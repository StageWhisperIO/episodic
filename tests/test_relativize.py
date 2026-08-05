from episodic.core.normalize import relativize_command
from episodic.core.episode import _build_commands, _build_tests


def test_relativizes_path_under_root():
    assert relativize_command("cargo test --manifest-path /repo/src/Cargo.toml", "/repo") \
        == "cargo test --manifest-path ./src/Cargo.toml"


def test_relativizes_cd_to_root():
    assert relativize_command("cd /repo && cargo test", "/repo") == "cd . && cargo test"


def test_leaves_paths_outside_root_untouched():
    assert relativize_command("python /usr/bin/tool.py", "/repo") == "python /usr/bin/tool.py"


def test_does_not_match_prefix_collision():
    assert relativize_command("cat /repo-backup/x", "/repo") == "cat /repo-backup/x"


def test_noop_without_root_or_command():
    assert relativize_command("cargo test --manifest-path /repo/x", None) \
        == "cargo test --manifest-path /repo/x"
    assert relativize_command("", "/repo") == ""


def test_trailing_slash_root():
    assert relativize_command("pytest /repo/tests", "/repo/") == "pytest ./tests"


def _shell(command, response="ok", exit_code=0):
    return [{"type": "shell_command", "ts": "t",
             "data": {"command": command, "cwd": "/repo", "exit_code": exit_code, "response": response}}]


def test_build_commands_relativizes_at_capture_time():
    events = _shell("cargo test --manifest-path /repo/src/Cargo.toml")
    assert _build_commands(events, "/repo")[0]["command"] == "cargo test --manifest-path ./src/Cargo.toml"


def test_build_commands_noop_without_root():
    events = _shell("cargo test --manifest-path /repo/src/Cargo.toml")
    assert _build_commands(events)[0]["command"] == "cargo test --manifest-path /repo/src/Cargo.toml"


def test_build_tests_relativizes_captured_test_command():
    events = _shell("python -m pytest /repo/tests", response="1 passed")
    detected = _build_tests(events, "/repo")
    assert detected and "/repo" not in detected[0]["command"]
    assert detected[0]["command"] == "python -m pytest ./tests"
