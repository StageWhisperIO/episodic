from episodic.core.normalize import _exit_code, event_from_hook


def test_exit_code_probes_common_keys():
    assert _exit_code({"exit_code": 0}) == 0
    assert _exit_code({"exitCode": 1}) == 1
    assert _exit_code({"returncode": 2}) == 2
    assert _exit_code({"code": "127"}) == 127
    assert _exit_code({"status": "-1"}) == -1


def test_exit_code_ignores_bools_and_non_numeric():
    assert _exit_code({"exit_code": True}) is None
    assert _exit_code({"code": "ok"}) is None
    assert _exit_code({"stdout": "x"}) is None
    assert _exit_code("not a dict") is None


def test_exit_code_rejects_malformed_numeric_strings():
    assert _exit_code({"exit_code": "--1"}) is None
    assert _exit_code({"code": "-"}) is None
    assert _exit_code({"code": "1-2"}) is None


def test_interrupted_maps_to_130():
    assert _exit_code({"interrupted": True}) == 130
    assert _exit_code({"interrupted": False}) is None


def test_shell_event_carries_derived_exit_code():
    event = event_from_hook({
        "hook_event_name": "PostToolUse",
        "session_id": "s1",
        "tool_name": "Bash",
        "tool_input": {"command": "cargo test"},
        "tool_response": {"stdout": "", "stderr": "boom", "interrupted": True},
    })
    assert event["type"] == "shell_command"
    assert event["data"]["exit_code"] == 130
