import json

from episodic.core import transcript
from episodic.core.episode import (
    _apply_transcript_exit_codes,
    _build_commands,
    _build_tests,
    _reconstruct_failed_commands,
)
from episodic.schema import new_event


def _write(path, entries):
    with open(path, "w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")


def _use(tool_id, command):
    return {"message": {"content": [
        {"type": "tool_use", "name": "Bash", "id": tool_id, "input": {"command": command}}]}}


def _result(tool_id, is_error, text=""):
    return {"message": {"content": [
        {"type": "tool_result", "tool_use_id": tool_id, "is_error": is_error, "content": text}]}}


def _session_start(path):
    return new_event("s1", "session_start", data={"transcript_path": str(path)})


def _pre(command):
    return new_event("s1", "tool_pre", tool_name="Bash", data={"tool_input": {"command": command}, "cwd": "/repo"})


def _post(command, exit_code=None):
    return new_event("s1", "shell_command", tool_name="Bash",
                     data={"command": command, "cwd": "/repo", "exit_code": exit_code, "response": ""})


def test_bash_outcomes_maps_command_to_is_error(tmp_path):
    path = tmp_path / "t.jsonl"
    _write(path, [_use("a", "pytest -q"), _result("a", True), _use("b", "ls"), _result("b", False)])
    outcomes = transcript.bash_outcomes(str(path))
    assert list(outcomes["pytest -q"]) == [True]
    assert list(outcomes["ls"]) == [False]


def test_apply_enriches_none_exit_codes(tmp_path):
    path = tmp_path / "t.jsonl"
    _write(path, [_use("a", "pytest -q"), _result("a", True), _use("b", "ls"), _result("b", False)])
    events = [
        {"type": "session_start", "data": {"transcript_path": str(path)}},
        {"type": "shell_command", "data": {"command": "pytest -q", "exit_code": None}},
        {"type": "shell_command", "data": {"command": "ls", "exit_code": None}},
    ]
    _apply_transcript_exit_codes(events)
    assert events[1]["data"]["exit_code"] == 1
    assert events[2]["data"]["exit_code"] == 0


def test_apply_preserves_existing_exit_code(tmp_path):
    path = tmp_path / "t.jsonl"
    _write(path, [_use("a", "cmd"), _result("a", True)])
    events = [
        {"type": "session_start", "data": {"transcript_path": str(path)}},
        {"type": "shell_command", "data": {"command": "cmd", "exit_code": 130}},
    ]
    _apply_transcript_exit_codes(events)
    assert events[1]["data"]["exit_code"] == 130


def test_apply_noop_without_transcript():
    events = [{"type": "shell_command", "data": {"command": "x", "exit_code": None}}]
    _apply_transcript_exit_codes(events)
    assert events[0]["data"]["exit_code"] is None


def test_apply_skips_on_count_mismatch(tmp_path):
    path = tmp_path / "t.jsonl"
    _write(path, [_use("a", "flaky"), _result("a", False), _use("b", "flaky"), _result("b", True)])
    events = [
        {"type": "session_start", "data": {"transcript_path": str(path)}},
        {"type": "shell_command", "data": {"command": "flaky", "exit_code": None}},
    ]
    _apply_transcript_exit_codes(events)
    assert events[1]["data"]["exit_code"] is None


def test_duplicate_commands_consumed_in_order(tmp_path):
    path = tmp_path / "t.jsonl"
    _write(path, [_use("a", "make test"), _result("a", False), _use("b", "make test"), _result("b", True)])
    events = [
        {"type": "session_start", "data": {"transcript_path": str(path)}},
        {"type": "shell_command", "data": {"command": "make test", "exit_code": None}},
        {"type": "shell_command", "data": {"command": "make test", "exit_code": None}},
    ]
    _apply_transcript_exit_codes(events)
    assert events[1]["data"]["exit_code"] == 0
    assert events[2]["data"]["exit_code"] == 1


def test_reconstructs_failed_command(tmp_path):
    path = tmp_path / "t.jsonl"
    _write(path, [_use("a", "pytest -q"), _result("a", True, "Exit code: 2\n\n1 failed")])
    events = [_session_start(path), _pre("pytest -q")]
    out = _reconstruct_failed_commands(events)
    shells = [event for event in out if event["type"] == "shell_command"]
    assert len(shells) == 1
    assert shells[0]["data"]["reconstructed"] is True
    assert shells[0]["data"]["exit_code"] == 2
    assert shells[0]["data"]["command"] == "pytest -q"


def test_reconstructed_failed_test_flows_to_commands_and_tests(tmp_path):
    path = tmp_path / "t.jsonl"
    _write(path, [_use("a", "pytest -q"), _result("a", True, "Exit code: 1\n\n2 failed, 3 passed")])
    out = _reconstruct_failed_commands([_session_start(path), _pre("pytest -q")])
    commands = _build_commands(out)
    tests = _build_tests(out)
    assert len(commands) == 1
    assert commands[0]["reconstructed"] is True
    assert commands[0]["exit_code"] == 1
    assert commands[0]["is_test"] is True
    assert len(tests) == 1
    assert tests[0]["reconstructed"] is True
    assert tests[0]["ok"] is False
    assert tests[0]["failed"] == 2


def test_reconstruction_skips_on_count_mismatch(tmp_path):
    path = tmp_path / "t.jsonl"
    _write(path, [_use("a", "flaky"), _result("a", False), _use("b", "flaky"), _result("b", True)])
    events = [_session_start(path), _pre("flaky")]
    out = _reconstruct_failed_commands(events)
    assert [event for event in out if event["type"] == "shell_command"] == []
    assert out == events


def test_reconstruction_interleaved_inserts_only_failure(tmp_path):
    path = tmp_path / "t.jsonl"
    _write(path, [_use("a", "make test"), _result("a", True, "Exit code: 1"),
                  _use("b", "make test"), _result("b", False)])
    pre1, post_ok, pre2 = _pre("make test"), _post("make test", 0), _pre("make test")
    events = [_session_start(path), pre1, post_ok, pre2]
    out = _reconstruct_failed_commands(events)
    reconstructed = [event for event in out if event["data"].get("reconstructed")]
    assert len(reconstructed) == 1
    assert reconstructed[0]["data"]["exit_code"] == 1
    assert out[out.index(pre1) + 1] is reconstructed[0]


def test_reconstruction_noop_without_transcript():
    events = [_pre("pytest")]
    assert _reconstruct_failed_commands(events) is events


def test_reconstruct_then_apply_fills_success_exit_code(tmp_path):
    path = tmp_path / "t.jsonl"
    _write(path, [_use("a", "make test"), _result("a", True, "Exit code: 1"),
                  _use("b", "make test"), _result("b", False)])
    pre1, post_ok, pre2 = _pre("make test"), _post("make test", None), _pre("make test")
    out = _reconstruct_failed_commands([_session_start(path), pre1, post_ok, pre2])
    _apply_transcript_exit_codes(out)
    assert post_ok["data"]["exit_code"] == 0
    synth = [event for event in out if event["data"].get("reconstructed")][0]
    assert synth["data"]["exit_code"] == 1
