import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

notify = importlib.import_module("notify")
import_rollout_mod = importlib.import_module("import_rollout")

codex_to_hook = notify.codex_to_hook
map_rollout_line = import_rollout_mod.map_rollout_line


def test_codex_to_hook_turn_complete():
    payload = {
        "type": "agent-turn-complete",
        "turn-id": "t1",
        "input-messages": ["hello world"],
        "last-assistant-message": "done",
    }
    results = codex_to_hook(payload)
    assert len(results) >= 1
    user_prompts = [r for r in results if r["hook_event_name"] == "UserPromptSubmit"]
    assert len(user_prompts) == 1
    assert "hello world" in user_prompts[0]["prompt"]
    stops = [r for r in results if r["hook_event_name"] == "Stop"]
    assert len(stops) == 1


def test_codex_to_hook_session_id_from_turn_id():
    payload = {
        "type": "agent-turn-complete",
        "turn-id": "myturnabc",
        "input-messages": ["test"],
    }
    results = codex_to_hook(payload)
    for r in results:
        assert r["session_id"] == "myturnabc"


def test_codex_to_hook_session_id_from_session_id():
    payload = {
        "type": "agent-turn-complete",
        "session-id": "mysession",
        "turn-id": "t2",
        "input-messages": ["test"],
    }
    results = codex_to_hook(payload)
    for r in results:
        assert r["session_id"] == "mysession"


def test_map_rollout_line_user_message():
    obj = {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "run the tests"}],
    }
    result = map_rollout_line(obj)
    assert result is not None
    assert result["hook_event_name"] == "UserPromptSubmit"
    assert "run the tests" in result["prompt"]


def test_map_rollout_line_shell_function_call():
    obj = {
        "type": "function_call",
        "name": "shell",
        "arguments": '{"command": ["bash", "-lc", "pytest"]}',
    }
    result = map_rollout_line(obj)
    assert result is not None
    assert result["hook_event_name"] == "PostToolUse"
    assert result["tool_name"] == "Bash"
    assert "pytest" in result["tool_input"]["command"]


def test_map_rollout_line_shell_string_command():
    obj = {
        "type": "function_call",
        "name": "shell",
        "arguments": '{"command": "ls -la"}',
    }
    result = map_rollout_line(obj)
    assert result is not None
    assert result["tool_input"]["command"] == "ls -la"


def test_map_rollout_line_apply_patch():
    patch = "--- a/src/foo.py\n+++ b/src/foo.py\n@@ -1 +1 @@\n-old\n+new"
    obj = {
        "type": "function_call",
        "name": "apply_patch",
        "arguments": f'{{"patch": "{patch}"}}',
    }
    result = map_rollout_line(obj)
    assert result is not None
    assert result["hook_event_name"] == "PostToolUse"
    assert result["tool_name"] == "Edit"


def test_map_rollout_line_unknown_returns_none():
    obj = {"type": "unknown_event", "data": "something"}
    result = map_rollout_line(obj)
    assert result is None


def test_codex_to_hook_no_input_messages():
    payload = {"type": "agent-turn-complete", "turn-id": "t3"}
    results = codex_to_hook(payload)
    stops = [r for r in results if r["hook_event_name"] == "Stop"]
    assert len(stops) == 1
    user_prompts = [r for r in results if r["hook_event_name"] == "UserPromptSubmit"]
    assert len(user_prompts) == 0


def main():
    tests = [
        test_codex_to_hook_turn_complete,
        test_codex_to_hook_session_id_from_turn_id,
        test_codex_to_hook_session_id_from_session_id,
        test_map_rollout_line_user_message,
        test_map_rollout_line_shell_function_call,
        test_map_rollout_line_shell_string_command,
        test_map_rollout_line_apply_patch,
        test_map_rollout_line_unknown_returns_none,
        test_codex_to_hook_no_input_messages,
    ]
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"ERROR {t.__name__}: {e}", file=sys.stderr)
            return 1
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
