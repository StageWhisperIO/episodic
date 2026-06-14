import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import import_rollout
import notify


ROLLOUT_ROWS = [
    {"type": "session_meta", "payload": {"id": "sess-abc", "cwd": "/repo", "cli_version": "0.135.0",
                                          "git": {"commit_hash": "abc123", "branch": "master"}}},
    {"type": "turn_context", "payload": {"type": "turn_context"}},
    {"type": "event_msg", "payload": {"type": "user_message", "message": "Create foo.py and run pytest"}},
    {"type": "response_item", "payload": {
        "type": "function_call",
        "name": "exec_command",
        "arguments": "{\"cmd\": \"python3 -m pytest -q\", \"workdir\": \"/repo/sub\"}",
        "call_id": "c1",
    }},
    {"type": "response_item", "payload": {
        "type": "function_call_output",
        "call_id": "c1",
        "output": "Chunk ID: 0\nProcess exited with code 0\nOutput:\n..\n2 passed in 0.01s\n",
    }},
    {"type": "response_item", "payload": {
        "type": "function_call",
        "name": "apply_patch",
        "arguments": "{\"input\": \"*** Begin Patch\\n*** Add File: src/new.py\\n+x = 1\\n*** End Patch\"}",
        "call_id": "c2",
    }},
    {"type": "event_msg", "payload": {
        "type": "token_count",
        "info": {"total_token_usage": {"input_tokens": 100, "output_tokens": 20}},
    }},
]


def test_map_rows():
    session_id, cwd, payloads, usage, git = import_rollout.map_rows(ROLLOUT_ROWS)
    assert session_id == "sess-abc"
    assert cwd == "/repo"
    assert git["commit_hash"] == "abc123"

    prompts = [p for p in payloads if p["hook_event_name"] == "UserPromptSubmit"]
    assert len(prompts) == 1 and "foo.py" in prompts[0]["prompt"]

    bash = [p for p in payloads if p.get("tool_name") == "Bash"]
    assert len(bash) == 1
    assert bash[0]["tool_input"]["command"] == "python3 -m pytest -q"
    assert bash[0]["tool_response"]["exit_code"] == 0
    assert "2 passed" in bash[0]["tool_response"]["stdout"]
    assert bash[0]["cwd"] == "/repo/sub"

    edits = [p for p in payloads if p.get("tool_name") == "Edit"]
    assert len(edits) == 1 and edits[0]["tool_input"]["file_path"].endswith("src/new.py")

    assert usage["input_tokens"] == 100 and usage["output_tokens"] == 20


def test_parse_output():
    body, code = import_rollout._parse_output("Process exited with code 1\nOutput:\nboom\n")
    assert code == 1 and body.strip() == "boom"


def test_notify():
    out = notify.codex_to_hook({
        "type": "agent-turn-complete",
        "turn-id": "t1",
        "input-messages": ["fix the bug"],
        "last-assistant-message": "done",
    })
    kinds = [p["hook_event_name"] for p in out]
    assert "UserPromptSubmit" in kinds
    assert "Stop" in kinds
    assert any("fix the bug" in p.get("prompt", "") for p in out)


def main():
    test_map_rows()
    test_parse_output()
    test_notify()
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
