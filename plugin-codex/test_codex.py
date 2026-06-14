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
    {"type": "response_item", "payload": {
        "type": "function_call",
        "name": "apply_patch",
        "arguments": "{\"input\": \"*** Begin Patch\\n*** Delete File: src/old.py\\n*** End Patch\"}",
        "call_id": "c3",
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

    writes = [p for p in payloads if p.get("tool_name") == "Write"]
    assert len(writes) == 1 and writes[0]["tool_input"]["file_path"].endswith("src/new.py")

    deletes = [p for p in payloads if p.get("tool_name") == "DeleteFile"]
    assert len(deletes) == 1 and deletes[0]["tool_input"]["file_path"].endswith("src/old.py")

    assert usage["input_tokens"] == 100 and usage["output_tokens"] == 20


CUSTOM_TOOL_ROWS = [
    {"type": "session_meta", "payload": {"id": "sess-real", "cwd": "/repo",
                                          "git": {"commit_hash": "deadbeef", "branch": "main"}}},
    {"type": "response_item", "payload": {
        "type": "custom_tool_call", "name": "apply_patch", "call_id": "p1",
        "input": "*** Begin Patch\n*** Update File: app/service.py\n@@\n-old = 1\n+new = 2\n*** End Patch",
    }},
    {"type": "response_item", "payload": {
        "type": "custom_tool_call_output", "call_id": "p1",
        "output": "{\"output\":\"Success. Updated the following files:\\nM app/service.py\\n\",\"metadata\":{\"exit_code\":0}}",
    }},
    {"type": "response_item", "payload": {
        "type": "custom_tool_call", "name": "apply_patch", "call_id": "p2",
        "input": "*** Begin Patch\n*** Delete File: app/gone.py\n*** End Patch",
    }},
    {"type": "response_item", "payload": {
        "type": "custom_tool_call_output", "call_id": "p2",
        "output": "{\"output\":\"apply_patch failed: File app/gone.py does not exist\",\"metadata\":{\"exit_code\":1}}",
    }},
]


def test_custom_tool_call_patch():
    _, _, payloads, _, git = import_rollout.map_rows(CUSTOM_TOOL_ROWS)
    assert git["commit_hash"] == "deadbeef"

    edits = [p for p in payloads if p.get("tool_name") == "Edit"]
    assert len(edits) == 1
    assert edits[0]["tool_input"]["file_path"].endswith("app/service.py")
    assert "Update File: app/service.py" in edits[0]["tool_input"]["patch"]
    assert edits[0]["tool_response"]["exit_code"] == 0

    failed = [p for p in payloads if p.get("tool_name") == "ApplyPatch"]
    assert len(failed) == 1
    assert failed[0]["tool_response"]["exit_code"] == 1
    assert not [p for p in payloads if p.get("tool_name") == "DeleteFile"]


def test_parse_output():
    body, code = import_rollout._parse_output("Process exited with code 1\nOutput:\nboom\n")
    assert code == 1 and body.strip() == "boom"
    body, code = import_rollout._parse_output(
        "{\"output\":\"done\",\"metadata\":{\"exit_code\":0}}"
    )
    assert code == 0 and body == "done"


def test_import_anchors_single_store():
    import json
    import tempfile

    work = tempfile.mkdtemp()
    other = tempfile.mkdtemp()
    rollout = os.path.join(work, "rollout.jsonl")
    rows = [
        {"type": "session_meta", "payload": {"id": "anchor-test", "cwd": other,
                                              "git": {"commit_hash": "c0ffee", "branch": "main"}}},
        {"type": "response_item", "payload": {
            "type": "function_call", "name": "exec_command", "call_id": "s1",
            "arguments": json.dumps({"cmd": "ls", "workdir": other}),
        }},
        {"type": "response_item", "payload": {
            "type": "function_call_output", "call_id": "s1",
            "output": "Process exited with code 0\nOutput:\nfile.txt\n",
        }},
        {"type": "response_item", "payload": {
            "type": "custom_tool_call", "name": "apply_patch", "call_id": "p1",
            "input": "*** Begin Patch\n*** Add File: new.py\n+x = 1\n*** End Patch",
        }},
        {"type": "response_item", "payload": {
            "type": "custom_tool_call_output", "call_id": "p1",
            "output": "{\"output\":\"Success. Updated the following files:\\nA new.py\\n\",\"metadata\":{\"exit_code\":0}}",
        }},
    ]
    with open(rollout, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    import_rollout.import_rollout(rollout, session_id="anchor-test", cwd=work)

    assert os.path.exists(os.path.join(work, ".episodic", "sessions", "anchor-test", "events.jsonl"))
    assert not os.path.exists(os.path.join(other, ".episodic"))

    from episodic import store
    from episodic.core.episode import build_episode
    episode = build_episode(store.get_session("anchor-test", start=work))
    assert __import__("episodic").validate_episode(episode) == []
    statuses = {d["status"] for d in episode["diffs"]}
    assert "added" in statuses
    assert any(s["type"] == "shell_command" for s in episode["steps"])


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
    test_custom_tool_call_patch()
    test_import_anchors_single_store()
    test_parse_output()
    test_notify()
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
