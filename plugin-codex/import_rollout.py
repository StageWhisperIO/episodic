import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

os.environ.setdefault("EPISODIC_AGENT", "codex")

from episodic.collector.hook import ingest


def _extract_patch_path(args_str):
    lines = args_str.splitlines()
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("+++ ") or stripped.startswith("--- "):
            candidate = stripped[4:].strip()
            if candidate and candidate != "/dev/null":
                return candidate.lstrip("b/")
    return ""


def map_rollout_line(obj):
    line_type = obj.get("type", "")

    if line_type == "message" and obj.get("role") == "user":
        content = obj.get("content", [])
        texts = []
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "input_text":
                    texts.append(part.get("text", ""))
                elif isinstance(part, str):
                    texts.append(part)
        elif isinstance(content, str):
            texts.append(content)
        return {
            "hook_event_name": "UserPromptSubmit",
            "prompt": " ".join(texts),
        }

    if line_type == "function_call":
        name = obj.get("name", "")
        args_raw = obj.get("arguments", "{}")
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        except Exception:
            args = {}

        if name == "shell":
            cmd = args.get("command", "")
            if isinstance(cmd, list):
                cmd = " ".join(cmd)
            return {
                "hook_event_name": "PostToolUse",
                "tool_name": "Bash",
                "tool_input": {"command": cmd},
            }

        if name == "apply_patch" or "apply_patch" in str(args_raw):
            patch_content = args.get("patch", "") or args_raw
            file_path = _extract_patch_path(str(patch_content))
            return {
                "hook_event_name": "PostToolUse",
                "tool_name": "Edit",
                "tool_input": {"file_path": file_path},
            }

    if line_type == "function_call_output":
        output = obj.get("output", "")
        return {
            "hook_event_name": "PostToolUse",
            "tool_name": "Bash",
            "tool_input": {"command": ""},
            "tool_response": {"stdout": output},
        }

    return None


def import_rollout(path, session_id=None, cwd=None):
    session_id = session_id or os.path.splitext(os.path.basename(path))[0]
    cwd = cwd or os.getcwd()

    ingest({"hook_event_name": "SessionStart", "session_id": session_id, "cwd": cwd})

    count = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    obj = json.loads(raw_line)
                except Exception:
                    continue
                mapped = map_rollout_line(obj)
                if mapped is None:
                    continue
                mapped["session_id"] = session_id
                mapped["cwd"] = cwd
                try:
                    ingest(mapped)
                    count += 1
                except Exception:
                    pass
    except Exception:
        pass

    ingest({"hook_event_name": "SessionEnd", "session_id": session_id, "cwd": cwd})
    return count


def main():
    if len(sys.argv) < 2:
        print("usage: import_rollout.py <rollout.jsonl> [session_id]", file=sys.stderr)
        return 1
    path = sys.argv[1]
    session_id = sys.argv[2] if len(sys.argv) > 2 else None
    count = import_rollout(path, session_id=session_id)
    print(f"ingested {count} events from {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
