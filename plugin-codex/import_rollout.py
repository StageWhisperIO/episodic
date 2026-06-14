import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

os.environ.setdefault("EPISODIC_AGENT", "codex")

from episodic.collector.hook import ingest

SHELL_FUNCTIONS = {"exec_command", "shell", "local_shell", "container.exec", "bash"}
PATCH_FUNCTIONS = {"apply_patch", "edit_file", "write_file"}

EXIT_CODE_RE = re.compile(r"Process exited with code (-?\d+)")
PATCH_PATH_RE = re.compile(r"^\*\*\* (?:Add|Update|Delete) File: (.+)$", re.M)
OUTPUT_MARKER = "Output:\n"


def _payload(row):
    payload = row.get("payload")
    return payload if isinstance(payload, dict) else {}


def _parse_arguments(arguments):
    if isinstance(arguments, dict):
        return arguments
    try:
        return json.loads(arguments)
    except (TypeError, ValueError):
        return {}


def _command_from_arguments(args):
    command = args.get("cmd")
    if command is None:
        command = args.get("command")
    if isinstance(command, list):
        return " ".join(str(part) for part in command)
    return command or ""


def _parse_output(raw):
    if not isinstance(raw, str):
        raw = json.dumps(raw) if raw is not None else ""
    exit_match = EXIT_CODE_RE.search(raw)
    exit_code = int(exit_match.group(1)) if exit_match else None
    index = raw.find(OUTPUT_MARKER)
    body = raw[index + len(OUTPUT_MARKER):] if index != -1 else raw
    return body, exit_code


def _patch_paths(args, raw_arguments, cwd):
    text = args.get("input") or args.get("patch") or args.get("file_path") or raw_arguments or ""
    paths = PATCH_PATH_RE.findall(text)
    if not paths and args.get("file_path"):
        paths = [args["file_path"]]
    resolved = []
    for path in paths:
        path = path.strip()
        resolved.append(path if os.path.isabs(path) else os.path.join(cwd, path))
    return resolved


def _outputs_by_call(rows):
    outputs = {}
    for row in rows:
        payload = _payload(row)
        if payload.get("type") == "function_call_output":
            outputs[payload.get("call_id")] = payload.get("output")
    return outputs


def map_rows(rows):
    session_meta = next((_payload(r) for r in rows if r.get("type") == "session_meta"), {})
    session_id = session_meta.get("id") or "codex-session"
    cwd = session_meta.get("cwd") or os.getcwd()
    git = session_meta.get("git") or {}
    outputs = _outputs_by_call(rows)

    payloads = []
    usage = None
    for row in rows:
        payload = _payload(row)
        kind = payload.get("type")

        if kind == "user_message":
            text = payload.get("message", "")
            if text:
                payloads.append({"hook_event_name": "UserPromptSubmit", "prompt": text})

        elif kind == "function_call":
            name = payload.get("name", "")
            args = _parse_arguments(payload.get("arguments"))
            call_id = payload.get("call_id")
            workdir = args.get("workdir") or args.get("cwd") or cwd
            if name in SHELL_FUNCTIONS:
                command = _command_from_arguments(args)
                body, exit_code = _parse_output(outputs.get(call_id))
                payloads.append({
                    "hook_event_name": "PostToolUse",
                    "tool_name": "Bash",
                    "tool_input": {"command": command},
                    "tool_response": {"stdout": body, "exit_code": exit_code},
                    "cwd": workdir,
                })
            elif name in PATCH_FUNCTIONS:
                for path in _patch_paths(args, payload.get("arguments"), workdir):
                    payloads.append({
                        "hook_event_name": "PostToolUse",
                        "tool_name": "Edit",
                        "tool_input": {"file_path": path},
                        "tool_response": {"stdout": "applied"},
                        "cwd": workdir,
                    })

        elif kind == "token_count":
            totals = (payload.get("info") or {}).get("total_token_usage") or {}
            if totals:
                usage = {
                    "input_tokens": totals.get("input_tokens", 0),
                    "output_tokens": totals.get("output_tokens", 0),
                    "cost_usd": 0.0,
                }

    return session_id, cwd, payloads, usage, git


def import_rollout(path, session_id=None, cwd=None):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    rollout_session, rollout_cwd, payloads, usage, git = map_rows(rows)
    session_id = session_id or rollout_session
    cwd = cwd or rollout_cwd

    ingest({"hook_event_name": "SessionStart", "session_id": session_id, "cwd": cwd, "source": "exec"})
    for payload in payloads:
        payload["session_id"] = session_id
        payload.setdefault("cwd", cwd)
        ingest(payload)

    from episodic import store
    meta = store.read_meta(session_id, cwd)
    repo_state = meta.get("repo_state") or {}
    if git.get("commit_hash"):
        repo_state["base_commit"] = git["commit_hash"]
    if git.get("branch"):
        repo_state["branch"] = git["branch"]
    if repo_state:
        meta["repo_state"] = repo_state
    if usage:
        meta["usage"] = usage
    store.write_meta(session_id, meta, cwd)

    ingest({"hook_event_name": "SessionEnd", "session_id": session_id, "cwd": cwd, "reason": "exec"})
    return len(payloads)


def main():
    if len(sys.argv) < 2:
        print("usage: import_rollout.py <rollout.jsonl> [session_id]", file=sys.stderr)
        return 1
    path = sys.argv[1]
    session_id = sys.argv[2] if len(sys.argv) > 2 else None
    count = import_rollout(path, session_id=session_id)
    print(f"ingested {count} tool/prompt events from {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
