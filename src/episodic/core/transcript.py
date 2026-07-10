import json
import re
from collections import defaultdict, deque

_EXIT_RE = re.compile(r"^\s*Exit code:?\s*(\d+)", re.IGNORECASE)


def _iter_entries(path):
    try:
        handle = open(path, encoding="utf-8")
    except OSError:
        return
    with handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except ValueError:
                continue


def _result_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def _record(block):
    text = _result_text(block.get("content"))
    is_error = bool(block.get("is_error"))
    match = _EXIT_RE.match(text)
    exit_code = int(match.group(1)) if match else (1 if is_error else 0)
    return {"is_error": is_error, "exit_code": exit_code, "output": text}


def bash_records(path):
    commands = {}
    records = defaultdict(deque)
    for entry in _iter_entries(path):
        content = (entry.get("message") or {}).get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use" and block.get("name") == "Bash":
                commands[block.get("id")] = (block.get("input") or {}).get("command", "")
            elif block.get("type") == "tool_result":
                command = commands.get(block.get("tool_use_id"))
                if command is not None:
                    records[command].append(_record(block))
    return records
