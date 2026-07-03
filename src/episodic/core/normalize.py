from ..schema import new_event

FILE_EDIT_TOOLS = {"Edit", "MultiEdit", "NotebookEdit"}
FILE_WRITE_TOOLS = {"Write"}
FILE_DELETE_TOOLS = {"DeleteFile"}
FILE_READ_TOOLS = {"Read", "NotebookRead"}
SHELL_TOOLS = {"Bash", "BashOutput"}

DENIAL_MARKERS = (
    "permission denied",
    "user denied",
    "operation not permitted",
    "request was rejected",
    "tool use was rejected",
)

MAX_RESPONSE_CHARS = 6000


def _stringify(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("stdout", "output", "content", "result", "text", "message"):
            if key in value and isinstance(value[key], str):
                primary = value[key]
                stderr = value.get("stderr")
                if isinstance(stderr, str) and stderr.strip():
                    return f"{primary}\n{stderr}"
                return primary
        return _safe_json(value)
    if isinstance(value, list):
        return "\n".join(_stringify(item) for item in value)
    return str(value)


def _safe_json(value):
    import json

    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _truncate(text, limit=MAX_RESPONSE_CHARS):
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} chars]"


def _file_path(tool_input):
    for key in ("file_path", "notebook_path", "path", "filePath"):
        value = tool_input.get(key)
        if isinstance(value, str):
            return value
    return None


EXIT_CODE_KEYS = ("exit_code", "exitCode", "returncode", "return_code", "code", "status")


def _exit_code(raw_response):
    if not isinstance(raw_response, dict):
        return None
    for key in EXIT_CODE_KEYS:
        value = raw_response.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            stripped = value.strip()
            digits = stripped[1:] if stripped[:1] == "-" else stripped
            if digits.isdigit():
                return int(stripped)
    if raw_response.get("interrupted") is True:
        return 130
    return None


def _looks_denied(response_text):
    lowered = response_text.lower()
    return any(marker in lowered for marker in DENIAL_MARKERS)


def event_from_hook(payload, source="claude-code"):
    session_id = payload.get("session_id") or "unknown-session"
    hook = payload.get("hook_event_name") or payload.get("hook") or ""
    cwd = payload.get("cwd")

    def build(event_type, tool_name=None, data=None):
        merged = {"cwd": cwd}
        if data:
            merged.update(data)
        return new_event(session_id, event_type, source=source, tool_name=tool_name, data=merged)

    if hook == "SessionStart":
        return build("session_start", data={
            "origin": payload.get("source"),
            "permission_mode": payload.get("permission_mode"),
            "transcript_path": payload.get("transcript_path"),
        })

    if hook in ("SessionEnd",):
        return build("session_end", data={"reason": payload.get("reason")})

    if hook == "UserPromptSubmit":
        return build("user_prompt", data={"prompt": payload.get("prompt", "")})

    if hook in ("Stop", "SubagentStop"):
        return build("note", data={"kind": "turn_end", "hook": hook})

    if hook in ("Notification", "PreCompact"):
        return build("note", data={"kind": hook.lower(), "message": payload.get("message")})

    tool_name = payload.get("tool_name")
    tool_input = payload.get("tool_input") or {}

    if hook == "PreToolUse":
        return build("tool_pre", tool_name=tool_name, data={
            "tool_input": tool_input,
            "permission_mode": payload.get("permission_mode"),
        })

    if hook == "PostToolUse":
        raw_response = payload.get("tool_response")
        response_text = _truncate(_stringify(raw_response))
        denied = _looks_denied(response_text)
        base_data = {
            "tool_input": tool_input,
            "response": response_text,
            "permission_mode": payload.get("permission_mode"),
            "approved": not denied,
        }
        if denied:
            return build("denial", tool_name=tool_name, data=base_data)
        if tool_name in SHELL_TOOLS:
            return build("shell_command", tool_name=tool_name, data={
                **base_data,
                "command": tool_input.get("command", ""),
                "exit_code": _exit_code(raw_response),
            })
        if tool_name in FILE_EDIT_TOOLS:
            return build("file_edit", tool_name=tool_name, data={
                **base_data,
                "file_path": _file_path(tool_input),
            })
        if tool_name in FILE_WRITE_TOOLS:
            return build("file_write", tool_name=tool_name, data={
                **base_data,
                "file_path": _file_path(tool_input),
            })
        if tool_name in FILE_DELETE_TOOLS:
            return build("file_delete", tool_name=tool_name, data={
                **base_data,
                "file_path": _file_path(tool_input),
            })
        if tool_name in FILE_READ_TOOLS:
            return build("file_read", tool_name=tool_name, data={
                **base_data,
                "file_path": _file_path(tool_input),
            })
        return build("tool_post", tool_name=tool_name, data=base_data)

    return build("note", data={"kind": "unhandled", "hook": hook})
