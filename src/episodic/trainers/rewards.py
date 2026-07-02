import re

KNOWN_TOOLS = (
    "Bash", "Edit", "Write", "Read", "Grep", "Glob", "MultiEdit",
    "shell_command", "file_edit", "file_write", "file_read", "file_delete", "user_prompt",
)

_ACTION = re.compile(r"^ACTION\s+(\S+?)\(")


def _text(completion):
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        return last.get("content", "") if isinstance(last, dict) else str(last)
    if isinstance(completion, dict):
        return completion.get("content", "")
    return str(completion)


def _score_action(text):
    text = (text or "").strip()
    score = 0.0
    if text.startswith("ACTION "):
        score += 0.5
    if "(" in text and ")" in text:
        score += 0.3
    match = _ACTION.match(text)
    if match and any(match.group(1).startswith(tool) for tool in KNOWN_TOOLS):
        score += 0.2
    return min(1.0, score)


def action_format_reward(prompts=None, completions=None, **kwargs):
    return [_score_action(_text(completion)) for completion in (completions or [])]
