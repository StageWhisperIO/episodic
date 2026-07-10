import json
import os
import sys

from .. import store
from ..core.normalize import event_from_hook

FINALIZE_HOOKS = {"SessionEnd", "Stop"}
LIGHT_HOOKS = {"PreToolUse", "PostToolUse", "UserPromptSubmit"}
AUTO_LABEL_DISABLED = {"0", "false", "no", "off", ""}


def _source():
    return os.environ.get("EPISODIC_AGENT") or os.environ.get("EPISODIC_SOURCE") or "claude-code"


def _label_timeout():
    raw = os.environ.get("EPISODIC_LABEL_TIMEOUT", "60")
    try:
        return int(raw)
    except ValueError:
        return 60


def _auto_label_generate(hook):
    if hook != "SessionEnd":
        return None
    if os.environ.get("EPISODIC_AUTO_LABEL", "").lower() in AUTO_LABEL_DISABLED:
        return None
    from ..core import feedback

    return feedback.command_generate(timeout=_label_timeout())


def ingest(payload):
    source = _source()
    cwd = payload.get("cwd")
    hook = payload.get("hook_event_name") or payload.get("hook") or ""
    event = event_from_hook(payload, source=source)
    session_id = event["session_id"]

    store.append_event(event, cwd)
    store.set_current(session_id, cwd)

    if hook == "SessionStart":
        from ..service import record_session_start

        record_session_start(session_id, source, cwd, cwd)

    if hook in FINALIZE_HOOKS:
        from ..service import finalize_session

        try:
            generate = _auto_label_generate(hook)
        except Exception:
            generate = None
        finalize_session(session_id, cwd, generate=generate)

    return event


def main():
    if os.environ.get("EPISODIC_DISABLE"):
        return 0
    raw = sys.stdin.read()
    if not raw.strip():
        return 0
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return 0
    try:
        ingest(payload)
    except Exception:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
