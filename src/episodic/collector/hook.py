import json
import os
import sys

from .. import store
from ..core.normalize import event_from_hook

FINALIZE_HOOKS = {"SessionEnd", "Stop"}
LIGHT_HOOKS = {"PreToolUse", "PostToolUse", "UserPromptSubmit"}


def _source():
    return os.environ.get("EPISODIC_AGENT") or os.environ.get("EPISODIC_SOURCE") or "claude-code"


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

        finalize_session(session_id, cwd)

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
