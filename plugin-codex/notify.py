import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

os.environ.setdefault("EPISODIC_AGENT", "codex")

from episodic.collector.hook import ingest


def codex_to_hook(payload):
    event_type = payload.get("type", "")
    session_id = (
        payload.get("session-id")
        or payload.get("turn-id")
        or "codex-session"
    )
    cwd = payload.get("cwd") or os.getcwd()
    results = []

    if "turn-start" in event_type or "turn-complete" in event_type:
        input_messages = payload.get("input-messages") or []
        if input_messages:
            prompt = " ".join(str(m) for m in input_messages)
            results.append({
                "hook_event_name": "UserPromptSubmit",
                "session_id": session_id,
                "cwd": cwd,
                "prompt": prompt,
            })

        if "turn-complete" in event_type:
            results.append({
                "hook_event_name": "Stop",
                "session_id": session_id,
                "cwd": cwd,
            })

    return results


def main():
    try:
        if len(sys.argv) > 1:
            raw = sys.argv[1]
        else:
            raw = sys.stdin.read()
        payload = json.loads(raw)
        for hook_payload in codex_to_hook(payload):
            try:
                ingest(hook_payload)
            except Exception:
                pass
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
