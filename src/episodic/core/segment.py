from .ids import episode_id_from_session

MIN_TASK_CHARS = 40
CONTINUATION_TOKENS = (
    "yes", "y", "ok", "okay", "sure", "continue", "go", "go on", "proceed", "do it", "do both",
    "do b", "do c", "do a", "next", "please", "implement", "push", "commit", "commit to main",
    "fix it", "try again", "again", "and", "also", "yeah", "yep", "do that", "run it",
)


def _is_new_task(prompt, is_first):
    if is_first:
        return True
    text = " ".join((prompt or "").split()).lower()
    if len(text) < MIN_TASK_CHARS:
        return False
    for token in CONTINUATION_TOKENS:
        if text == token or text.startswith(token + " ") or text.startswith(token + ","):
            return False
    return True


def segment_events(events):
    prelude = [event for event in events if event["type"] == "session_start"]
    segments = []
    current = None
    seen_user = False
    for event in events:
        if event["type"] == "session_start":
            continue
        if event["type"] == "user_prompt":
            if _is_new_task(event["data"].get("prompt", ""), not seen_user):
                current = list(prelude)
                segments.append(current)
            seen_user = True
        if current is None:
            current = list(prelude)
            segments.append(current)
        current.append(event)
    return segments


def segment_session(session, generate=None):
    from .episode import build_episode

    parent_id = episode_id_from_session(session["id"])
    meta = session.get("meta") or {}
    children = []
    for index, segment in enumerate(segment_events(session["events"])):
        sub_session = {"id": f"{session['id']}#s{index}", "meta": meta, "events": segment}
        child = build_episode(sub_session, generate=generate)
        child["parent_id"] = parent_id
        child["segment_index"] = index
        children.append(child)
    return children
