from datetime import datetime

from .ids import episode_id_from_session

MIN_TASK_CHARS = 40
CONTINUATION_CONNECTORS = (
    "yes", "y", "ok", "okay", "sure", "continue", "proceed", "next", "yeah", "yep",
    "and", "also", "then", "again", "try again", "run it",
    "please do", "go on", "do it", "do that", "do both", "do a", "do b", "do c",
)


def _is_new_task(prompt, is_first):
    if is_first:
        return True
    text = " ".join((prompt or "").split()).lower()
    if len(text) < MIN_TASK_CHARS:
        return False
    for token in CONTINUATION_CONNECTORS:
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


def _parse_ts(value):
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _segment_start(segment):
    stamps = sorted(
        ts for ts in (
            _parse_ts(event.get("ts")) for event in segment if event.get("type") != "session_start"
        ) if ts
    )
    return stamps[0] if stamps else None


def _bucket_feedback(feedback_items, starts):
    if not starts:
        return []
    buckets = [[] for _ in starts]
    last_index = len(starts) - 1
    for item in feedback_items:
        ts = _parse_ts(item.get("ts"))
        if ts is None:
            buckets[last_index].append(item)
            continue
        target = 0
        for index, start in enumerate(starts):
            if start is not None and start <= ts:
                target = index
        buckets[target].append(item)
    return buckets


def segment_session(session, generate=None):
    from .episode import build_episode

    parent_id = episode_id_from_session(session["id"])
    meta = session.get("meta") or {}
    segments = segment_events(session["events"])
    starts = [_segment_start(segment) for segment in segments]
    buckets = _bucket_feedback(meta.get("human_feedback") or [], starts)
    last_index = len(segments) - 1

    children = []
    for index, segment in enumerate(segments):
        child_meta = dict(meta)
        child_meta["human_feedback"] = buckets[index]
        if index == last_index:
            child_meta["outcome"] = meta.get("outcome")
        else:
            child_meta.pop("outcome", None)
        sub_session = {"id": f"{session['id']}#s{index}", "meta": child_meta, "events": segment}
        child = build_episode(sub_session, generate=generate)
        child["parent_id"] = parent_id
        child["segment_index"] = index
        children.append(child)
    return children
