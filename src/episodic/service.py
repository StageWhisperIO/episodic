from . import store
from .schema import new_event, now_iso
from .core.episode import build_episode
from .core import gitinfo, reward


def resolve_session_id(session_id=None, start=None):
    if session_id:
        return session_id
    return store.get_current(start)


def ensure_meta(session_id, start=None, **patch):
    meta = store.read_meta(session_id, start)
    if "created_at" not in meta:
        meta["created_at"] = now_iso()
    meta.update({key: value for key, value in patch.items() if value is not None})
    store.write_meta(session_id, meta, start)
    return meta


def record_session_start(session_id, agent, cwd, start=None):
    repo_state = gitinfo.repo_state(cwd) if cwd else None
    ensure_meta(
        session_id,
        start=start,
        agent=agent,
        cwd=cwd,
        repo_state=repo_state,
    )
    store.set_current(session_id, start)


def set_intent(intent, session_id=None, start=None):
    session_id = resolve_session_id(session_id, start)
    if not session_id:
        return None
    ensure_meta(session_id, start=start, intent=intent)
    store.append_event(new_event(session_id, "note", data={"kind": "intent", "intent": intent}), start)
    return session_id


def add_feedback(label, note=None, session_id=None, start=None):
    session_id = resolve_session_id(session_id, start)
    if not session_id:
        return None
    meta = store.read_meta(session_id, start)
    feedback = meta.get("human_feedback", [])
    feedback.append({"ts": now_iso(), "label": label, "note": note})
    ensure_meta(session_id, start=start, human_feedback=feedback)
    store.append_event(new_event(session_id, "feedback", data={"label": label, "note": note}), start)
    return finalize_session(session_id, start)


def set_outcome(outcome, session_id=None, start=None):
    session_id = resolve_session_id(session_id, start)
    if not session_id:
        return None
    ensure_meta(session_id, start=start, outcome=outcome)
    store.append_event(new_event(session_id, "outcome", data=outcome), start)
    return finalize_session(session_id, start)


def update_episode(episode, start=None):
    episode["reward_vector"] = reward.reward_vector(episode)
    store.save_episode(episode, start)
    return episode


def _has_content(episode):
    return bool(episode["steps"] or episode["commands"] or episode["diffs"] or episode["tests"])


def finalize_session(session_id=None, start=None, generate=None):
    session_id = resolve_session_id(session_id, start)
    if not session_id:
        return None
    session = store.get_session(session_id, start)
    if not session["events"] and not session["meta"]:
        return None
    episode = build_episode(session, generate=generate)
    if not _has_content(episode):
        existing = store.get_episode(episode["id"], start)
        if existing is not None and _has_content(existing):
            return existing
    store.save_episode(episode, start)
    return episode


def renormalize(start=None):
    rebuilt = []
    for session_id in store.list_sessions(start):
        episode = finalize_session(session_id, start)
        if episode is not None:
            rebuilt.append(episode["id"])
    return rebuilt
