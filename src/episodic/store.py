import json
from pathlib import Path

from . import paths


def _read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_event(event, start=None):
    session_id = event["session_id"]
    _append_jsonl(paths.events_path(session_id, start), event)
    return event


def read_events(session_id, start=None):
    return _read_jsonl(paths.events_path(session_id, start))


def read_meta(session_id, start=None):
    path = paths.meta_path(session_id, start)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_meta(session_id, meta, start=None):
    path = paths.meta_path(session_id, start)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def update_meta(session_id, patch, start=None):
    meta = read_meta(session_id, start)
    meta.update(patch)
    return write_meta(session_id, meta, start)


def set_current(session_id, start=None):
    path = paths.current_path(start)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(session_id, encoding="utf-8")


def get_current(start=None):
    path = paths.current_path(start)
    if not path.exists():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return value or None


def list_sessions(start=None):
    directory = paths.sessions_dir(start)
    if not directory.exists():
        return []
    return sorted(child.name for child in directory.iterdir() if child.is_dir())


def get_session(session_id, start=None):
    return {
        "id": session_id,
        "meta": read_meta(session_id, start),
        "events": read_events(session_id, start),
    }


def save_episode(episode, start=None):
    path = paths.episode_path(episode["id"], start)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(episode, indent=2, ensure_ascii=False), encoding="utf-8")
    index_episode(episode, start)
    return path


def get_episode(episode_id, start=None):
    path = paths.episode_path(episode_id, start)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def index_row(episode):
    return {
        "id": episode["id"],
        "created_at": episode["created_at"],
        "agent": episode["agent"],
        "intent": episode["intent"][:140],
        "branch": episode["repo_state"].get("branch"),
        "outcome": episode["outcome"]["status"],
        "composite_reward": episode["reward_vector"]["composite"],
        "tests_run": episode["stats"]["tests_run"],
        "file_edits": episode["stats"]["file_edits"],
        "labels": episode["labels"],
    }


def index_episode(episode, start=None):
    path = paths.index_path(start)
    rows = [row for row in _read_jsonl(path) if row.get("id") != episode["id"]]
    rows.append(index_row(episode))
    rows.sort(key=lambda row: row.get("created_at", ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def list_episodes(start=None):
    rows = _read_jsonl(paths.index_path(start))
    if rows:
        return rows
    directory = paths.episodes_dir(start)
    if not directory.exists():
        return []
    fallback = []
    for child in sorted(directory.glob("*.json")):
        episode = json.loads(child.read_text(encoding="utf-8"))
        fallback.append(index_row(episode))
    return fallback


def iter_episodes(start=None):
    directory = paths.episodes_dir(start)
    if not directory.exists():
        return
    for child in sorted(directory.glob("*.json")):
        yield json.loads(child.read_text(encoding="utf-8"))


def load_episodes(start=None):
    return list(iter_episodes(start))
