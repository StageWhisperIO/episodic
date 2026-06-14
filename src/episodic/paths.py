import os
from pathlib import Path

ENV_HOME = "EPISODIC_HOME"
STORE_DIRNAME = ".episodic"
ANCHORS = (STORE_DIRNAME, ".git")


def resolve_base(start=None):
    current = Path(start).resolve() if start else Path.cwd().resolve()
    for candidate in (current, *current.parents):
        for anchor in ANCHORS:
            if (candidate / anchor).exists():
                return candidate
    return current


def home(start=None):
    override = os.environ.get(ENV_HOME)
    if override:
        return Path(override).expanduser().resolve()
    return resolve_base(start) / STORE_DIRNAME


def sessions_dir(start=None):
    return home(start) / "sessions"


def episodes_dir(start=None):
    return home(start) / "episodes"


def exports_dir(start=None):
    return home(start) / "exports"


def replays_dir(start=None):
    return home(start) / "replays"


def session_dir(session_id, start=None):
    return sessions_dir(start) / session_id


def events_path(session_id, start=None):
    return session_dir(session_id, start) / "events.jsonl"


def meta_path(session_id, start=None):
    return session_dir(session_id, start) / "meta.json"


def episode_path(episode_id, start=None):
    return episodes_dir(start) / f"{episode_id}.json"


def index_path(start=None):
    return episodes_dir(start) / "index.jsonl"


def current_path(start=None):
    return home(start) / "current"
