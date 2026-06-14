from episodic.schema import new_event, validate_episode
from episodic.core.episode import build_episode


def _session():
    sid = "sess-del"
    meta = {"imported": True, "repo_state": {"root": None, "base_commit": None}, "cwd": None}
    events = [
        new_event(sid, "file_write", source="codex", tool_name="Write", data={"file_path": "a.py"}),
        new_event(sid, "file_edit", source="codex", tool_name="Edit", data={"file_path": "b.py"}),
        new_event(sid, "file_delete", source="codex", tool_name="DeleteFile", data={"file_path": "c.py"}),
    ]
    return {"id": sid, "events": events, "meta": meta}


def test_event_diffs_preserve_operation():
    episode = build_episode(_session())
    assert validate_episode(episode) == []
    status_by_file = {d["file"]: d["status"] for d in episode["diffs"]}
    assert status_by_file == {"a.py": "added", "b.py": "modified", "c.py": "deleted"}


def test_delete_is_a_step():
    episode = build_episode(_session())
    delete_steps = [s for s in episode["steps"] if s["type"] == "file_delete"]
    assert len(delete_steps) == 1
    assert delete_steps[0]["intent"] == "delete c.py"
    assert episode["stats"]["file_edits"] == 3
