from episodic import store
from episodic.collector import hook
from episodic.core.ids import episode_id_from_session


def test_label_timeout_defaults_on_malformed_value(monkeypatch):
    monkeypatch.delenv("EPISODIC_LABEL_TIMEOUT", raising=False)
    assert hook._label_timeout() == 60

    monkeypatch.setenv("EPISODIC_LABEL_TIMEOUT", "45")
    assert hook._label_timeout() == 45

    monkeypatch.setenv("EPISODIC_LABEL_TIMEOUT", "60s")
    assert hook._label_timeout() == 60


def test_ingest_finalizes_session_despite_malformed_label_timeout(tmp_path, monkeypatch):
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path / ".episodic"))
    monkeypatch.setenv("EPISODIC_AUTO_LABEL", "1")
    monkeypatch.setenv("EPISODIC_LABEL_TIMEOUT", "60s")
    monkeypatch.setenv("EPISODIC_LABELER_CMD", "echo '{}'")

    session_id = "sess-malformed-timeout"
    cwd = str(tmp_path)
    hook.ingest({
        "session_id": session_id, "hook_event_name": "UserPromptSubmit",
        "cwd": cwd, "prompt": "fix the bug",
    })
    hook.ingest({
        "session_id": session_id, "hook_event_name": "SessionEnd",
        "cwd": cwd, "reason": "clear",
    })

    episode = store.get_episode(episode_id_from_session(session_id))
    assert episode is not None
    assert len(episode["steps"]) >= 1


def test_ingest_finalizes_session_when_auto_label_generate_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path / ".episodic"))

    def _boom(hook_name):
        raise RuntimeError("labeler unavailable")

    monkeypatch.setattr(hook, "_auto_label_generate", _boom)

    session_id = "sess-generate-raises"
    cwd = str(tmp_path)
    hook.ingest({
        "session_id": session_id, "hook_event_name": "UserPromptSubmit",
        "cwd": cwd, "prompt": "fix the bug",
    })
    hook.ingest({
        "session_id": session_id, "hook_event_name": "SessionEnd",
        "cwd": cwd, "reason": "clear",
    })

    episode = store.get_episode(episode_id_from_session(session_id))
    assert episode is not None
    assert len(episode["steps"]) >= 1
