import json

from episodic import service, store
from episodic.schema import new_event


def _shell_event(session_id, command, response, exit_code):
    return new_event(session_id, "shell_command", data={
        "command": command, "response": response, "exit_code": exit_code, "cwd": "/tmp/repo"})


def test_renormalize_rebuilds_episodes_with_corrected_tests(tmp_path, monkeypatch):
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path / ".episodic"))
    session_id = "sess-1"
    store.append_event(new_event(session_id, "session_start", data={"cwd": "/tmp/repo"}))
    store.append_event(_shell_event(session_id, "cargo test --no-run", "Finished test [unoptimized]", 0))
    store.append_event(_shell_event(session_id, "pytest -q", "3 passed in 0.1s", 0))

    rebuilt = service.renormalize()
    assert len(rebuilt) == 1

    episode = store.get_episode(rebuilt[0])
    frameworks = [t["framework"] for t in episode["tests"]]
    assert "pytest" in frameworks
    assert "cargo-test" not in frameworks
    assert all("output_excerpt" in t for t in episode["tests"])
    assert episode["reward_vector"]["test_pass"] == 1.0


def test_renormalize_empty_store_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path / ".episodic"))
    assert service.renormalize() == []


def test_renormalize_does_not_overwrite_content_with_empty_rebuild(tmp_path, monkeypatch):
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path / ".episodic"))
    cwd = str(tmp_path)
    session_id = "sess-pruned"
    store.write_meta(session_id, {"cwd": cwd, "repo_state": {"root": None}})
    store.append_event(new_event(session_id, "session_start", data={"cwd": cwd}))
    store.append_event(new_event(session_id, "shell_command", data={
        "command": "pytest -q", "response": "3 passed in 0.1s", "exit_code": 0, "cwd": cwd}))
    episode_id = service.finalize_session(session_id)["id"]
    assert store.get_episode(episode_id)["tests"]

    store.paths.events_path(session_id).write_text(
        json.dumps(new_event(session_id, "session_start", data={"cwd": cwd})) + "\n", encoding="utf-8")

    service.renormalize()
    preserved = store.get_episode(episode_id)
    assert preserved["tests"], "content-bearing episode must survive rebuild from pruned events"


def test_finalize_creates_empty_episode_when_none_exists(tmp_path, monkeypatch):
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path / ".episodic"))
    session_id = "sess-bare"
    store.write_meta(session_id, {"cwd": str(tmp_path), "repo_state": {"root": None}})
    store.append_event(new_event(session_id, "session_start", data={"cwd": str(tmp_path)}))
    episode = service.finalize_session(session_id)
    assert episode is not None and episode["steps"] == []
