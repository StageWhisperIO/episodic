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
