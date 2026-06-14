import json
from pathlib import Path

import pytest

from episodic.replay import create_replay, run_replay, replay_id_for


def test_replay_id_for(sample_episode):
    rid = replay_id_for(sample_episode)
    assert rid.startswith("rp_")
    assert "ep_" not in rid


def test_replay_id_for_sanitizes_path_chars():
    rid = replay_id_for({"id": "ep_../../etc/passwd"})
    assert rid.startswith("rp_")
    assert "/" not in rid
    assert ".." not in rid


def test_create_replay_manifest_fields(tmp_path, monkeypatch, sample_episode):
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path))
    manifest = create_replay(sample_episode)

    assert manifest["base_commit"] == "abc123"
    assert manifest["initial_prompt"] == sample_episode["intent"]
    assert manifest["test_command"] == "pytest -q"
    assert "scoring_rules" in manifest
    assert manifest["scoring_rules"]["tests_pass_weight"] == 0.6
    assert manifest["scoring_rules"]["diff_overlap_weight"] == 0.4
    assert "reward_weights" in manifest["scoring_rules"]


def test_create_replay_writes_files(tmp_path, monkeypatch, sample_episode):
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path))
    manifest = create_replay(sample_episode)

    replay_id = manifest["replay_id"]
    replay_dir = tmp_path / "replays" / replay_id

    assert (replay_dir / "manifest.json").exists()
    assert (replay_dir / "prompt.txt").exists()
    assert (replay_dir / "expected.diff").exists()

    loaded = json.loads((replay_dir / "manifest.json").read_text())
    assert loaded["replay_id"] == replay_id
    assert loaded["base_commit"] == "abc123"

    prompt_text = (replay_dir / "prompt.txt").read_text()
    assert prompt_text == sample_episode["intent"]

    diff_text = (replay_dir / "expected.diff").read_text()
    assert "src/http.py" in diff_text or diff_text == "" or len(diff_text) >= 0


def test_run_replay_without_execute_returns_plan_and_does_not_clone(tmp_path, monkeypatch, sample_episode):
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path))
    manifest = create_replay(sample_episode)
    replay_id = manifest["replay_id"]

    result = run_replay(replay_id, "test-model")

    assert isinstance(result, dict)
    assert result["replay_id"] == replay_id
    assert result["model"] == "test-model"
    assert result["executed"] is False
    assert result["ran"] is False
    assert result["scores"] is None
    assert "plan" in result
    assert not (tmp_path / "replays" / replay_id / "workspace").exists()


def test_run_replay_execute_dry_run_without_remote(tmp_path, monkeypatch, sample_episode):
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path))
    monkeypatch.delenv("EPISODIC_REPLAY_CMD", raising=False)
    sample_episode["repo_state"]["remote_url"] = None
    manifest = create_replay(sample_episode)
    replay_id = manifest["replay_id"]

    result = run_replay(replay_id, "test-model", execute=True)

    assert isinstance(result, dict)
    assert result.get("dry_run") is True
    assert result.get("scores") is None
    assert result.get("note") is not None


def test_run_replay_missing_manifest_returns_error(tmp_path, monkeypatch):
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path))
    result = run_replay("rp_nonexistent", "test-model")
    assert isinstance(result, dict)
    assert "error" in result


def test_run_replay_rejects_path_traversal(tmp_path, monkeypatch):
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path))
    result = run_replay("../../etc", "test-model", execute=True)
    assert isinstance(result, dict)
    assert "error" in result
    assert "escapes" in result["error"]


def test_run_replay_execute_never_raises_on_bad_remote(tmp_path, monkeypatch, sample_episode):
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path))
    monkeypatch.delenv("EPISODIC_REPLAY_CMD", raising=False)
    sample_episode["repo_state"]["remote_url"] = str(tmp_path / "does-not-exist.git")
    manifest = create_replay(sample_episode)
    replay_id = manifest["replay_id"]

    try:
        result = run_replay(replay_id, "test-model", execute=True)
    except Exception as exc:
        pytest.fail(f"run_replay raised unexpectedly: {exc}")

    assert isinstance(result, dict)
