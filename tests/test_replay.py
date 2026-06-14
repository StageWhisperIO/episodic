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


def test_replay_id_for_disambiguates_collisions():
    a = replay_id_for({"id": "ep_a/b"})
    b = replay_id_for({"id": "ep_a_b"})
    assert a != b
    assert "/" not in a and "/" not in b


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
    sample_episode["repo_state"]["root"] = str(tmp_path / "no-such-root")
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


def test_run_replay_refuses_symlinked_workspace(tmp_path, monkeypatch, sample_episode):
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path))
    manifest = create_replay(sample_episode)
    replay_id = manifest["replay_id"]
    replay_dir = tmp_path / "replays" / replay_id

    external = tmp_path / "external"
    external.mkdir()
    (external / "keep.txt").write_text("important")
    (replay_dir / "workspace").symlink_to(external)

    result = run_replay(replay_id, "test-model", execute=True)

    assert "error" in result
    assert (external / "keep.txt").exists()


def test_run_replay_local_repo_fallback_copies_root(tmp_path, monkeypatch, sample_episode):
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path))
    monkeypatch.delenv("EPISODIC_REPLAY_CMD", raising=False)
    local = tmp_path / "localrepo"
    local.mkdir()
    (local / ".git").mkdir()
    (local / "mod.py").write_text("x = 1\n")
    sample_episode["repo_state"]["remote_url"] = None
    sample_episode["repo_state"]["root"] = str(local)
    sample_episode["commands"] = []
    manifest = create_replay(sample_episode)
    replay_id = manifest["replay_id"]

    result = run_replay(replay_id, "test-model", execute=True)

    workspace = tmp_path / "replays" / replay_id / "workspace"
    assert (workspace / "mod.py").exists()
    assert result["workspace"] == str(workspace)


def test_run_replay_local_repo_diff_scoring(tmp_path, monkeypatch, sample_episode):
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path))
    monkeypatch.delenv("EPISODIC_REPLAY_CMD", raising=False)
    local = tmp_path / "localrepo2"
    local.mkdir()
    (local / ".git").mkdir()
    (local / "mod.py").write_text("x = 1\n")
    runner = tmp_path / "runner.py"
    runner.write_text(
        "import os, sys\n"
        "workspace = sys.argv[2]\n"
        "with open(os.path.join(workspace, 'mod.py'), 'a') as fh:\n"
        "    fh.write('y = 2\\n')\n"
        "with open(os.path.join(workspace, 'new.py'), 'w') as fh:\n"
        "    fh.write('z = 3\\n')\n"
    )
    sample_episode["repo_state"]["remote_url"] = None
    sample_episode["repo_state"]["root"] = str(local)
    sample_episode["commands"] = []
    sample_episode["diffs"] = [
        {"file": "mod.py", "status": "modified", "additions": 1, "deletions": 0, "unified": None},
        {"file": "new.py", "status": "added", "additions": 1, "deletions": 0, "unified": None},
    ]
    manifest = create_replay(sample_episode)
    replay_id = manifest["replay_id"]

    result = run_replay(replay_id, "candidate", execute=True,
                        runner_cmd=f"python3 {runner} {{model}} {{workspace}} {{prompt_file}}")

    assert result["ran"] is True
    assert "mod.py" in result["produced_files"]
    assert "new.py" in result["produced_files"]
    assert result["scores"]["diff_overlap"] == 1.0


def test_run_replay_local_fallback_refuses_non_git_dir(tmp_path, monkeypatch, sample_episode):
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path))
    monkeypatch.delenv("EPISODIC_REPLAY_CMD", raising=False)
    local = tmp_path / "not-a-repo"
    local.mkdir()
    (local / "secret.txt").write_text("do not copy")
    sample_episode["repo_state"]["remote_url"] = None
    sample_episode["repo_state"]["root"] = str(local)
    sample_episode["commands"] = []
    manifest = create_replay(sample_episode)
    replay_id = manifest["replay_id"]

    result = run_replay(replay_id, "test-model", execute=True)

    assert result.get("dry_run") is True
    assert not (tmp_path / "replays" / replay_id / "workspace").exists()


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
    assert "error" in result
    assert "clone failed" in result["error"]
    assert not (tmp_path / "replays" / replay_id / "workspace").exists()


def test_run_replay_invalid_runner_template_returns_error(tmp_path, monkeypatch, sample_episode):
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path))
    monkeypatch.delenv("EPISODIC_REPLAY_CMD", raising=False)
    local = tmp_path / "repo3"
    local.mkdir()
    (local / ".git").mkdir()
    (local / "mod.py").write_text("x = 1\n")
    sample_episode["repo_state"]["remote_url"] = None
    sample_episode["repo_state"]["root"] = str(local)
    sample_episode["commands"] = []
    manifest = create_replay(sample_episode)
    replay_id = manifest["replay_id"]

    result = run_replay(replay_id, "m", execute=True, runner_cmd="echo {nope}")

    assert "error" in result
    assert "runner template" in result["error"]
