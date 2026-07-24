import json

import pytest

from episodic import exporters


def _task_dir(out_dir, ep_id):
    return out_dir / "tasks" / ep_id


def test_harbor_in_registry_and_formats():
    assert "harbor" in exporters.FORMATS
    assert "harbor" in exporters._EXPORTERS


def test_mints_task_from_good_episode(episodes, tmp_path):
    result = exporters.export(episodes, "harbor", tmp_path)

    assert result["format"] == "harbor"
    assert result["tasks"] == 1
    assert result["count"] == 1

    task = _task_dir(tmp_path, "ep_test_good")
    assert (task / "task.toml").exists()
    assert (task / "Dockerfile").exists()
    assert (task / "tests" / "run-tests.sh").exists()
    assert (task / "metadata.json").exists()
    assert (task / "solution.patch").exists()


def test_verifier_script_carries_captured_command_not_a_guess(episodes, tmp_path):
    exporters.export(episodes, "harbor", tmp_path)
    script = (_task_dir(tmp_path, "ep_test_good") / "tests" / "run-tests.sh").read_text()
    assert script.startswith("#!/usr/bin/env bash")
    assert "pytest -q" in script


def test_task_toml_instruction_matches_intent(episodes, tmp_path):
    tomllib = pytest.importorskip("tomllib")
    exporters.export(episodes, "harbor", tmp_path)
    doc = tomllib.loads((_task_dir(tmp_path, "ep_test_good") / "task.toml").read_text())
    assert doc["task"]["instruction"] == "Add a retry helper to the http client"
    assert doc["verifier"]["command"] == "pytest -q"
    assert doc["verifier"]["framework"] == "pytest"
    assert doc["environment"]["os"] == "linux"
    assert doc["metadata"]["episode_id"] == "ep_test_good"
    assert doc["metadata"]["source"] == "episodic"


def test_dockerfile_clones_repo_at_base_commit(episodes, tmp_path):
    exporters.export(episodes, "harbor", tmp_path)
    dockerfile = (_task_dir(tmp_path, "ep_test_good") / "Dockerfile").read_text()
    assert "FROM python:3.12-slim" in dockerfile
    assert "git clone" in dockerfile
    assert "github.com/acme/demo.git" in dockerfile
    assert "checkout" in dockerfile and "abc123" in dockerfile


def test_solution_patch_is_the_recorded_diff(episodes, tmp_path):
    exporters.export(episodes, "harbor", tmp_path)
    patch = (_task_dir(tmp_path, "ep_test_good") / "solution.patch").read_text()
    assert "diff --git a/src/http.py b/src/http.py" in patch


def test_bad_episode_is_skipped_with_reason(episodes, tmp_path):
    result = exporters.export(episodes, "harbor", tmp_path)
    assert not _task_dir(tmp_path, "ep_test_bad").exists()
    reasons = {row["id"]: row["reason"] for row in result["skipped"]}
    assert reasons["ep_test_bad"] in {"bad_outcome", "no_verifier", "low_trust"}


def test_low_trust_episode_is_skipped(sample_episode, tmp_path):
    sample_episode["validity"] = {"trust": "low"}
    result = exporters.export([sample_episode], "harbor", tmp_path)
    assert result["tasks"] == 0
    assert result["skipped"][0]["reason"] == "low_trust"


def test_episode_without_captured_verifier_is_skipped(sample_episode, tmp_path):
    sample_episode["tests"] = []
    sample_episode["commands"] = []
    result = exporters.export([sample_episode], "harbor", tmp_path)
    assert result["tasks"] == 0
    assert result["skipped"][0]["reason"] == "no_verifier"


def test_dataset_manifest_and_readme_written(episodes, tmp_path):
    exporters.export(episodes, "harbor", tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["task_count"] == 1
    assert "ep_test_good" in manifest["minted"]
    assert (tmp_path / "dataset.toml").exists()
    assert (tmp_path / "README.md").exists()


def test_malicious_remote_url_is_not_embedded(sample_episode, tmp_path):
    sample_episode["repo_state"]["remote_url"] = "https://x.com/r.git; rm -rf /"
    exporters.export([sample_episode], "harbor", tmp_path)
    dockerfile = (_task_dir(tmp_path, "ep_test_good") / "Dockerfile").read_text()
    assert "rm -rf" not in dockerfile
    assert "git clone" not in dockerfile
    assert "Mount the target repository" in dockerfile


def test_harbor_rejects_stdout(episodes):
    with pytest.raises(ValueError, match="stdout"):
        exporters.export(episodes, "harbor", "-")


def test_unsafe_episode_id_is_skipped(sample_episode, tmp_path):
    sample_episode["id"] = "../../etc/passwd"
    result = exporters.export([sample_episode], "harbor", tmp_path)
    assert result["tasks"] == 0
    assert result["skipped"][0]["reason"] == "unsafe_id"
    assert not (tmp_path / "etc").exists()
