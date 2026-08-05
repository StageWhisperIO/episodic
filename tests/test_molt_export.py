import asyncio
import importlib.util
import json
import subprocess
import sys
import tempfile
import types

import pytest

from episodic import exporters


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=str(repo), check=True, capture_output=True)


def _failing_origin(tmp_path):
    origin = tmp_path / "origin"
    origin.mkdir()
    (origin / "app.py").write_text("VALUE = 1\n")
    (origin / "test_app.py").write_text("from app import VALUE\n\n\ndef test_value():\n    assert VALUE == 2\n")
    _git(origin, "init", "-q")
    _git(origin, "config", "user.email", "t@t.dev")
    _git(origin, "config", "user.name", "t")
    _git(origin, "add", "-A")
    _git(origin, "commit", "-q", "-m", "base")
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(origin),
                         capture_output=True, text=True).stdout.strip()
    return str(origin), sha


def _gold_patch(origin):
    work = tempfile.mkdtemp()
    subprocess.run(["git", "clone", "--quiet", origin, work], check=True, capture_output=True)
    with open(f"{work}/app.py", "w") as handle:
        handle.write("VALUE = 2\n")
    diff = subprocess.run(["git", "-C", work, "diff"], capture_output=True, text=True).stdout
    return "```diff\n" + diff + "```"


def _load_generated_env(path):
    molt = types.ModuleType("molt")
    agents = types.ModuleType("molt.agents")

    class Env:
        pass

    class Result:
        def __init__(self, reward=None, observation=None, terminated=False, info=None):
            self.reward, self.observation, self.terminated, self.info = reward, observation, terminated, info

    class StepEnvRunner:
        def __init__(self, env_cls):
            self.env_cls = env_cls

    agents.Env, agents.Result, agents.StepEnvRunner = Env, Result, StepEnvRunner
    molt.agents = agents
    sys.modules["molt"], sys.modules["molt.agents"] = molt, agents
    try:
        spec = importlib.util.spec_from_file_location("episodic_env_gen", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop("molt", None)
        sys.modules.pop("molt.agents", None)
    return mod, StepEnvRunner


def _agent_path(out_dir):
    return out_dir / "agents" / "episodic_env.py"


def test_molt_in_registry_and_formats():
    assert "molt" in exporters.FORMATS
    assert "molt" in exporters._EXPORTERS


def test_export_writes_bundle(episodes, tmp_path):
    result = exporters.export(episodes, "molt", tmp_path)

    assert result["format"] == "molt"
    assert result["tasks"] == 1 and result["count"] == 1
    assert (tmp_path / "prompts.jsonl").exists()
    assert _agent_path(tmp_path).exists()
    assert (tmp_path / "README.md").exists()
    assert (tmp_path / "manifest.json").exists()


def test_prompts_jsonl_rows(episodes, tmp_path):
    exporters.export(episodes, "molt", tmp_path)
    rows = [json.loads(l) for l in (tmp_path / "prompts.jsonl").read_text().splitlines() if l.strip()]
    assert len(rows) == 1
    row = rows[0]
    assert row["input"].startswith("Add a retry helper to the http client")
    assert "```diff" in row["input"]
    assert row["episode_id"] == "ep_test_good"
    label = json.loads(row["label"])
    assert label["remote_url"] == "https://github.com/acme/demo.git"
    assert label["base_commit"] == "abc123"
    assert label["test_command"] == "pytest -q"
    assert label["framework"] == "pytest"
    assert row["portable"] is True


def test_generated_agent_is_valid_python_and_defines_contract(episodes, tmp_path):
    exporters.export(episodes, "molt", tmp_path)
    source = _agent_path(tmp_path).read_text()
    compile(source, "episodic_env.py", "exec")
    mod, StepEnvRunner = _load_generated_env(str(_agent_path(tmp_path)))
    assert issubclass(mod.AgentRunner, StepEnvRunner)
    assert mod.AgentRunner().env_cls is mod.EpisodicEnv
    assert hasattr(mod.EpisodicEnv, "step")


def test_generated_extract_patch(episodes, tmp_path):
    exporters.export(episodes, "molt", tmp_path)
    mod, _ = _load_generated_env(str(_agent_path(tmp_path)))
    assert mod._extract_patch("```diff\ndiff --git a/x b/x\n```").strip() == "diff --git a/x b/x"
    assert mod._extract_patch("diff --git a/x b/x\n+y").startswith("diff --git")
    assert mod._extract_patch("no code here") == "no code here"


def test_generated_grade_rewards_passing_patch(sample_episode, tmp_path):
    origin, sha = _failing_origin(tmp_path)
    patch = _gold_patch(origin)
    exporters.export([sample_episode], "molt", tmp_path / "bundle")
    mod, _ = _load_generated_env(str(_agent_path(tmp_path / "bundle")))
    label = json.dumps({"remote_url": origin, "base_commit": sha,
                        "test_command": "python3 -m pytest -q", "framework": "pytest"})

    assert mod._grade(label, patch) == 1.0
    assert mod._grade(label, "") == 0.0
    assert mod._grade(label, "```diff\nnot a real patch\n```") == 0.0


def test_generated_step_is_async_and_returns_tensor_reward(sample_episode, tmp_path):
    origin, sha = _failing_origin(tmp_path)
    patch = _gold_patch(origin)
    exporters.export([sample_episode], "molt", tmp_path / "bundle")
    mod, _ = _load_generated_env(str(_agent_path(tmp_path / "bundle")))
    label = json.dumps({"remote_url": origin, "base_commit": sha,
                        "test_command": "python3 -m pytest -q", "framework": "pytest"})

    result = asyncio.run(mod.EpisodicEnv().step({"label": label, "action_text": patch}))
    assert float(result.reward) == 1.0
    assert "tests_pass" in result.info


def test_piped_command_respects_pipefail(sample_episode, tmp_path):
    origin, sha = _failing_origin(tmp_path)
    patch = _gold_patch(origin)
    exporters.export([sample_episode], "molt", tmp_path / "bundle")
    mod, _ = _load_generated_env(str(_agent_path(tmp_path / "bundle")))
    label = json.dumps({"remote_url": origin, "base_commit": sha,
                        "test_command": "python3 -m pytest -q | tail -1", "framework": "pytest"})

    assert mod._grade(label, patch) == 1.0
    assert mod._grade(label, "") == 0.0


def test_absolute_path_command_flagged_non_portable(sample_episode, tmp_path):
    sample_episode["commands"] = []
    sample_episode["tests"] = [{
        "ts": "t", "framework": "cargo",
        "command": "cargo test --manifest-path /Users/x/repo/Cargo.toml",
        "passed": 1, "failed": 0, "total": 1, "ok": True,
    }]
    result = exporters.export([sample_episode], "molt", tmp_path)

    assert result["tasks"] == 1
    assert result["non_portable"] == 1
    rows = [json.loads(l) for l in (tmp_path / "prompts.jsonl").read_text().splitlines() if l.strip()]
    assert rows[0]["portable"] is False
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert "ep_test_good" in manifest["non_portable"]


def test_molt_relativizes_repo_path_and_marks_portable(sample_episode, tmp_path):
    sample_episode["repo_state"]["root"] = "/repo"
    sample_episode["commands"] = []
    sample_episode["tests"] = [{
        "ts": "t", "framework": "cargo",
        "command": "cargo test --manifest-path /repo/src-tauri/Cargo.toml",
        "passed": 1, "failed": 0, "total": 1, "ok": True,
    }]
    result = exporters.export([sample_episode], "molt", tmp_path)

    assert result["non_portable"] == 0
    rows = [json.loads(l) for l in (tmp_path / "prompts.jsonl").read_text().splitlines() if l.strip()]
    assert rows[0]["portable"] is True
    assert json.loads(rows[0]["label"])["test_command"] == "cargo test --manifest-path ./src-tauri/Cargo.toml"


def test_grade_rejects_option_injection_remote(episodes, tmp_path):
    exporters.export(episodes, "molt", tmp_path)
    mod, _ = _load_generated_env(str(_agent_path(tmp_path)))
    label = json.dumps({"remote_url": "--upload-pack=touch /tmp/pwned",
                        "test_command": "true", "framework": "x"})
    assert mod._grade(label, "") == 0.0


def test_episode_without_remote_is_skipped(sample_episode, tmp_path):
    sample_episode["repo_state"]["remote_url"] = None
    result = exporters.export([sample_episode], "molt", tmp_path)
    assert result["tasks"] == 0
    assert result["skipped"][0]["reason"] == "no_remote"


def test_bad_episode_is_skipped(episodes, tmp_path):
    result = exporters.export(episodes, "molt", tmp_path)
    reasons = {row["id"]: row["reason"] for row in result["skipped"]}
    assert reasons["ep_test_bad"] in {"bad_outcome", "no_verifier", "no_remote", "low_trust"}


def test_low_trust_episode_is_skipped(sample_episode, tmp_path):
    sample_episode["validity"] = {"trust": "low"}
    result = exporters.export([sample_episode], "molt", tmp_path)
    assert result["skipped"][0]["reason"] == "low_trust"


def test_episode_without_verifier_is_skipped(sample_episode, tmp_path):
    sample_episode["tests"] = []
    sample_episode["commands"] = []
    result = exporters.export([sample_episode], "molt", tmp_path)
    assert result["skipped"][0]["reason"] == "no_verifier"


def test_unsafe_id_is_skipped(sample_episode, tmp_path):
    sample_episode["id"] = "../../etc/passwd"
    result = exporters.export([sample_episode], "molt", tmp_path)
    assert result["tasks"] == 0
    assert result["skipped"][0]["reason"] == "unsafe_id"


def test_unsafe_base_commit_is_dropped_not_embedded(sample_episode, tmp_path):
    sample_episode["repo_state"]["base_commit"] = "--output=/tmp/x"
    exporters.export([sample_episode], "molt", tmp_path)
    rows = [json.loads(l) for l in (tmp_path / "prompts.jsonl").read_text().splitlines() if l.strip()]
    assert json.loads(rows[0]["label"])["base_commit"] is None


def test_readme_has_train_command(episodes, tmp_path):
    exporters.export(episodes, "molt", tmp_path)
    readme = (tmp_path / "README.md").read_text()
    assert "molt.cli.train_rl_ray" in readme
    assert "--train.agent_path ./agents/episodic_env.py" in readme
    assert "--data.input_key input" in readme


def test_molt_rejects_stdout(episodes):
    with pytest.raises(ValueError, match="stdout"):
        exporters.export(episodes, "molt", "-")
