import json
import subprocess

import pytest

from episodic import store, loop
from episodic.schema import new_episode
from episodic.core import reward


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _origin_repo(tmp_path):
    origin = tmp_path / "origin"
    origin.mkdir()
    (origin / "f.py").write_text("x = 1\n")
    (origin / "test_f.py").write_text("def test_ok():\n    assert True\n")
    _git(str(origin), "init", "-q")
    _git(str(origin), "config", "user.email", "t@t.dev")
    _git(str(origin), "config", "user.name", "t")
    _git(str(origin), "add", "-A")
    _git(str(origin), "commit", "-q", "-m", "base")
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(origin),
                         capture_output=True, text=True).stdout.strip()
    return str(origin), sha


def _episode(ep_id, origin, sha):
    episode = new_episode(id=ep_id, intent="edit f.py")
    episode["repo_state"].update({"root": origin, "remote_url": origin, "base_commit": sha})
    episode["steps"] = [{
        "index": 0, "ts": "t", "type": "file_edit", "tool": "Edit", "intent": "edit f.py",
        "input": {"file_path": "f.py"}, "observation": "done", "approved": True,
        "cwd": origin, "duration_ms": None,
    }]
    episode["diffs"] = [{"file": "f.py", "status": "modified", "additions": 1, "deletions": 0, "unified": None}]
    episode["commands"] = [{"ts": "t", "command": "python3 -m pytest -q", "cwd": origin,
                            "exit_code": 0, "output_excerpt": "1 passed", "is_test": True}]
    episode["outcome"]["status"] = "merged"
    episode["reward_vector"] = reward.reward_vector(episode)
    return episode


def _split_ids(seed, frac):
    pool = [f"ep_{i:02d}" for i in range(60)]
    low = [i for i in pool if loop._hash_frac(i, seed) < frac]
    high = [i for i in pool if loop._hash_frac(i, seed) >= frac]
    return low[:2], high[:2]


def test_split_is_deterministic_and_total():
    good = [{"id": f"ep_{i}"} for i in range(20)]
    train_a, holdout_a = loop.split_episodes(good, 0.3, seed=0)
    train_b, holdout_b = loop.split_episodes(good, 0.3, seed=0)
    assert [e["id"] for e in train_a] == [e["id"] for e in train_b]
    assert len(train_a) + len(holdout_a) == 20
    assert set(e["id"] for e in train_a).isdisjoint(e["id"] for e in holdout_a)


def test_execute_flag_parsing():
    assert loop._execute_flag(True) is True
    assert loop._execute_flag(False) is False
    assert loop._execute_flag("true") is True
    assert loop._execute_flag("1") is True
    assert loop._execute_flag("false") is False
    assert loop._execute_flag("0") is False
    assert loop._execute_flag("") is False
    assert loop._execute_flag(None) is False
    assert loop._execute_flag(1) is False


def test_composite_coerces_non_numeric():
    assert loop._composite({"reward_vector": {"composite": 0.7}}) == 0.7
    assert loop._composite({"reward_vector": {"composite": "1.0"}}) == 0.0
    assert loop._composite({"reward_vector": {"composite": None}}) == 0.0
    assert loop._composite({}) == 0.0


def test_json_safe_strips_non_finite():
    cleaned = loop._json_safe({"a": float("nan"), "b": [float("inf"), 1.0],
                               "c": {"d": float("-inf")}})
    assert cleaned == {"a": None, "b": [None, 1.0], "c": {"d": None}}
    text = json.dumps(cleaned)
    assert "NaN" not in text and "Infinity" not in text


def test_finite_rejects_non_numeric_and_non_finite():
    assert loop._finite(0.5) and loop._finite(0) and loop._finite(-1.0)
    assert not loop._finite(None)
    assert not loop._finite(float("nan"))
    assert not loop._finite(float("inf"))
    assert not loop._finite(True)
    assert not loop._finite("0.5")


def test_partition_is_order_independent():
    pool = [{"id": f"ep_{i}", "reward_vector": {"composite": 0.9}} for i in range(10)]
    forward = loop.partition(list(pool), 0.0, 0.3, seed=0)
    backward = loop.partition(list(reversed(pool)), 0.0, 0.3, seed=0)
    assert [e["id"] for e in forward[0]] == [e["id"] for e in backward[0]]
    assert [e["id"] for e in forward[1]] == [e["id"] for e in backward[1]]


def test_run_loop_rejects_bad_config(tmp_path, monkeypatch):
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path / ".episodic"))
    for key, value in [("holdout_frac", 1.5), ("max_holdout", -1),
                       ("eval_concurrency", 0), ("promote_margin", "nope"),
                       ("promote_margin", -0.1), ("holdout_frac", float("nan")),
                       ("eval_concurrency", float("inf")), ("promote_margin", float("inf"))]:
        with pytest.raises(ValueError):
            loop.run_loop({"trainer": "command", "format": "sft", key: value,
                           "train_config": {"command": "true"}, "out": str(tmp_path / "o")})


def test_partition_streams_filter_and_split_in_one_pass():
    consumed = []

    def episodes():
        for i in range(20):
            composite = 0.9 if i % 2 == 0 else 0.1
            ep = {"id": f"ep_{i}", "reward_vector": {"composite": composite}}
            consumed.append(ep["id"])
            yield ep

    train, holdout = loop.partition(episodes(), min_composite=0.5, holdout_frac=0.3, seed=0)

    kept = {e["id"] for e in train} | {e["id"] for e in holdout}
    assert kept == {f"ep_{i}" for i in range(20) if i % 2 == 0}
    assert len(consumed) == 20
    assert set(e["id"] for e in train).isdisjoint(e["id"] for e in holdout)


def test_loop_dry_run_does_not_train_or_execute(tmp_path, monkeypatch):
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path / ".episodic"))
    origin, sha = _origin_repo(tmp_path)
    holdout_ids, train_ids = _split_ids(seed=0, frac=0.5)
    for ep_id in holdout_ids + train_ids:
        store.save_episode(_episode(ep_id, origin, sha))

    config = {"trainer": "command", "format": "sft", "holdout_frac": 0.5, "seed": 0,
              "min_composite": 0.0, "train_config": {"command": "false"},
              "out": str(tmp_path / "loopout")}
    manifest = loop.run_loop(config)

    assert manifest["executed"] is False
    assert manifest["decision"] == "dry_run"
    assert manifest["scores"] == []
    assert "train_manifest" not in manifest
    assert manifest["candidate_model"] is None
    assert not (tmp_path / "loopout" / "candidate").exists()
    assert set(manifest["holdout_ids"]) == set(holdout_ids)
    assert set(manifest["train_ids"]) == set(train_ids)

    plan = manifest["plan"]
    assert plan["trainer"] == "command"
    assert plan["dataset"].endswith(".jsonl")
    assert plan["dataset_rows"] > 0
    assert plan["train_config"] == {"command": "false"}
    assert plan["candidate_model_dir"].endswith("candidate")
    assert plan["holdout_count"] == len(holdout_ids)


def test_loop_string_false_execute_stays_dry_run(tmp_path, monkeypatch):
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path / ".episodic"))
    origin, sha = _origin_repo(tmp_path)
    holdout_ids, train_ids = _split_ids(seed=0, frac=0.5)
    for ep_id in holdout_ids + train_ids:
        store.save_episode(_episode(ep_id, origin, sha))

    config = {"trainer": "command", "format": "sft", "holdout_frac": 0.5, "seed": 0,
              "min_composite": 0.0, "train_config": {"command": "false"},
              "execute": "false", "out": str(tmp_path / "lo")}
    manifest = loop.run_loop(config)

    assert manifest["executed"] is False
    assert manifest["decision"] == "dry_run"
    assert "train_manifest" not in manifest


def test_loop_dry_run_auto_mints_harbor_tasks(tmp_path, monkeypatch):
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path / ".episodic"))
    origin, sha = _origin_repo(tmp_path)
    holdout_ids, train_ids = _split_ids(seed=0, frac=0.5)
    for ep_id in holdout_ids + train_ids:
        store.save_episode(_episode(ep_id, origin, sha))

    config = {"trainer": "command", "format": "sft", "holdout_frac": 0.5, "seed": 0,
              "min_composite": 0.0, "train_config": {"command": "false"},
              "out": str(tmp_path / "loopout")}
    manifest = loop.run_loop(config)

    assert manifest["harbor"]["tasks"] == len(train_ids)
    harbor_manifest = json.loads((tmp_path / "loopout" / "harbor" / "manifest.json").read_text())
    assert harbor_manifest["task_count"] == len(train_ids)
    for ep_id in train_ids:
        assert (tmp_path / "loopout" / "harbor" / "tasks" / ep_id / "task.toml").exists()
        script = (tmp_path / "loopout" / "harbor" / "tasks" / ep_id / "tests" / "run-tests.sh").read_text()
        assert "pytest" in script


def test_loop_mint_harbor_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path / ".episodic"))
    origin, sha = _origin_repo(tmp_path)
    holdout_ids, train_ids = _split_ids(seed=0, frac=0.5)
    for ep_id in holdout_ids + train_ids:
        store.save_episode(_episode(ep_id, origin, sha))

    config = {"trainer": "command", "format": "sft", "holdout_frac": 0.5, "seed": 0,
              "min_composite": 0.0, "train_config": {"command": "false"}, "mint_harbor": False,
              "out": str(tmp_path / "loopout")}
    manifest = loop.run_loop(config)

    assert manifest["harbor"] is None
    assert not (tmp_path / "loopout" / "harbor").exists()


def test_loop_executes_evaluates_and_promotes(tmp_path, monkeypatch):
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path / ".episodic"))
    origin, sha = _origin_repo(tmp_path)
    holdout_ids, train_ids = _split_ids(seed=0, frac=0.5)
    for ep_id in holdout_ids + train_ids:
        store.save_episode(_episode(ep_id, origin, sha))

    runner = tmp_path / "runner.py"
    runner.write_text(
        "import os, sys\n"
        "model, workspace = sys.argv[1], sys.argv[2]\n"
        "if 'candidate' in model:\n"
        "    with open(os.path.join(workspace, 'f.py'), 'a') as fh:\n"
        "        fh.write('# edit\\n')\n"
    )
    config = {
        "trainer": "command", "format": "sft", "holdout_frac": 0.5, "seed": 0,
        "min_composite": 0.0, "train_config": {"command": "true"},
        "base_model": "base", "execute": True,
        "replay_cmd": f"python3 {runner} {{model}} {{workspace}} {{prompt_file}}",
        "out": str(tmp_path / "loopout"),
    }
    manifest = loop.run_loop(config)

    assert manifest["executed"] is True
    assert manifest["evaluated"] >= 1
    assert manifest["candidate_mean"] > manifest["base_mean"]
    assert manifest["decision"] == "promote"
    promoted = json.loads((tmp_path / "loopout" / "promoted.json").read_text())
    assert "candidate" in promoted["model_dir"]
