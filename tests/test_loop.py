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


def test_loop_dry_run_trains_but_does_not_execute(tmp_path, monkeypatch):
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path / ".episodic"))
    origin, sha = _origin_repo(tmp_path)
    holdout_ids, train_ids = _split_ids(seed=0, frac=0.5)
    for ep_id in holdout_ids + train_ids:
        store.save_episode(_episode(ep_id, origin, sha))

    config = {"trainer": "command", "format": "sft", "holdout_frac": 0.5, "seed": 0,
              "min_composite": 0.0, "train_config": {"command": "true"},
              "out": str(tmp_path / "loopout")}
    manifest = loop.run_loop(config)

    assert manifest["executed"] is False
    assert manifest["decision"] == "dry_run"
    assert manifest["scores"] == []
    assert "train_manifest" in manifest
    assert set(manifest["holdout_ids"]) == set(holdout_ids)
    assert set(manifest["train_ids"]) == set(train_ids)


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
