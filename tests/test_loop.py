import json
import subprocess
from pathlib import Path

import pytest

from episodic import store, loop
from episodic.schema import new_episode
from episodic.core import reward, rubric


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


def _partition_ids(seed, frac, n_holdout, n_train):
    pool = [f"ep_{i:03d}" for i in range(200)]
    holdout_ids = [i for i in pool if loop._hash_frac(i, seed) < frac][:n_holdout]
    train_ids = [i for i in pool if loop._hash_frac(i, seed) >= frac][:n_train]
    return holdout_ids, train_ids


def _episode_with_observation(ep_id, origin, sha, observation):
    episode = new_episode(id=ep_id, intent="edit f.py")
    episode["repo_state"].update({"root": origin, "remote_url": origin, "base_commit": sha})
    episode["steps"] = [{
        "index": 0, "ts": "t", "type": "file_edit", "tool": "Edit", "intent": "edit f.py",
        "input": {"file_path": "f.py"}, "observation": observation, "approved": True,
        "cwd": origin, "duration_ms": None,
    }]
    episode["diffs"] = [{"file": "f.py", "status": "modified", "additions": 1, "deletions": 0, "unified": None}]
    episode["commands"] = [{"ts": "t", "command": "python3 -m pytest -q", "cwd": origin,
                            "exit_code": 0, "output_excerpt": "1 passed", "is_test": True}]
    episode["outcome"]["status"] = "merged"
    episode["reward_vector"] = reward.reward_vector(episode)
    return episode


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
                       ("eval_concurrency", float("inf")), ("promote_margin", float("inf")),
                       ("epochs", 0), ("epochs", 1.5), ("epochs", -1), ("epochs", "nope")]:
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
    assert promoted["served_ref"] == promoted["model_dir"]


def test_served_ref_prefers_trainer_reported_sampler_path():
    train_manifest = {"result": {"sampler_path": "tinker://run-id/weights/checkpoint-001"}}
    assert loop._served_ref("/local/candidate", train_manifest) == "tinker://run-id/weights/checkpoint-001"


def test_served_ref_falls_back_to_candidate_model_dir():
    assert loop._served_ref("/local/candidate", {"result": {}}) == "/local/candidate"
    assert loop._served_ref("/local/candidate", None) == "/local/candidate"


def test_ensure_reward_is_a_noop_without_a_judge():
    episode = new_episode(id="ep_noop")
    episode["reward_vector"] = reward.reward_vector(episode)
    before = episode["reward_vector"]
    episodes = [episode]

    result = loop.ensure_reward(episodes, judge=None)

    assert result is episodes
    assert episode["reward_vector"] is before


def test_ensure_reward_recomputes_rubric_with_a_judge():
    episode = new_episode(id="ep_judged")
    episode["reward_vector"] = reward.reward_vector(episode)
    baseline_rubric = episode["reward_vector"]["rubric"]

    judge = rubric.openrubrics_judge(lambda prompt: "SCORE: 0.9 solid")
    loop.ensure_reward([episode], judge=judge)

    assert episode["reward_vector"]["rubric"] > baseline_rubric
    assert episode["reward_vector"]["components"]["has_rubric"] is True


def test_ensure_reward_skips_rejudging_cache_hits():
    calls = []
    judge = rubric.openrubrics_judge(lambda prompt: calls.append(prompt) or "SCORE: 0.9 solid")
    episode = new_episode(id="ep_cache")
    episode["reward_vector"] = reward.reward_vector(episode)
    cache = {}

    loop.ensure_reward([episode], judge, cache=cache, judge_sig="sig-a")
    first = len(calls)
    assert first > 0 and cache

    loop.ensure_reward([episode], judge, cache=cache, judge_sig="sig-a")
    assert len(calls) == first


def test_ensure_reward_cache_misses_when_judge_signature_changes():
    calls = []
    judge = rubric.openrubrics_judge(lambda prompt: calls.append(prompt) or "SCORE: 0.9 solid")
    episode = new_episode(id="ep_cache2")
    episode["reward_vector"] = reward.reward_vector(episode)
    cache = {}

    loop.ensure_reward([episode], judge, cache=cache, judge_sig="sig-a")
    first = len(calls)
    loop.ensure_reward([episode], judge, cache=cache, judge_sig="sig-b")
    assert len(calls) > first


def test_run_loop_writes_the_judge_reward_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path / ".episodic"))
    origin, sha = _origin_repo(tmp_path)
    holdout_ids, train_ids = _split_ids(seed=0, frac=0.5)
    for ep_id in holdout_ids + train_ids:
        store.save_episode(_episode(ep_id, origin, sha))

    config = {"trainer": "command", "format": "sft", "holdout_frac": 0.5, "seed": 0,
              "min_composite": 0.0, "train_config": {"command": "false"},
              "judge": True, "judge_cmd": "sh -c \"printf 'SCORE: 0.9 solid'\"", "judge_timeout": 10,
              "out": str(tmp_path / "loopout")}
    loop.run_loop(config)

    cache_path = loop._judge_cache_path(None)
    assert cache_path.exists()
    assert json.loads(cache_path.read_text(encoding="utf-8"))


def test_resolve_judge_returns_none_when_disabled():
    assert loop._resolve_judge({}) is None
    assert loop._resolve_judge({"judge": False}) is None


def test_resolve_judge_builds_a_working_default_judge():
    judge = loop._resolve_judge({
        "judge": True, "judge_cmd": "sh -c \"printf 'SCORE: 1.0 great'\"", "judge_timeout": 5,
    })
    assert callable(judge)
    satisfied, _ = judge({}, {"desc": "x"})
    assert satisfied == 1.0


def test_resolve_judge_swallows_an_unauthenticated_labeler():
    judge = loop._resolve_judge({
        "judge": True,
        "judge_cmd": "sh -c \"printf 'Not logged in' 1>&2; exit 1\"",
        "judge_timeout": 5,
    })
    satisfied, reason = judge({}, {"desc": "x"})
    assert satisfied is None
    assert "judge unavailable" in reason


def test_run_loop_rejects_bad_judge_timeout(tmp_path, monkeypatch):
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path / ".episodic"))
    with pytest.raises(ValueError):
        loop.run_loop({"trainer": "command", "format": "sft", "judge": True, "judge_timeout": -1,
                       "train_config": {"command": "true"}, "out": str(tmp_path / "o")})


def test_run_loop_single_epoch_manifest_has_no_epoch_bookkeeping_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path / ".episodic"))
    origin, sha = _origin_repo(tmp_path)
    holdout_ids, train_ids = _split_ids(seed=0, frac=0.5)
    for ep_id in holdout_ids + train_ids:
        store.save_episode(_episode(ep_id, origin, sha))

    config = {"trainer": "command", "format": "sft", "holdout_frac": 0.5, "seed": 0,
              "min_composite": 0.0, "train_config": {"command": "false"},
              "out": str(tmp_path / "loopout")}
    manifest = loop.run_loop(config)

    assert "epoch" not in manifest
    assert "epoch_count" not in manifest
    assert "epochs" not in manifest
    assert manifest["decision"] == "dry_run"


def test_run_loop_multi_epoch_produces_epoch_history_and_final_fields(tmp_path, monkeypatch):
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
        "base_model": "base", "execute": True, "epochs": 2,
        "replay_cmd": f"python3 {runner} {{model}} {{workspace}} {{prompt_file}}",
        "out": str(tmp_path / "loopout"),
    }
    manifest = loop.run_loop(config)

    assert manifest["epoch_count"] == 2
    assert len(manifest["epochs"]) == 2
    assert manifest["final_decision"] == "promote"
    assert manifest["final_model"] != "base"

    epoch0, epoch1 = manifest["epochs"]
    assert epoch0["epoch"] == 0
    assert epoch1["epoch"] == 1
    assert epoch0["evaluator_type"] == "rubric_judge"
    assert epoch1["base_model"] == epoch0["candidate_model"]
    assert (tmp_path / "loopout" / "epoch_0" / "loop.json").exists()
    assert (tmp_path / "loopout" / "epoch_1" / "loop.json").exists()


def test_run_loop_router_flag_writes_a_router_model(tmp_path, monkeypatch):
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path / ".episodic"))
    origin, sha = _origin_repo(tmp_path)
    holdout_ids, train_ids = _split_ids(seed=0, frac=0.5)
    for ep_id in holdout_ids + train_ids:
        store.save_episode(_episode(ep_id, origin, sha))

    config = {"trainer": "command", "format": "sft", "holdout_frac": 0.5, "seed": 0,
              "min_composite": 0.0, "train_config": {"command": "false"}, "router": True,
              "out": str(tmp_path / "loopout")}
    manifest = loop.run_loop(config)

    assert manifest["router"] is not None
    assert manifest["router"]["trained_on"] == len(holdout_ids) + len(train_ids)
    assert (tmp_path / "loopout" / "router_model.json").exists()


def test_run_loop_router_flag_off_by_default_does_not_write_a_model(tmp_path, monkeypatch):
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path / ".episodic"))
    origin, sha = _origin_repo(tmp_path)
    holdout_ids, train_ids = _split_ids(seed=0, frac=0.5)
    for ep_id in holdout_ids + train_ids:
        store.save_episode(_episode(ep_id, origin, sha))

    config = {"trainer": "command", "format": "sft", "holdout_frac": 0.5, "seed": 0,
              "min_composite": 0.0, "train_config": {"command": "false"},
              "out": str(tmp_path / "loopout")}
    manifest = loop.run_loop(config)

    assert manifest["router"] is None
    assert not (tmp_path / "loopout" / "router_model.json").exists()


def test_loop_default_judge_lifts_a_no_verifier_episode_into_the_training_pool(tmp_path, monkeypatch):
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path / ".episodic"))
    episode = new_episode(id="ep_neutral", intent="task with no runnable verifier")
    store.save_episode(episode)

    base_config = {
        "trainer": "command", "format": "sft", "holdout_frac": 0.0, "seed": 0,
        "min_composite": 0.4, "train_config": {"command": "false"},
        "out": str(tmp_path / "loop_base"),
    }
    baseline = loop.run_loop(base_config)
    assert baseline["decision"] == "no_train_data"
    assert baseline["train_ids"] == []

    judged_config = dict(base_config, out=str(tmp_path / "loop_judged"), judge=True,
                         judge_cmd="sh -c \"printf 'SCORE: 1.0 clear and correct'\"")
    judged = loop.run_loop(judged_config)
    assert judged["train_ids"] == ["ep_neutral"]


def test_sim_prefilter_flag_parsing():
    assert loop._sim_prefilter_flag({}) is False
    assert loop._sim_prefilter_flag({"sim_prefilter": True}) is True
    assert loop._sim_prefilter_flag({"sim_prefilter": False}) is False
    assert loop._sim_prefilter_flag({"sim_prefilter": "true"}) is True
    assert loop._sim_prefilter_flag({"sim_prefilter": "false"}) is False


def test_resolve_sim_predictor_defaults_to_the_prefix_persistence_baseline():
    from episodic import worldbench

    assert loop._resolve_sim_predictor({}) is worldbench.NAMED_PREDICTORS["prefix"]


def test_resolve_sim_predictor_accepts_a_callable_unchanged():
    custom = lambda sample: "x"
    assert loop._resolve_sim_predictor({"sim_predictor": custom}) is custom


def test_resolve_sim_predictor_rejects_an_unknown_name():
    with pytest.raises(ValueError):
        loop._resolve_sim_predictor({"sim_predictor": "not-a-real-predictor"})


def test_sim_rank_holdout_prioritizes_episodes_the_predictor_reproduces_worst():
    easy = [{"id": f"ep_easy_{i}", "steps": [{"index": 0, "observation": "same"}]} for i in range(3)]
    hard = [{"id": f"ep_hard_{i}", "steps": [{"index": 0, "observation": "totally different " * 8}]}
            for i in range(3)]
    holdout = easy + hard

    def predictor(sample):
        return "same" if "easy" in sample["episode_id"] else ""

    ranked = loop._sim_rank_holdout(holdout, {"sim_predictor": predictor})
    ranked_ids = [e["id"] for e in ranked]

    assert set(ranked_ids[:3]) == {e["id"] for e in hard}
    assert set(ranked_ids[3:]) == {e["id"] for e in easy}


def test_sim_rank_holdout_rejects_bad_sim_max_turns():
    holdout = [{"id": f"ep_{i}", "steps": []} for i in range(3)]
    with pytest.raises(ValueError):
        loop._sim_rank_holdout(holdout, {"sim_max_turns": -1})


def test_run_loop_sim_prefilter_off_by_default_uses_id_sorted_holdout(tmp_path, monkeypatch):
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path / ".episodic"))
    origin, sha = _origin_repo(tmp_path)
    holdout_ids, train_ids = _partition_ids(seed=0, frac=0.5, n_holdout=6, n_train=1)

    for ep_id in holdout_ids:
        store.save_episode(_episode_with_observation(ep_id, origin, sha, "same"))
    for ep_id in train_ids:
        store.save_episode(_episode_with_observation(ep_id, origin, sha, "same"))

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
        "base_model": "base", "execute": True, "max_holdout": 3,
        "replay_cmd": f"python3 {runner} {{model}} {{workspace}} {{prompt_file}}",
        "out": str(tmp_path / "loopout"),
    }
    manifest = loop.run_loop(config)

    evaluated_ids = {row["episode_id"] for row in manifest["scores"]}
    assert evaluated_ids == set(sorted(holdout_ids)[:3])


def test_run_loop_sim_prefilter_selects_hardest_holdout_episodes_for_real_replay(tmp_path, monkeypatch):
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path / ".episodic"))
    origin, sha = _origin_repo(tmp_path)
    holdout_ids, train_ids = _partition_ids(seed=0, frac=0.5, n_holdout=6, n_train=1)
    hard_ids = set(sorted(holdout_ids)[-3:])

    for ep_id in holdout_ids:
        observation = "totally different unmatched content " * 8 if ep_id in hard_ids else "same"
        store.save_episode(_episode_with_observation(ep_id, origin, sha, observation))
    for ep_id in train_ids:
        store.save_episode(_episode_with_observation(ep_id, origin, sha, "same"))

    def sim_predictor(sample):
        return "" if sample["episode_id"] in hard_ids else "same"

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
        "base_model": "base", "execute": True, "max_holdout": 3,
        "sim_prefilter": True, "sim_predictor": sim_predictor,
        "replay_cmd": f"python3 {runner} {{model}} {{workspace}} {{prompt_file}}",
        "out": str(tmp_path / "loopout"),
    }
    manifest = loop.run_loop(config)

    evaluated_ids = {row["episode_id"] for row in manifest["scores"]}
    assert evaluated_ids == hard_ids
    assert evaluated_ids != set(sorted(holdout_ids)[:3])


def _model_diff(sha):
    return (
        "diff --git a/f.py b/f.py\n"
        f"index {sha[:7]}..0000000 100644\n"
        "--- a/f.py\n"
        "+++ b/f.py\n"
        "@@ -1 +1 @@\n"
        "-x = 1\n"
        "+x = 2\n"
    )


def test_resolve_eval_runner_is_none_when_eval_backend_is_unset():
    assert loop._resolve_eval_runner({}, None) is None


def test_resolve_eval_runner_builds_a_working_callable_for_the_stub_backend(tmp_path, monkeypatch):
    origin, sha = _origin_repo(tmp_path)
    diff = _model_diff(sha)
    runner = loop._resolve_eval_runner(
        {"eval_backend": "stub", "eval_stub": {"candidate-model": diff}}, None)

    assert callable(runner)
    output, rc = runner("candidate-model", Path(origin), "fix f.py")
    assert rc == 0
    assert (Path(origin) / "f.py").read_text() == "x = 2\n"


def test_eval_one_with_stub_runner_produces_distinct_paired_scores(tmp_path, monkeypatch):
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path / ".episodic"))
    origin, sha = _origin_repo(tmp_path)
    episode = _episode("ep_pairtest", origin, sha)

    runner = loop._resolve_eval_runner(
        {"eval_backend": "stub", "eval_stub": {"candidate-model": _model_diff(sha)}}, None)

    row = loop._eval_one(episode, "candidate-model", "base-model", None, None, runner=runner)

    assert row["candidate"] is not None
    assert row["base"] is not None
    assert row["candidate"] > row["base"]


def _source_dep_origin(tmp_path):
    origin = tmp_path / "origin_src"
    origin.mkdir()
    (origin / "f.py").write_text("def val():\n    return 1\n")
    (origin / "test_f.py").write_text("from f import val\n\n\ndef test_ok():\n    assert val() == 1\n")
    _git(str(origin), "init", "-q")
    _git(str(origin), "config", "user.email", "t@t.dev")
    _git(str(origin), "config", "user.name", "t")
    _git(str(origin), "add", "-A")
    _git(str(origin), "commit", "-q", "-m", "base")
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(origin),
                         capture_output=True, text=True).stdout.strip()
    return str(origin), sha


def _source_breaking_diff():
    return (
        "diff --git a/f.py b/f.py\n"
        "--- a/f.py\n"
        "+++ b/f.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def val():\n"
        "-    return 1\n"
        "+    return 2\n"
    )


def test_run_loop_eval_backend_stub_keeps_base_when_the_candidate_diff_regresses(tmp_path, monkeypatch):
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path / ".episodic"))
    origin, sha = _source_dep_origin(tmp_path)
    holdout_ids, train_ids = _split_ids(seed=0, frac=0.5)
    for ep_id in holdout_ids + train_ids:
        store.save_episode(_episode(ep_id, origin, sha))

    def stub(model, messages):
        return _source_breaking_diff() if "candidate" in model else ""

    config = {
        "trainer": "command", "format": "sft", "holdout_frac": 0.5, "seed": 0,
        "min_composite": 0.0, "train_config": {"command": "true"},
        "base_model": "base-model", "execute": True,
        "eval_backend": "stub", "eval_stub": stub,
        "out": str(tmp_path / "loopout"),
    }
    manifest = loop.run_loop(config)

    assert manifest["executed"] is True
    assert manifest["evaluated"] >= 1
    assert manifest["candidate_mean"] < manifest["base_mean"]
    assert manifest["decision"] == "keep_base"
    assert not (tmp_path / "loopout" / "promoted.json").exists()


def test_run_loop_eval_backend_stub_promotes_when_the_candidate_diff_helps(tmp_path, monkeypatch):
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path / ".episodic"))
    origin, sha = _origin_repo(tmp_path)
    holdout_ids, train_ids = _split_ids(seed=0, frac=0.5)
    for ep_id in holdout_ids + train_ids:
        store.save_episode(_episode(ep_id, origin, sha))

    calls = []

    def stub(model, messages):
        calls.append(model)
        return _model_diff(sha) if "candidate" in model else ""

    config = {
        "trainer": "command", "format": "sft", "holdout_frac": 0.5, "seed": 0,
        "min_composite": 0.0, "train_config": {"command": "true"},
        "base_model": "base-model", "execute": True,
        "eval_backend": "stub", "eval_stub": stub,
        "out": str(tmp_path / "loopout"),
    }
    manifest = loop.run_loop(config)

    assert manifest["executed"] is True
    assert manifest["evaluated"] >= 1
    assert manifest["candidate_mean"] > manifest["base_mean"]
    assert manifest["decision"] == "promote"
    assert any("candidate" in model for model in calls)
    assert any("candidate" not in model for model in calls)
    promoted = json.loads((tmp_path / "loopout" / "promoted.json").read_text())
    assert "candidate" in promoted["model_dir"]


def test_run_loop_eval_backend_unavailable_propagates_trainer_unavailable(tmp_path, monkeypatch):
    from episodic import trainers
    from episodic.trainers import mlx as mlx_trainer

    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path / ".episodic"))
    origin, sha = _origin_repo(tmp_path)
    holdout_ids, train_ids = _split_ids(seed=0, frac=0.5)
    for ep_id in holdout_ids + train_ids:
        store.save_episode(_episode(ep_id, origin, sha))

    def raise_unavailable():
        raise trainers.TrainerUnavailable("mlx-sft", "install mlx-lm")

    monkeypatch.setattr(mlx_trainer, "_require_mlx", raise_unavailable)

    config = {
        "trainer": "command", "format": "sft", "holdout_frac": 0.5, "seed": 0,
        "min_composite": 0.0, "train_config": {"command": "true"},
        "base_model": "base-model", "execute": True,
        "eval_backend": "mlx", "eval_model_dir": "fake/model",
        "out": str(tmp_path / "loopout"),
    }

    with pytest.raises(trainers.TrainerUnavailable):
        loop.run_loop(config)


def test_run_loop_sim_prefilter_is_a_noop_when_holdout_is_not_capped(tmp_path, monkeypatch):
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path / ".episodic"))
    origin, sha = _origin_repo(tmp_path)
    holdout_ids, train_ids = _split_ids(seed=0, frac=0.5)
    for ep_id in holdout_ids + train_ids:
        store.save_episode(_episode(ep_id, origin, sha))

    def sim_predictor(sample):
        raise AssertionError("sim predictor should not run when holdout is not capped")

    config = {
        "trainer": "command", "format": "sft", "holdout_frac": 0.5, "seed": 0,
        "min_composite": 0.0, "train_config": {"command": "false"},
        "sim_prefilter": True, "sim_predictor": sim_predictor,
        "out": str(tmp_path / "loopout"),
    }
    manifest = loop.run_loop(config)
    assert manifest["decision"] == "dry_run"
