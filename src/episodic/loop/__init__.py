import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from . import evaluator
from .. import store, exporters, trainers, replay, paths, worldbench, fidelity
from ..exporters import is_bad, is_trusted
from ..core import reward, rubric, validity
from ..serving import difficulty
from ..worldmodel import env as wm_env

SCHEMA_VERSION = "0.1.0"


def _composite(episode):
    value = (episode.get("reward_vector") or {}).get("composite")
    return value if _finite(value) else 0.0


def _apply_judged_reward(episode, judge):
    episode["reward_vector"] = reward.reward_vector(episode, judge=judge)
    return episode


def _judge_fingerprint():
    judged = [(item["id"], item.get("desc")) for item in rubric.CODING_RUBRIC if item.get("judge")]
    basis = json.dumps([judged, rubric.JUDGE_TRAJECTORY_LIMIT], sort_keys=True)
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


_JUDGE_FINGERPRINT = _judge_fingerprint()


def _judge_signature(config, state):
    etype = state.get("type")
    if etype == "trl_reward":
        return f"trl_reward:{state.get('model_dir')}:{_JUDGE_FINGERPRINT}"
    if etype == "local_critic":
        return f"local_critic:{id(state.get('critic'))}:{_JUDGE_FINGERPRINT}"
    return f"rubric_judge:{config.get('judge_cmd')}:{config.get('judge_timeout')}:{_JUDGE_FINGERPRINT}"


def _reward_cache_key(episode, judge_sig):
    payload = {key: value for key, value in episode.items() if key != "reward_vector"}
    basis = json.dumps({"episode": payload, "judge": judge_sig}, sort_keys=True, default=str)
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


def _judge_cache_path(start):
    return paths.home(start) / "cache" / "judge_reward.json"


def _load_judge_cache(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_judge_cache(path, cache):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache), encoding="utf-8")


def ensure_reward(episodes, judge, concurrency=4, cache=None, judge_sig=""):
    if judge is None:
        return episodes
    store_cache = {} if cache is None else cache
    pending = []
    for episode in episodes:
        key = _reward_cache_key(episode, judge_sig)
        cached = store_cache.get(key)
        if cached is not None:
            episode["reward_vector"] = cached
        else:
            pending.append((key, episode))
    if pending:
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            list(pool.map(lambda item: _apply_judged_reward(item[1], judge), pending))
        for key, episode in pending:
            store_cache[key] = episode["reward_vector"]
    return episodes


def _resolve_judge(config):
    if not config.get("judge"):
        return None
    timeout = _number(config, "judge_timeout", 120, low=1)
    return rubric.default_judge(command=config.get("judge_cmd"), timeout=timeout)


def ensure_validity(episodes):
    for episode in episodes:
        prior_llm = (episode.get("validity") or {}).get("llm")
        fresh = validity.assess(episode)
        if prior_llm is not None:
            fresh["llm"] = prior_llm
            if not prior_llm.get("trustworthy"):
                fresh["trust"] = "low"
                fresh["source"] = "rules+llm"
        episode["validity"] = fresh
    return episodes


def _execute_flag(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "on")
    return False


def _mint_flag(config):
    value = config.get("mint_harbor", True)
    if isinstance(value, bool):
        return value
    return _execute_flag(value)


def _mint_harbor(train, out):
    result = exporters.export(train, "harbor", str(out / "harbor"))
    return {
        "tasks": result.get("tasks", 0),
        "skipped": len(result.get("skipped", [])),
        "out_dir": result.get("out_dir"),
    }


def _router_flag(config):
    value = config.get("router", False)
    if isinstance(value, bool):
        return value
    return _execute_flag(value)


def _train_router(episodes, out):
    model = difficulty.learn_router(episodes)
    if model is None:
        return None
    path = difficulty.save_router_model(model, out / "router_model.json")
    return {
        "path": path,
        "trained_on": model.get("trained_on"),
        "positive_rate": model.get("positive_rate"),
        "fallback": model.get("fallback"),
    }


def _sim_prefilter_flag(config):
    value = config.get("sim_prefilter", False)
    if isinstance(value, bool):
        return value
    return _execute_flag(value)


def _resolve_sim_predictor(config):
    predictor = config.get("sim_predictor", "prefix")
    if isinstance(predictor, str):
        if predictor not in worldbench.NAMED_PREDICTORS:
            raise ValueError(
                f"unknown sim_predictor {predictor!r}; choose from {sorted(worldbench.NAMED_PREDICTORS)}")
        return worldbench.NAMED_PREDICTORS[predictor]
    return predictor


def _sim_max_turns(config):
    if config.get("sim_max_turns") is None:
        return None
    return _number(config, "sim_max_turns", None, low=1, integer=True)


def _sim_score(episode, predictor, max_turns):
    result = wm_env.rollout(episode, predictor, max_turns=max_turns)
    trajectory = fidelity.trajectory_score(result["turns"])
    return trajectory["mean_composite"]


def _sim_rank_holdout(holdout, config):
    predictor = _resolve_sim_predictor(config)
    max_turns = _sim_max_turns(config)
    scored = [(episode, _sim_score(episode, predictor, max_turns)) for episode in holdout]
    scored.sort(key=lambda pair: (pair[1] if pair[1] is not None else 1.0, pair[0]["id"]))
    return [episode for episode, _ in scored]


def _hash_frac(episode_id, seed):
    digest = hashlib.sha256(f"{seed}:{episode_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0x100000000


def select_good(episodes, min_composite):
    for episode in episodes:
        if _composite(episode) >= min_composite and not is_bad(episode) and is_trusted(episode):
            yield episode


def split_episodes(good, holdout_frac, seed):
    train, holdout = [], []
    for episode in good:
        target = holdout if _hash_frac(episode["id"], seed) < holdout_frac else train
        target.append(episode)
    return train, holdout


def partition(episodes, min_composite, holdout_frac, seed):
    train, holdout = split_episodes(select_good(episodes, min_composite), holdout_frac, seed)
    train.sort(key=lambda episode: episode["id"])
    holdout.sort(key=lambda episode: episode["id"])
    return train, holdout


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _mean(values):
    return sum(values) / len(values) if values else None


def _score(result):
    total = (result.get("scores") or {}).get("total")
    return total if _finite(total) else None


def _eval_one(episode, candidate_model, base_model, runner_cmd, start):
    replay.create_replay(episode, start=start)
    replay_id = replay.replay_id_for(episode)
    candidate = replay.run_replay(replay_id, candidate_model, start=start, runner_cmd=runner_cmd, execute=True)
    base = replay.run_replay(replay_id, base_model, start=start, runner_cmd=runner_cmd, execute=True)
    return {
        "episode_id": episode["id"],
        "candidate": _score(candidate),
        "base": _score(base),
    }


def _evaluate(holdout, candidate_model, base_model, runner_cmd, concurrency, start):
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        return list(pool.map(
            lambda episode: _eval_one(episode, candidate_model, base_model, runner_cmd, start),
            holdout,
        ))


def _now():
    return datetime.now(timezone.utc).isoformat()


def _number(config, key, default, low=None, high=None, integer=False):
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{key} must be finite, got {value!r}")
    if integer and int(value) != value:
        raise ValueError(f"{key} must be an integer, got {value!r}")
    if low is not None and value < low:
        raise ValueError(f"{key} must be >= {low}, got {value!r}")
    if high is not None and value > high:
        raise ValueError(f"{key} must be <= {high}, got {value!r}")
    return int(value) if integer else value


def _run_epoch(config, base_model, out, start, judge, epoch_index, judge_cache=None, judge_sig=""):
    fmt = config.get("format", "sft")
    trainer_name = config.get("trainer", "trl-sft")
    min_composite = _number(config, "min_composite", 0.5)
    holdout_frac = _number(config, "holdout_frac", 0.2, low=0.0, high=1.0)
    seed = config.get("seed", 0)
    margin = _number(config, "promote_margin", 0.0, low=0.0)
    concurrency = _number(config, "eval_concurrency", 4, low=1, integer=True)
    max_holdout = _number(config, "max_holdout", 50, low=0, integer=True)
    train_config = config.get("train_config", {})
    runner_cmd = config.get("replay_cmd")
    execute = _execute_flag(config.get("execute"))

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    episodes = ensure_reward(list(store.iter_episodes(start)), judge, cache=judge_cache, judge_sig=judge_sig)
    episodes = ensure_validity(episodes)
    config["_dropped_low_trust"] = [ep["id"] for ep in episodes if not is_trusted(ep)]
    router_summary = _train_router(episodes, out) if _router_flag(config) else None
    train, holdout = partition(episodes, min_composite, holdout_frac, seed)

    capped = len(holdout) > max_holdout
    if capped and _sim_prefilter_flag(config):
        holdout_eval = _sim_rank_holdout(holdout, config)[:max_holdout]
    else:
        holdout_eval = holdout[:max_holdout]

    if not train:
        manifest = _manifest(config, [], holdout, None, None, [], "no_train_data", capped, executed=False)
        manifest["router"] = router_summary
        _write(out, manifest)
        return manifest, train

    export_result = exporters.export(train, fmt, str(out / "dataset"))
    dataset_path = export_result["files"][0]

    harbor_summary = _mint_harbor(train, out) if _mint_flag(config) else None

    if not execute:
        candidate_model = str(out / "candidate")
        plan = {
            "trainer": trainer_name,
            "dataset": dataset_path,
            "dataset_rows": export_result.get("count"),
            "train_config": train_config,
            "candidate_model_dir": candidate_model,
            "holdout_count": len(holdout_eval),
            "runner_cmd": runner_cmd,
            "note": "set execute=true to train and run replay-eval (clones repos and runs recorded test commands)",
        }
        manifest = _manifest(
            config, train, holdout, None, base_model, [], "dry_run", capped, executed=False, plan=plan,
        )
        manifest["harbor"] = harbor_summary
        manifest["router"] = router_summary
        _write(out, manifest)
        return manifest, train

    train_manifest = trainers.train(trainer_name, dataset_path, str(out / "candidate"), train_config, cwd=start)
    candidate_model = (train_manifest.get("result") or {}).get("model_dir") or str(out / "candidate")

    scores = _evaluate(holdout_eval, candidate_model, base_model, runner_cmd, concurrency, start)
    paired = [row for row in scores if _finite(row["candidate"]) and _finite(row["base"])]
    candidate_mean = _mean([row["candidate"] for row in paired])
    base_mean = _mean([row["base"] for row in paired])

    if paired and candidate_mean >= base_mean + margin:
        decision = "promote"
    else:
        decision = "keep_base"

    manifest = _manifest(config, train, holdout, candidate_model, base_model, scores, decision, capped, executed=True)
    manifest["harbor"] = harbor_summary
    manifest["router"] = router_summary
    manifest["train_manifest"] = train_manifest
    manifest["candidate_mean"] = candidate_mean
    manifest["base_mean"] = base_mean
    manifest["evaluated"] = len(paired)
    if decision == "promote":
        (out / "promoted.json").write_text(
            json.dumps(_json_safe({
                "model_dir": candidate_model,
                "served_ref": _served_ref(candidate_model, train_manifest),
                "candidate_mean": candidate_mean,
                "base_mean": base_mean,
                "decided_at": _now(),
            }), indent=2),
            encoding="utf-8",
        )
    _write(out, manifest)
    return manifest, train


def run_loop(config, start=None):
    epochs = _number(config, "epochs", 1, low=1, integer=True)
    out_root = Path(config.get("out") or (paths.exports_dir(start) / "loop"))
    base_model = config.get("base_model", "base")

    judge = _resolve_judge(config)
    state = evaluator.build(config, judge)

    judge_cache_path = _judge_cache_path(start)
    judge_cache = _load_judge_cache(judge_cache_path)

    history = []
    for epoch_index in range(epochs):
        epoch_out = out_root if epochs == 1 else out_root / f"epoch_{epoch_index}"
        judge_sig = _judge_signature(config, state)
        manifest, train_episodes = _run_epoch(
            config, base_model, epoch_out, start, state["judge"], epoch_index, judge_cache, judge_sig)
        if state.get("judge") is not None:
            _save_judge_cache(judge_cache_path, judge_cache)

        if epochs > 1:
            manifest["epoch"] = epoch_index
            manifest["evaluator_type"] = state["type"]
        history.append(manifest)

        if manifest["decision"] == "promote":
            base_model = manifest["candidate_model"]

        if epoch_index < epochs - 1:
            state = evaluator.refresh(state, config, train_episodes, epoch_index, out_root, start)

    if epochs == 1:
        return history[0]

    final = history[-1]
    return {
        "schema_version": SCHEMA_VERSION,
        "epochs": history,
        "epoch_count": epochs,
        "final_decision": final["decision"],
        "final_model": base_model,
    }


def _served_ref(candidate_model, train_manifest):
    result = (train_manifest or {}).get("result") or {}
    for key in ("sampler_path", "served_ref", "model_path"):
        value = result.get(key)
        if value:
            return value
    return candidate_model


def _manifest(config, train, holdout, candidate_model, base_model, scores, decision, capped, executed, plan=None):
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _now(),
        "trainer": config.get("trainer", "trl-sft"),
        "format": config.get("format", "sft"),
        "base_model": base_model,
        "candidate_model": candidate_model,
        "executed": executed,
        "decision": decision,
        "preflight_dropped": config.get("_dropped_low_trust", []),
        "train_ids": [episode["id"] for episode in train],
        "holdout_ids": [episode["id"] for episode in holdout],
        "holdout_capped": capped,
        "scores": scores,
    }
    if plan is not None:
        manifest["plan"] = plan
    return manifest


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _write(out, manifest):
    (out / "loop.json").write_text(json.dumps(_json_safe(manifest), indent=2) + "\n", encoding="utf-8")
