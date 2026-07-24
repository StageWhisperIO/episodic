import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from .. import store, exporters, trainers, replay, paths
from ..exporters import is_bad, is_trusted
from ..core import validity

SCHEMA_VERSION = "0.1.0"


def _composite(episode):
    value = (episode.get("reward_vector") or {}).get("composite")
    return value if _finite(value) else 0.0


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


def run_loop(config, start=None):
    fmt = config.get("format", "sft")
    trainer_name = config.get("trainer", "trl-sft")
    min_composite = _number(config, "min_composite", 0.5)
    holdout_frac = _number(config, "holdout_frac", 0.2, low=0.0, high=1.0)
    seed = config.get("seed", 0)
    margin = _number(config, "promote_margin", 0.0, low=0.0)
    concurrency = _number(config, "eval_concurrency", 4, low=1, integer=True)
    max_holdout = _number(config, "max_holdout", 50, low=0, integer=True)
    base_model = config.get("base_model", "base")
    train_config = config.get("train_config", {})
    runner_cmd = config.get("replay_cmd")
    execute = _execute_flag(config.get("execute"))

    out = Path(config.get("out") or (paths.exports_dir(start) / "loop"))
    out.mkdir(parents=True, exist_ok=True)

    episodes = ensure_validity(list(store.iter_episodes(start)))
    config["_dropped_low_trust"] = [ep["id"] for ep in episodes if not is_trusted(ep)]
    train, holdout = partition(episodes, min_composite, holdout_frac, seed)

    capped = len(holdout) > max_holdout
    holdout_eval = holdout[:max_holdout]

    if not train:
        manifest = _manifest(config, [], holdout, None, None, [], "no_train_data", capped, executed=False)
        _write(out, manifest)
        return manifest

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
        _write(out, manifest)
        return manifest

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
    manifest["train_manifest"] = train_manifest
    manifest["candidate_mean"] = candidate_mean
    manifest["base_mean"] = base_mean
    manifest["evaluated"] = len(paired)
    if decision == "promote":
        (out / "promoted.json").write_text(
            json.dumps(_json_safe({"model_dir": candidate_model, "candidate_mean": candidate_mean,
                                   "base_mean": base_mean, "decided_at": _now()}), indent=2),
            encoding="utf-8",
        )
    _write(out, manifest)
    return manifest


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
