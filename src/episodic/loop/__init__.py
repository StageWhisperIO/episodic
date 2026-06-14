import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from .. import store, exporters, trainers, replay, paths
from ..exporters import is_bad

SCHEMA_VERSION = "0.1.0"


def _composite(episode):
    return (episode.get("reward_vector") or {}).get("composite") or 0.0


def _hash_frac(episode_id, seed):
    digest = hashlib.sha256(f"{seed}:{episode_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0x100000000


def select_good(episodes, min_composite):
    for episode in episodes:
        if _composite(episode) >= min_composite and not is_bad(episode):
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


def _mean(values):
    return sum(values) / len(values) if values else None


def _eval_one(episode, candidate_model, base_model, runner_cmd, start):
    replay.create_replay(episode, start=start)
    replay_id = replay.replay_id_for(episode)
    candidate = replay.run_replay(replay_id, candidate_model, start=start, runner_cmd=runner_cmd, execute=True)
    base = replay.run_replay(replay_id, base_model, start=start, runner_cmd=runner_cmd, execute=True)
    return {
        "episode_id": episode["id"],
        "candidate": (candidate.get("scores") or {}).get("total"),
        "base": (base.get("scores") or {}).get("total"),
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
    execute = bool(config.get("execute"))

    out = Path(config.get("out") or (paths.exports_dir(start) / "loop"))
    out.mkdir(parents=True, exist_ok=True)

    train, holdout = partition(store.iter_episodes(start), min_composite, holdout_frac, seed)

    capped = len(holdout) > max_holdout
    holdout_eval = holdout[:max_holdout]

    if not train:
        manifest = _manifest(config, [], holdout, None, None, [], "no_train_data", capped, executed=False)
        _write(out, manifest)
        return manifest

    export_result = exporters.export(train, fmt, str(out / "dataset"))
    dataset_path = export_result["files"][0]

    train_manifest = trainers.train(trainer_name, dataset_path, str(out / "candidate"), train_config, cwd=start)
    candidate_model = (train_manifest.get("result") or {}).get("model_dir") or str(out / "candidate")

    if not execute:
        manifest = _manifest(
            config, train, holdout, candidate_model, base_model, [], "dry_run", capped, executed=False,
            plan={"holdout_count": len(holdout_eval), "runner_cmd": runner_cmd, "note":
                  "set execute=true to run replay-eval (clones repos and runs recorded test commands)"},
        )
        manifest["train_manifest"] = train_manifest
        _write(out, manifest)
        return manifest

    scores = _evaluate(holdout_eval, candidate_model, base_model, runner_cmd, concurrency, start)
    paired = [row for row in scores if row["candidate"] is not None and row["base"] is not None]
    candidate_mean = _mean([row["candidate"] for row in paired])
    base_mean = _mean([row["base"] for row in paired])

    if paired and candidate_mean >= base_mean + margin:
        decision = "promote"
    else:
        decision = "keep_base"

    manifest = _manifest(config, train, holdout, candidate_model, base_model, scores, decision, capped, executed=True)
    manifest["train_manifest"] = train_manifest
    manifest["candidate_mean"] = candidate_mean
    manifest["base_mean"] = base_mean
    manifest["evaluated"] = len(paired)
    if decision == "promote":
        (out / "promoted.json").write_text(
            json.dumps({"model_dir": candidate_model, "candidate_mean": candidate_mean,
                        "base_mean": base_mean, "decided_at": _now()}, indent=2),
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
        "train_ids": [episode["id"] for episode in train],
        "holdout_ids": [episode["id"] for episode in holdout],
        "holdout_capped": capped,
        "scores": scores,
    }
    if plan is not None:
        manifest["plan"] = plan
    return manifest


def _write(out, manifest):
    (out / "loop.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
