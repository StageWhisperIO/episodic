import importlib
import json
import math
import os
import tempfile
from pathlib import Path

from episodic import paths


def _check_imports():
    modules = [
        "episodic.schema", "episodic.store", "episodic.service", "episodic.exporters",
        "episodic.replay", "episodic.loop", "episodic.trainers", "episodic.worldmodel",
        "episodic.worldbench", "episodic.fidelity", "episodic.testing",
        "episodic.core.reward", "episodic.core.summary", "episodic.core.testdetect",
    ]
    for name in modules:
        importlib.import_module(name)
    return True, f"{len(modules)} modules import"


def _check_schema_valid():
    from episodic.schema import validate_episode
    from episodic.testing import make_episode

    errors = validate_episode(make_episode("ep_doctor"))
    return not errors, "synthetic episode is schema-valid" if not errors else str(errors[:2])


def _check_schema_file_sync(required=False):
    from episodic.schema import EPISODE_SCHEMA

    target = paths.resolve_base() / "schemas" / "episode.schema.json"
    if not target.exists():
        return None, "schemas/episode.schema.json not present (run `episodic schema dump`)"
    on_disk = json.loads(target.read_text())
    in_sync = on_disk == EPISODE_SCHEMA
    return in_sync, "schema file in sync" if in_sync else "schema file drifted from schema.py"


def _check_store_roundtrip():
    from episodic import store
    from episodic.testing import populate_store

    saved = populate_store(6, seed=0)
    loaded = store.load_episodes()
    ok = len(loaded) == 6 and {e["id"] for e in loaded} == {e["id"] for e in saved}
    return ok, f"saved/loaded {len(loaded)} episodes"


def _check_exporters():
    from episodic import exporters
    from episodic.testing import make_population

    pop = make_population(8, seed=1)
    with tempfile.TemporaryDirectory() as tmp:
        results = {}
        for fmt in exporters.FORMATS:
            result = exporters.export(pop, fmt, os.path.join(tmp, fmt))
            assert Path(result["files"][0]).exists(), f"{fmt} produced no file"
            results[fmt] = result.get("count")
    return True, f"all {len(results)} formats exported: {results}"


def _check_reward_finite():
    from episodic.core import reward
    from episodic.testing import make_population

    for episode in make_population(10, seed=2):
        composite = reward.reward_vector(episode)["composite"]
        assert math.isfinite(composite) and 0.0 <= composite <= 1.0, composite
    return True, "reward composites finite and in [0,1]"


def _check_worldmodel():
    from episodic import worldmodel
    from episodic.testing import make_population

    pop = make_population(20, seed=3, sources=[f"r{i}" for i in range(6)])
    one = worldmodel.wm_samples(pop, one_per_trajectory=True, seed=0)
    assert len(one) == len(pop), "expected one sample per trajectory"
    train, holdout, _ = worldmodel.ood_split(pop, holdout_frac=0.4, seed=1)
    train_sources = {worldmodel.source_key(e) for e in train}
    holdout_sources = {worldmodel.source_key(e) for e in holdout}
    assert train_sources.isdisjoint(holdout_sources), "OOD sources overlap"
    return True, f"turn-expansion + OOD split ok (train={len(train)}, holdout={len(holdout)})"


def _check_fidelity():
    from episodic import fidelity

    assert fidelity.score_observation("x", "x")["composite"] == 1.0
    masked = fidelity.score_observation(
        "done at 2026-06-14T10:00:00Z", "done at 2026-06-14T11:22:33Z")
    assert masked["factuality"] == 1.0, "runtime metadata penalized"
    return True, "exact-match and runtime-mask scoring ok"


def _check_worldbench():
    from episodic import worldbench
    from episodic.testing import make_population

    pop = make_population(12, seed=4)
    oracle = worldbench.run_bench(pop, "oracle")["overall"]["composite"]
    empty = worldbench.run_bench(pop, "empty")["overall"]["composite"]
    assert oracle == 1.0 and empty < oracle, (oracle, empty)
    turing = worldbench.turing_test(pop, "oracle")
    assert turing["indistinguishability"] == 1.0, turing
    return True, f"worldbench oracle={oracle} empty={empty} turing_indist={turing['indistinguishability']}"


def _check_replay_plan():
    from episodic import replay
    from episodic.testing import make_episode

    episode = make_episode("ep_doctor_replay", remote_url=None)
    replay.create_replay(episode)
    result = replay.run_replay(replay.replay_id_for(episode), "doctor-model")
    assert result.get("executed") is False and result.get("scores") is None, result
    return True, "replay returns a plan without executing"


def _check_loop_dry_run():
    from episodic import loop
    from episodic.testing import populate_store

    with tempfile.TemporaryDirectory() as tmp:
        populate_store(8, seed=5)
        manifest = loop.run_loop({
            "trainer": "command", "format": "sft", "min_composite": 0.0,
            "holdout_frac": 0.5, "seed": 0, "train_config": {"command": "true"},
            "out": os.path.join(tmp, "loop"),
        })
    assert manifest["executed"] is False and manifest["decision"] in ("dry_run", "no_train_data"), manifest["decision"]
    return True, f"loop dry-run decision={manifest['decision']}"


def _check_optional_deps():
    available = {}
    for name in ("numpy", "pyarrow", "datasets", "torch", "trl", "transformers", "unsloth"):
        try:
            importlib.import_module(name)
            available[name] = True
        except ImportError:
            available[name] = False
    return True, available


_CHECKS = [
    ("imports", True, _check_imports),
    ("schema_valid", True, _check_schema_valid),
    ("schema_file_sync", False, _check_schema_file_sync),
    ("store_roundtrip", True, _check_store_roundtrip),
    ("exporters", True, _check_exporters),
    ("reward_finite", True, _check_reward_finite),
    ("worldmodel", True, _check_worldmodel),
    ("fidelity", True, _check_fidelity),
    ("worldbench", True, _check_worldbench),
    ("replay_plan", True, _check_replay_plan),
    ("loop_dry_run", True, _check_loop_dry_run),
    ("optional_deps", False, _check_optional_deps),
]


def run_checks():
    results = []
    with tempfile.TemporaryDirectory() as home:
        previous = os.environ.get(paths.ENV_HOME)
        os.environ[paths.ENV_HOME] = os.path.join(home, ".episodic")
        try:
            for name, required, fn in _CHECKS:
                try:
                    ok, detail = fn()
                except Exception as exc:
                    ok, detail = False, f"{type(exc).__name__}: {exc}"
                results.append({"name": name, "required": required, "ok": ok, "detail": detail})
        finally:
            if previous is None:
                os.environ.pop(paths.ENV_HOME, None)
            else:
                os.environ[paths.ENV_HOME] = previous

    failed = [r for r in results if r["required"] and r["ok"] is False]
    return {"ok": not failed, "passed": sum(1 for r in results if r["ok"] is True),
            "total": len(results), "failed": [r["name"] for r in failed], "checks": results}
