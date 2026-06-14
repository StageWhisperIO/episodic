import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = "0.1.0"


class TrainerUnavailable(RuntimeError):
    def __init__(self, trainer, hint):
        self.trainer = trainer
        self.hint = hint
        super().__init__(hint)


_REGISTRY = {}
_DISCOVERED = False


def register(trainer):
    _REGISTRY[trainer.name] = trainer
    return trainer


def _discover():
    global _DISCOVERED
    if _DISCOVERED:
        return
    from . import command, trl, unsloth  # noqa: F401  built-ins self-register on import
    try:
        from importlib.metadata import entry_points

        selected = entry_points(group="episodic.trainers")
    except Exception:
        selected = []
    for entry in selected:
        try:
            obj = entry.load()
            register(obj() if isinstance(obj, type) else obj)
        except Exception:
            continue
    _DISCOVERED = True


def get(name):
    _discover()
    if name not in _REGISTRY:
        raise KeyError(f"unknown trainer '{name}'; available: {', '.join(available())}")
    return _REGISTRY[name]


def available():
    _discover()
    return sorted(_REGISTRY)


def _episode_ids(rows):
    ids = []
    seen = set()
    for row in rows:
        meta = row.get("meta") or {}
        for key in ("episode_id", "chosen_id", "rejected_id"):
            value = meta.get(key)
            if value and value not in seen:
                seen.add(value)
                ids.append(value)
    return ids


def _base_commit(cwd):
    from ..core import gitinfo

    root = cwd or "."
    try:
        return gitinfo.head_commit(root) if gitinfo.git_available(root) else None
    except Exception:
        return None


def build_manifest(trainer, dataset_path, config, result, cwd):
    data = Path(dataset_path).read_bytes()
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    return {
        "schema_version": SCHEMA_VERSION,
        "trainer": trainer.name,
        "trainer_version": getattr(trainer, "version", "0"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "dataset_sha256": hashlib.sha256(data).hexdigest(),
        "dataset_rows": len(rows),
        "episode_ids": _episode_ids(rows),
        "base_commit": _base_commit(cwd),
        "config": config,
        "result": result,
    }


def train(trainer_name, dataset_path, out_dir, config=None, cwd=None):
    trainer = get(trainer_name)
    config = dict(config or {})
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = trainer.train(str(dataset_path), str(out), config)
    manifest = build_manifest(trainer, dataset_path, config, result, cwd)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
