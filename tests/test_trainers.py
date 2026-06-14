import json

import pytest

from episodic import trainers


def _write_sft(path):
    rows = [
        {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}],
         "meta": {"episode_id": "ep_a", "reward": 0.9}},
        {"messages": [{"role": "user", "content": "go"}, {"role": "assistant", "content": "ok"}],
         "meta": {"episode_id": "ep_b", "reward": 0.7}},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return rows


def test_builtin_trainers_registered():
    names = trainers.available()
    assert {"command", "trl-sft", "trl-dpo"}.issubset(set(names))
    assert trainers.get("trl-sft").consumes == ("sft",)
    assert trainers.get("trl-dpo").consumes == ("dpo",)


def test_unknown_trainer_raises():
    with pytest.raises(KeyError):
        trainers.get("does-not-exist")


def test_command_trainer_runs_and_manifest(tmp_path):
    dataset = tmp_path / "sft.jsonl"
    _write_sft(dataset)

    script = tmp_path / "fake_train.py"
    script.write_text(
        "import json, os, sys\n"
        "dataset, out = sys.argv[1], sys.argv[2]\n"
        "rows = sum(1 for line in open(dataset) if line.strip())\n"
        "open(os.path.join(out, 'metrics.json'), 'w').write(json.dumps({'rows': rows}))\n",
        encoding="utf-8",
    )
    out = tmp_path / "run"
    config = {"command": f"python3 {script} {{dataset}} {{out}}"}

    manifest = trainers.train("command", str(dataset), str(out), config, cwd=str(tmp_path))

    assert (out / "manifest.json").exists()
    assert manifest["trainer"] == "command"
    assert manifest["dataset_rows"] == 2
    assert manifest["episode_ids"] == ["ep_a", "ep_b"]
    assert len(manifest["dataset_sha256"]) == 64
    assert manifest["result"]["returncode"] == 0
    assert manifest["result"]["metrics"] == {"rows": 2}


def test_command_trainer_requires_command(tmp_path):
    dataset = tmp_path / "sft.jsonl"
    _write_sft(dataset)
    with pytest.raises(ValueError, match="config.command"):
        trainers.train("command", str(dataset), str(tmp_path / "out"), {}, cwd=str(tmp_path))


def test_trl_unavailable_is_graceful(tmp_path):
    pytest.importorskip  # marker; we assert the unavailable path only when trl is absent
    try:
        import trl  # noqa: F401
        pytest.skip("trl installed; unavailable path not exercised")
    except ImportError:
        pass
    dataset = tmp_path / "sft.jsonl"
    _write_sft(dataset)
    with pytest.raises(trainers.TrainerUnavailable) as info:
        trainers.train("trl-sft", str(dataset), str(tmp_path / "out"), {}, cwd=str(tmp_path))
    assert "episodic[trl]" in info.value.hint
