import json
import pytest
from episodic import exporters


def test_formats_constant():
    assert set(exporters.FORMATS) == {"sft", "dpo", "reward", "rlds", "wm", "jsonl", "parquet"}


def test_unknown_format_raises(episodes, tmp_path):
    with pytest.raises(ValueError, match="Unknown format"):
        exporters.export(episodes, "csv", tmp_path)


def test_jsonl_export(episodes, tmp_path):
    result = exporters.export(episodes, "jsonl", tmp_path)
    path = tmp_path / "episodes.jsonl"
    assert path.exists()
    lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    assert len(lines) == 2
    assert result["count"] == 2
    assert result["format"] == "jsonl"


def test_sft_export_only_good(episodes, tmp_path):
    result = exporters.export(episodes, "sft", tmp_path)
    path = tmp_path / "sft.jsonl"
    assert path.exists()
    lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    assert result["count"] == 1
    row = lines[0]
    assert row["messages"][0]["role"] == "user"
    assert row["messages"][1]["role"] == "assistant"
    assert "episode_id" in row["meta"]
    assert "reward" in row["meta"]


def test_reward_export_both_episodes(episodes, tmp_path):
    result = exporters.export(episodes, "reward", tmp_path)
    path = tmp_path / "reward.jsonl"
    assert path.exists()
    lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    assert len(lines) == 2
    assert result["count"] == 2
    for row in lines:
        assert "prompt" in row
        assert "trajectory" in row
        assert "reward_vector" in row
        assert "scalar_reward" in row


def test_rlds_export(episodes, tmp_path):
    result = exporters.export(episodes, "rlds", tmp_path)
    path = tmp_path / "rlds.jsonl"
    assert path.exists()
    lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    assert len(lines) == 2
    assert result["count"] == 2
    good_ep_row = next(r for r in lines if r["episode_id"] == "ep_test_good")
    steps = good_ep_row["steps"]
    assert len(steps) > 0
    last_step = steps[-1]
    assert last_step["is_terminal"] is True
    assert last_step["is_last"] is True
    assert last_step["discount"] == 0.0
    good_ep = next(ep for ep in episodes if ep["id"] == "ep_test_good")
    assert last_step["next_observation"] == good_ep["steps"][-1].get("observation", "")
    expected_reward = (good_ep.get("reward_vector") or {}).get("composite") or 0.0
    assert last_step["reward"] == pytest.approx(expected_reward)
    first_step = steps[0]
    assert first_step["is_first"] is True
    assert first_step["reward"] == 0.0
    for step in steps[:-1]:
        assert step["discount"] == 1.0


def test_dpo_export(episodes, tmp_path):
    result = exporters.export(episodes, "dpo", tmp_path)
    path = tmp_path / "dpo.jsonl"
    assert path.exists()
    lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    assert result["pairs"] >= 1
    for row in lines:
        assert "prompt" in row
        assert "chosen" in row
        assert "rejected" in row
        assert "meta" in row


def test_parquet_export_or_fallback(episodes, tmp_path):
    result = exporters.export(episodes, "parquet", tmp_path)
    assert result["count"] == 2
    assert len(result["files"]) == 1
    out_file = tmp_path / result["files"][0].split("/")[-1]
    assert out_file.exists()
    try:
        import pyarrow
        assert out_file.suffix == ".parquet"
    except ImportError:
        assert out_file.suffix == ".jsonl"
        assert "fallback" in result
        lines = [json.loads(l) for l in out_file.read_text().splitlines() if l.strip()]
        assert len(lines) == 2


def test_trajectory_text_structure(episodes):
    good_ep = episodes[0]
    text = exporters.trajectory_text(good_ep)
    assert text.startswith("USER: ")
    assert "ACTION" in text
    assert "OBS:" in text


def test_is_good_and_is_bad(episodes):
    good_ep, bad_ep = episodes
    assert exporters.is_good(good_ep) is True
    assert exporters.is_bad(good_ep) is False
    assert exporters.is_bad(bad_ep) is True


def test_export_creates_out_dir(episodes, tmp_path):
    new_dir = tmp_path / "nested" / "exports"
    result = exporters.export(episodes, "jsonl", new_dir)
    assert new_dir.exists()
    assert result["out_dir"] == str(new_dir)
