import json

from episodic.schema import new_episode
from episodic.exporters import export, trajectory_text


def _episode_two_workdirs():
    ep = new_episode(id="ep_cwd", intent="run the suite in each package")
    ep["steps"] = [
        {
            "index": 0,
            "ts": "2026-06-14T00:00:00+00:00",
            "type": "shell_command",
            "tool": "Bash",
            "intent": "pytest -q",
            "input": {"command": "pytest -q"},
            "observation": "1 passed",
            "approved": True,
            "cwd": "/repo/frontend",
            "duration_ms": None,
        },
        {
            "index": 1,
            "ts": "2026-06-14T00:00:01+00:00",
            "type": "shell_command",
            "tool": "Bash",
            "intent": "pytest -q",
            "input": {"command": "pytest -q"},
            "observation": "1 passed",
            "approved": True,
            "cwd": "/repo/backend",
            "duration_ms": None,
        },
    ]
    return ep


def test_rlds_export_preserves_distinct_cwds(tmp_path):
    export([_episode_two_workdirs()], "rlds", str(tmp_path))
    rows = [json.loads(line) for line in (tmp_path / "rlds.jsonl").read_text().splitlines()]
    actions = rows[0]["steps"]
    assert [a["action"]["cwd"] for a in actions] == ["/repo/frontend", "/repo/backend"]


def test_trajectory_text_preserves_distinct_cwds():
    text = trajectory_text(_episode_two_workdirs())
    assert "@ /repo/frontend" in text
    assert "@ /repo/backend" in text
