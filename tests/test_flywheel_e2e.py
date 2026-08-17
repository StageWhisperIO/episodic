import json
import subprocess
import threading
import urllib.request
from pathlib import Path

import pytest

from episodic import loop, paths, service, store
from episodic.collector import otel
from episodic.schema import new_event
from episodic.serving.server import build_server

STAGEWHISPER_HOME = Path("/Users/piotrmrzyglowski/Documents/projects/stagewhisper/.episodic")
STAGEWHISPER_EPISODES = STAGEWHISPER_HOME / "episodes"
MAX_REAL_EPISODES = 10
MAX_STEPS = 200

pytestmark = pytest.mark.skipif(
    not STAGEWHISPER_EPISODES.is_dir(),
    reason="real stagewhisper episode store not available at ../stagewhisper/.episodic",
)


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _sandbox_repo(tmp_path):
    repo = tmp_path / "sandbox_repo"
    repo.mkdir()
    (repo / "f.py").write_text("x = 1\n")
    (repo / "test_f.py").write_text("def test_ok():\n    assert True\n")
    _git(str(repo), "init", "-q")
    _git(str(repo), "config", "user.email", "e2e@episodic.local")
    _git(str(repo), "config", "user.name", "episodic-e2e")
    _git(str(repo), "add", "-A")
    _git(str(repo), "commit", "-q", "-m", "base")
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(repo), capture_output=True, text=True
    ).stdout.strip()
    return str(repo), sha


def _load_real_episodes(repo_root, sha):
    episodes = []
    for path in sorted(STAGEWHISPER_EPISODES.glob("ep_*.json")):
        episode = json.loads(path.read_text(encoding="utf-8"))
        if len(episode.get("steps", [])) >= MAX_STEPS:
            continue
        episode["repo_state"] = {
            "root": repo_root, "repo": "sandbox", "remote_url": repo_root,
            "branch": "main", "base_commit": sha, "dirty": False,
        }
        episode["diffs"] = [
            {"file": "f.py", "status": "modified", "additions": 1, "deletions": 0, "unified": None}
        ]
        episodes.append(episode)
        if len(episodes) >= MAX_REAL_EPISODES:
            break
    return episodes


def _write_runner(tmp_path):
    runner = tmp_path / "runner.py"
    runner.write_text(
        "import os, sys\n"
        "model, workspace = sys.argv[1], sys.argv[2]\n"
        "if 'candidate' in model:\n"
        "    with open(os.path.join(workspace, 'f.py'), 'a') as fh:\n"
        "        fh.write('# edit\\n')\n"
    )
    return runner


def _fake_opener(json_body):
    class _Response:
        def read(self):
            return json.dumps(json_body).encode("utf-8")

    def opener(request, timeout=None):
        return _Response()

    return opener


def _run_server(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return f"http://{host}:{port}", thread


def _shutdown(server, thread):
    server.shutdown()
    thread.join(timeout=5)
    server.server_close()


def _post(base_url, path, body):
    request = urllib.request.Request(
        base_url + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(request)


def test_flywheel_loop_promote_serve_ingest_new_episode(tmp_path, monkeypatch):
    home = tmp_path / ".episodic"
    monkeypatch.setenv("EPISODIC_HOME", str(home))

    repo_root, sha = _sandbox_repo(tmp_path)
    real_episodes = _load_real_episodes(repo_root, sha)
    assert len(real_episodes) >= 4, "expected several small real episodes in the stagewhisper store"
    for episode in real_episodes:
        store.save_episode(episode)

    runner = _write_runner(tmp_path)
    loop_out = tmp_path / "loopout"
    loop_config = {
        "trainer": "command", "format": "sft", "holdout_frac": 0.5, "seed": 0,
        "min_composite": 0.0, "train_config": {"command": "true"},
        "base_model": "base", "execute": True,
        "judge": True, "judge_cmd": "sh -c \"printf 'SCORE: 0.9 solid fix with a clear explanation'\"",
        "judge_timeout": 10,
        "replay_cmd": f"python3 {runner} {{model}} {{workspace}} {{prompt_file}}",
        "out": str(loop_out),
    }
    manifest = loop.run_loop(loop_config)
    assert manifest["train_ids"], "real episode sample produced no train split"
    assert manifest["holdout_ids"], "real episode sample produced no holdout split"
    assert manifest["decision"] == "promote"
    assert (paths.home() / "cache" / "judge_reward.json").exists()

    promoted = json.loads((loop_out / "promoted.json").read_text(encoding="utf-8"))
    assert promoted["served_ref"]

    upstream_body = {
        "id": "chatcmpl-e2e", "object": "chat.completion", "model": promoted["served_ref"],
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "diff applied, tests green"},
            "finish_reason": "stop",
        }],
    }
    serve_config = {
        "loop_out": str(loop_out),
        "distilled": {
            "backend": "openai", "base_url": "http://distilled.local",
            "opener": _fake_opener(upstream_body),
        },
    }
    server = build_server("127.0.0.1", 0, serve_config)
    base_url, thread = _run_server(server)
    try:
        response = _post(base_url, "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "fix f.py"}],
        })
        served = json.loads(response.read())
    finally:
        _shutdown(server, thread)

    assert served["episodic_tier"] == "distilled"
    assert served["choices"][0]["message"]["content"] == "diff applied, tests green"

    session_id = "e2e-session-" + real_episodes[0]["id"]
    service.record_session_start(session_id, agent="claude-code", cwd=repo_root)
    store.append_event(new_event(session_id, "user_prompt", data={"prompt": "fix f.py"}))
    store.append_event(new_event(session_id, "shell_command", tool_name="Bash", data={
        "cwd": repo_root, "command": "python3 -m pytest -q",
        "response": served["choices"][0]["message"]["content"],
        "exit_code": 0, "approved": True,
    }))

    otel_payload = {
        "resourceMetrics": [{
            "resource": {"attributes": []},
            "scopeMetrics": [{
                "metrics": [
                    {"name": "claude_code.token.usage", "sum": {"dataPoints": [
                        {"asInt": "128", "attributes": [
                            {"key": "type", "value": {"stringValue": "input"}},
                            {"key": "session.id", "value": {"stringValue": session_id}},
                        ]},
                        {"asInt": "64", "attributes": [
                            {"key": "type", "value": {"stringValue": "output"}},
                            {"key": "session.id", "value": {"stringValue": session_id}},
                        ]},
                    ]}},
                    {"name": "claude_code.cost.usage", "sum": {"dataPoints": [
                        {"asDouble": 0.012, "attributes": [
                            {"key": "session.id", "value": {"stringValue": session_id}},
                        ]},
                    ]}},
                ],
            }],
        }],
    }

    otel_server = otel.build_otel_server("127.0.0.1", 0)
    otel_base_url, otel_thread = _run_server(otel_server)
    try:
        otel_response = _post(otel_base_url, "/v1/metrics", otel_payload)
        assert otel_response.status == 200
    finally:
        _shutdown(otel_server, otel_thread)

    meta = store.read_meta(session_id)
    assert meta["usage"]["input_tokens"] == 128
    assert meta["usage"]["output_tokens"] == 64
    assert meta["usage"]["cost_usd"] == pytest.approx(0.012)

    existing_ids = {p.stem for p in (home / "episodes").glob("ep_*.json")}

    new_episode = service.finalize_session(session_id)
    assert new_episode is not None
    assert new_episode["id"] not in existing_ids
    assert new_episode["stats"]["input_tokens"] == 128
    assert new_episode["stats"]["output_tokens"] == 64
    assert new_episode["stats"]["cost_usd"] == pytest.approx(0.012)

    episode_path = paths.episode_path(new_episode["id"])
    assert episode_path.exists()
    saved = store.get_episode(new_episode["id"])
    assert saved["id"] == new_episode["id"]
