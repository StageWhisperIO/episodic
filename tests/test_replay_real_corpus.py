import json
from pathlib import Path

import pytest

from episodic import loop
from episodic.core import gitinfo
from episodic.exporters import _captured_verifier
from episodic.replay import _resolve_test_command

STAGEWHISPER_HOME = Path("/Users/piotrmrzyglowski/Documents/projects/stagewhisper/.episodic")
STAGEWHISPER_EPISODES = STAGEWHISPER_HOME / "episodes"

FAST_OFFLINE_EPISODE_ID = "ep_edb8ff40807b"

pytestmark = pytest.mark.skipif(
    not STAGEWHISPER_EPISODES.is_dir(),
    reason="real stagewhisper episode store not available at ../stagewhisper/.episodic",
)


def _load_real_episodes():
    episodes = []
    for path in sorted(STAGEWHISPER_EPISODES.glob("ep_*.json")):
        episodes.append(json.loads(path.read_text(encoding="utf-8")))
    return episodes


def _runnable(ep):
    repo_state = ep.get("repo_state") or {}
    root = repo_state.get("root")
    verifier = _captured_verifier(ep)
    if verifier is None or verifier.get("total") is None:
        return False
    return bool(root) and gitinfo.git_available(root)


def test_real_corpus_runnable_funnel_is_non_trivial():
    episodes = _load_real_episodes()
    assert len(episodes) >= 90

    solid = [ep for ep in episodes if _runnable(ep)]

    assert 10 <= len(solid) <= len(episodes)


def test_real_corpus_test_commands_are_relativized_with_no_embedded_host_paths():
    episodes = _load_real_episodes()
    solid = [ep for ep in episodes if _runnable(ep)]
    assert solid

    for ep in solid:
        root = ep["repo_state"]["root"]
        command, test_cwd = _resolve_test_command(ep, root)
        assert command is not None
        assert root not in command
        if test_cwd is not None:
            assert test_cwd not in command


def test_real_corpus_known_subdir_episode_gets_cwd_threaded_when_present():
    episodes = {ep["id"]: ep for ep in _load_real_episodes()}
    expected_threaded = {
        "ep_0bcb5d29879d": "stagewhisper-mobile/src-tauri",
        "ep_8f9a957edf43": "web/backend",
        "ep_edb8ff40807b": "integrations/hermes-stagewhisper-plugin",
    }
    seen = 0
    for ep_id, expected_cwd in expected_threaded.items():
        ep = episodes.get(ep_id)
        if ep is None or not _runnable(ep):
            continue
        seen += 1
        _, test_cwd = _resolve_test_command(ep, ep["repo_state"]["root"])
        assert test_cwd == expected_cwd
    if seen == 0:
        pytest.skip("none of the known subdir-threaded episodes are present in the live store anymore")


def test_real_corpus_replay_eval_executes_for_real_against_the_live_repo(tmp_path, monkeypatch):
    episodes = {ep["id"]: ep for ep in _load_real_episodes()}
    ep = episodes.get(FAST_OFFLINE_EPISODE_ID)
    if ep is None or not _runnable(ep):
        pytest.skip(f"{FAST_OFFLINE_EPISODE_ID} is no longer present/runnable in the live store")

    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path / ".episodic"))
    ep = dict(ep)
    ep["repo_state"] = dict(ep["repo_state"])
    ep["repo_state"]["remote_url"] = ep["repo_state"]["root"]

    runner = loop._resolve_eval_runner({"eval_backend": "stub"}, None)
    row = loop._eval_one(ep, "candidate", "base", None, None, runner=runner)

    assert row["episode_id"] == FAST_OFFLINE_EPISODE_ID
    assert row["candidate"] is not None
    assert row["base"] is not None
