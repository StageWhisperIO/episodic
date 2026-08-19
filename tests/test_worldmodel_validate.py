import math
import subprocess

import pytest

from episodic.testing import make_episode
from episodic.worldbench import NAMED_PREDICTORS
from episodic.worldmodel import validate as wm_validate


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=str(repo), check=True, capture_output=True)


def _git_repo(root, files):
    root.mkdir(parents=True, exist_ok=True)
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@t.dev")
    _git(root, "config", "user.name", "t")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(root), capture_output=True, text=True
    ).stdout.strip()


def test_sim_trajectory_score_oracle_is_perfect_and_empty_is_lower():
    episode = make_episode("ep_wm_validate_oracle")
    oracle_score = wm_validate.sim_trajectory_score(episode, NAMED_PREDICTORS["oracle"])
    empty_score = wm_validate.sim_trajectory_score(episode, NAMED_PREDICTORS["empty"])
    assert oracle_score == 1.0
    assert empty_score < oracle_score


def test_sim_scores_maps_every_episode_id():
    episodes = [make_episode(f"ep_wm_validate_sim_{i}") for i in range(3)]
    scores = wm_validate.sim_scores(episodes, NAMED_PREDICTORS["oracle"])
    assert set(scores) == {ep["id"] for ep in episodes}
    assert all(value == 1.0 for value in scores.values())


def test_sim_trajectory_score_respects_history_budget_without_raising():
    episode = make_episode("ep_wm_validate_budget", files=tuple(f"f{i}.py" for i in range(20)))
    score = wm_validate.sim_trajectory_score(episode, NAMED_PREDICTORS["oracle"], history_budget=50)
    assert score == 1.0


def test_pearson_perfect_positive_and_negative_correlation():
    assert wm_validate.pearson([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert wm_validate.pearson([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_pearson_returns_none_for_constant_series():
    assert wm_validate.pearson([1, 1, 1], [1, 2, 3]) is None
    assert wm_validate.pearson([1], [1]) is None


def test_spearman_is_robust_to_monotonic_nonlinear_transform():
    xs = [1, 2, 3, 4, 5]
    ys = [x ** 3 for x in xs]
    assert wm_validate.spearman(xs, ys) == pytest.approx(1.0)
    assert wm_validate.pearson(xs, ys) < 1.0


def test_correlate_pairs_only_shared_finite_ids():
    sim = {"a": 0.9, "b": 0.1, "c": None, "d": 0.5}
    real = {"a": 1.0, "b": 0.0, "c": 0.5, "e": 0.2}
    report = wm_validate.correlate(sim, real)
    assert report["n"] == 2
    assert report["episode_ids"] == ["a", "b"]
    assert report["pearson"] == pytest.approx(1.0)
    assert report["spearman"] == pytest.approx(1.0)


def test_correlate_with_no_overlap_returns_zero_n_and_none_correlations():
    report = wm_validate.correlate({"a": 0.5}, {"b": 0.5})
    assert report["n"] == 0
    assert report["pearson"] is None
    assert report["spearman"] is None


def test_offline_replay_scores_runs_locally_without_network(tmp_path, monkeypatch, sample_episode):
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path / ".episodic"))
    origin = tmp_path / "origin"
    command = "python3 -m pytest -q test_ok.py"
    sha = _git_repo(origin, {"test_ok.py": "def test_ok():\n    assert True\n"})

    sample_episode["repo_state"].update({"root": str(origin), "remote_url": "https://example.com/nope.git",
                                         "base_commit": sha})
    sample_episode["commands"] = [{"ts": "t", "command": command, "cwd": str(origin), "exit_code": 0,
                                   "output_excerpt": "1 passed", "is_test": True}]
    sample_episode["tests"] = [{"ts": "t", "framework": "pytest", "command": command,
                                "passed": 1, "failed": 0, "skipped": 0, "total": 1, "ok": True}]

    scores = wm_validate.offline_replay_scores([sample_episode])

    assert scores[sample_episode["id"]] is not None
    assert 0.0 <= scores[sample_episode["id"]] <= 1.0


def test_offline_replay_scores_never_dials_the_original_remote_url(tmp_path, monkeypatch, sample_episode):
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path / ".episodic"))
    origin = tmp_path / "origin2"
    sha = _git_repo(origin, {"mod.py": "x = 1\n"})
    sample_episode["repo_state"].update({
        "root": str(origin), "remote_url": "https://this-host-does-not-exist.invalid/repo.git",
        "base_commit": sha,
    })
    sample_episode["commands"] = []

    scores = wm_validate.offline_replay_scores([sample_episode])

    assert sample_episode["id"] in scores


def test_offline_replay_scores_returns_none_when_no_local_repo(tmp_path, monkeypatch, sample_episode):
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path / ".episodic"))
    sample_episode["repo_state"].update({"root": None, "remote_url": None, "base_commit": None})
    sample_episode["commands"] = []

    scores = wm_validate.offline_replay_scores([sample_episode])

    assert scores[sample_episode["id"]] is None


def test_offline_replay_scores_uses_injected_runner(tmp_path, monkeypatch, sample_episode):
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path / ".episodic"))
    origin = tmp_path / "origin3"
    sha = _git_repo(origin, {"mod.py": "x = 1\n"})
    sample_episode["repo_state"].update({"root": str(origin), "remote_url": None, "base_commit": sha})
    sample_episode["commands"] = []

    calls = []

    def runner(model, workspace, prompt_text):
        calls.append(model)
        return "ran", 0

    wm_validate.offline_replay_scores([sample_episode], model="offline-check", runner=runner)

    assert calls == ["offline-check"]


def test_offline_replay_scores_cleans_up_each_workspace(tmp_path, monkeypatch, sample_episode):
    from episodic import paths

    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path / ".episodic"))
    origin = tmp_path / "origin_cleanup"
    sha = _git_repo(origin, {"mod.py": "x = 1\n"})
    sample_episode["repo_state"].update({"root": str(origin), "remote_url": None, "base_commit": sha})
    sample_episode["commands"] = []

    wm_validate.offline_replay_scores([sample_episode], runner=lambda *a: ("ran", 0))

    replays_root = paths.replays_dir()
    assert not replays_root.exists() or not any(replays_root.iterdir())


def test_offline_replay_scores_respects_max_episodes(tmp_path, monkeypatch):
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path / ".episodic"))
    episodes = [make_episode(f"ep_cap_{i}") for i in range(3)]

    scores = wm_validate.offline_replay_scores(episodes, runner=lambda *a: ("ran", 0), max_episodes=2)

    assert set(scores) == {"ep_cap_0", "ep_cap_1"}


def test_has_captured_verifier_tracks_test_signal(sample_episode):
    command = "pytest -q"
    sample_episode["tests"] = []
    sample_episode["commands"] = []
    assert wm_validate.has_captured_verifier(sample_episode) is False

    sample_episode["tests"] = [{"ts": "t", "framework": "pytest", "command": command,
                                "passed": 1, "failed": 0, "skipped": 0, "total": 1, "ok": True}]
    assert wm_validate.has_captured_verifier(sample_episode) is True


def test_finite_excludes_bool_and_non_finite():
    assert wm_validate._finite(1.0) is True
    assert wm_validate._finite(True) is False
    assert wm_validate._finite(math.nan) is False
    assert wm_validate._finite(math.inf) is False
    assert wm_validate._finite(None) is False
