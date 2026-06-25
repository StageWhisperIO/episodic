from episodic import store
from episodic.schema import validate_episode
from episodic.testing import (
    make_episode,
    make_trajectory,
    make_population,
    populate_store,
    make_test_observation,
    terminal_observation,
)


def test_make_episode_is_schema_valid_and_scored():
    ep = make_episode("ep_factory_1", outcome="merged", feedback=["useful"])
    assert validate_episode(ep) == []
    assert ep["outcome"]["status"] == "merged"
    assert ep["tests"][0]["ok"] is True
    assert ep["reward_vector"]["composite"] > 0.5


def test_make_episode_failure_path_scores_low():
    ep = make_episode("ep_factory_2", outcome="reverted", feedback=["wrong"], passed=1, failed=2)
    assert validate_episode(ep) == []
    assert ep["tests"][0]["ok"] is False
    assert ep["reward_vector"]["composite"] < 0.5


def test_make_episode_is_deterministic():
    a = make_episode("ep_same", cost_usd=0.2)
    b = make_episode("ep_same", cost_usd=0.2)
    assert a == b


def test_make_trajectory_observations_present():
    turns = [
        {"action": {"command": "ls"}, "observation": terminal_observation("ls", "a.py b.py")},
        {"action": {"command": "pytest -q"}, "observation": make_test_observation(2, 0),
         "is_test": True, "passed": 2, "failed": 0},
    ]
    ep = make_trajectory("ep_traj", "list and test", turns)
    assert validate_episode(ep) == []
    assert len(ep["steps"]) == 2
    assert ep["commands"][0]["command"] == "ls"
    assert ep["tests"][0]["passed"] == 2


def test_population_is_schema_valid_and_diverse():
    pop = make_population(15, seed=3)
    assert all(validate_episode(ep) == [] for ep in pop)
    sources = {ep["repo_state"]["repo"] for ep in pop}
    outcomes = {ep["outcome"]["status"] for ep in pop}
    assert len(sources) >= 3
    assert {"merged", "reverted"}.issubset(outcomes)
    assert len({ep["id"] for ep in pop}) == 15


def test_populate_store_round_trips(tmp_path, monkeypatch):
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path / ".episodic"))
    saved = populate_store(8, seed=0)
    loaded = store.load_episodes()
    assert len(loaded) == 8
    assert {ep["id"] for ep in loaded} == {ep["id"] for ep in saved}
