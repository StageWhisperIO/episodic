import json

from episodic.serving import difficulty
from episodic import testing


def test_episode_difficulty_is_low_for_a_clean_high_reward_episode():
    episode = testing.make_episode("ep_easy", outcome="merged", feedback=["useful"], passed=3, failed=0)
    episode["validity"] = {"trust": "high"}
    assert difficulty.episode_difficulty(episode) < 0.5


def test_episode_difficulty_is_high_for_a_failed_low_trust_episode():
    episode = testing.make_episode("ep_hard", outcome="reverted", feedback=["wrong"], passed=0, failed=3)
    episode["validity"] = {"trust": "low"}
    episode["stats"]["denials"] = 3
    assert difficulty.episode_difficulty(episode) > 0.5


def test_episode_difficulty_defaults_to_medium_without_a_reward_vector():
    assert difficulty.episode_difficulty({}) == 0.5


def test_text_features_pick_up_keywords_and_length():
    plain = difficulty.text_features("fix typo in readme")
    hard = difficulty.text_features("traceback: race condition causes a security vulnerability crash " * 5)
    assert hard["keyword_hits"] > plain["keyword_hits"]
    assert hard["log_length"] > plain["log_length"]


def test_learn_router_returns_a_fallback_model_with_too_little_data():
    episodes = testing.make_population(2)
    model = difficulty.learn_router(episodes)
    assert model["fallback"] is True
    assert model["trained_on"] <= 2


def test_learn_router_fits_weights_that_separate_easy_and_hard_examples():
    easy = []
    hard = []
    for i in range(15):
        e = testing.make_episode(f"ep_easy_{i}", intent="polish the docs", outcome="merged",
                                 feedback=["useful"], passed=3, failed=0)
        e["validity"] = {"trust": "high"}
        easy.append(e)
        h = testing.make_episode(f"ep_hard_{i}",
                                 intent="fix a nasty race condition traceback crash in the security module",
                                 outcome="reverted", feedback=["wrong"], passed=0, failed=3)
        h["validity"] = {"trust": "low"}
        hard.append(h)

    model = difficulty.learn_router(easy + hard)
    assert model["fallback"] is False
    assert model["trained_on"] == 30

    easy_prob = difficulty.predict_proba("polish the docs some more", model)
    hard_prob = difficulty.predict_proba(
        "fix a nasty race condition traceback crash in the security module today", model)
    assert hard_prob > easy_prob


def test_save_and_load_router_model_round_trips(tmp_path):
    model = {"weights": [0.1], "bias": 0.2, "feature_names": ["log_length"]}
    path = difficulty.save_router_model(model, tmp_path / "router_model.json")
    assert json.loads((tmp_path / "router_model.json").read_text()) == model
    assert difficulty.load_router_model(path) == model


def test_learned_escalate_returns_none_without_a_configured_model():
    assert difficulty.learned_escalate({"messages": []}, {}) is None


def test_learned_escalate_uses_an_inline_model():
    model = {
        "feature_names": list(difficulty.FEATURE_NAMES),
        "mean": [0.0] * len(difficulty.FEATURE_NAMES),
        "std": [1.0] * len(difficulty.FEATURE_NAMES),
        "weights": [0.0] * len(difficulty.FEATURE_NAMES),
        "bias": 10.0,
        "hard_threshold": 0.5,
    }
    decision = difficulty.learned_escalate({"messages": [{"content": "hi"}]}, {"router_model": model})
    assert decision is True

    model_low = dict(model, bias=-10.0)
    decision_low = difficulty.learned_escalate({"messages": [{"content": "hi"}]}, {"router_model": model_low})
    assert decision_low is False


def test_learned_escalate_loads_and_caches_a_model_from_disk(tmp_path):
    model = {
        "feature_names": list(difficulty.FEATURE_NAMES),
        "mean": [0.0] * len(difficulty.FEATURE_NAMES),
        "std": [1.0] * len(difficulty.FEATURE_NAMES),
        "weights": [0.0] * len(difficulty.FEATURE_NAMES),
        "bias": 10.0,
        "hard_threshold": 0.5,
    }
    path = tmp_path / "router_model.json"
    difficulty.save_router_model(model, path)

    config = {"router_model_path": str(path)}
    assert difficulty.learned_escalate({"messages": [{"content": "hi"}]}, config) is True

    difficulty._MODEL_CACHE.clear()
