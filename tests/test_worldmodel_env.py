from episodic.worldmodel import env as wm_env
from episodic.worldbench import NAMED_PREDICTORS
from episodic.testing import make_episode


def test_reset_returns_not_done_for_a_nonempty_episode():
    episode = make_episode("ep_env_reset")
    env = wm_env.WorldModelEnv(episode, NAMED_PREDICTORS["oracle"])
    state = env.reset()
    assert state["done"] is False
    assert state["turn_index"] == 0


def test_reset_on_empty_episode_is_immediately_done():
    episode = make_episode("ep_env_empty")
    episode["steps"] = []
    env = wm_env.WorldModelEnv(episode, NAMED_PREDICTORS["oracle"])
    state = env.reset()
    assert state["done"] is True


def test_step_raises_after_episode_ends():
    episode = make_episode("ep_env_step_raise")
    episode["steps"] = episode["steps"][:1]
    env = wm_env.WorldModelEnv(episode, NAMED_PREDICTORS["oracle"])
    env.reset()
    env.step()
    assert env.done is True
    try:
        env.step()
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_oracle_predictor_reproduces_target_observation_every_turn():
    episode = make_episode("ep_env_oracle")
    result = wm_env.rollout(episode, NAMED_PREDICTORS["oracle"])
    assert result["truncated"] is False
    assert len(result["turns"]) == len(episode["steps"])
    for turn in result["turns"]:
        assert turn["predicted_observation"] == turn["target_observation"]


def test_rollout_feeds_predicted_observations_back_into_context_not_ground_truth():
    episode = make_episode("ep_env_closed_loop")
    seen_histories = []

    def predictor(sample):
        seen_histories.append(sample["history"])
        return "SIMULATED_OBSERVATION"

    result = wm_env.rollout(episode, predictor)
    assert len(result["turns"]) >= 2
    assert "SIMULATED_OBSERVATION" in seen_histories[-1]
    for turn in result["turns"][1:]:
        assert turn["predicted_observation"] == "SIMULATED_OBSERVATION"


def test_max_turns_truncates_and_flags_truncated():
    episode = make_episode("ep_env_maxturns")
    result = wm_env.rollout(episode, NAMED_PREDICTORS["oracle"], max_turns=1)
    assert len(result["turns"]) == 1
    assert result["truncated"] is True


def test_max_turns_at_or_above_episode_length_is_not_truncated():
    episode = make_episode("ep_env_maxturns_full")
    result = wm_env.rollout(episode, NAMED_PREDICTORS["oracle"], max_turns=len(episode["steps"]) + 5)
    assert result["truncated"] is False
    assert len(result["turns"]) == len(episode["steps"])


def test_history_budget_truncates_growing_context():
    episode = make_episode("ep_env_budget", files=("a.py", "b.py", "c.py", "d.py"))
    env = wm_env.WorldModelEnv(episode, NAMED_PREDICTORS["oracle"], history_budget=40)
    env.reset()
    for _ in range(len(episode["steps"])):
        env.step()
    assert len(env._context()) <= 40


def test_policy_overrides_recorded_action_sequence():
    episode = make_episode("ep_env_policy")
    seen_actions = []

    def policy(ep, index, transcript):
        action = f"custom-action-{index}"
        seen_actions.append(action)
        return action

    def predictor(sample):
        assert sample["action"] == seen_actions[sample["turn_index"]]
        return "obs"

    wm_env.rollout(episode, predictor, policy=policy)
    assert seen_actions


def test_prefix_predictor_is_a_persistence_baseline_of_the_models_own_last_prediction():
    episode = make_episode("ep_env_prefix")
    result = wm_env.rollout(episode, NAMED_PREDICTORS["prefix"])
    assert result["turns"][0]["predicted_observation"] == ""
    for prev_turn, turn in zip(result["turns"], result["turns"][1:]):
        assert turn["predicted_observation"] == prev_turn["predicted_observation"]


def test_rollout_of_empty_episode_returns_no_turns():
    episode = make_episode("ep_env_empty_rollout")
    episode["steps"] = []
    result = wm_env.rollout(episode, NAMED_PREDICTORS["oracle"])
    assert result["turns"] == []
    assert result["truncated"] is False
