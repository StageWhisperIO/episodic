import math
from collections import deque

from episodic.trainers import sao


def test_dis_mask_keeps_in_region_and_masks_outside():
    advantages, masked = sao.dis_mask(
        [1.0, 1.0, 1.0], [0.0, 0.5, -0.5], [0.0, 0.0, 0.0], 0.2, 0.2)
    assert advantages == [1.0, 0.0, 0.0]
    assert masked == 2


def test_dis_mask_asymmetric_bounds():
    ratio_high = math.log(1.25)
    advantages, masked = sao.dis_mask([1.0], [ratio_high], [0.0], 0.2, 0.3)
    assert advantages == [1.0]
    assert masked == 0


def test_running_baseline_prefers_prompt_window_then_global():
    windows = {"p": deque([0.5, 1.0])}
    global_window = deque([0.0])
    assert sao.running_baseline("p", windows, global_window) == 0.75
    assert sao.running_baseline("q", windows, global_window) == 0.0
    assert sao.running_baseline("q", {}, deque()) == 0.0


def test_unique_prompts_dedupes_and_skips_userless_rows():
    rows = [
        {"messages": [{"role": "user", "content": "a"}], "meta": {"episode_id": "1"}},
        {"messages": [{"role": "user", "content": "a"}], "meta": {"episode_id": "2"}},
        {"messages": [{"role": "assistant", "content": "x"}]},
        {"messages": [{"role": "user", "content": "b"}]},
    ]
    prompts = sao.unique_prompts(rows)
    assert [p["user"]["content"] for p in prompts] == ["a", "b"]
    assert prompts[0]["meta"] == {"episode_id": "1"}


def test_token_gae_constant_value_single_turn():
    advantages = sao.token_gae([0.0, 0.0, 1.0], [0.5, 0.5, 0.5], gamma=1.0, lam=1.0)
    assert advantages == [0.5, 0.5, 0.5]


def test_token_gae_skips_observation_tokens():
    rewards = [0.0, 0.0, 0.0, 0.0, 1.0]
    values = [0.2, 0.4, 9.0, 9.0, 0.6]
    mask = [True, True, False, False, True]
    advantages = sao.token_gae(rewards, values, gamma=1.0, lam=1.0, action_mask=mask)
    assert advantages[2] == 0.0 and advantages[3] == 0.0
    assert advantages[4] == 0.4
    assert abs(advantages[1] - 0.6) < 1e-9
    assert abs(advantages[0] - 0.8) < 1e-9


def test_terminal_rewards_places_reward_last():
    assert sao.terminal_rewards(2.0, 3) == [0.0, 0.0, 2.0]
    assert sao.terminal_rewards(2.0, 0) == []


def test_resolve_reward_defaults_to_action_format():
    score = sao.resolve_reward({})
    assert isinstance(score("prompt", "completion"), float)


def _constant_reward(prompts=None, completions=None, meta=None, **kwargs):
    return [0.7 for _ in (completions or [])]


def test_resolve_reward_uses_configured_funcs():
    score = sao.resolve_reward({"reward_funcs": [f"{__name__}:_constant_reward"]})
    assert score("p", "c") == 0.7
