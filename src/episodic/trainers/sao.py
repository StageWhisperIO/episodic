import math
import random

DEFAULT_EPSILON_LOW = 0.2
DEFAULT_EPSILON_HIGH = 0.2
DEFAULT_BASELINE_WINDOW = 8
DEFAULT_SAMPLER_REFRESH_STEPS = 4
DEFAULT_NGU_K = 4
DEFAULT_NGU_P = 0.9
DEFAULT_NGU_MAX_ROUNDS = 6


def unique_prompts(rows):
    prompts = []
    seen = set()
    for row in rows:
        user = next((m for m in (row.get("messages") or []) if m.get("role") == "user"), None)
        if not user:
            continue
        content = user.get("content", "")
        if not content or content in seen:
            continue
        seen.add(content)
        prompts.append({"user": user, "meta": row.get("meta")})
    return prompts


def resolve_reward(config):
    import importlib

    funcs = []
    for ref in config.get("reward_funcs", []):
        module_name, _, attr = ref.partition(":")
        funcs.append(getattr(importlib.import_module(module_name), attr))
    if not funcs:
        from .rewards import action_format_reward
        funcs = [action_format_reward]

    def score(prompt_text, completion_text, meta=None):
        total = 0.0
        for func in funcs:
            out = func(prompts=[prompt_text], completions=[completion_text], meta=[meta])
            total += float(out[0]) if out else 0.0
        return total / len(funcs)

    return score


def running_baseline(key, prompt_windows, global_window):
    window = prompt_windows.get(key)
    if window:
        return sum(window) / len(window)
    if global_window:
        return sum(global_window) / len(global_window)
    return 0.0


def group_advantages(rewards, normalize=True, eps=1e-6):
    count = len(rewards)
    if count == 0:
        return []
    mean = sum(rewards) / count
    centered = [reward - mean for reward in rewards]
    if not normalize or count < 2:
        return centered
    std = math.sqrt(sum(value * value for value in centered) / count)
    if std < eps:
        return [0.0] * count
    return [value / std for value in centered]


def dis_mask(advantages, current_logprobs, rollout_logprobs, epsilon_low, epsilon_high):
    masked_advantages = []
    masked = 0
    for advantage, current, rollout in zip(advantages, current_logprobs, rollout_logprobs):
        ratio = math.exp(current - rollout)
        if (1.0 - epsilon_low) < ratio < (1.0 + epsilon_high):
            masked_advantages.append(advantage)
        else:
            masked_advantages.append(0.0)
            masked += 1
    return masked_advantages, masked


def token_gae(rewards, values, gamma=1.0, lam=1.0, action_mask=None):
    length = len(rewards)
    if action_mask is None:
        action_mask = [True] * length
    action_indices = [i for i in range(length) if action_mask[i]]
    advantages = [0.0] * length
    running = 0.0
    next_value = 0.0
    for i in reversed(action_indices):
        delta = rewards[i] + gamma * next_value - values[i]
        running = delta + gamma * lam * running
        advantages[i] = running
        next_value = values[i]
    return advantages


def terminal_rewards(reward, length):
    rewards = [0.0] * length
    if length:
        rewards[-1] = reward
    return rewards


def ngu_classify(rewards, max_reward=1.0, eps=1e-9):
    if not rewards:
        return "filter"
    if all(reward >= max_reward - eps for reward in rewards):
        return "filter"
    if max(rewards) - min(rewards) > eps:
        return "train"
    return "retry"


class NGUScheduler:
    def __init__(self, k=DEFAULT_NGU_K, p=DEFAULT_NGU_P, max_rounds=DEFAULT_NGU_MAX_ROUNDS,
                 rng=None, max_reward=1.0, eps=1e-9):
        self.k = k
        self.p = p
        self.max_rounds = max_rounds
        self.rng = rng if rng is not None else random.Random()
        self.max_reward = max_reward
        self.eps = eps
        self._state = {}

    def pending(self, key):
        return key in self._state

    def submit(self, key, samples, rewards):
        entry = self._state.setdefault(key, {"samples": [], "rewards": [], "rounds": 0})
        entry["samples"] = entry["samples"] + list(samples)
        entry["rewards"] = entry["rewards"] + list(rewards)
        entry["rounds"] += 1

        action = ngu_classify(entry["rewards"], self.max_reward, self.eps)
        if action == "train":
            result = {"action": "train", "rounds": entry["rounds"],
                      "samples": entry["samples"], "rewards": entry["rewards"]}
            del self._state[key]
            return result
        if action == "filter":
            del self._state[key]
            return {"action": "filter", "rounds": entry["rounds"], "samples": None, "rewards": None}

        if entry["rounds"] >= self.max_rounds or self.rng.random() >= self.p:
            del self._state[key]
            return {"action": "drop", "rounds": entry["rounds"], "samples": None, "rewards": None}
        return {"action": "retry", "rounds": entry["rounds"], "samples": None, "rewards": None}
