import hashlib
import json

WM_SYSTEM = (
    "You are a language world model for a coding agent. Given the interaction history and "
    "the agent's current action, predict the environment observation that results. Produce only "
    "the observation, faithfully reflecting tool execution, file-system state, and prior turns."
)


def _frac(*parts):
    digest = hashlib.sha256(":".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0x100000000


def _pick(seed_key, count, seed):
    return int(hashlib.sha256(f"{seed}:{seed_key}".encode("utf-8")).hexdigest()[:8], 16) % count


def source_key(episode):
    repo_state = episode.get("repo_state") or {}
    return (
        repo_state.get("repo")
        or repo_state.get("remote_url")
        or (episode.get("labels") or [None])[0]
        or "unknown"
    )


def _action_repr(step):
    tool = step.get("tool") or step.get("type") or "unknown"
    raw = step.get("input") or {}
    if raw.get("prompt") is not None:
        return f"{tool}: {str(raw['prompt'])[:200]}"
    compact = json.dumps(raw, ensure_ascii=False)[:200]
    cwd = step.get("cwd")
    return f"{tool}({compact})" + (f" @ {cwd}" if cwd else "")


def render_history(episode, upto_index):
    lines = [f"INTENT: {episode.get('intent', '')}"]
    steps = episode.get("steps", [])
    for j in range(upto_index):
        step = steps[j]
        lines.append(f"ACTION: {_action_repr(step)}")
        obs = (step.get("observation") or "")[:400]
        lines.append(f"OBSERVATION: {obs}")
    lines.append(f"ACTION: {_action_repr(steps[upto_index])}")
    lines.append("OBSERVATION:")
    return "\n".join(lines)


def expand_turns(episode):
    samples = []
    steps = episode.get("steps", [])
    for index, step in enumerate(steps):
        observation = step.get("observation") or ""
        if not observation.strip():
            continue
        samples.append({
            "episode_id": episode["id"],
            "turn_index": index,
            "source": source_key(episode),
            "domain": (episode.get("labels") or ["unknown"])[0],
            "intent": episode.get("intent", ""),
            "history": render_history(episode, index),
            "action": _action_repr(step),
            "target_observation": observation,
        })
    return samples


def wm_samples(episodes, one_per_trajectory=False, seed=0):
    out = []
    for episode in episodes:
        samples = expand_turns(episode)
        if not samples:
            continue
        if one_per_trajectory:
            out.append(samples[_pick(episode["id"], len(samples), seed)])
        else:
            out.extend(samples)
    return out


def ood_split(episodes, holdout_frac=0.3, seed=0):
    sources = sorted({source_key(episode) for episode in episodes})
    holdout_sources = {src for src in sources if _frac(src, seed) < holdout_frac}
    train, holdout = [], []
    for episode in episodes:
        (holdout if source_key(episode) in holdout_sources else train).append(episode)
    mapping = {src: ("holdout" if src in holdout_sources else "train") for src in sources}
    return train, holdout, mapping


def to_messages(sample):
    return {
        "messages": [
            {"role": "system", "content": WM_SYSTEM},
            {"role": "user", "content": sample["history"]},
            {"role": "assistant", "content": sample["target_observation"]},
        ],
        "meta": {
            "episode_id": sample["episode_id"],
            "turn_index": sample["turn_index"],
            "source": sample["source"],
            "domain": sample["domain"],
        },
    }
