import math

from .. import fidelity, replay
from ..exporters import _captured_verifier
from . import env as wm_env


def has_captured_verifier(episode):
    return _captured_verifier(episode) is not None


def sim_trajectory_score(episode, predictor, max_turns=None, history_budget=wm_env.HISTORY_BUDGET):
    result = wm_env.rollout(episode, predictor, history_budget=history_budget, max_turns=max_turns)
    trajectory = fidelity.trajectory_score(result["turns"])
    return trajectory["mean_composite"]


def sim_scores(episodes, predictor, max_turns=None, history_budget=wm_env.HISTORY_BUDGET):
    return {
        episode["id"]: sim_trajectory_score(
            episode, predictor, max_turns=max_turns, history_budget=history_budget)
        for episode in episodes
    }


def _local_clone_episode(episode):
    repo_state = dict(episode.get("repo_state") or {})
    root = repo_state.get("root")
    if root:
        repo_state["remote_url"] = root
    clone = dict(episode)
    clone["repo_state"] = repo_state
    return clone


def _unified_diff(episode):
    from ..core import diffparse

    return diffparse.join_unified(diff.get("unified") for diff in episode.get("diffs", []))


def _oracle_diff_runner(unified_diff):
    from ..replay import modelrun

    def runner(model, workspace, prompt_text):
        ok, log = modelrun.apply_diff(unified_diff, workspace)
        return log, 0 if ok else 1

    return runner


def offline_replay_scores(episodes, model="offline-oracle-diff", start=None, runner=None, max_episodes=None):
    scores = {}
    selected = episodes[:max_episodes] if max_episodes else episodes
    for episode in selected:
        local_episode = _local_clone_episode(episode)
        replay.create_replay(local_episode, start=start)
        replay_id = replay.replay_id_for(local_episode)
        episode_runner = runner or _oracle_diff_runner(_unified_diff(episode))
        try:
            result = replay.run_replay(replay_id, model, start=start, execute=True, runner=episode_runner)
            total = (result.get("scores") or {}).get("total")
            scores[episode["id"]] = total if _finite(total) else None
        finally:
            replay.cleanup_replay(replay_id, start=start)
    return scores


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _mean(values):
    return sum(values) / len(values)


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mean_x, mean_y = _mean(xs), _mean(ys)
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return None
    return covariance / math.sqrt(var_x * var_y)


def _rank(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average_rank = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = average_rank
        i = j + 1
    return ranks


def spearman(xs, ys):
    if len(xs) < 2:
        return None
    return pearson(_rank(xs), _rank(ys))


def correlate(sim_scores_map, real_scores_map):
    episode_ids = [
        episode_id for episode_id in sorted(set(sim_scores_map) & set(real_scores_map))
        if _finite(sim_scores_map[episode_id]) and _finite(real_scores_map[episode_id])
    ]
    xs = [sim_scores_map[episode_id] for episode_id in episode_ids]
    ys = [real_scores_map[episode_id] for episode_id in episode_ids]
    r_pearson = pearson(xs, ys)
    r_spearman = spearman(xs, ys)
    return {
        "n": len(episode_ids),
        "episode_ids": episode_ids,
        "sim_scores": xs,
        "real_scores": ys,
        "pearson": round(r_pearson, 4) if r_pearson is not None else None,
        "spearman": round(r_spearman, 4) if r_spearman is not None else None,
    }
