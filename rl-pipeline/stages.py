import os
import sys
import json
import math
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from episodic import store as _store

try:
    from episodic import exporters as _exporters
    _trajectory_text = _exporters.trajectory_text
    _is_good = _exporters.is_good
    _is_bad = _exporters.is_bad
except ImportError:
    _exporters = None
    _trajectory_text = None
    _is_good = None
    _is_bad = None


def load_episodes(home=None):
    return _store.load_episodes(home)


def _composite(ep):
    return (ep.get("reward_vector") or {}).get("composite") or 0.0


def quality_filter(episodes, min_composite=0.5):
    bad_statuses = {"failed", "reverted"}
    kept = []
    dropped = []
    for ep in episodes:
        composite = _composite(ep)
        status = (ep.get("outcome") or {}).get("status", "")
        has_wrong = any(
            fb.get("label") == "wrong"
            for fb in (ep.get("human_feedback") or [])
        )
        if composite >= min_composite and status not in bad_statuses and not has_wrong:
            kept.append(ep)
        else:
            dropped.append(ep)
    return kept, {"kept": len(kept), "dropped": len(dropped), "total": len(episodes)}


def trajectory_text(ep):
    if _trajectory_text is not None:
        return _trajectory_text(ep)
    parts = [f"USER: {ep.get('intent', '')}"]
    for step in ep.get("steps", []):
        tool = step.get("tool") or step.get("type") or "unknown"
        raw_input = step.get("input") or {}
        compact = json.dumps(raw_input, ensure_ascii=False)[:120]
        parts.append(f"ACTION {tool}({compact})")
        obs = (step.get("observation") or "")[:200]
        parts.append(f"OBS: {obs}")
    return "\n".join(parts)


def sft_dataset(good):
    rows = []
    for ep in good:
        rv = ep.get("reward_vector") or {}
        rows.append({
            "messages": [
                {"role": "user", "content": ep.get("intent", "")},
                {"role": "assistant", "content": trajectory_text(ep)},
            ],
            "meta": {
                "episode_id": ep["id"],
                "reward": rv.get("composite", 0.0),
            },
        })
    return rows


def _norm_intent(ep):
    return (ep.get("intent") or "").lower().strip()


def _is_good_ep(ep):
    if _is_good is not None:
        return _is_good(ep)
    return _composite(ep) >= 0.5


def _is_bad_ep(ep):
    if _is_bad is not None:
        return _is_bad(ep)
    bad_statuses = {"failed", "reverted"}
    if (ep.get("outcome") or {}).get("status") in bad_statuses:
        return True
    return any(fb.get("label") == "wrong" for fb in (ep.get("human_feedback") or []))


def preference_pairs(episodes):
    groups = {}
    for ep in episodes:
        key = _norm_intent(ep)
        groups.setdefault(key, []).append(ep)

    rows = []
    for intent_key, group in groups.items():
        if len(group) < 2:
            continue
        good = [ep for ep in group if _is_good_ep(ep)]
        bad = [ep for ep in group if _is_bad_ep(ep)]

        if good and bad:
            chosen = max(good, key=_composite)
            rejected = min(bad, key=_composite)
        else:
            sorted_group = sorted(group, key=_composite)
            if sorted_group[-1] is sorted_group[0]:
                continue
            chosen = sorted_group[-1]
            rejected = sorted_group[0]
            if _composite(chosen) == _composite(rejected):
                continue

        rows.append({
            "prompt": chosen.get("intent", ""),
            "chosen": trajectory_text(chosen),
            "rejected": trajectory_text(rejected),
            "meta": {
                "chosen_id": chosen["id"],
                "rejected_id": rejected["id"],
                "chosen_reward": _composite(chosen),
                "rejected_reward": _composite(rejected),
            },
        })
    return rows


def featurize(ep):
    rv = ep.get("reward_vector") or {}
    stats = ep.get("stats") or {}
    test_pass = rv.get("test_pass", 0.0)
    edit_focus = rv.get("edit_focus", 0.0)
    file_edits = stats.get("file_edits", 0)
    tests_run = stats.get("tests_run", 0)
    file_reads = stats.get("file_reads", 0)
    has_feedback = 1.0 if ep.get("human_feedback") else 0.0
    return [
        float(test_pass),
        float(edit_focus),
        min(float(file_edits) / 10.0, 1.0),
        min(float(tests_run) / 5.0, 1.0),
        has_feedback,
        min(float(file_reads) / 20.0, 1.0),
        1.0,
    ]


_FEATURE_NAMES = [
    "test_pass",
    "edit_focus",
    "file_edits_norm",
    "tests_run_norm",
    "has_feedback",
    "file_reads_norm",
    "bias",
]


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _mat_vec(mat, vec):
    return [_dot(row, vec) for row in mat]


def _transpose(mat):
    if not mat:
        return []
    return [[mat[r][c] for r in range(len(mat))] for c in range(len(mat[0]))]


def _mat_mul(a, b):
    bt = _transpose(b)
    return [[_dot(row_a, col_b) for col_b in bt] for row_a in a]


def _identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _solve_normal_equations(X, y):
    n_feat = len(X[0])
    Xt = _transpose(X)
    XtX = _mat_mul(Xt, X)
    Xty = _mat_vec(Xt, y)
    aug = [row[:] + [Xty[i]] for i, row in enumerate(XtX)]
    n = len(aug)
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        if abs(aug[col][col]) < 1e-12:
            return [0.0] * n_feat
        factor = aug[col][col]
        aug[col] = [v / factor for v in aug[col]]
        for r in range(n):
            if r != col:
                mult = aug[r][col]
                aug[r] = [aug[r][k] - mult * aug[col][k] for k in range(len(aug[r]))]
    return [aug[i][-1] for i in range(n)]


def reward_model_train(episodes):
    n_feat = len(_FEATURE_NAMES)
    zero_weights = [0.0] * n_feat

    valid = [(featurize(ep), _composite(ep)) for ep in episodes]
    if len(valid) < 2:
        return {"weights": zero_weights, "features": _FEATURE_NAMES, "r2": 0.0}

    X = [row for row, _ in valid]
    y = [label for _, label in valid]

    try:
        import numpy as np
        X_np = np.array(X, dtype=float)
        y_np = np.array(y, dtype=float)
        result = np.linalg.lstsq(X_np, y_np, rcond=None)
        weights = result[0].tolist()
    except ImportError:
        weights = _solve_normal_equations(X, y)

    y_mean = sum(y) / len(y)
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
    y_pred = [_dot(weights, xi) for xi in X]
    ss_res = sum((yi - yp) ** 2 for yi, yp in zip(y, y_pred))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    return {"weights": weights, "features": _FEATURE_NAMES, "r2": round(r2, 6)}


def reward_model_predict(model, ep):
    return _dot(model["weights"], featurize(ep))


def rl_batches(episodes):
    transitions = []
    for ep in episodes:
        steps = ep.get("steps", [])
        composite = _composite(ep)
        n = len(steps)
        for i, step in enumerate(steps):
            is_terminal = i == n - 1
            prev_obs = steps[i - 1].get("observation", "") if i > 0 else ""
            transitions.append({
                "state": {
                    "intent": ep.get("intent", ""),
                    "step_index": i,
                    "prev_observation": prev_obs,
                },
                "action": {
                    "type": step.get("type"),
                    "tool": step.get("tool"),
                    "input": step.get("input"),
                },
                "observation": step.get("observation", ""),
                "reward": composite if is_terminal else 0.0,
                "terminal": is_terminal,
                "discount": 1.0,
            })
    return transitions


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx < 1e-12 or dy < 1e-12:
        return 0.0
    return num / (dx * dy)


def evaluate(episodes, model):
    actual = [_composite(ep) for ep in episodes]
    predicted = [reward_model_predict(model, ep) for ep in episodes]
    n = len(episodes)
    pearson = _pearson(predicted, actual)
    mae = sum(abs(p - a) for p, a in zip(predicted, actual)) / n if n > 0 else 0.0
    return {
        "n": n,
        "pearson": round(pearson, 6),
        "mean_abs_err": round(mae, 6),
    }


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
