import json
from pathlib import Path

FORMATS = ("sft", "dpo", "reward", "rlds", "wm", "jsonl", "parquet")

_GOOD_LABELS = {"useful", "accepted_as_is", "accepted_after_edits"}
_BAD_LABELS = {"wrong"}
_GOOD_STATUSES = {"accepted", "merged"}
_BAD_STATUSES = {"failed", "reverted"}


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def trajectory_text(ep):
    parts = [f"USER: {ep['intent']}"]
    for step in ep.get("steps", []):
        if step.get("type") == "user_prompt":
            prompt = (step.get("input") or {}).get("prompt", "")
            parts.append(f"ACTION user_prompt({prompt[:120]})")
        else:
            tool = step.get("tool") or step.get("type") or "unknown"
            raw_input = step.get("input") or {}
            compact = json.dumps(raw_input, ensure_ascii=False)[:120]
            cwd = step.get("cwd")
            location = f" @ {cwd}" if cwd else ""
            parts.append(f"ACTION {tool}({compact}){location}")
        obs = (step.get("observation") or "")[:200]
        parts.append(f"OBS: {obs}")
    return "\n".join(parts)


def is_good(ep):
    rv = ep.get("reward_vector") or {}
    if (rv.get("composite") or 0.0) >= 0.5:
        return True
    if ep.get("outcome", {}).get("status") in _GOOD_STATUSES:
        return True
    for fb in ep.get("human_feedback", []):
        if fb.get("label") in _GOOD_LABELS:
            return True
    return False


def is_bad(ep):
    outcome = ep.get("outcome", {})
    if outcome.get("status") in _BAD_STATUSES:
        return True
    if outcome.get("caused_regression"):
        return True
    for fb in ep.get("human_feedback", []):
        if fb.get("label") in _BAD_LABELS:
            return True
    return False


def norm_intent(ep):
    return (ep.get("intent") or "").lower().strip()


def _composite(ep):
    return (ep.get("reward_vector") or {}).get("composite") or 0.0


def _export_jsonl(episodes, out_dir):
    path = out_dir / "episodes.jsonl"
    write_jsonl(path, episodes)
    return {"files": [str(path)], "count": len(episodes)}


def _export_sft(episodes, out_dir):
    rows = []
    for ep in episodes:
        if not is_good(ep):
            continue
        rv = ep.get("reward_vector") or {}
        rows.append({
            "messages": [
                {"role": "user", "content": ep["intent"]},
                {"role": "assistant", "content": trajectory_text(ep)},
            ],
            "meta": {"episode_id": ep["id"], "reward": rv.get("composite")},
        })
    path = out_dir / "sft.jsonl"
    write_jsonl(path, rows)
    return {"files": [str(path)], "count": len(rows)}


def _export_dpo(episodes, out_dir):
    groups = {}
    for ep in episodes:
        key = norm_intent(ep)
        groups.setdefault(key, []).append(ep)

    rows = []
    for intent_key, group in groups.items():
        good = [ep for ep in group if is_good(ep)]
        bad = [ep for ep in group if is_bad(ep)]
        if not good:
            continue
        chosen = max(good, key=_composite)
        if bad:
            rejected = min(bad, key=_composite)
        else:
            non_good = [ep for ep in group if not is_good(ep)]
            if not non_good:
                all_sorted = sorted(group, key=_composite)
                if len(all_sorted) < 2:
                    continue
                rejected = all_sorted[0]
                chosen = all_sorted[-1]
            else:
                rejected = min(non_good, key=_composite)

        rows.append({
            "prompt": chosen["intent"],
            "chosen": trajectory_text(chosen),
            "rejected": trajectory_text(rejected),
            "meta": {
                "chosen_id": chosen["id"],
                "rejected_id": rejected["id"],
                "chosen_reward": _composite(chosen),
                "rejected_reward": _composite(rejected),
            },
        })

    path = out_dir / "dpo.jsonl"
    write_jsonl(path, rows)
    return {"files": [str(path)], "count": len(rows), "pairs": len(rows)}


def _export_reward(episodes, out_dir):
    rows = []
    for ep in episodes:
        rows.append({
            "prompt": ep["intent"],
            "trajectory": trajectory_text(ep),
            "reward_vector": ep.get("reward_vector"),
            "scalar_reward": _composite(ep),
        })
    path = out_dir / "reward.jsonl"
    write_jsonl(path, rows)
    return {"files": [str(path)], "count": len(rows)}


def _export_rlds(episodes, out_dir):
    rows = []
    for ep in episodes:
        steps = ep.get("steps", [])
        n = len(steps)
        composite = _composite(ep)
        rlds_steps = []
        for i, step in enumerate(steps):
            prev_obs = steps[i - 1].get("observation", "") if i > 0 else ""
            is_last = i == n - 1
            rlds_steps.append({
                "observation": prev_obs,
                "action": {
                    "tool": step.get("tool"),
                    "input": step.get("input"),
                    "type": step.get("type"),
                    "cwd": step.get("cwd"),
                },
                "next_observation": step.get("observation", ""),
                "reward": composite if is_last else 0.0,
                "is_first": i == 0,
                "is_last": is_last,
                "is_terminal": is_last,
                "discount": 0.0 if is_last else 1.0,
            })
        rows.append({"episode_id": ep["id"], "steps": rlds_steps})
    path = out_dir / "rlds.jsonl"
    write_jsonl(path, rows)
    return {"files": [str(path)], "count": len(rows)}


def _export_wm(episodes, out_dir):
    from episodic.worldmodel import wm_samples, to_messages

    rows = [to_messages(sample) for sample in wm_samples(episodes)]
    path = out_dir / "wm.jsonl"
    write_jsonl(path, rows)
    return {"files": [str(path)], "count": len(rows), "turns": len(rows)}


def _flatten_row(ep):
    rv = ep.get("reward_vector") or {}
    stats = ep.get("stats") or {}
    diffs = ep.get("diffs") or []
    additions = sum(d.get("additions", 0) for d in diffs)
    deletions = sum(d.get("deletions", 0) for d in diffs)
    return {
        "id": ep.get("id"),
        "intent": ep.get("intent"),
        "agent": ep.get("agent"),
        "branch": (ep.get("repo_state") or {}).get("branch"),
        "outcome_status": (ep.get("outcome") or {}).get("status"),
        "composite": rv.get("composite"),
        "test_pass": rv.get("test_pass"),
        "file_edits": stats.get("file_edits"),
        "tests_run": stats.get("tests_run"),
        "additions": additions,
        "deletions": deletions,
    }


def _export_parquet(episodes, out_dir):
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        rows = [_flatten_row(ep) for ep in episodes]
        if not rows:
            table = pa.table({})
        else:
            keys = list(rows[0].keys())
            table = pa.table({k: [r[k] for r in rows] for k in keys})
        path = out_dir / "episodes.parquet"
        pq.write_table(table, str(path))
        return {"files": [str(path)], "count": len(rows)}
    except ImportError:
        rows = [_flatten_row(ep) for ep in episodes]
        path = out_dir / "episodes.jsonl"
        write_jsonl(path, rows)
        return {
            "files": [str(path)],
            "count": len(rows),
            "fallback": "pyarrow not installed; wrote jsonl",
        }


_EXPORTERS = {
    "jsonl": _export_jsonl,
    "sft": _export_sft,
    "dpo": _export_dpo,
    "reward": _export_reward,
    "rlds": _export_rlds,
    "wm": _export_wm,
    "parquet": _export_parquet,
}


def export(episodes, fmt, out_dir):
    if fmt not in _EXPORTERS:
        raise ValueError(f"Unknown format {fmt!r}. Choose from: {FORMATS}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = _EXPORTERS[fmt](episodes, out_dir)
    result["format"] = fmt
    result["out_dir"] = str(out_dir)
    return result
