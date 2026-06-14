from datetime import datetime

from ..schema import new_episode, default_stats
from . import gitinfo, diffparse, testdetect, reward
from .ids import episode_id_from_session

STEP_EVENT_TYPES = {
    "user_prompt",
    "file_read",
    "file_edit",
    "file_write",
    "shell_command",
    "tool_post",
    "denial",
}

OBSERVATION_LIMIT = 600


def _parse_ts(value):
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _first_line(text):
    for line in (text or "").splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _step_intent(event):
    data = event["data"]
    if event["type"] == "user_prompt":
        return _first_line(data.get("prompt", ""))[:200]
    if event["type"] == "shell_command":
        return data.get("command", "")[:200]
    if event["type"] in ("file_edit", "file_write", "file_read"):
        verb = {"file_edit": "edit", "file_write": "write", "file_read": "read"}[event["type"]]
        return f"{verb} {data.get('file_path') or '?'}"
    if event["type"] == "denial":
        return f"denied {event.get('tool_name') or 'tool'}"
    return event.get("tool_name") or event["type"]


def _observation(event):
    data = event["data"]
    text = data.get("response") or data.get("prompt") or ""
    return text[:OBSERVATION_LIMIT]


def _build_steps(events):
    steps = []
    for event in events:
        if event["type"] not in STEP_EVENT_TYPES:
            continue
        data = event["data"]
        steps.append({
            "index": len(steps),
            "ts": event["ts"],
            "type": event["type"],
            "tool": event.get("tool_name"),
            "intent": _step_intent(event),
            "input": data.get("tool_input") or _prompt_input(event),
            "observation": _observation(event),
            "approved": data.get("approved"),
            "duration_ms": None,
        })
    return steps


def _prompt_input(event):
    if event["type"] == "user_prompt":
        return {"prompt": event["data"].get("prompt", "")}
    return {}


def _build_commands(events):
    commands = []
    for event in events:
        if event["type"] != "shell_command":
            continue
        data = event["data"]
        command = data.get("command", "")
        commands.append({
            "ts": event["ts"],
            "command": command,
            "cwd": data.get("cwd"),
            "exit_code": data.get("exit_code"),
            "output_excerpt": (data.get("response") or "")[:OBSERVATION_LIMIT],
            "is_test": testdetect.classify_command(command) is not None,
        })
    return commands


def _build_tests(events):
    tests = []
    for event in events:
        if event["type"] != "shell_command":
            continue
        data = event["data"]
        detected = testdetect.detect_test_run(
            data.get("command", ""), data.get("response", ""), event["ts"]
        )
        if detected:
            tests.append(detected)
    return tests


def _build_diffs(repo_state, cwd, events):
    base_commit = repo_state.get("base_commit")
    if repo_state.get("root") and gitinfo.git_available(repo_state["root"]):
        patch = gitinfo.working_diff(repo_state["root"], base_commit)
        parsed = diffparse.parse_unified_diff(patch)
        if parsed:
            return parsed
    touched = {}
    for event in events:
        if event["type"] in ("file_edit", "file_write"):
            path = event["data"].get("file_path")
            if path:
                status = "added" if event["type"] == "file_write" else "modified"
                touched[path] = status
    return [
        {"file": path, "status": status, "additions": 0, "deletions": 0, "unified": None}
        for path, status in sorted(touched.items())
    ]


def _build_stats(events, meta):
    stats = default_stats()
    timestamps = [_parse_ts(event["ts"]) for event in events]
    timestamps = [ts for ts in timestamps if ts]
    if timestamps:
        stats["started_at"] = min(timestamps).isoformat()
        stats["ended_at"] = max(timestamps).isoformat()
        stats["duration_ms"] = int((max(timestamps) - min(timestamps)).total_seconds() * 1000)
    for event in events:
        kind = event["type"]
        if kind in ("file_edit", "file_write", "file_read", "shell_command", "tool_post"):
            stats["tool_calls"] += 1
        if kind == "file_read":
            stats["file_reads"] += 1
        if kind in ("file_edit", "file_write"):
            stats["file_edits"] += 1
        if kind == "shell_command":
            stats["shell_commands"] += 1
        if kind == "denial":
            stats["denials"] += 1
        if event["data"].get("approved") is True:
            stats["approvals"] += 1
    usage = meta.get("usage") or {}
    stats["input_tokens"] = usage.get("input_tokens", 0)
    stats["output_tokens"] = usage.get("output_tokens", 0)
    stats["cost_usd"] = usage.get("cost_usd", 0.0)
    return stats


def _resolve_intent(meta, events):
    if meta.get("intent"):
        return meta["intent"]
    for event in events:
        if event["type"] == "user_prompt":
            return event["data"].get("prompt", "")
    return ""


def _resolve_repo_state(meta, cwd):
    state = meta.get("repo_state")
    if state and state.get("root"):
        return state
    if cwd:
        return gitinfo.repo_state(cwd)
    return state or gitinfo.repo_state(".")


def _resolve_cwd(meta, events):
    if meta.get("cwd"):
        return meta["cwd"]
    for event in events:
        cwd = event["data"].get("cwd")
        if cwd:
            return cwd
    return None


def build_episode(session):
    events = session["events"]
    meta = session.get("meta") or {}
    agent = meta.get("agent") or (events[0]["source"] if events else "claude-code")
    cwd = _resolve_cwd(meta, events)
    repo_state = _resolve_repo_state(meta, cwd)

    episode = new_episode(
        id=episode_id_from_session(session["id"]),
        agent=agent,
        intent=_resolve_intent(meta, events),
        repo_state=repo_state,
        created_at=meta.get("created_at"),
    )
    episode["steps"] = _build_steps(events)
    episode["commands"] = _build_commands(events)
    episode["tests"] = _build_tests(events)
    episode["diffs"] = _build_diffs(repo_state, cwd, events)
    episode["human_feedback"] = meta.get("human_feedback", [])
    episode["outcome"] = meta.get("outcome") or episode["outcome"]
    episode["stats"] = _build_stats(events, meta)
    episode["stats"]["tests_run"] = len(episode["tests"])
    episode["labels"] = sorted(set(meta.get("labels", []) + [f["label"] for f in episode["human_feedback"]]))
    episode["reward_vector"] = reward.reward_vector(episode)
    return episode
