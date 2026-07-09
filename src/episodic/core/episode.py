import os
from collections import defaultdict
from datetime import datetime

from ..schema import new_episode, new_event, default_stats
from . import gitinfo, diffparse, testdetect, reward, transcript, deploydetect, validity
from .normalize import MAX_RESPONSE_CHARS
from .ids import episode_id_from_session

STEP_EVENT_TYPES = {
    "user_prompt",
    "file_read",
    "file_edit",
    "file_write",
    "file_delete",
    "shell_command",
    "tool_post",
    "denial",
}

DIFF_STATUS_BY_TYPE = {
    "file_write": "added",
    "file_edit": "modified",
    "file_delete": "deleted",
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
    if event["type"] in ("file_edit", "file_write", "file_read", "file_delete"):
        verb = {"file_edit": "edit", "file_write": "write", "file_read": "read", "file_delete": "delete"}[event["type"]]
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
            "cwd": data.get("cwd"),
            "duration_ms": None,
        })
    return steps


def _prompt_input(event):
    if event["type"] == "user_prompt":
        return {"prompt": event["data"].get("prompt", "")}
    return {}


def _transcript_path(events):
    for event in events:
        if event["type"] == "session_start":
            path = event["data"].get("transcript_path")
            if path:
                return path
    return None


def _synth_shell_event(pre_event, command, record):
    return new_event(
        pre_event["session_id"],
        "shell_command",
        source=pre_event.get("source", "claude-code"),
        tool_name="Bash",
        data={
            "cwd": pre_event["data"].get("cwd"),
            "command": command,
            "response": (record["output"] or "")[:MAX_RESPONSE_CHARS],
            "exit_code": record["exit_code"],
            "approved": True,
            "reconstructed": True,
        },
        ts=pre_event["ts"],
    )


def _reconstruct_failed_commands(events):
    path = _transcript_path(events)
    if not path:
        return events
    records = transcript.bash_records(path)
    pre_by_command = defaultdict(list)
    post_by_command = defaultdict(int)
    for event in events:
        if event["type"] == "tool_pre" and event.get("tool_name") == "Bash":
            pre_by_command[(event["data"].get("tool_input") or {}).get("command", "")].append(event)
        elif event["type"] == "shell_command":
            post_by_command[event["data"].get("command", "")] += 1

    inserts = defaultdict(list)
    for command, records_deque in records.items():
        entries = list(records_deque)
        if not any(entry["is_error"] for entry in entries):
            continue
        pre_events = pre_by_command.get(command, [])
        successes = sum(1 for entry in entries if not entry["is_error"])
        if len(pre_events) != len(entries) or post_by_command.get(command, 0) != successes:
            continue
        for pre_event, entry in zip(pre_events, entries):
            if entry["is_error"]:
                inserts[id(pre_event)].append(_synth_shell_event(pre_event, command, entry))
    if not inserts:
        return events

    result = []
    for event in events:
        result.append(event)
        result.extend(inserts.get(id(event), ()))
    return result


def _apply_transcript_exit_codes(events):
    path = _transcript_path(events)
    if not path:
        return
    outcomes = transcript.bash_outcomes(path)
    by_command = defaultdict(list)
    for event in events:
        if event["type"] == "shell_command":
            by_command[event["data"].get("command", "")].append(event)
    for command, group in by_command.items():
        results = outcomes.get(command)
        if results is None or len(results) != len(group):
            continue
        for event, is_error in zip(group, results):
            if event["data"].get("exit_code") is None:
                event["data"]["exit_code"] = 1 if is_error else 0


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
            "reconstructed": bool(data.get("reconstructed")),
        })
    return commands


def _build_tests(events):
    tests = []
    for event in events:
        if event["type"] != "shell_command":
            continue
        data = event["data"]
        detected = testdetect.detect_test_run(
            data.get("command", ""), data.get("response", ""), event["ts"], data.get("exit_code")
        )
        if detected:
            detected["reconstructed"] = bool(data.get("reconstructed"))
            tests.append(detected)
    return tests


def _build_deployments(events):
    deployments = []
    for event in events:
        if event["type"] != "shell_command":
            continue
        data = event["data"]
        classified = deploydetect.classify_deploy(data.get("command", ""))
        if not classified:
            continue
        exit_code = data.get("exit_code")
        deployments.append({
            "ts": event["ts"],
            "command": data.get("command", ""),
            "method": classified["method"],
            "target_env": classified["target_env"],
            "exit_code": exit_code,
            "verified": False if exit_code not in (0, None) else None,
            "reconstructed": bool(data.get("reconstructed")),
        })
    return deployments


def _verify_deployments(episode):
    negatives = {"wrong", "too_broad", "needed_human_rescue"}
    negative_after = [
        item["ts"] for item in episode["human_feedback"]
        if item.get("source") == "mined" and item.get("label") in negatives and item.get("ts")
    ]
    for deployment in episode["deployments"]:
        if deployment["verified"] is False:
            continue
        if any(ts > deployment["ts"] for ts in negative_after):
            deployment["verified"] = False


def _relativize(path, base):
    if base and path and path.startswith(base):
        try:
            return os.path.relpath(path, base)
        except ValueError:
            return path
    return path


def _touched_files(repo_state, cwd, events):
    base = repo_state.get("root") or cwd
    touched = {}
    for event in events:
        if event["type"] in DIFF_STATUS_BY_TYPE:
            path = event["data"].get("file_path")
            if path:
                touched[_relativize(path, base)] = DIFF_STATUS_BY_TYPE[event["type"]]
    return touched


def _event_diffs(repo_state, cwd, events):
    return [
        {"file": path, "status": status, "additions": 0, "deletions": 0, "unified": None}
        for path, status in sorted(_touched_files(repo_state, cwd, events).items())
    ]


def _build_diffs(repo_state, cwd, events, live=True):
    root = repo_state.get("root")
    base_commit = repo_state.get("base_commit")
    git_ok = bool(root and base_commit and gitinfo.git_available(root))
    if live and git_ok and gitinfo.head_commit(root) == base_commit:
        touched = _touched_files(repo_state, cwd, events)
        parsed = diffparse.parse_unified_diff(gitinfo.working_diff(root, base_commit))
        scoped = [entry for entry in parsed if entry.get("file") in touched]
        if scoped:
            return scoped, "git-working-tree"
        return _event_diffs(repo_state, cwd, events), "events"
    return _event_diffs(repo_state, cwd, events), "events-untrusted" if git_ok else "events"


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
        if kind in ("file_edit", "file_write", "file_delete", "file_read", "shell_command", "tool_post"):
            stats["tool_calls"] += 1
        if kind == "file_read":
            stats["file_reads"] += 1
        if kind in ("file_edit", "file_write", "file_delete"):
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


def _apply_mined_signal(episode, generate):
    from . import feedback as feedback_mod
    try:
        mined = feedback_mod.mine(episode, generate)
    except Exception:
        return
    existing = {(item.get("label"), item.get("evidence_step_index")) for item in episode["human_feedback"]}
    for item in mined["feedback"]:
        key = (item["label"], item.get("evidence_step_index"))
        if key not in existing:
            episode["human_feedback"].append(item)
            existing.add(key)
    if mined["outcome_hint"]:
        episode["outcome_hint"] = mined["outcome_hint"]


def build_episode(session, generate=None):
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
    events = _reconstruct_failed_commands(events)
    _apply_transcript_exit_codes(events)
    episode["steps"] = _build_steps(events)
    episode["commands"] = _build_commands(events)
    episode["tests"] = _build_tests(events)
    episode["diffs"], episode["diff_source"] = _build_diffs(
        repo_state, cwd, events, live=not meta.get("imported")
    )
    episode["human_feedback"] = meta.get("human_feedback", [])
    episode["deployments"] = _build_deployments(events)
    episode["outcome"] = meta.get("outcome") or episode["outcome"]
    if generate is not None:
        _apply_mined_signal(episode, generate)
    _verify_deployments(episode)
    episode["stats"] = _build_stats(events, meta)
    episode["stats"]["tests_run"] = len(episode["tests"])
    labels = set(meta.get("labels", []) + [f["label"] for f in episode["human_feedback"]])
    if any(f.get("source") == "mined" for f in episode["human_feedback"]):
        labels.add("mined_feedback")
    if reward.terminal_test_signal(episode["tests"])[2]:
        labels.add("blocked_on_env")
    episode["labels"] = sorted(labels)
    episode["reward_vector"] = reward.reward_vector(episode)
    episode["validity"] = validity.assess(episode)
    return episode
