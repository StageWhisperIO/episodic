import uuid
from datetime import datetime, timezone

SCHEMA_VERSION = "0.1.0"

AGENTS = ("claude-code", "codex", "unknown")

EVENT_TYPES = (
    "session_start",
    "session_end",
    "user_prompt",
    "agent_message",
    "tool_pre",
    "tool_post",
    "file_read",
    "file_edit",
    "file_write",
    "shell_command",
    "test_run",
    "approval",
    "denial",
    "feedback",
    "outcome",
    "note",
)

FEEDBACK_LABELS = (
    "useful",
    "wrong",
    "too_broad",
    "too_slow",
    "needed_human_rescue",
    "accepted_as_is",
    "accepted_after_edits",
)

OUTCOME_STATUSES = ("open", "accepted", "merged", "failed", "reverted", "abandoned")


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def short_id():
    return uuid.uuid4().hex[:12]


def new_event(session_id, type, source="claude-code", tool_name=None, data=None, ts=None, id=None):
    return {
        "id": id or short_id(),
        "ts": ts or now_iso(),
        "session_id": session_id,
        "type": type,
        "source": source,
        "tool_name": tool_name,
        "data": data or {},
    }


def default_repo_state():
    return {
        "root": None,
        "repo": None,
        "remote_url": None,
        "branch": None,
        "base_commit": None,
        "dirty": False,
    }


def default_outcome():
    return {
        "status": "open",
        "commit": None,
        "branch": None,
        "pr_url": None,
        "pr_number": None,
        "pr_state": None,
        "ci_status": None,
        "review_decision": None,
        "merged": False,
        "reverted": False,
        "manual_edits_after_agent": False,
        "linked_at": None,
    }


def default_reward():
    return {
        "test_pass": 0.0,
        "human_label": 0.0,
        "outcome": 0.0,
        "cost_efficiency": 0.0,
        "edit_focus": 0.0,
        "composite": 0.0,
        "components": {},
    }


def default_stats():
    return {
        "started_at": None,
        "ended_at": None,
        "duration_ms": 0,
        "tool_calls": 0,
        "file_reads": 0,
        "file_edits": 0,
        "shell_commands": 0,
        "tests_run": 0,
        "approvals": 0,
        "denials": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
    }


def new_episode(id, agent="claude-code", intent="", repo_state=None, created_at=None):
    return {
        "id": id,
        "schema_version": SCHEMA_VERSION,
        "created_at": created_at or now_iso(),
        "agent": agent,
        "intent": intent,
        "repo_state": repo_state or default_repo_state(),
        "steps": [],
        "diffs": [],
        "commands": [],
        "tests": [],
        "human_feedback": [],
        "outcome": default_outcome(),
        "reward_vector": default_reward(),
        "stats": default_stats(),
        "labels": [],
    }


EPISODE_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "$id": "https://episodic.dev/schemas/coding-episode.json",
    "title": "CodingEpisode",
    "type": "object",
    "required": [
        "id",
        "schema_version",
        "created_at",
        "agent",
        "intent",
        "repo_state",
        "steps",
        "diffs",
        "commands",
        "tests",
        "human_feedback",
        "outcome",
        "reward_vector",
        "stats",
        "labels",
    ],
    "properties": {
        "id": {"type": "string"},
        "schema_version": {"type": "string"},
        "created_at": {"type": "string"},
        "agent": {"type": "string", "enum": list(AGENTS)},
        "intent": {"type": "string"},
        "repo_state": {
            "type": "object",
            "properties": {
                "root": {"type": ["string", "null"]},
                "repo": {"type": ["string", "null"]},
                "remote_url": {"type": ["string", "null"]},
                "branch": {"type": ["string", "null"]},
                "base_commit": {"type": ["string", "null"]},
                "dirty": {"type": "boolean"},
            },
        },
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["index", "ts", "type", "intent", "input", "observation"],
                "properties": {
                    "index": {"type": "integer"},
                    "ts": {"type": "string"},
                    "type": {"type": "string"},
                    "tool": {"type": ["string", "null"]},
                    "intent": {"type": "string"},
                    "input": {"type": "object"},
                    "observation": {"type": "string"},
                    "approved": {"type": ["boolean", "null"]},
                    "duration_ms": {"type": ["integer", "null"]},
                },
            },
        },
        "diffs": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["file", "status", "additions", "deletions"],
                "properties": {
                    "file": {"type": "string"},
                    "status": {"type": "string"},
                    "additions": {"type": "integer"},
                    "deletions": {"type": "integer"},
                    "unified": {"type": ["string", "null"]},
                },
            },
        },
        "commands": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["ts", "command", "is_test"],
                "properties": {
                    "ts": {"type": "string"},
                    "command": {"type": "string"},
                    "cwd": {"type": ["string", "null"]},
                    "exit_code": {"type": ["integer", "null"]},
                    "output_excerpt": {"type": "string"},
                    "is_test": {"type": "boolean"},
                },
            },
        },
        "tests": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["ts", "framework", "passed", "failed", "ok"],
                "properties": {
                    "ts": {"type": "string"},
                    "framework": {"type": "string"},
                    "command": {"type": "string"},
                    "passed": {"type": "integer"},
                    "failed": {"type": "integer"},
                    "skipped": {"type": "integer"},
                    "total": {"type": "integer"},
                    "ok": {"type": "boolean"},
                },
            },
        },
        "human_feedback": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["ts", "label"],
                "properties": {
                    "ts": {"type": "string"},
                    "label": {"type": "string", "enum": list(FEEDBACK_LABELS)},
                    "note": {"type": ["string", "null"]},
                },
            },
        },
        "outcome": {
            "type": "object",
            "required": ["status"],
            "properties": {
                "status": {"type": "string", "enum": list(OUTCOME_STATUSES)},
                "commit": {"type": ["string", "null"]},
                "branch": {"type": ["string", "null"]},
                "pr_url": {"type": ["string", "null"]},
                "pr_number": {"type": ["integer", "null"]},
                "pr_state": {"type": ["string", "null"]},
                "ci_status": {"type": ["string", "null"]},
                "review_decision": {"type": ["string", "null"]},
                "merged": {"type": "boolean"},
                "reverted": {"type": "boolean"},
                "manual_edits_after_agent": {"type": "boolean"},
                "linked_at": {"type": ["string", "null"]},
            },
        },
        "reward_vector": {
            "type": "object",
            "required": ["composite"],
            "properties": {
                "test_pass": {"type": "number"},
                "human_label": {"type": "number"},
                "outcome": {"type": "number"},
                "cost_efficiency": {"type": "number"},
                "edit_focus": {"type": "number"},
                "composite": {"type": "number"},
                "components": {"type": "object"},
            },
        },
        "stats": {"type": "object"},
        "labels": {"type": "array", "items": {"type": "string"}},
    },
}

_JSON_TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "null": type(None),
}


def _type_ok(value, declared):
    names = declared if isinstance(declared, list) else [declared]
    if "number" in names and isinstance(value, bool):
        return False
    if "integer" in names and isinstance(value, bool):
        return False
    return any(isinstance(value, _JSON_TYPES[name]) for name in names)


def _validate(instance, schema, path, errors):
    declared = schema.get("type")
    if declared is not None and not _type_ok(instance, declared):
        errors.append(f"{path or '<root>'}: expected {declared}, got {type(instance).__name__}")
        return
    enum = schema.get("enum")
    if enum is not None and instance not in enum:
        errors.append(f"{path or '<root>'}: {instance!r} not in {enum}")
    if isinstance(instance, dict):
        for key in schema.get("required", []):
            if key not in instance:
                errors.append(f"{path or '<root>'}: missing required '{key}'")
        for key, subschema in schema.get("properties", {}).items():
            if key in instance:
                _validate(instance[key], subschema, f"{path}.{key}" if path else key, errors)
    if isinstance(instance, list) and "items" in schema:
        for index, item in enumerate(instance):
            _validate(item, schema["items"], f"{path}[{index}]", errors)


def validate_episode(episode):
    errors = []
    _validate(episode, EPISODE_SCHEMA, "", errors)
    return errors
