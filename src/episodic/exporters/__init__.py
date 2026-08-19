import json
import re
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

FORMATS = ("sft", "dpo", "reward", "rlds", "wm", "jsonl", "parquet", "harbor", "molt")

STDOUT = "-"

_GOOD_LABELS = {"useful", "accepted_as_is", "accepted_after_edits"}
_BAD_LABELS = {"wrong"}
_GOOD_STATUSES = {"accepted", "merged"}
_BAD_STATUSES = {"failed", "reverted"}


def _out_path(out_dir, filename):
    return STDOUT if out_dir == STDOUT else out_dir / filename


def write_jsonl(path, rows):
    if path == STDOUT:
        for row in rows:
            sys.stdout.write(json.dumps(row, ensure_ascii=False) + "\n")
        sys.stdout.flush()
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


SFT_HISTORY_BUDGET = 4000
_ACTION_STEP_TYPES = {"file_read", "file_edit", "file_write", "file_delete", "shell_command", "tool_post"}


def _action_line(step):
    if step.get("type") == "user_prompt":
        prompt = (step.get("input") or {}).get("prompt", "")
        return f"ACTION user_prompt({prompt[:120]})"
    tool = step.get("tool") or step.get("type") or "unknown"
    compact = json.dumps(step.get("input") or {}, ensure_ascii=False)[:120]
    cwd = step.get("cwd")
    location = f" @ {cwd}" if cwd else ""
    return f"ACTION {tool}({compact}){location}"


def _obs_line(step):
    return f"OBS: {(step.get('observation') or '')[:200]}"


def trajectory_text(ep):
    parts = [f"USER: {ep.get('intent', '')}"]
    for step in ep.get("steps", []):
        parts.append(_action_line(step))
        parts.append(_obs_line(step))
    return "\n".join(parts)


SFT_INTENT_BUDGET = 600


def segment_episode(ep, history_budget=SFT_HISTORY_BUDGET):
    intent = (ep.get("intent") or "")[:SFT_INTENT_BUDGET]
    reward = (ep.get("reward_vector") or {}).get("composite")
    examples = []
    history = []
    for step in ep.get("steps", []):
        if step.get("type") in _ACTION_STEP_TYPES:
            context = "\n".join(history)[-history_budget:]
            user_content = f"USER: {intent}" + (f"\n{context}" if context else "")
            examples.append({
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": _action_line(step)},
                ],
                "meta": {"episode_id": ep["id"], "reward": reward, "step_index": step.get("index")},
            })
            history.append(_action_line(step))
            history.append(_obs_line(step))
        else:
            history.append(_action_line(step))
    return examples


def is_trusted(ep):
    return ((ep.get("validity") or {}).get("trust") or "high") != "low"


def is_good(ep):
    if not is_trusted(ep):
        return False
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
    path = _out_path(out_dir, "episodes.jsonl")
    write_jsonl(path, episodes)
    return {"files": [str(path)], "count": len(episodes)}


def _export_sft(episodes, out_dir):
    good = [ep for ep in episodes if is_good(ep)]
    rows = []
    for ep in good:
        rows.extend(segment_episode(ep))
    path = _out_path(out_dir, "sft.jsonl")
    write_jsonl(path, rows)
    return {"files": [str(path)], "count": len(rows), "episodes": len(good)}


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

    path = _out_path(out_dir, "dpo.jsonl")
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
    path = _out_path(out_dir, "reward.jsonl")
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
    path = _out_path(out_dir, "rlds.jsonl")
    write_jsonl(path, rows)
    return {"files": [str(path)], "count": len(rows)}


def _export_wm(episodes, out_dir):
    from episodic.worldmodel import wm_samples, to_messages

    rows = [to_messages(sample) for sample in wm_samples(episodes)]
    path = _out_path(out_dir, "wm.jsonl")
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


HARBOR_INTENT_BUDGET = 4000
HARBOR_TIMEOUT_SEC = 120

_FRAMEWORK_IMAGES = {
    "pytest": "python:3.12-slim",
    "unittest": "python:3.12-slim",
    "jest": "node:20-slim",
    "mocha": "node:20-slim",
    "vitest": "node:20-slim",
    "go": "golang:1.22",
    "cargo": "rust:1-slim",
}

_GIT_REMOTE = re.compile(r"(?:https?://|ssh://|git@)[A-Za-z0-9@:/._~-]+\Z")


def _captured_verifier(ep):
    for test in ep.get("tests", []):
        command = test.get("command")
        if command and test.get("ok"):
            return {
                "command": command,
                "framework": test.get("framework") or "unknown",
                "passed": test.get("passed"),
                "failed": test.get("failed"),
                "total": test.get("total"),
            }
    for cmd in ep.get("commands", []):
        if cmd.get("is_test") and cmd.get("command") and cmd.get("exit_code") == 0:
            return {
                "command": cmd["command"],
                "framework": "unknown",
                "passed": None,
                "failed": None,
                "total": None,
            }
    return None


def harbor_skip_reason(ep):
    if not is_trusted(ep):
        return "low_trust"
    if is_bad(ep):
        return "bad_outcome"
    if _captured_verifier(ep) is None:
        return "no_verifier"
    return None


def _base_image(framework, command):
    if framework in _FRAMEWORK_IMAGES:
        return _FRAMEWORK_IMAGES[framework]
    first = (command or "").strip().split()[:1]
    head = first[0] if first else ""
    if head in ("npm", "yarn", "pnpm", "node", "npx"):
        return "node:20-slim"
    if head == "go":
        return "golang:1.22"
    if head == "cargo":
        return "rust:1-slim"
    return "python:3.12-slim"


def _safe_remote(url):
    return url if url and _GIT_REMOTE.fullmatch(url) else None


def _toml_escape(value):
    out = []
    for ch in value:
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\r":
            out.append("\\r")
        elif ord(ch) < 0x20:
            out.append("\\u%04x" % ord(ch))
        else:
            out.append(ch)
    return "".join(out)


def _toml_scalar(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_scalar(item) for item in value) + "]"
    return '"' + _toml_escape(str(value)) + '"'


def _toml_table(name, table):
    lines = [f"[{name}]"]
    for key, value in table.items():
        if value is None:
            continue
        lines.append(f"{key} = {_toml_scalar(value)}")
    return "\n".join(lines)


def _toml_document(preamble, tables):
    parts = [preamble] if preamble else []
    for name, table in tables:
        parts.append(_toml_table(name, table))
    return "\n\n".join(parts) + "\n"


def _task_toml(ep, verifier, portable):
    repo = ep.get("repo_state") or {}
    rv = ep.get("reward_vector") or {}
    preamble = f'# Harbor task minted by Episodic from episode {ep.get("id")}\nversion = "1.0"'
    tables = [
        ("task", {"instruction": (ep.get("intent") or "")[:HARBOR_INTENT_BUDGET]}),
        ("environment", {"os": "linux", "build": "Dockerfile"}),
        ("verifier", {
            "command": verifier["command"],
            "framework": verifier["framework"],
            "timeout_sec": HARBOR_TIMEOUT_SEC,
        }),
        ("metadata", {
            "episode_id": ep.get("id"),
            "agent": ep.get("agent"),
            "source": "episodic",
            "portable": portable,
            "composite_reward": rv.get("composite"),
            "outcome": (ep.get("outcome") or {}).get("status"),
            "branch": repo.get("branch"),
            "base_commit": repo.get("base_commit"),
            "remote_url": repo.get("remote_url"),
            "labels": ep.get("labels") or [],
        }),
    ]
    return _toml_document(preamble, tables)


def _dockerfile(ep, verifier):
    repo = ep.get("repo_state") or {}
    lines = [f"FROM {_base_image(verifier['framework'], verifier['command'])}", "WORKDIR /workspace"]
    remote = _safe_remote(repo.get("remote_url") or "")
    base_commit = repo.get("base_commit")
    if remote:
        lines.append(f"RUN git clone {shlex.quote(remote)} /workspace")
        if base_commit:
            lines.append(f"RUN git -C /workspace checkout {shlex.quote(str(base_commit))}")
    else:
        lines.append("# Mount the target repository at /workspace before running this task.")
    lines.append("COPY tests/run-tests.sh /workspace/run-tests.sh")
    lines.append("RUN chmod +x /workspace/run-tests.sh")
    return "\n".join(lines) + "\n"


def _test_script(verifier):
    return "#!/usr/bin/env bash\nset -euo pipefail\ncd /workspace\n" + verifier["command"] + "\n"


def _solution_patch(ep):
    from ..core import diffparse

    patch = diffparse.join_unified(d.get("unified") for d in ep.get("diffs", []))
    return patch or None


def _task_metadata(ep, verifier):
    return {
        "episode_id": ep.get("id"),
        "agent": ep.get("agent"),
        "intent": ep.get("intent"),
        "source": "episodic",
        "verifier": verifier,
        "reward_vector": ep.get("reward_vector"),
        "outcome": ep.get("outcome"),
        "repo_state": ep.get("repo_state"),
        "labels": ep.get("labels") or [],
        "human_feedback": ep.get("human_feedback") or [],
    }


def _write_task(task_dir, ep, verifier, portable):
    (task_dir / "tests").mkdir(parents=True, exist_ok=True)
    (task_dir / "task.toml").write_text(_task_toml(ep, verifier, portable), encoding="utf-8")
    (task_dir / "Dockerfile").write_text(_dockerfile(ep, verifier), encoding="utf-8")
    (task_dir / "tests" / "run-tests.sh").write_text(_test_script(verifier), encoding="utf-8")
    (task_dir / "metadata.json").write_text(
        json.dumps(_task_metadata(ep, verifier), indent=2, ensure_ascii=False), encoding="utf-8")
    patch = _solution_patch(ep)
    if patch is not None:
        (task_dir / "solution.patch").write_text(patch, encoding="utf-8")


def _harbor_readme(minted, skipped, non_portable):
    portability = (
        f"\n\n{len(non_portable)} of the minted verifiers reference absolute host paths "
        "(`portable = false` in task.toml); normalize those before running in the container."
        if non_portable else ""
    )
    return (
        "# Episodic-minted Harbor dataset\n\n"
        f"{len(minted)} task(s) minted from captured coding episodes; "
        f"{len(skipped)} episode(s) skipped (no real verifier, low trust, or bad outcome)."
        f"{portability}\n\n"
        "Each task carries the captured test command as its verifier and the recorded diff as "
        "`solution.patch`. Run with Harbor:\n\n"
        "```bash\nharbor run --dataset ./ --agent claude-code --model <model>\n```\n"
    )


def _write_dataset(out_dir, minted, skipped, non_portable):
    dataset_toml = _toml_document(
        "# Episodic-minted Harbor dataset",
        [("dataset", {
            "name": "episodic-minted",
            "version": "0.1.0",
            "source": "episodic",
            "task_count": len(minted),
            "created_at": datetime.now(timezone.utc).isoformat(),
        })],
    )
    (out_dir / "dataset.toml").write_text(dataset_toml, encoding="utf-8")
    (out_dir / "manifest.json").write_text(
        json.dumps({"minted": minted, "skipped": skipped, "task_count": len(minted),
                    "non_portable": non_portable}, indent=2),
        encoding="utf-8")
    (out_dir / "README.md").write_text(_harbor_readme(minted, skipped, non_portable), encoding="utf-8")


def _export_harbor(episodes, out_dir):
    from .. import paths
    from ..core import normalize

    tasks_root = out_dir / "tasks"
    minted, skipped, non_portable = [], [], []
    for ep in episodes:
        ep_id = ep.get("id") or ""
        reason = harbor_skip_reason(ep)
        if reason:
            skipped.append({"id": ep_id, "reason": reason})
            continue
        try:
            task_id = paths.safe_id(ep_id, "episode_id")
        except ValueError:
            skipped.append({"id": ep_id, "reason": "unsafe_id"})
            continue
        verifier = _captured_verifier(ep)
        command = normalize.relativize_command(verifier["command"], (ep.get("repo_state") or {}).get("root"))
        verifier = {**verifier, "command": command}
        portable = _command_is_portable(command)
        if not portable:
            non_portable.append(task_id)
        _write_task(tasks_root / task_id, ep, verifier, portable)
        minted.append(task_id)
    _write_dataset(out_dir, minted, skipped, non_portable)
    return {
        "files": [str(out_dir / "dataset.toml"), str(out_dir / "manifest.json")],
        "count": len(minted),
        "tasks": len(minted),
        "skipped": skipped,
        "non_portable": len(non_portable),
    }


_MOLT_PROMPT_SUFFIX = "\n\nRespond with a unified diff of your changes inside a ```diff code block."

_SAFE_REF = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*\Z")

_MOLT_ENV_TEMPLATE = '''import json
import shutil
import subprocess
import tempfile

import torch

from molt.agents import Env, Result, StepEnvRunner

CLONE_TIMEOUT = 120
TEST_TIMEOUT = 120


def _extract_patch(text):
    fence = "```"
    if fence in text:
        blocks = text.split(fence)
        for i in range(1, len(blocks), 2):
            body = blocks[i]
            newline = body.find("\\n")
            lang = body[:newline].strip().lower() if newline != -1 else ""
            content = body[newline + 1:] if newline != -1 else body
            if lang in ("diff", "patch") or content.lstrip().startswith("diff --git"):
                return content
    return text


def _safe_arg(value):
    return isinstance(value, str) and value != "" and not value.startswith("-") and "\\x00" not in value


def _run(args, cwd=None, timeout=60):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=timeout).returncode


def _grade(label, action_text):
    spec = json.loads(label) if label else {}
    remote = spec.get("remote_url")
    command = spec.get("test_command")
    if not _safe_arg(remote) or not command:
        return 0.0
    workspace = tempfile.mkdtemp(prefix="episodic_molt_")
    try:
        if _run(["git", "clone", "--quiet", remote, workspace], timeout=CLONE_TIMEOUT) != 0:
            return 0.0
        base = spec.get("base_commit")
        if _safe_arg(base) and _run(["git", "-C", workspace, "checkout", "--quiet", base], timeout=30) != 0:
            return 0.0
        patch = _extract_patch(action_text)
        if patch.strip():
            applied = subprocess.run(["git", "-C", workspace, "apply", "-"], input=patch,
                                     capture_output=True, text=True, timeout=30).returncode
            if applied != 0:
                return 0.0
        rc = subprocess.run(["bash", "-c", "set -o pipefail\\n" + command], cwd=workspace,
                            capture_output=True, text=True, timeout=TEST_TIMEOUT).returncode
        return 1.0 if rc == 0 else 0.0
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


class EpisodicEnv(Env):
    async def step(self, state) -> Result:
        reward = torch.tensor(_grade(state.get("label") or "", state.get("action_text", "")), dtype=torch.float32)
        return Result(reward=reward, info={"tests_pass": reward})


class AgentRunner(StepEnvRunner):
    def __init__(self):
        super().__init__(EpisodicEnv)
'''


_ABS_PATH_IN_CMD = re.compile(r"(?:^|\s)/")


def _command_is_portable(command):
    return not _ABS_PATH_IN_CMD.search(command or "")


def _clonable_remote(remote):
    return isinstance(remote, str) and remote.strip() != "" and not remote.startswith("-") and "\x00" not in remote


def molt_skip_reason(ep):
    if not is_trusted(ep):
        return "low_trust"
    if is_bad(ep):
        return "bad_outcome"
    if _captured_verifier(ep) is None:
        return "no_verifier"
    if not _clonable_remote((ep.get("repo_state") or {}).get("remote_url")):
        return "no_remote"
    return None


def _molt_verifier_spec(ep):
    from ..core import normalize

    verifier = _captured_verifier(ep)
    repo = ep.get("repo_state") or {}
    base = repo.get("base_commit")
    return {
        "remote_url": repo.get("remote_url"),
        "base_commit": base if base and _SAFE_REF.fullmatch(base) else None,
        "test_command": normalize.relativize_command(verifier["command"], repo.get("root")),
        "framework": verifier["framework"],
    }


def _molt_readme(minted, skipped, non_portable):
    portability = (
        f"\n\n{non_portable} of the minted verifiers reference absolute host paths (`portable: false` in "
        "prompts.jsonl); normalize those paths before running in a fresh clone."
        if non_portable else ""
    )
    return (
        "# Episodic-minted Molt RL bundle\n\n"
        f"{minted} prompt(s) minted from verified coding episodes; {skipped} episode(s) skipped."
        f"{portability}\n\n"
        "Each rollout clones `remote_url` at `base_commit`, applies the model's unified diff, and runs "
        "the captured test command (under `bash -c` with `set -o pipefail`) as the verifier "
        "(reward 1.0 on pass, else 0.0). Run with Molt:\n\n"
        "```bash\n"
        "python3 -m molt.cli.train_rl_ray \\\n"
        "  --actor.model_name_or_path /path/to/automodel \\\n"
        "  --data.prompt_dataset ./prompts.jsonl \\\n"
        "  --data.input_key input \\\n"
        "  --train.agent_path ./agents/episodic_env.py \\\n"
        "  --algo.advantage.estimator grpo \\\n"
        "  --ckpt.output_dir ./ckpt/rl\n"
        "```\n"
    )


def _export_molt(episodes, out_dir):
    from .. import paths

    rows, minted, skipped, non_portable = [], [], [], []
    for ep in episodes:
        ep_id = ep.get("id") or ""
        reason = molt_skip_reason(ep)
        if reason:
            skipped.append({"id": ep_id, "reason": reason})
            continue
        try:
            safe = paths.safe_id(ep_id, "episode_id")
        except ValueError:
            skipped.append({"id": ep_id, "reason": "unsafe_id"})
            continue
        spec = _molt_verifier_spec(ep)
        portable = _command_is_portable(spec["test_command"])
        if not portable:
            non_portable.append(safe)
        rows.append({
            "input": (ep.get("intent") or "") + _MOLT_PROMPT_SUFFIX,
            "label": json.dumps(spec, ensure_ascii=False),
            "episode_id": safe,
            "composite_reward": (ep.get("reward_vector") or {}).get("composite"),
            "portable": portable,
        })
        minted.append(safe)

    (out_dir / "agents").mkdir(parents=True, exist_ok=True)
    (out_dir / "agents" / "episodic_env.py").write_text(_MOLT_ENV_TEMPLATE, encoding="utf-8")
    write_jsonl(out_dir / "prompts.jsonl", rows)
    (out_dir / "README.md").write_text(_molt_readme(len(minted), len(skipped), len(non_portable)), encoding="utf-8")
    (out_dir / "manifest.json").write_text(
        json.dumps({"minted": minted, "skipped": skipped, "task_count": len(minted),
                    "non_portable": non_portable}, indent=2),
        encoding="utf-8")
    return {
        "files": [str(out_dir / "prompts.jsonl"), str(out_dir / "agents" / "episodic_env.py")],
        "count": len(minted),
        "tasks": len(minted),
        "skipped": skipped,
        "non_portable": len(non_portable),
    }


_EXPORTERS = {
    "jsonl": _export_jsonl,
    "sft": _export_sft,
    "dpo": _export_dpo,
    "reward": _export_reward,
    "rlds": _export_rlds,
    "wm": _export_wm,
    "parquet": _export_parquet,
    "harbor": _export_harbor,
    "molt": _export_molt,
}


def export(episodes, fmt, out_dir):
    if fmt not in _EXPORTERS:
        raise ValueError(f"Unknown format {fmt!r}. Choose from: {FORMATS}")
    if str(out_dir) == STDOUT:
        if fmt == "parquet":
            raise ValueError("parquet is a binary format and cannot be written to stdout; pass a real --out path")
        if fmt == "harbor":
            raise ValueError("harbor writes a task directory tree and cannot be written to stdout; pass a real --out path")
        if fmt == "molt":
            raise ValueError("molt writes a bundle directory and cannot be written to stdout; pass a real --out path")
        result = _EXPORTERS[fmt](episodes, STDOUT)
        result["format"] = fmt
        result["out_dir"] = STDOUT
        return result
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = _EXPORTERS[fmt](episodes, out_dir)
    result["format"] = fmt
    result["out_dir"] = str(out_dir)
    return result
