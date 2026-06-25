import hashlib

from episodic.schema import new_episode
from episodic.core import reward

DOMAINS = ("terminal", "swe", "search", "mcp", "web")

_TERMINAL_OBS = "$ {cmd}\n{output}\nuser@host:{cwd}$ "
_TEST_OK = "============================= test session starts =============================\ncollected {n} items\n\n{dots}\n\n============================== {n} passed in 0.1s =============================="
_TEST_FAIL = "============================= test session starts =============================\ncollected {n} items\n\n{marks}\n\n=================== {failed} failed, {passed} passed in 0.1s ==================="


def _frac(*parts):
    digest = hashlib.sha256(":".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0x100000000


def _ts(index):
    return f"2026-06-14T10:{index:02d}:00+00:00"


def make_step(index, type="file_edit", tool="Edit", input=None, observation="applied",
              approved=True, cwd=None, intent=None):
    return {
        "index": index,
        "ts": _ts(index),
        "type": type,
        "tool": tool,
        "intent": intent or f"step {index}",
        "input": input or {},
        "observation": observation,
        "approved": approved,
        "cwd": cwd,
        "duration_ms": None,
    }


def terminal_observation(command, output, cwd="/repo"):
    return _TERMINAL_OBS.format(cmd=command, output=output, cwd=cwd)


def make_test_observation(passed, failed=0):
    total = passed + failed
    if failed == 0:
        return _TEST_OK.format(n=total, dots="." * total)
    marks = "." * passed + "F" * failed
    return _TEST_FAIL.format(n=total, marks=marks, failed=failed, passed=passed)


def make_trajectory(episode_id, intent, turns, *, agent="claude-code", domain="swe",
                    source=None, repo_root="/repo", remote_url=None, base_commit="base000"):
    episode = new_episode(id=episode_id, agent=agent, intent=intent, created_at=_ts(0))
    episode["repo_state"].update({
        "root": repo_root,
        "repo": source or domain,
        "remote_url": remote_url,
        "base_commit": base_commit,
        "branch": "main",
    })
    episode["labels"] = [domain]
    steps, commands, tests = [], [], []
    for index, turn in enumerate(turns):
        action = turn["action"]
        observation = turn["observation"]
        step_type = turn.get("type", "shell_command" if action.get("command") else "file_edit")
        tool = turn.get("tool", "Bash" if action.get("command") else "Edit")
        steps.append(make_step(index, type=step_type, tool=tool, input=action,
                               observation=observation, cwd=repo_root, intent=turn.get("intent")))
        if action.get("command"):
            commands.append({
                "ts": _ts(index),
                "command": action["command"],
                "cwd": repo_root,
                "exit_code": turn.get("exit_code", 0),
                "output_excerpt": observation[:400],
                "is_test": turn.get("is_test", False),
            })
            if turn.get("is_test"):
                tests.append({
                    "ts": _ts(index),
                    "framework": turn.get("framework", "pytest"),
                    "command": action["command"],
                    "passed": turn.get("passed", 0),
                    "failed": turn.get("failed", 0),
                    "skipped": 0,
                    "total": turn.get("passed", 0) + turn.get("failed", 0),
                    "ok": turn.get("failed", 0) == 0 and turn.get("exit_code", 0) == 0,
                })
    episode["steps"] = steps
    episode["commands"] = commands
    episode["tests"] = tests
    return episode


def make_episode(episode_id, intent="add retry to http client", *, agent="claude-code",
                 domain="swe", source=None, outcome="merged", feedback=None,
                 passed=3, failed=0, files=("src/http.py",), additions=12, deletions=2,
                 remote_url=None, repo_root="/repo", base_commit="base000",
                 cost_usd=0.0, file_reads=2):
    test_cmd = "pytest -q"
    turns = [
        {"action": {"prompt": intent}, "observation": "", "type": "user_prompt", "tool": None,
         "intent": intent},
    ]
    for f in files:
        turns.append({"action": {"file_path": f}, "observation": f"applied edit to {f}",
                      "type": "file_edit", "tool": "Edit", "intent": f"edit {f}"})
    turns.append({
        "action": {"command": test_cmd}, "observation": make_test_observation(passed, failed),
        "type": "shell_command", "tool": "Bash", "is_test": True, "framework": "pytest",
        "passed": passed, "failed": failed, "exit_code": 0 if failed == 0 else 1,
        "intent": "run tests",
    })
    episode = make_trajectory(episode_id, intent, turns, agent=agent, domain=domain,
                              source=source, repo_root=repo_root, remote_url=remote_url,
                              base_commit=base_commit)
    episode["diffs"] = [{
        "file": f, "status": "modified",
        "additions": additions if i == 0 else 1,
        "deletions": deletions if i == 0 else 0,
        "unified": f"diff --git a/{f} b/{f}\n@@ -1 +1 @@\n-old\n+new\n",
    } for i, f in enumerate(files)]
    episode["outcome"]["status"] = outcome
    if outcome == "merged":
        episode["outcome"]["merged"] = True
    if outcome == "reverted":
        episode["outcome"]["reverted"] = True
    if feedback:
        episode["human_feedback"] = [{"ts": _ts(99), "label": label, "note": None}
                                     for label in feedback]
        episode["labels"] = sorted(set(episode["labels"]) | set(feedback))
    episode["stats"].update({
        "file_edits": len(files),
        "file_reads": file_reads,
        "shell_commands": 1,
        "tests_run": 1,
        "cost_usd": cost_usd,
    })
    episode["reward_vector"] = reward.reward_vector(episode)
    return episode


_OUTCOME_CYCLE = ("merged", "merged", "accepted", "abandoned", "reverted")
_SOURCE_CYCLE = ("repo-alpha", "repo-beta", "repo-gamma")


def make_population(n, *, seed=0, sources=None):
    sources = sources or _SOURCE_CYCLE
    episodes = []
    for i in range(n):
        outcome = _OUTCOME_CYCLE[i % len(_OUTCOME_CYCLE)]
        source = sources[i % len(sources)]
        good = outcome in ("merged", "accepted")
        episodes.append(make_episode(
            f"ep_{seed}_{i:03d}",
            intent=f"task {i % 7}: refactor module {i % 4}",
            domain=DOMAINS[i % len(DOMAINS)],
            source=source,
            outcome=outcome,
            feedback=["useful"] if good else (["wrong"] if outcome == "reverted" else None),
            passed=3 if good else 1,
            failed=0 if good else 2,
            files=("src/http.py",) if i % 2 == 0 else ("src/a.py", "src/b.py"),
            cost_usd=round(0.05 + _frac(seed, i) * 0.5, 4),
        ))
    return episodes


def populate_store(n, start=None, *, seed=0, sources=None):
    from episodic import store

    episodes = make_population(n, seed=seed, sources=sources)
    for episode in episodes:
        store.save_episode(episode, start=start)
    return episodes
