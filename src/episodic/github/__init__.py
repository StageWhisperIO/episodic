import json
import shutil

from episodic.schema import default_outcome, now_iso
from episodic.core import gitinfo


def gh_available():
    return shutil.which("gh") is not None


def _aggregate_ci(checks):
    if not checks:
        return None
    failure_states = {"FAILURE", "ERROR", "CANCELLED"}
    conclusions = {c.get("conclusion") or c.get("state") or "" for c in checks}
    if conclusions & failure_states:
        return "failure"
    if all(s in {"SUCCESS"} for s in conclusions):
        return "success"
    return "pending"


def outcome_from_pr_json(pr_json, episode):
    repo_state = episode.get("repo_state", {})
    outcome = default_outcome()

    merged = pr_json.get("merged") or bool(pr_json.get("mergedAt"))
    state = pr_json.get("state", "")

    if merged:
        outcome["status"] = "merged"
        outcome["merged"] = True
    elif state == "OPEN":
        outcome["status"] = "open"
    elif state == "CLOSED":
        outcome["status"] = "abandoned"

    outcome["pr_url"] = pr_json.get("url")
    outcome["pr_number"] = pr_json.get("number")
    outcome["pr_state"] = state
    outcome["branch"] = pr_json.get("headRefName") or repo_state.get("branch")
    outcome["commit"] = pr_json.get("headRefOid") or repo_state.get("base_commit")
    outcome["ci_status"] = _aggregate_ci(pr_json.get("statusCheckRollup") or [])
    outcome["review_decision"] = pr_json.get("reviewDecision")
    outcome["linked_at"] = now_iso()
    outcome["reverted"] = False
    outcome["manual_edits_after_agent"] = False

    return outcome


def fetch_pr(arg, cwd):
    fields = "state,mergedAt,merged,number,url,headRefName,headRefOid,statusCheckRollup,reviewDecision"
    cmd = ["gh", "pr", "view"]
    if arg:
        cmd.append(arg)
    cmd += ["--json", fields]
    raw = gitinfo._run(cmd, cwd)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def should_refresh(outcome):
    if not (outcome.get("pr_url") or outcome.get("pr_number")):
        return False
    if outcome.get("status") in {"abandoned", "reverted"}:
        return False
    return (
        outcome.get("status") in {"open", "accepted", None}
        or outcome.get("pr_state") in {"OPEN", None}
        or outcome.get("ci_status") in {"pending", None}
    )


def refresh_outcome(episode, cwd=None):
    outcome = episode.get("outcome") or {}
    ref = outcome.get("pr_url") or outcome.get("pr_number")
    if not ref or not gh_available():
        return None
    repo_state = episode.get("repo_state", {})
    effective_cwd = cwd or repo_state.get("root") or "."
    pr_json = fetch_pr(str(ref), effective_cwd)
    if not pr_json:
        return None
    new_outcome = outcome_from_pr_json(pr_json, episode)
    for key in ("reverted", "manual_edits_after_agent", "caused_regression", "regression_commits"):
        new_outcome[key] = outcome.get(key, new_outcome[key])
    return new_outcome


def link_episode(episode, pr=None, auto=False, cwd=None):
    repo_state = episode.get("repo_state", {})
    effective_cwd = cwd or repo_state.get("root") or "."

    outcome = None
    if gh_available() and (pr or auto):
        pr_json = fetch_pr(pr or "", effective_cwd)
        if pr_json:
            outcome = outcome_from_pr_json(pr_json, episode)

    if outcome is None:
        outcome = default_outcome()
        outcome["commit"] = repo_state.get("base_commit")
        outcome["branch"] = repo_state.get("branch")
        outcome["linked_at"] = now_iso()
        outcome["status"] = "open"

    base = repo_state.get("base_commit")
    if base and gitinfo.git_available(effective_cwd):
        extra_raw = gitinfo._run(["git", "rev-list", f"{base}..HEAD", "--count"], effective_cwd)
        try:
            extra = int(extra_raw or 0)
        except ValueError:
            extra = 0
        outcome["manual_edits_after_agent"] = extra > 0
        revert_log = gitinfo._run(
            ["git", "log", "--grep=revert", "-i", "--oneline", f"{base}..HEAD"],
            effective_cwd,
        )
        outcome["reverted"] = bool(revert_log)

    return outcome
