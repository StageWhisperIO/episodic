import re

from . import reward, testdetect

HARD_PENALTY = 0.5
JUDGE_TRAJECTORY_LIMIT = 4000
SCOPED_FILES_TIGHT = 3
SCOPED_FILES_LOOSE = 8


def _steps(episode):
    return episode.get("steps", [])


def _first_index(steps, predicate):
    for step in steps:
        if predicate(step):
            return step.get("index", 0)
    return None


def _is_edit(step):
    return step.get("type") in ("file_edit", "file_write")


def _is_test_command(step):
    if step.get("type") != "shell_command":
        return False
    command = (step.get("input") or {}).get("command", "")
    return testdetect.classify_command(command) is not None


def check_has_test_evidence(episode):
    runs = [test for test in episode.get("tests", []) if test.get("total", 0) > 0]
    if runs:
        return 1.0, f"{len(runs)} executed test run(s)"
    return 0.0, "no executed tests"


def check_not_blocked_on_env(episode):
    _, has_tests, blocked = reward.terminal_test_signal(episode.get("tests", []))
    if not has_tests:
        return None, "no tests"
    return (0.0, "final run is all env errors") if blocked else (1.0, "not env-blocked")


def check_tests_end_green(episode):
    score, has_tests, _ = reward.terminal_test_signal(episode.get("tests", []))
    if not has_tests:
        return None, "no tests"
    return score, f"terminal test signal {score}"


def check_reproduce_before_fix(episode):
    steps = _steps(episode)
    first_edit = _first_index(steps, _is_edit)
    first_test = _first_index(steps, _is_test_command)
    if first_edit is None or first_test is None:
        return None, "no edit or no test step"
    if first_test < first_edit:
        return 1.0, "ran a test before the first edit"
    return 0.2, "edited before any test"


def check_explored_before_edit(episode):
    steps = _steps(episode)
    first_edit = _first_index(steps, _is_edit)
    if first_edit is None:
        return None, "no edits"
    reads_before = sum(1 for step in steps if step.get("type") == "file_read" and step.get("index", 0) < first_edit)
    if reads_before > 0:
        return 1.0, f"{reads_before} read(s) before first edit"
    return 0.3, "edited without reading first"


def check_change_is_scoped(episode):
    files = {diff.get("file") for diff in episode.get("diffs", []) if diff.get("file")}
    if not files:
        return None, "no diffs"
    count = len(files)
    if count <= SCOPED_FILES_TIGHT:
        return 1.0, f"{count} file(s) touched"
    if count <= SCOPED_FILES_LOOSE:
        return 0.6, f"{count} files touched"
    return 0.3, f"{count} files touched (broad)"


def check_low_denials(episode):
    denials = (episode.get("stats") or {}).get("denials", 0)
    if denials == 0:
        return 1.0, "no denied actions"
    return max(0.0, 1.0 - 0.25 * denials), f"{denials} denied action(s)"


CODING_RUBRIC = [
    {"id": "has_test_evidence", "kind": "hard", "weight": 2,
     "desc": "The trajectory actually executed tests.", "check": check_has_test_evidence},
    {"id": "not_blocked_on_env", "kind": "hard", "weight": 2,
     "desc": "The final test run was not defeated by environment/collection errors.", "check": check_not_blocked_on_env},
    {"id": "tests_end_green", "kind": "principle", "weight": 3,
     "desc": "Tests end passing (with credit for red-to-green progression).", "check": check_tests_end_green},
    {"id": "reproduce_before_fix", "kind": "principle", "weight": 1,
     "desc": "A failing/again test was run before editing code.", "check": check_reproduce_before_fix},
    {"id": "explored_before_edit", "kind": "principle", "weight": 1,
     "desc": "Relevant files were read before being changed.", "check": check_explored_before_edit},
    {"id": "change_is_scoped", "kind": "principle", "weight": 1,
     "desc": "The change stays focused on a small number of files.", "check": check_change_is_scoped},
    {"id": "low_denials", "kind": "principle", "weight": 1,
     "desc": "The agent did not accumulate denied actions.", "check": check_low_denials},
    {"id": "explanation_quality", "kind": "principle", "weight": 2,
     "desc": "The agent clearly explained the change and why it works.", "judge": True},
    {"id": "correct_beyond_tests", "kind": "principle", "weight": 2,
     "desc": "The change is correct and complete beyond what the tests assert.", "judge": True},
]


def _judge_prompt(episode, criterion):
    from ..exporters import trajectory_text

    trajectory = trajectory_text(episode)[:JUDGE_TRAJECTORY_LIMIT]
    return (
        "You are grading a coding agent trajectory against one rubric criterion.\n"
        f"Criterion: {criterion['desc']}\n"
        f"Trajectory:\n{trajectory}\n"
        "Reply with a line 'SCORE: <0..1>' followed by a one-line reason."
    )


def _parse_verdict(text):
    match = re.search(r"SCORE:\s*(1(?:\.0+)?|0(?:\.\d+)?)", text or "")
    score = float(match.group(1)) if match else 0.0
    return max(0.0, min(1.0, score)), (text or "").strip()[:200]


def openrubrics_judge(generate):
    def judge(episode, criterion):
        return _parse_verdict(generate(_judge_prompt(episode, criterion)))
    return judge


def score_episode(episode, rubric=CODING_RUBRIC, judge=None):
    criteria = []
    for item in rubric:
        if item.get("judge"):
            if judge is None:
                satisfied, reason = None, "requires a judge model"
            else:
                satisfied, reason = judge(episode, item)
        else:
            satisfied, reason = item["check"](episode)
        criteria.append({"id": item["id"], "kind": item["kind"], "weight": item["weight"],
                         "satisfied": satisfied, "reason": reason, "applicable": satisfied is not None})

    applicable = [row for row in criteria if row["applicable"]]
    total_weight = sum(row["weight"] for row in applicable)
    hard_violations = [row["id"] for row in applicable if row["kind"] == "hard" and row["satisfied"] < 1.0]

    if not total_weight:
        return {"score": None, "hard_pass": True, "hard_violations": [], "criteria": criteria, "applicable_weight": 0}

    base = sum(row["weight"] * row["satisfied"] for row in applicable) / total_weight
    score = base * HARD_PENALTY if hard_violations else base
    return {
        "score": round(score, 4),
        "base": round(base, 4),
        "hard_pass": not hard_violations,
        "hard_violations": hard_violations,
        "criteria": criteria,
        "applicable_weight": total_weight,
    }


def rubric_reward(episode):
    return score_episode(episode)["score"]
