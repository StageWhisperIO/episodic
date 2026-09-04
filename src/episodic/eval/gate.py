import pathlib

from .. import replay
from ..replay import modelrun
from ..worldmodel.validate import _local_clone_episode, _oracle_diff_runner, _unified_diff
from . import editfmt

_MAX_FORMAT_FILES = 3
_MAX_FORMAT_DIFF_CHARS = 8000
_MIN_GRADIENT_HEADROOM = 0.05


def score_episode(episode, runner):
    local = _local_clone_episode(episode)
    replay.create_replay(local)
    replay_id = replay.replay_id_for(local)
    try:
        result = replay.run_replay(replay_id, "candidate", execute=True, runner=runner)
    finally:
        replay.cleanup_replay(replay_id)
    scores = result.get("scores") or {}
    tests = result.get("tests") or {}
    return {"total": scores.get("total"), "ok": bool(tests.get("ok")),
            "verifier_reverted": result.get("verifier_reverted") or []}


def graded_score(episode, runner):
    local = _local_clone_episode(episode)
    replay.create_replay(local)
    replay_id = replay.replay_id_for(local)
    try:
        result = replay.run_replay(replay_id, "candidate", execute=True, runner=runner)
    finally:
        replay.cleanup_replay(replay_id)
    scores = result.get("scores") or {}
    tests = result.get("tests") or {}
    passed = tests.get("passed") or 0
    failed = tests.get("failed") or 0
    errors = tests.get("errors") or 0
    denom = passed + failed + errors
    fraction = passed / denom if denom else (1.0 if tests.get("ok") else 0.0)
    return {"ok": bool(tests.get("ok")), "pass_fraction": fraction,
            "tests_pass": scores.get("tests_pass") or 0.0,
            "diff_overlap": scores.get("diff_overlap") or 0.0,
            "passed": passed, "failed": failed, "errors": errors}


def empty_pass_fraction(episode):
    return graded_score(episode, empty_runner)["pass_fraction"]


def graded_advantage(episode, runner, baseline_fraction=None):
    if baseline_fraction is None:
        baseline_fraction = empty_pass_fraction(episode)
    graded = graded_score(episode, runner)
    return {**graded, "baseline_fraction": baseline_fraction,
            "advantage": graded["pass_fraction"] - baseline_fraction}


def _variance(values):
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def _correlation(xs, ys):
    if len(xs) < 2:
        return 0.0
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    vx, vy = _variance(xs), _variance(ys)
    return cov / (vx * vy) ** 0.5 if vx > 0 and vy > 0 else 0.0


def reward_components_report(episodes, runner):
    rows = [{"id": episode["id"], **graded_score(episode, runner)} for episode in episodes]
    fractions = [row["pass_fraction"] for row in rows]
    overlaps = [row["diff_overlap"] for row in rows]
    return {"n": len(rows),
            "pass_fraction": {"mean": sum(fractions) / len(fractions) if fractions else 0.0,
                              "var": _variance(fractions)},
            "diff_overlap": {"mean": sum(overlaps) / len(overlaps) if overlaps else 0.0,
                             "var": _variance(overlaps)},
            "corr_fraction_overlap": _correlation(fractions, overlaps),
            "rows": rows}


def empty_runner(model, workspace, prompt_text):
    return "noop", 0


def broken_runner(unified_diff):
    def runner(model, workspace, prompt_text):
        modelrun.apply_diff(unified_diff, workspace)
        path = pathlib.Path(workspace) / "solution.py"
        path.write_text(path.read_text() + "\nraise RuntimeError('broken')\n")
        return "broke", 0
    return runner


def verify_gate(episode):
    diff = _unified_diff(episode)
    oracle = score_episode(episode, _oracle_diff_runner(diff))
    empty = score_episode(episode, empty_runner)
    broken = score_episode(episode, broken_runner(diff))
    oracle_total = oracle["total"] or 0
    clean = bool(oracle["ok"] and not empty["ok"] and not broken["ok"]
                 and oracle_total > (empty["total"] or 0)
                 and oracle_total > (broken["total"] or 0))
    return {"oracle": oracle, "empty": empty, "broken": broken, "clean": clean}


def gate_report(episodes):
    rows = [{"id": ep["id"], **verify_gate(ep)} for ep in episodes]
    clean = sum(1 for row in rows if row["clean"])
    return {"total": len(rows), "clean": clean, "all_clean": clean == len(rows), "rows": rows}


def certify_episode(episode):
    diff = _unified_diff(episode)
    if not diff.strip():
        return {"green_ok": False, "red_ok": None, "test_necessary": False, "reason": "no diff"}
    green = score_episode(episode, _oracle_diff_runner(diff))
    if not green["ok"]:
        return {"green_ok": False, "red_ok": None, "test_necessary": False, "reason": "diff does not pass"}
    red = score_episode(episode, empty_runner)
    necessary = bool(green["ok"] and not red["ok"])
    reason = "certified" if necessary else "test not necessary (passes without the diff)"
    return {"green_ok": green["ok"], "red_ok": red["ok"], "test_necessary": necessary, "reason": reason}


def certify_corpus(episodes):
    rows = []
    for episode in episodes:
        try:
            result = certify_episode(episode)
        except Exception as exc:
            result = {"green_ok": False, "red_ok": None, "test_necessary": False,
                      "reason": f"{type(exc).__name__}: {exc}"}
        rows.append({"id": episode["id"], **result})
    certified = [row for row in rows if row["test_necessary"]]
    return {"total": len(rows), "certified": len(certified),
            "certified_ids": [row["id"] for row in certified], "rows": rows}


def _format_reachable(episode, unified_diff, max_files=_MAX_FORMAT_FILES,
                       max_diff_chars=_MAX_FORMAT_DIFF_CHARS):
    files = editfmt._files_of(episode)
    if not files:
        return False, "no files"
    if len(files) > max_files:
        return False, f"{len(files)} files > {max_files}"
    if any(not path.endswith(".py") for path in files):
        return False, "touches non-.py file"
    if len(unified_diff) > max_diff_chars:
        return False, f"diff {len(unified_diff)} chars > {max_diff_chars}"
    return True, "ok"


def gradient_capable(episode, max_files=_MAX_FORMAT_FILES, max_diff_chars=_MAX_FORMAT_DIFF_CHARS,
                      headroom_threshold=_MIN_GRADIENT_HEADROOM):
    empty_shape = {"gold_pass_fraction": None, "empty_pass_fraction": None, "headroom": None,
                   "format_reachable": None, "format_reason": None}
    diff = _unified_diff(episode)
    if not diff.strip():
        return {"gradient_capable": False, "reason": "no diff", **empty_shape}

    reachable, format_reason = _format_reachable(episode, diff, max_files, max_diff_chars)

    try:
        oracle = graded_score(episode, _oracle_diff_runner(diff))
    except Exception as exc:
        return {"gradient_capable": False, "reason": f"oracle run failed: {type(exc).__name__}: {exc}",
                **{**empty_shape, "format_reachable": reachable, "format_reason": format_reason}}

    if oracle["pass_fraction"] == 0:
        return {"gradient_capable": False, "reason": "gold pass_fraction is 0 (harness can't reach green)",
                **{**empty_shape, "format_reachable": reachable, "format_reason": format_reason,
                   "gold_pass_fraction": 0.0}}

    try:
        empty = graded_score(episode, empty_runner)
    except Exception as exc:
        return {"gradient_capable": False, "reason": f"empty run failed: {type(exc).__name__}: {exc}",
                **{**empty_shape, "format_reachable": reachable, "format_reason": format_reason,
                   "gold_pass_fraction": oracle["pass_fraction"]}}

    headroom = oracle["pass_fraction"] - empty["pass_fraction"]
    not_discriminating = (not oracle["ok"]) or empty["ok"] or headroom <= headroom_threshold
    capable = bool(reachable and not not_discriminating)

    if not_discriminating:
        reason = "not test-necessary (near-zero oracle advantage)"
    elif not reachable:
        reason = f"not format-reachable ({format_reason})"
    else:
        reason = "gradient-capable"

    return {"gradient_capable": capable, "reason": reason, "format_reachable": reachable,
            "format_reason": format_reason, "gold_pass_fraction": oracle["pass_fraction"],
            "empty_pass_fraction": empty["pass_fraction"], "headroom": headroom}


def gradient_capable_report(episodes, **kwargs):
    rows = []
    for episode in episodes:
        try:
            result = gradient_capable(episode, **kwargs)
        except Exception as exc:
            result = {"gradient_capable": False, "reason": f"{type(exc).__name__}: {exc}",
                      "format_reachable": None, "format_reason": None,
                      "gold_pass_fraction": None, "empty_pass_fraction": None, "headroom": None}
        rows.append({"id": episode["id"], **result})
    capable = [row for row in rows if row["gradient_capable"]]
    return {"total": len(rows), "capable": len(capable),
            "capable_ids": [row["id"] for row in capable], "rows": rows}
