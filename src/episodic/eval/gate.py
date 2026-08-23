import pathlib

from .. import replay
from ..replay import modelrun
from ..worldmodel.validate import _local_clone_episode, _oracle_diff_runner, _unified_diff


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
