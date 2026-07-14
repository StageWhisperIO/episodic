from episodic import exporters, loop
from episodic.schema import new_episode


def _test(ok, passed, failed=0, errors=0):
    return {"ts": "t", "framework": "pytest", "ok": ok, "passed": passed, "failed": failed,
            "errors": errors, "total": passed + failed + errors}


def test_ensure_validity_recomputes_stale_high_trust():
    ep = new_episode(id="ep_stale", intent="well specified task that quietly regressed after later feedback")
    ep["tests"] = [_test(True, 3)]
    ep["human_feedback"] = [{"ts": "t", "label": "wrong"}]
    ep["validity"] = {"trust": "high", "flags": [], "categories": [], "severity": None, "source": "rules"}
    loop.ensure_validity([ep])
    assert ep["validity"]["trust"] == "low"


def test_ensure_validity_preserves_llm_downgrade():
    ep = new_episode(id="ep_llm", intent="a perfectly clean and well specified task with green tests")
    ep["tests"] = [_test(True, 3)]
    ep["validity"] = {"trust": "high", "flags": [], "categories": [], "severity": None,
                      "source": "rules+llm", "llm": {"trustworthy": False, "categories": ["low_coverage_tests"]}}
    loop.ensure_validity([ep])
    assert ep["validity"]["trust"] == "low"
    assert ep["validity"]["llm"]["trustworthy"] is False


def test_ensure_validity_attaches_and_preflight_drops_low_trust():
    good = new_episode(id="ep_good", intent="clean well-specified task with passing behaviour")
    good["tests"] = [_test(True, 3)]
    bad = new_episode(id="ep_bad", intent="x")
    bad["tests"] = [_test(True, 3)]
    bad["human_feedback"] = [{"ts": "t", "label": "wrong"}]
    loop.ensure_validity([good, bad])
    assert good["validity"]["trust"] != "low"
    assert bad["validity"]["trust"] == "low"
    assert exporters.is_trusted(bad) is False
