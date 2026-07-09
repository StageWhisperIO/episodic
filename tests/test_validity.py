from episodic.core import validity
from episodic.core.episode import build_episode
from episodic.exporters import is_good, is_trusted
from episodic.schema import new_episode, new_event, validate_episode


def _ep(**over):
    ep = new_episode(id="ep_v", intent=over.get("intent", "make the authentication flow reject expired tokens correctly"))
    ep["reward_vector"]["composite"] = over.get("composite", 0.5)
    ep["reward_vector"]["human_label"] = over.get("human_label", 0.0)
    ep["tests"] = over.get("tests", [])
    ep["human_feedback"] = over.get("human_feedback", [])
    ep["diffs"] = over.get("diffs", [])
    ep["deployments"] = over.get("deployments", [])
    ep["steps"] = over.get("steps", [{"index": i, "type": "shell_command"} for i in range(6)])
    ep["outcome_hint"] = over.get("outcome_hint")
    return ep


def _test(ok, passed=1, failed=0, errors=0):
    return {"framework": "pytest", "ok": ok, "passed": passed, "failed": failed, "errors": errors,
            "total": passed + failed + errors}


def test_test_reward_false_positive_low_coverage():
    ep = _ep(tests=[_test(True)], human_feedback=[{"label": "wrong"}])
    flags = validity.flag_episode(ep)
    codes = {f["code"] for f in flags}
    assert "test_reward_false_positive" in codes
    assert validity.trust_tier(flags) == "low"


def test_test_reward_false_negative_overly_strict():
    ep = _ep(tests=[_test(False, passed=0, failed=2)], human_feedback=[{"label": "accepted_as_is"}])
    flags = validity.flag_episode(ep)
    assert any(f["category"] == validity.OVERLY_STRICT for f in flags)
    assert validity.trust_tier(flags) == "low"


def test_outcome_contradiction_verified_deploy_vs_negative():
    ep = _ep(deployments=[{"target_env": "prod", "verified": True}],
             human_feedback=[{"label": "needed_human_rescue"}])
    flags = validity.flag_episode(ep)
    assert any(f["category"] == validity.CONTRADICTION for f in flags)


def test_unverified_success_medium():
    ep = _ep(composite=0.7, tests=[], human_feedback=[], diffs=[])
    codes = {f["code"] for f in validity.flag_episode(ep)}
    assert "unverified_success" in codes


def test_underspecified_intent():
    ep = _ep(intent="fix it", tests=[], composite=0.4)
    assert any(f["category"] == validity.UNDERSPECIFIED for f in validity.flag_episode(ep))


def test_clean_episode_is_high_trust():
    ep = _ep(tests=[_test(True)], human_feedback=[{"label": "accepted_as_is"}],
             diffs=[{"file": "auth.py"}, {"file": "test_auth.py"}])
    result = validity.assess(ep)
    assert result["trust"] == "high"
    assert result["flags"] == []


def test_low_trust_excluded_from_training_set():
    bad = _ep(tests=[_test(True)], human_feedback=[{"label": "wrong"}], composite=0.8)
    bad["validity"] = validity.assess(bad)
    assert bad["validity"]["trust"] == "low"
    assert is_trusted(bad) is False
    assert is_good(bad) is False


def test_llm_validate_conservative_majority():
    calls = iter([
        '{"trustworthy":false,"categories":["low_coverage_tests"],"severity":"high","confidence":0.8,"rationale":"tests do not cover it"}',
        '{"trustworthy":false,"categories":["low_coverage_tests"],"severity":"high","confidence":0.7,"rationale":"x"}',
        '{"trustworthy":true,"categories":[],"severity":"low","confidence":0.4,"rationale":"maybe fine"}',
    ])
    ep = _ep(tests=[_test(True)])
    result = validity.validate(ep, lambda prompt: next(calls), passes=3)
    assert result["trustworthy"] is False
    assert "low_coverage_tests" in result["categories"]
    assert result["escalate"] is True


def test_assess_merges_llm_and_persists_via_build():
    session = {
        "id": "sv", "meta": {"cwd": "/tmp/none"},
        "events": [
            new_event("sv", "user_prompt", data={"prompt": "add token expiry validation to the auth guard"}),
            new_event("sv", "file_edit", data={"file_path": "/tmp/none/auth.py"}),
        ],
    }
    episode = build_episode(session)
    assert episode["validity"]["trust"] in ("high", "medium", "low")
    assert validate_episode(episode) == []
