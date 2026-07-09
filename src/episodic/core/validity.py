from . import reward

OVERLY_STRICT = "overly_strict_tests"
UNDERSPECIFIED = "underspecified_intent"
LOW_COVERAGE = "low_coverage_tests"
MISLEADING = "misleading_intent"
CONTRADICTION = "reward_contradiction"

CATEGORIES = (OVERLY_STRICT, UNDERSPECIFIED, LOW_COVERAGE, MISLEADING, CONTRADICTION)
_TRUST_ORDER = {"low": 0, "medium": 1, "high": 2}
_SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2}
POSITIVE_LABELS = {"accepted_as_is", "accepted_after_edits", "useful"}
NEGATIVE_LABELS = {"wrong", "needed_human_rescue", "too_broad"}
INTENT_MIN_CHARS = 40


def _rv(episode):
    return episode.get("reward_vector") or {}


def _composite(episode):
    return _rv(episode).get("composite") or 0.0


def _human_label(episode):
    return _rv(episode).get("human_label") or 0.0


def _tests(episode):
    return [test for test in (episode.get("tests") or []) if test.get("total", 0) > 0]


def _has_passing_test(episode):
    return any(test.get("ok") for test in _tests(episode))


def _has_failing_test(episode):
    return any(not test.get("ok") for test in _tests(episode))


def _labels(episode):
    return {item.get("label") for item in episode.get("human_feedback", [])}


def _positive_human(episode):
    if _labels(episode) & POSITIVE_LABELS:
        return True
    if (episode.get("outcome_hint") or {}).get("success") == "yes":
        return True
    if episode.get("outcome", {}).get("status") in ("merged", "accepted"):
        return True
    return _human_label(episode) > 0.1


def _negative_human(episode):
    if _labels(episode) & NEGATIVE_LABELS:
        return True
    if (episode.get("outcome_hint") or {}).get("success") == "no":
        return True
    if episode.get("outcome", {}).get("status") in ("reverted", "failed"):
        return True
    return _human_label(episode) < -0.1


def _verified_deploy(episode):
    return any(d.get("verified") is True for d in episode.get("deployments", []))


def _failed_deploy(episode):
    return any(d.get("verified") is False for d in episode.get("deployments", []))


def _flag(code, category, severity, evidence):
    return {"code": code, "category": category, "severity": severity, "evidence": evidence}


def flag_episode(episode):
    flags = []

    if _has_passing_test(episode) and _negative_human(episode):
        flags.append(_flag("test_reward_false_positive", LOW_COVERAGE, "high",
                           "tests pass but the human/outcome signal is negative"))

    if _has_failing_test(episode) and _positive_human(episode):
        flags.append(_flag("test_reward_false_negative", OVERLY_STRICT, "high",
                           "tests fail but the human/outcome signal is positive"))

    if _positive_human(episode) and _failed_deploy(episode):
        flags.append(_flag("outcome_contradiction", CONTRADICTION, "high",
                           "positive human signal but a prod deploy was not verified"))
    if _negative_human(episode) and _verified_deploy(episode):
        flags.append(_flag("outcome_contradiction", CONTRADICTION, "high",
                           "negative human signal but a prod deploy was verified"))

    if reward.terminal_test_signal(episode.get("tests") or [])[2]:
        flags.append(_flag("env_blocked", LOW_COVERAGE, "medium",
                           "final test run is all environment/collection errors"))

    if (_composite(episode) >= 0.5 and not _tests(episode)
            and not _verified_deploy(episode) and not _positive_human(episode)):
        flags.append(_flag("unverified_success", LOW_COVERAGE, "medium",
                           "high reward with no tests, no verified deploy, no positive human signal"))

    diffs = episode.get("diffs") or []
    edits_source = any(not (d.get("file") or "").endswith((".md", ".txt", ".rst")) for d in diffs)
    touches_tests = any("test" in (d.get("file") or "").lower() for d in diffs)
    if (edits_source and not touches_tests and not _tests(episode) and _composite(episode) >= 0.5):
        flags.append(_flag("low_coverage_proxy", LOW_COVERAGE, "medium",
                           "source edited with no test changes and no executed tests, yet reward is high"))

    intent = (episode.get("intent") or "").strip()
    total_steps = len(episode.get("steps") or [])
    rescued = bool(_labels(episode) & {"needed_human_rescue", "wrong"})
    if len(intent) < INTENT_MIN_CHARS:
        severity = "high" if rescued else "medium"
        flags.append(_flag("underspecified_intent", UNDERSPECIFIED, severity,
                           "intent is very short/vague" + (" and required a correction" if rescued else "")))

    for item in episode.get("human_feedback", []):
        index = item.get("evidence_step_index")
        if item.get("label") == "wrong" and isinstance(index, int) and index <= max(1, total_steps // 3):
            flags.append(_flag("misleading_early_correction", MISLEADING, "medium",
                               "an early 'wrong' correction suggests the intent pointed the wrong way"))
            break

    return flags


def trust_tier(flags):
    if any(flag["severity"] == "high" for flag in flags):
        return "low"
    if any(flag["severity"] == "medium" for flag in flags):
        return "medium"
    return "high"


def _worst_severity(flags):
    if not flags:
        return None
    return max((flag["severity"] for flag in flags), key=lambda level: _SEVERITY_RANK[level])


def _min_trust(a, b):
    return a if _TRUST_ORDER[a] <= _TRUST_ORDER[b] else b


def _render(episode):
    tests = _tests(episode)
    test_line = "; ".join(
        f"{t.get('framework')} ok={t.get('ok')} passed={t.get('passed')} failed={t.get('failed')} errors={t.get('errors')}"
        for t in tests) or "none executed"
    feedback = "; ".join(
        f"{f.get('label')}({f.get('confidence')})" for f in episode.get("human_feedback", [])) or "none"
    diffs = ", ".join((d.get("file") or "?") for d in (episode.get("diffs") or [])[:12]) or "none"
    hint = episode.get("outcome_hint") or {}
    rv = _rv(episode)
    return (
        f"INTENT: {(episode.get('intent') or '')[:300]}\n"
        f"STEPS: {len(episode.get('steps') or [])}  DIFFS: {diffs}\n"
        f"TESTS: {test_line}\n"
        f"HUMAN_FEEDBACK: {feedback}\n"
        f"OUTCOME: status={episode.get('outcome', {}).get('status')} hint={hint.get('success')} "
        f"deploys={[d.get('verified') for d in episode.get('deployments', [])]}\n"
        f"RECORDED_REWARD: composite={rv.get('composite')} test_pass={rv.get('test_pass')} "
        f"human_label={rv.get('human_label')} outcome_src={(rv.get('components') or {}).get('outcome_source')}"
    )


def build_prompt(episode):
    return (
        "You are auditing whether a coding-agent episode's RECORDED REWARD reflects whether the task "
        "was truly and completely solved. First form your own independent judgment from the evidence "
        "below; do NOT assume the recorded reward is correct. Then flag any reward-quality issues.\n"
        f"Categories (multi-label allowed): {', '.join(CATEGORIES)}.\n"
        "- overly_strict_tests: tests reject a functionally-correct solution.\n"
        "- underspecified_intent: the intent omits requirements needed to judge success.\n"
        "- low_coverage_tests: success is not actually verified by the tests/evidence.\n"
        "- misleading_intent: the intent points at the wrong behavior.\n"
        "- reward_contradiction: independent signals (tests vs human/outcome) disagree.\n"
        "Reply with ONLY a JSON object:\n"
        '{"trustworthy":<bool>,"categories":[<category>...],"severity":"low|medium|high",'
        '"confidence":<0..1>,"rationale":"<short>"}\n\n'
        f"EPISODE:\n{_render(episode)}"
    )


def validate(episode, generate, passes=3):
    from .feedback import _extract_json

    verdicts = []
    for _ in range(passes):
        data = _extract_json(generate(build_prompt(episode)))
        if isinstance(data, dict):
            verdicts.append(data)
    if not verdicts:
        return None

    broken_votes = sum(1 for v in verdicts if v.get("trustworthy") is False)
    majority = len(verdicts) // 2 + 1
    trustworthy = broken_votes < majority

    category_votes = {}
    for verdict in verdicts:
        for category in verdict.get("categories") or []:
            if category in CATEGORIES:
                category_votes[category] = category_votes.get(category, 0) + 1
    categories = sorted(c for c, n in category_votes.items() if n >= majority)

    severity = "high" if not trustworthy else "medium"
    confidences = [v.get("confidence") for v in verdicts if isinstance(v.get("confidence"), (int, float))]
    confidence = round(sum(confidences) / len(confidences), 3) if confidences else None
    agreement = round(max(broken_votes, len(verdicts) - broken_votes) / len(verdicts), 3)
    rationale = next((v.get("rationale") for v in verdicts if v.get("rationale")), "")

    return {
        "trustworthy": trustworthy,
        "categories": categories,
        "severity": severity,
        "confidence": confidence,
        "agreement": agreement,
        "escalate": agreement < 1.0,
        "rationale": (rationale or "")[:300],
        "passes": len(verdicts),
        "source": "mined",
    }


def assess(episode, generate=None, passes=3):
    flags = flag_episode(episode)
    trust = trust_tier(flags)
    categories = set(flag["category"] for flag in flags)
    severity = _worst_severity(flags)
    source = "rules"
    llm = None

    if generate is not None:
        llm = validate(episode, generate, passes=passes)
        if llm is not None:
            source = "rules+llm"
            categories.update(llm["categories"])
            if not llm["trustworthy"]:
                trust = _min_trust(trust, "low")
                severity = "high"
            elif llm["categories"]:
                trust = _min_trust(trust, "medium")

    result = {
        "trust": trust,
        "flags": flags,
        "categories": sorted(categories),
        "severity": severity,
        "source": source,
    }
    if llm is not None:
        result["llm"] = llm
    return result
