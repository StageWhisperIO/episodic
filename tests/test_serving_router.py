import json

import pytest

from episodic.serving import BackendUnavailable
from episodic.serving.router import Router, resolve_served_ref
from episodic.serving.openai import OpenAIBackend


def test_route_defaults_to_distilled_tier():
    router = Router({"distilled": {"base_url": "http://d.local"}, "frontier": {"base_url": "http://f.local"}})
    tier, backend, _config = router.route({"messages": []})
    assert tier == "distilled"
    assert isinstance(backend, OpenAIBackend)
    assert backend.base_url == "http://d.local"


def test_route_honors_explicit_model_tier_request():
    router = Router({"distilled": {"base_url": "http://d.local"}, "frontier": {"base_url": "http://f.local"}})
    tier, backend, _config = router.route({"model": "frontier", "messages": []})
    assert tier == "frontier"
    assert backend.base_url == "http://f.local"


def test_route_escalates_on_explicit_flag():
    router = Router({"distilled": {"base_url": "http://d.local"}, "frontier": {"base_url": "http://f.local"}})
    tier, _backend, _config = router.route({"episodic_escalate": True, "messages": []})
    assert tier == "frontier"


def test_route_escalates_on_message_length_budget():
    router = Router({
        "distilled": {"base_url": "http://d.local"},
        "frontier": {"base_url": "http://f.local"},
        "escalate_chars": 10,
    })
    tier, _backend, _config = router.route({"messages": [{"role": "user", "content": "x" * 50}]})
    assert tier == "frontier"


def test_route_stays_distilled_under_message_length_budget():
    router = Router({
        "distilled": {"base_url": "http://d.local"},
        "frontier": {"base_url": "http://f.local"},
        "escalate_chars": 1000,
    })
    tier, _backend, _config = router.route({"messages": [{"role": "user", "content": "short"}]})
    assert tier == "distilled"


def test_route_uses_custom_escalate_hook():
    calls = []

    def escalate(payload, config):
        calls.append(payload)
        return payload.get("messages", [{}])[0].get("content") == "urgent"

    router = Router({
        "distilled": {"base_url": "http://d.local"},
        "frontier": {"base_url": "http://f.local"},
        "escalate": escalate,
    })

    tier, _backend, _config = router.route({"messages": [{"content": "urgent"}]})
    assert tier == "frontier"
    tier, _backend, _config = router.route({"messages": [{"content": "normal"}]})
    assert tier == "distilled"
    assert len(calls) == 2


def test_resolve_served_ref_reads_promoted_json(tmp_path):
    loop_out = tmp_path / "loopout"
    loop_out.mkdir()
    (loop_out / "promoted.json").write_text(
        json.dumps({"model_dir": "/models/candidate", "served_ref": "tinker://run/weights/ckpt"}),
        encoding="utf-8",
    )
    assert resolve_served_ref(out=str(loop_out)) == "tinker://run/weights/ckpt"


def test_resolve_served_ref_falls_back_to_model_dir(tmp_path):
    loop_out = tmp_path / "loopout"
    loop_out.mkdir()
    (loop_out / "promoted.json").write_text(json.dumps({"model_dir": "/models/candidate"}), encoding="utf-8")
    assert resolve_served_ref(out=str(loop_out)) == "/models/candidate"


def test_resolve_served_ref_returns_none_when_missing(tmp_path):
    assert resolve_served_ref(out=str(tmp_path / "does-not-exist")) is None


def test_router_distilled_tier_auto_fills_served_ref_from_promoted_json(tmp_path):
    loop_out = tmp_path / "loopout"
    loop_out.mkdir()
    (loop_out / "promoted.json").write_text(
        json.dumps({"model_dir": "/models/candidate", "served_ref": "/models/candidate"}),
        encoding="utf-8",
    )
    router = Router({
        "distilled": {"base_url": "http://d.local"},
        "loop_out": str(loop_out),
    })
    _tier, backend, tier_config = router.route({"messages": []})
    assert tier_config["served_ref"] == "/models/candidate"
    assert backend.model == "/models/candidate"


def test_list_models_returns_both_tiers():
    router = Router({
        "distilled": {"model": "distilled-1"},
        "frontier": {"model": "gpt-4o-mini"},
    })
    models = router.list_models()
    assert {m["id"] for m in models} == {"distilled-1", "gpt-4o-mini"}
    assert {m["tier"] for m in models} == {"distilled", "frontier"}


def test_list_models_falls_back_to_tier_name_when_unconfigured(tmp_path):
    router = Router({"loop_out": str(tmp_path / "no-such-loopout")})
    models = router.list_models()
    ids = {m["id"] for m in models}
    assert ids == {"distilled", "frontier"}


def test_router_uses_cost_aware_escalate_when_router_model_configured():
    hard_model = {
        "feature_names": ["log_length", "keyword_hits", "question_marks", "code_fences", "avg_word_length"],
        "mean": [0.0] * 5, "std": [1.0] * 5, "weights": [0.0] * 5, "bias": 10.0, "hard_threshold": 0.5,
    }
    router = Router({
        "distilled": {"base_url": "http://d.local"},
        "frontier": {"base_url": "http://f.local"},
        "router_model": hard_model,
    })
    tier, _backend, _config = router.route({"messages": [{"content": "fix the bug"}]})
    assert tier == "frontier"


def test_router_falls_back_to_default_escalate_when_router_model_path_missing(tmp_path):
    router = Router({
        "distilled": {"base_url": "http://d.local"},
        "frontier": {"base_url": "http://f.local"},
        "router_model_path": str(tmp_path / "does-not-exist.json"),
    })
    tier, _backend, _config = router.route({"messages": [{"content": "hello there"}]})
    assert tier == "distilled"


def test_router_without_router_model_uses_plain_default_escalate():
    router = Router({"distilled": {"base_url": "http://d.local"}, "frontier": {"base_url": "http://f.local"}})
    assert router.escalate_fn.__name__ == "default_escalate"


def test_backend_for_unconfigured_tier_fails_closed_instead_of_public_openai():
    router = Router({"distilled": {"base_url": "http://d.local"}})
    with pytest.raises(BackendUnavailable):
        router.backend_for("frontier")


def test_escalation_to_unconfigured_frontier_fails_closed():
    router = Router({"distilled": {"base_url": "http://d.local"}})
    with pytest.raises(BackendUnavailable):
        router.route({"episodic_escalate": True, "messages": []})


def test_openai_tier_is_allowed_with_an_explicit_api_key():
    router = Router({"frontier": {"backend": "openai", "api_key": "sk-test"}})
    backend, _config = router.backend_for("frontier")
    assert backend.base_url == "https://api.openai.com"


def test_distilled_served_ref_without_upstream_fails_closed(tmp_path):
    loop_out = tmp_path / "loopout"
    loop_out.mkdir()
    (loop_out / "promoted.json").write_text(
        json.dumps({"model_dir": "/models/candidate", "served_ref": "/models/candidate"}),
        encoding="utf-8",
    )
    router = Router({"loop_out": str(loop_out)})
    with pytest.raises(BackendUnavailable):
        router.backend_for("distilled")
