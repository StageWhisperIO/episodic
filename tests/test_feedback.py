from episodic.core import feedback, reward
from episodic.core.episode import build_episode
from episodic.schema import new_event, new_episode, validate_episode


def test_extract_json_ignores_prose_and_braced_strings():
    text = 'sure, here:\n{"outcome":{"success":"no","rationale":"has a } brace"}}\nthanks'
    parsed = feedback._extract_json(text)
    assert parsed["outcome"]["success"] == "no"
    assert parsed["outcome"]["rationale"] == "has a } brace"


def test_extract_json_returns_none_on_garbage():
    assert feedback._extract_json("no json here") is None


def test_extract_json_skips_unparseable_brace_block_before_real_json():
    text = 'Analyzing the {transcript} carefully: {"outcome":{"success":"yes","rationale":"ok"}}'
    parsed = feedback._extract_json(text)
    assert parsed["outcome"]["success"] == "yes"
    assert parsed["outcome"]["rationale"] == "ok"


def test_extract_json_plain_json():
    text = '{"outcome":{"success":"no","rationale":"x"}}'
    assert feedback._extract_json(text)["outcome"]["success"] == "no"


def test_extract_json_fenced_code_block():
    text = '```json\n{"outcome":{"success":"unclear","rationale":"n/a"}}\n```'
    parsed = feedback._extract_json(text)
    assert parsed["outcome"]["success"] == "unclear"


def test_clamp01_defaults_missing_or_invalid_confidence_to_half():
    assert feedback._clamp01(None) == 0.5
    assert feedback._clamp01("not-a-number") == 0.5
    assert feedback._clamp01(0.9) == 0.9
    assert feedback._clamp01(-5) == 0.0
    assert feedback._clamp01(5) == 1.0


def test_mine_defaults_missing_confidence_to_half():
    ep = _episode_with_two_prompts()
    gen = _fake_generate(
        '{"feedback":[{"step_index":2,"label":"wrong","evidence":"still broken"}],'
        '"outcome":{"success":"no","rationale":"did not fix"}}'
    )
    mined = feedback.mine(ep, gen)
    assert mined["feedback"][0]["confidence"] == 0.5
    assert mined["outcome_hint"]["confidence"] == 0.5


def test_command_generate_raises_runtimeerror_on_nonzero_exit():
    generate = feedback.command_generate(
        command="sh -c \"printf 'Not logged in' 1>&2; exit 1\"", timeout=5
    )
    try:
        generate("prompt")
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        message = str(exc)
        assert "1" in message
        assert "Not logged in" in message


def test_command_generate_exposes_resolved_command():
    generate = feedback.command_generate(command="echo hi")
    assert generate.command == "echo hi"


def test_mine_model_provenance_prefers_env_override_then_resolved_command_then_default():
    ep = _episode_with_two_prompts()
    payload = '{"feedback":[{"step_index":2,"label":"wrong","confidence":0.9,"evidence":"x"}]}'

    gen_with_command = _fake_generate(payload)
    gen_with_command.command = "opus -p --model custom"
    mined = feedback.mine(ep, gen_with_command)
    assert mined["feedback"][0]["model"] == "opus -p --model custom"

    gen_no_command = _fake_generate(payload)
    mined_default = feedback.mine(ep, gen_no_command)
    assert mined_default["feedback"][0]["model"] == feedback.DEFAULT_MODEL


def test_mine_model_provenance_env_override_wins(monkeypatch):
    ep = _episode_with_two_prompts()
    payload = '{"feedback":[{"step_index":2,"label":"wrong","confidence":0.9,"evidence":"x"}]}'
    gen = _fake_generate(payload)
    gen.command = "opus -p --model custom"
    monkeypatch.setenv("EPISODIC_LABELER_MODEL", "explicit-model")
    mined = feedback.mine(ep, gen)
    assert mined["feedback"][0]["model"] == "explicit-model"


def test_command_generate_resolves_command_exactly_once(monkeypatch):
    calls = []
    original = feedback._resolve_command

    def counting(command):
        calls.append(command)
        return original(command)

    monkeypatch.setattr(feedback, "_resolve_command", counting)
    generate = feedback.command_generate(command="echo hi")
    generate("prompt")
    assert calls == ["echo hi"]


def test_probe_resolves_command_exactly_once(monkeypatch):
    calls = []
    original = feedback._resolve_command

    def counting(command):
        calls.append(command)
        return original(command)

    monkeypatch.setattr(feedback, "_resolve_command", counting)
    result = feedback.probe("hi", command="sh -c \"cat; printf ERR 1>&2; exit 3\"", timeout=5)
    assert calls == ['sh -c "cat; printf ERR 1>&2; exit 3"']
    assert result["command"] == 'sh -c "cat; printf ERR 1>&2; exit 3"'
    assert result["code"] == 3
    assert result["stdout"] == "hi"
    assert result["stderr"] == "ERR"


def _episode_with_two_prompts():
    ep = new_episode(id="ep_x", intent="fix the thing")
    ep["steps"] = [
        {"index": 0, "type": "user_prompt", "ts": "t0", "intent": "fix the thing", "input": {"prompt": "fix the thing"}},
        {"index": 1, "type": "file_edit", "ts": "t1", "intent": "edit a.py", "input": {}},
        {"index": 2, "type": "user_prompt", "ts": "t2", "intent": "still broken", "input": {"prompt": "it is still broken, are you sure?"}},
    ]
    return ep


def _fake_generate(payload):
    return lambda prompt: payload


def test_mine_maps_labels_and_outcome_and_drops_invalid():
    ep = _episode_with_two_prompts()
    gen = _fake_generate(
        '{"feedback":['
        '{"step_index":2,"label":"wrong","confidence":0.9,"evidence":"still broken"},'
        '{"step_index":2,"label":"not_a_label","confidence":0.5,"evidence":"x"}],'
        '"outcome":{"success":"no","confidence":0.7,"rationale":"did not fix"}}'
    )
    mined = feedback.mine(ep, gen)
    assert len(mined["feedback"]) == 1
    item = mined["feedback"][0]
    assert item["label"] == "wrong"
    assert item["source"] == "mined"
    assert item["evidence_step_index"] == 2
    assert item["ts"] == "t2"
    assert item["confidence"] == 0.9
    assert mined["outcome_hint"]["success"] == "no"
    assert mined["outcome_hint"]["confidence"] == 0.7


def test_build_episode_merges_mined_feedback_and_hint():
    session = {
        "id": "s1",
        "meta": {"cwd": "/tmp/none-such-repo"},
        "events": [
            new_event("s1", "user_prompt", data={"prompt": "fix the thing"}),
            new_event("s1", "file_edit", data={"file_path": "/tmp/none-such-repo/a.py"}),
            new_event("s1", "user_prompt", data={"prompt": "it is still broken, are you sure?"}),
        ],
    }
    gen = _fake_generate(
        '{"feedback":[{"step_index":2,"label":"wrong","confidence":0.8,"evidence":"still broken"}],'
        '"outcome":{"success":"no","confidence":0.6,"rationale":"unresolved"}}'
    )
    episode = build_episode(session, generate=gen)
    mined = [f for f in episode["human_feedback"] if f.get("source") == "mined"]
    assert len(mined) == 1
    assert "mined_feedback" in episode["labels"]
    assert "wrong" in episode["labels"]
    assert episode["outcome_hint"]["success"] == "no"
    assert episode["reward_vector"]["human_label"] < 0
    assert episode["reward_vector"]["components"]["outcome_source"] == "mined"
    assert validate_episode(episode) == []


def test_build_episode_without_generate_is_unchanged():
    session = {
        "id": "s2",
        "meta": {"cwd": "/tmp/none-such-repo"},
        "events": [new_event("s2", "user_prompt", data={"prompt": "hi"})],
    }
    episode = build_episode(session)
    assert episode["human_feedback"] == []
    assert episode.get("outcome_hint") is None
    assert "mined_feedback" not in episode["labels"]


def test_reward_confidence_weights_mined_feedback():
    ep = new_episode(id="ep_w")
    ep["human_feedback"] = [
        {"ts": "t", "label": "wrong", "source": "mined", "confidence": 0.5},
        {"ts": "t", "label": "useful"},
    ]
    rv = reward.reward_vector(ep)
    assert abs(rv["human_label"] - ((-1.0 * 0.5 + 1.0 * 1.0) / 1.5)) < 1e-3


def test_auto_label_generate_gating(monkeypatch):
    from episodic.collector import hook

    monkeypatch.delenv("EPISODIC_AUTO_LABEL", raising=False)
    assert hook._auto_label_generate("SessionEnd") is None
    assert hook._auto_label_generate("Stop") is None

    monkeypatch.setenv("EPISODIC_AUTO_LABEL", "1")
    assert hook._auto_label_generate("Stop") is None
    assert callable(hook._auto_label_generate("SessionEnd"))

    monkeypatch.setenv("EPISODIC_AUTO_LABEL", "off")
    assert hook._auto_label_generate("SessionEnd") is None


def test_reward_outcome_hint_only_when_open():
    open_ep = new_episode(id="ep_o")
    open_ep["outcome_hint"] = {"success": "yes", "confidence": 1.0, "source": "mined"}
    rv_open = reward.reward_vector(open_ep)
    assert rv_open["components"]["outcome_source"] == "mined"
    assert rv_open["components"]["normalized"]["outcome"] == 1.0

    merged_ep = new_episode(id="ep_m")
    merged_ep["outcome"]["status"] = "merged"
    merged_ep["outcome_hint"] = {"success": "no", "confidence": 1.0, "source": "mined"}
    rv_merged = reward.reward_vector(merged_ep)
    assert rv_merged["components"]["outcome_source"] == "authoritative"
    assert rv_merged["components"]["normalized"]["outcome"] == 1.0
