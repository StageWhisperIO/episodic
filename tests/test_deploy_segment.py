from episodic.core import deploydetect, reward, segment
from episodic.core.episode import build_episode
from episodic.core.ids import episode_id_from_session
from episodic.schema import new_event, new_episode, validate_episode


def test_classify_deploy_detects_method_and_env():
    assert deploydetect.classify_deploy("wrangler pages deploy ./dist --env production") == {
        "method": "wrangler", "target_env": "prod"}
    assert deploydetect.classify_deploy("npm run deploy -- --staging")["target_env"] == "staging"
    assert deploydetect.classify_deploy("kubectl apply -f k8s/dev.yaml")["method"] == "kubectl"
    assert deploydetect.classify_deploy("ls -la") is None
    assert deploydetect.classify_deploy("echo 'wrangler pages deploy'") is None


def _deploy_session(exit_code, trailing=None):
    events = [
        new_event("s", "user_prompt", data={"prompt": "ship the landing page to prod"}, ts="2026-01-01T00:00:01+00:00"),
        new_event("s", "shell_command", data={"command": "wrangler pages deploy ./dist --production", "exit_code": exit_code, "cwd": "/tmp/x"}, ts="2026-01-01T00:00:02+00:00"),
    ]
    if trailing:
        events.append(new_event("s", "user_prompt", data={"prompt": trailing}, ts="2026-01-01T00:00:03+00:00"))
    return {"id": "s", "meta": {"cwd": "/tmp/x"}, "events": events}


def test_build_deployments_records_prod_deploy():
    episode = build_episode(_deploy_session(0))
    assert len(episode["deployments"]) == 1
    deployment = episode["deployments"][0]
    assert deployment["method"] == "wrangler"
    assert deployment["target_env"] == "prod"
    assert deployment["verified"] is None
    assert validate_episode(episode) == []


def test_failed_deploy_is_unverified_false():
    episode = build_episode(_deploy_session(1))
    assert episode["deployments"][0]["verified"] is False
    rv = episode["reward_vector"]
    assert rv["components"]["outcome_source"] == "deploy"
    assert rv["components"]["normalized"]["outcome"] == 0.0


def test_deploy_marked_false_by_later_negative_feedback():
    session = _deploy_session(0, trailing="it is still broken on production, are you sure?")
    gen = lambda prompt: '{"feedback":[{"step_index":2,"label":"wrong","confidence":0.9,"evidence":"still broken on production"}],"outcome":{"success":"no","confidence":0.7,"rationale":"prod broken"}}'
    episode = build_episode(session, generate=gen)
    assert episode["deployments"][0]["verified"] is False
    assert episode["reward_vector"]["components"]["outcome_source"] == "deploy"


def test_reward_verified_prod_deploy_is_positive():
    ep = new_episode(id="ep_d")
    ep["deployments"] = [{"ts": "t", "method": "wrangler", "target_env": "prod", "verified": True}]
    rv = reward.reward_vector(ep)
    assert rv["components"]["outcome_source"] == "deploy"
    assert rv["components"]["normalized"]["outcome"] == 1.0


def test_segment_events_splits_tasks_and_merges_continuations():
    events = [
        new_event("s", "session_start", data={"transcript_path": "/none"}),
        new_event("s", "user_prompt", data={"prompt": "please add a sonarqube full inspection workflow to the desktop app"}),
        new_event("s", "file_edit", data={"file_path": "/r/a.yml"}),
        new_event("s", "user_prompt", data={"prompt": "push"}),
        new_event("s", "user_prompt", data={"prompt": "now write a full overview of the Signals product and its recent changes"}),
        new_event("s", "file_edit", data={"file_path": "/r/b.md"}),
    ]
    segments = segment.segment_events(events)
    assert len(segments) == 2
    assert all(seg[0]["type"] == "session_start" for seg in segments)
    assert any(e["type"] == "user_prompt" and e["data"]["prompt"] == "push" for e in segments[0])


def test_segment_session_yields_children_with_parent_and_index():
    events = [
        new_event("s7", "user_prompt", data={"prompt": "build a brand new authentication subsystem end to end"}),
        new_event("s7", "file_edit", data={"file_path": "/r/a.py"}),
        new_event("s7", "user_prompt", data={"prompt": "completely separate task: rewrite the billing documentation from scratch"}),
        new_event("s7", "file_edit", data={"file_path": "/r/b.py"}),
    ]
    session = {"id": "s7", "meta": {"cwd": "/tmp/x"}, "events": events}
    children = segment.segment_session(session)
    assert len(children) == 2
    assert children[0]["parent_id"] == episode_id_from_session("s7")
    assert children[0]["segment_index"] == 0
    assert children[1]["segment_index"] == 1
    assert children[0]["id"] != children[1]["id"]
    assert validate_episode(children[0]) == []
