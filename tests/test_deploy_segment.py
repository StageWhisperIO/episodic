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


def test_deploy_stays_unverified_when_later_negative_has_no_prior_positive():
    session = _deploy_session(0, trailing="it is still broken on production, are you sure?")
    gen = lambda prompt: '{"feedback":[{"step_index":2,"label":"wrong","confidence":0.9,"evidence":"still broken on production"}],"outcome":{"success":"no","confidence":0.7,"rationale":"prod broken"}}'
    episode = build_episode(session, generate=gen)
    assert episode["deployments"][0]["verified"] is None
    assert episode["reward_vector"]["components"]["outcome_source"] == "mined"


def test_successful_deploy_verified_true_after_later_positive_feedback():
    session = {
        "id": "sp", "meta": {"cwd": "/tmp/x", "human_feedback": [
            {"ts": "2026-01-01T00:00:03+00:00", "label": "accepted_as_is"},
        ]},
        "events": [
            new_event("sp", "user_prompt", data={"prompt": "ship the landing page to prod"},
                      ts="2026-01-01T00:00:01+00:00"),
            new_event("sp", "shell_command", data={
                "command": "wrangler pages deploy ./dist --production", "exit_code": 0, "cwd": "/tmp/x"},
                      ts="2026-01-01T00:00:02+00:00"),
        ],
    }
    episode = build_episode(session)
    assert episode["deployments"][0]["verified"] is True
    rv = episode["reward_vector"]
    assert rv["components"]["outcome_source"] == "deploy"
    assert rv["components"]["normalized"]["outcome"] == 1.0


def test_successful_deploy_stays_unverified_after_unrelated_later_negative():
    session = {
        "id": "sn", "meta": {"cwd": "/tmp/x", "human_feedback": [
            {"ts": "2026-01-01T00:00:03+00:00", "label": "wrong"},
        ]},
        "events": [
            new_event("sn", "user_prompt", data={"prompt": "ship the landing page to prod"},
                      ts="2026-01-01T00:00:01+00:00"),
            new_event("sn", "shell_command", data={
                "command": "wrangler pages deploy ./dist --production", "exit_code": 0, "cwd": "/tmp/x"},
                      ts="2026-01-01T00:00:02+00:00"),
        ],
    }
    episode = build_episode(session)
    assert episode["deployments"][0]["verified"] is None
    rv = episode["reward_vector"]
    assert rv["outcome"] != -1.0
    assert rv["components"]["outcome_source"] == "none"


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


def test_segment_session_isolates_feedback_per_segment_and_inherits_outcome_on_last_only():
    events = [
        new_event("s9", "user_prompt", data={"prompt": "build a brand new authentication subsystem end to end"},
                  ts="2026-01-01T00:00:00+00:00"),
        new_event("s9", "file_edit", data={"file_path": "/r/a.py"}, ts="2026-01-01T00:00:01+00:00"),
        new_event("s9", "user_prompt",
                  data={"prompt": "completely separate task: rewrite the billing documentation from scratch"},
                  ts="2026-01-01T00:10:00+00:00"),
        new_event("s9", "file_edit", data={"file_path": "/r/b.py"}, ts="2026-01-01T00:10:01+00:00"),
    ]
    meta = {
        "cwd": "/tmp/x",
        "outcome": {"status": "merged"},
        "human_feedback": [
            {"ts": "2026-01-01T00:00:02+00:00", "label": "accepted_as_is"},
            {"ts": "2026-01-01T00:10:02+00:00", "label": "wrong"},
        ],
    }
    session = {"id": "s9", "meta": meta, "events": events}
    children = segment.segment_session(session)
    assert len(children) == 2
    assert [f["label"] for f in children[0]["human_feedback"]] == ["accepted_as_is"]
    assert [f["label"] for f in children[1]["human_feedback"]] == ["wrong"]
    assert children[0]["outcome"]["status"] == "open"
    assert children[1]["outcome"]["status"] == "merged"


def test_segment_session_does_not_leak_mined_feedback_across_children():
    events = [
        new_event("s10", "user_prompt", data={"prompt": "build a brand new authentication subsystem end to end"},
                  ts="2026-01-01T00:00:00+00:00"),
        new_event("s10", "file_edit", data={"file_path": "/r/a.py"}, ts="2026-01-01T00:00:01+00:00"),
        new_event("s10", "user_prompt",
                  data={"prompt": "completely separate task: rewrite the billing documentation from scratch"},
                  ts="2026-01-01T00:10:00+00:00"),
        new_event("s10", "file_edit", data={"file_path": "/r/b.py"}, ts="2026-01-01T00:10:01+00:00"),
    ]
    session = {"id": "s10", "meta": {"cwd": "/tmp/x"}, "events": events}
    gen = lambda prompt: (
        '{"feedback":[{"step_index":0,"label":"wrong","confidence":0.9,"evidence":"x"}],'
        '"outcome":{"success":"no","confidence":0.5,"rationale":"x"}}'
    )
    children = segment.segment_session(session, generate=gen)
    assert len(children[0]["human_feedback"]) == 1
    assert len(children[1]["human_feedback"]) == 1


def test_is_new_task_action_verb_starts_new_task():
    assert segment._is_new_task("implement dark mode for the settings page with tests", False) is True


def test_is_new_task_connector_prefix_is_continuation():
    assert segment._is_new_task("and also add tests for the parser module please", False) is False


def test_is_new_task_bare_please_verb_starts_new_task():
    assert segment._is_new_task("please refactor the payment handler into a service layer", False) is True


def test_is_new_task_short_ack_is_continuation():
    assert segment._is_new_task("ok", False) is False


def test_reward_regression_clamp_applies_to_open_deploy_and_mined_sources():
    ep = new_episode(id="ep_reg")
    ep["outcome"]["caused_regression"] = True
    ep["deployments"] = [{"ts": "t", "method": "wrangler", "target_env": "prod", "verified": True}]
    rv = reward.reward_vector(ep)
    assert rv["components"]["outcome_source"] == "deploy"
    assert rv["outcome"] == -1.0

    ep_mined = new_episode(id="ep_reg_m")
    ep_mined["outcome"]["caused_regression"] = True
    ep_mined["outcome_hint"] = {"success": "yes", "confidence": 1.0}
    rv_mined = reward.reward_vector(ep_mined)
    assert rv_mined["components"]["outcome_source"] == "mined"
    assert rv_mined["outcome"] == -1.0


def test_deploy_env_keyword_not_fired_by_path_or_image_substring():
    assert deploydetect.classify_deploy("docker push registry.io/main-app:latest")["target_env"] == "unknown"
    assert deploydetect.classify_deploy("kubectl apply -f main-deployment.yaml")["target_env"] == "unknown"


def test_deploy_env_keyword_still_fires_on_whole_word():
    assert deploydetect._target_env("git push origin main") == "prod"
    assert deploydetect._target_env("wrangler pages deploy ./dist --staging") == "staging"


def test_deploy_segment_split_isolates_env_keyword_from_piped_prefix():
    result = deploydetect.classify_deploy("echo prod status | make deploy")
    assert result is not None
    assert result["method"] == "make-deploy"
    assert result["target_env"] == "unknown"
