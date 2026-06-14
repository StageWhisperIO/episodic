import pytest
from episodic.dashboard.server import render_index, render_episode, apply_feedback


def test_render_index(episodes):
    import episodic.store as store
    rows = [store.index_row(ep) for ep in episodes]
    html = render_index(rows)
    assert isinstance(html, str)
    assert "<table" in html
    for ep in episodes:
        assert ep["id"] in html


def test_render_episode(sample_episode):
    html = render_episode(sample_episode, "summary text")
    assert isinstance(html, str)
    assert sample_episode["intent"] in html
    assert "useful" in html


def test_apply_feedback(monkeypatch, tmp_path, sample_episode):
    import episodic.store as store
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path))
    store.save_episode(sample_episode)
    row = apply_feedback(sample_episode["id"], "useful")
    assert "error" not in row
    updated = store.get_episode(sample_episode["id"])
    assert "useful" in updated["labels"]
    assert isinstance(updated["reward_vector"]["composite"], float)
