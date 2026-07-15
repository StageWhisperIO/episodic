import pytest

from episodic import paths


@pytest.mark.parametrize("bad", ["../../etc/passwd", "a/b", "..", "/abs", "", "ep\x00x", "a\\b", "."])
def test_safe_id_rejects_traversal(bad):
    with pytest.raises(ValueError):
        paths.safe_id(bad)


@pytest.mark.parametrize("good", ["ep_5ff3b7119e8b", "0359d5d8-f7d2-48dc-b1cd-394113c74db4", "sv", "s7"])
def test_safe_id_accepts_real_ids(good):
    assert paths.safe_id(good) == good


def test_episode_path_blocks_traversal():
    with pytest.raises(ValueError):
        paths.episode_path("../../../etc/passwd")


def test_session_dir_blocks_traversal():
    with pytest.raises(ValueError):
        paths.session_dir("../../secrets")


def test_get_episode_blocks_traversal_even_if_target_exists(tmp_path):
    from episodic import store
    from episodic.schema import new_episode

    (tmp_path / ".episodic" / "episodes").mkdir(parents=True)
    (tmp_path / "secret.json").write_text('{"leaked": true}', encoding="utf-8")
    store.save_episode(new_episode(id="ep_real1"), start=str(tmp_path))

    assert store.get_episode("../../secret", start=str(tmp_path)) is None
    assert store.get_episode("ep_real1", start=str(tmp_path))["id"] == "ep_real1"
