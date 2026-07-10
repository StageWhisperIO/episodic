import pytest

from episodic import cli, store
from episodic.schema import new_event


def test_label_cli_exits_nonzero_and_prints_error_on_labeler_failure(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path / ".episodic"))
    session_id = "sess-label-fail"
    store.write_meta(session_id, {"cwd": str(tmp_path), "repo_state": {"root": None}})
    store.append_event(new_event(session_id, "session_start", data={"cwd": str(tmp_path)}))
    store.append_event(new_event(session_id, "user_prompt", data={"prompt": "fix the bug"}))

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["label", "--session", session_id, "--cmd",
                  "sh -c \"printf 'Not logged in' 1>&2; exit 1\""])
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "labeler failed" in captured.err
    assert "Not logged in" in captured.err


def test_segment_cli_exits_nonzero_and_prints_error_on_labeler_failure(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path / ".episodic"))
    session_id = "sess-segment-fail"
    store.write_meta(session_id, {"cwd": str(tmp_path), "repo_state": {"root": None}})
    store.append_event(new_event(session_id, "session_start", data={"cwd": str(tmp_path)}))
    store.append_event(new_event(session_id, "user_prompt", data={
        "prompt": "build a brand new authentication subsystem end to end"}))

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["segment", "--session", session_id, "--label", "--cmd",
                  "sh -c \"printf boom 1>&2; exit 1\""])
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "labeler failed" in captured.err
