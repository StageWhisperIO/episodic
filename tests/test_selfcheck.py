from episodic import selfcheck
from episodic.cli import main


def test_run_checks_all_required_pass():
    report = selfcheck.run_checks()
    assert report["ok"] is True
    assert report["failed"] == []
    names = {c["name"] for c in report["checks"]}
    assert {"exporters", "worldbench", "loop_dry_run", "fidelity", "worldmodel", "replay_plan"}.issubset(names)


def test_doctor_cli_exit_zero(capsys):
    rc = main(["doctor"])
    assert rc == 0
    assert "install is healthy" in capsys.readouterr().out


def test_doctor_json_mode(capsys):
    import json

    rc = main(["doctor", "--json"])
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    assert report["passed"] >= 11
