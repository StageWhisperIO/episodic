from episodic import cli, loop


def _run_loop_cli(monkeypatch, argv):
    captured = {}

    def fake_run_loop(config, start=None):
        captured["config"] = config
        return {"decision": "no_train_data"}

    monkeypatch.setattr(loop, "run_loop", fake_run_loop)
    cli.main(argv)
    return captured["config"]


def test_cmd_loop_enables_judge_by_default(monkeypatch, capsys):
    config = _run_loop_cli(monkeypatch, ["loop"])
    capsys.readouterr()
    assert config["judge"] is True


def test_cmd_loop_no_judge_disables_it(monkeypatch, capsys):
    config = _run_loop_cli(monkeypatch, ["loop", "--no-judge"])
    capsys.readouterr()
    assert config["judge"] is False


def test_cmd_loop_wires_judge_cmd_and_timeout(monkeypatch, capsys):
    config = _run_loop_cli(monkeypatch, [
        "loop", "--judge-cmd", "sh -c \"printf 'SCORE: 1.0'\"", "--judge-timeout", "30",
    ])
    capsys.readouterr()
    assert config["judge"] is True
    assert config["judge_cmd"] == "sh -c \"printf 'SCORE: 1.0'\""
    assert config["judge_timeout"] == 30


def test_cmd_loop_config_file_judge_false_is_not_overridden(monkeypatch, capsys, tmp_path):
    config_path = tmp_path / "loop.json"
    config_path.write_text('{"judge": false}', encoding="utf-8")
    config = _run_loop_cli(monkeypatch, ["loop", "--config", str(config_path)])
    capsys.readouterr()
    assert config["judge"] is False


def test_cmd_loop_epochs_defaults_to_unset(monkeypatch, capsys):
    config = _run_loop_cli(monkeypatch, ["loop"])
    capsys.readouterr()
    assert "epochs" not in config


def test_cmd_loop_wires_epochs_flag(monkeypatch, capsys):
    config = _run_loop_cli(monkeypatch, ["loop", "--epochs", "3"])
    capsys.readouterr()
    assert config["epochs"] == 3


def test_cmd_loop_wires_evaluator_flag(monkeypatch, capsys):
    config = _run_loop_cli(monkeypatch, ["loop", "--evaluator", "local_critic"])
    capsys.readouterr()
    assert config["evaluator"] == {"type": "local_critic"}


def test_cmd_loop_router_flag_is_off_by_default(monkeypatch, capsys):
    config = _run_loop_cli(monkeypatch, ["loop"])
    capsys.readouterr()
    assert "router" not in config


def test_cmd_loop_wires_router_flag(monkeypatch, capsys):
    config = _run_loop_cli(monkeypatch, ["loop", "--router"])
    capsys.readouterr()
    assert config["router"] is True


def test_cmd_loop_sim_prefilter_is_off_by_default(monkeypatch, capsys):
    config = _run_loop_cli(monkeypatch, ["loop"])
    capsys.readouterr()
    assert "sim_prefilter" not in config
    assert "sim_predictor" not in config


def test_cmd_loop_wires_sim_prefilter_and_named_backend(monkeypatch, capsys):
    config = _run_loop_cli(monkeypatch, [
        "loop", "--sim-prefilter", "--sim-backend", "echo", "--sim-max-turns", "5",
    ])
    capsys.readouterr()
    assert config["sim_prefilter"] is True
    assert config["sim_predictor"] == "echo"
    assert config["sim_max_turns"] == 5


def test_cmd_loop_sim_backend_mlx_needs_sim_model_dir(monkeypatch, capsys):
    try:
        _run_loop_cli(monkeypatch, ["loop", "--sim-backend", "mlx"])
        assert False, "expected SystemExit"
    except SystemExit as exc:
        assert exc.code == 1
    capsys.readouterr()


def test_cmd_loop_sim_backend_mlx_resolves_a_real_predictor(monkeypatch, capsys):
    from episodic.worldmodel import inference as wm_inference

    monkeypatch.setattr(wm_inference, "mlx_predictor", lambda model, **kw: (lambda sample: "predicted"))
    config = _run_loop_cli(monkeypatch, ["loop", "--sim-backend", "mlx", "--sim-model-dir", "fake/model"])
    capsys.readouterr()
    assert callable(config["sim_predictor"])
    assert config["sim_predictor"]({"history": "x"}) == "predicted"


def test_cmd_loop_sim_backend_unavailable_prints_hint_and_exits_zero(monkeypatch, capsys):
    from episodic import trainers
    from episodic.worldmodel import inference as wm_inference

    def raise_unavailable(model, **kw):
        raise trainers.TrainerUnavailable("mlx-sft", "install mlx-lm")

    monkeypatch.setattr(wm_inference, "mlx_predictor", raise_unavailable)
    monkeypatch.setattr(loop, "run_loop", lambda config, start=None: (_ for _ in ()).throw(
        AssertionError("run_loop should not be called")))
    rc = cli.main(["loop", "--sim-backend", "mlx", "--sim-model-dir", "fake/model"])
    assert rc == 0
    assert "install mlx-lm" in capsys.readouterr().err
