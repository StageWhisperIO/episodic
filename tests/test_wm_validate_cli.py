import json

from episodic.cli import main
from episodic.testing import populate_store


def test_cmd_wm_validate_fails_without_episodes(tmp_path, monkeypatch):
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path / ".episodic"))
    try:
        main(["wm-validate"])
        assert False, "expected SystemExit"
    except SystemExit as exc:
        assert exc.code == 1


def test_cmd_wm_validate_reports_baselines_without_execute_or_adapter(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path / ".episodic"))
    populate_store(20, seed=1)

    rc = main(["wm-validate", "--holdout-frac", "0.5"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)

    assert set(("oracle", "prefix", "empty")).issubset(out["fidelity"])
    assert "trained" not in out["fidelity"]
    assert out["predictor"]["trained"] is False
    assert out["dataset"] is None
    assert out["fidelity"]["oracle"]["mean_composite"] == 1.0
    assert out["n_holdout"] > 0
    assert (tmp_path / ".episodic" / "exports" / "wm_validate" / "report.json").exists()


def test_cmd_wm_validate_adapter_path_skips_training_and_scores_a_trained_predictor(tmp_path, monkeypatch, capsys):
    from episodic.worldmodel import inference as wm_inference

    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path / ".episodic"))
    populate_store(20, seed=2)

    calls = []

    def fake_mlx_predictor(model, **kw):
        calls.append((model, kw.get("adapter_path")))
        return lambda sample: sample["target_observation"]

    monkeypatch.setattr(wm_inference, "mlx_predictor", fake_mlx_predictor)

    rc = main([
        "wm-validate", "--holdout-frac", "0.5",
        "--model", "fake/base", "--adapter-path", "fake/adapters",
    ])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)

    assert calls == [("fake/base", "fake/adapters")]
    assert out["predictor"] == {
        "backend": "mlx", "base_model": "fake/base", "adapter_path": "fake/adapters", "trained": False,
    }
    assert out["fidelity"]["trained"]["mean_composite"] == 1.0
    assert out["dataset"]["count"] > 0


def test_cmd_wm_validate_execute_trains_a_fresh_adapter(tmp_path, monkeypatch, capsys):
    from episodic import trainers
    from episodic.worldmodel import inference as wm_inference

    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path / ".episodic"))
    populate_store(20, seed=3)

    train_calls = []

    def fake_train(trainer_name, dataset_path, out_dir, config, cwd=None):
        train_calls.append((trainer_name, dataset_path, out_dir, config))
        return {"result": {"model_dir": str(tmp_path / "adapters"), "base_model": "trained/base"}}

    monkeypatch.setattr(trainers, "train", fake_train)
    monkeypatch.setattr(wm_inference, "mlx_predictor", lambda model, **kw: (lambda sample: "predicted"))

    rc = main(["wm-validate", "--holdout-frac", "0.5", "--execute"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)

    assert train_calls and train_calls[0][0] == "mlx-sft"
    assert out["predictor"] == {
        "backend": "mlx", "base_model": "trained/base",
        "adapter_path": str(tmp_path / "adapters"), "trained": True,
    }
    assert "trained" in out["fidelity"]


def test_cmd_wm_validate_execute_trainer_unavailable_prints_hint_and_exits_zero(tmp_path, monkeypatch, capsys):
    from episodic import trainers

    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path / ".episodic"))
    populate_store(10, seed=4)

    def raise_unavailable(trainer_name, dataset_path, out_dir, config, cwd=None):
        raise trainers.TrainerUnavailable("mlx-sft", "install mlx-lm")

    monkeypatch.setattr(trainers, "train", raise_unavailable)

    rc = main(["wm-validate", "--holdout-frac", "0.5", "--execute"])
    assert rc == 0
    assert "install mlx-lm" in capsys.readouterr().err


def test_cmd_wm_validate_replay_correlate_needs_execute(tmp_path, monkeypatch):
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path / ".episodic"))
    populate_store(10, seed=5)
    try:
        main(["wm-validate", "--holdout-frac", "0.5", "--replay-correlate"])
        assert False, "expected SystemExit"
    except SystemExit as exc:
        assert exc.code == 1


def test_cmd_wm_validate_replay_correlate_wires_sim_and_real_scores(tmp_path, monkeypatch, capsys):
    from episodic import trainers
    from episodic.worldmodel import inference as wm_inference
    from episodic.worldmodel import validate as wm_validate

    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path / ".episodic"))
    populate_store(10, seed=6)

    def fake_train(trainer_name, dataset_path, out_dir, config, cwd=None):
        return {"result": {"model_dir": "adapters/x", "base_model": "trained/base"}}

    monkeypatch.setattr(trainers, "train", fake_train)
    monkeypatch.setattr(wm_inference, "mlx_predictor", lambda model, **kw: (lambda sample: "predicted"))

    sim_calls = []
    real_calls = []

    def fake_sim_scores(episodes, predictor, max_turns=None, history_budget=None):
        sim_calls.append(len(episodes))
        return {ep["id"]: 0.5 for ep in episodes}

    def fake_offline_replay_scores(episodes, model="offline-oracle-diff", start=None, runner=None):
        real_calls.append(len(episodes))
        return {ep["id"]: 0.5 for ep in episodes}

    monkeypatch.setattr(wm_validate, "sim_scores", fake_sim_scores)
    monkeypatch.setattr(wm_validate, "offline_replay_scores", fake_offline_replay_scores)

    rc = main(["wm-validate", "--holdout-frac", "0.5", "--execute", "--replay-correlate"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)

    assert sim_calls and real_calls
    assert out["replay_correlation"]["n"] == sim_calls[0]
    assert out["replay_correlation"]["pearson"] is None
