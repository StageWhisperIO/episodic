import json

from episodic import store, worldbench
from episodic.cli import main
from episodic.testing import make_population, populate_store


def test_oracle_scores_perfect_and_empty_scores_low():
    pop = make_population(10, seed=1)
    oracle = worldbench.run_bench(pop, "oracle", seed=0)
    empty = worldbench.run_bench(pop, "empty", seed=0)
    assert oracle["overall"]["composite"] == 1.0
    assert oracle["overall"]["exact_rate"] == 1.0
    assert empty["overall"]["composite"] < oracle["overall"]["composite"]


def test_predictor_ordering_oracle_ge_prefix_ge_empty():
    pop = make_population(12, seed=2)
    oracle = worldbench.run_bench(pop, "oracle")["overall"]["composite"]
    prefix = worldbench.run_bench(pop, "prefix")["overall"]["composite"]
    empty = worldbench.run_bench(pop, "empty")["overall"]["composite"]
    assert oracle >= prefix >= empty


def test_run_bench_is_deterministic_and_grouped():
    pop = make_population(9, seed=3)
    a = worldbench.run_bench(pop, "prefix", seed=5)
    b = worldbench.run_bench(pop, "prefix", seed=5)
    assert a["overall"] == b["overall"]
    assert a["n"] == len(pop)
    assert set(a["by_domain"])
    assert set(a["by_source"])


def test_source_holdout_restricts_to_ood_sources():
    sources = [f"repo-{i}" for i in range(8)]
    pop = make_population(32, seed=1, sources=sources)
    report = worldbench.run_bench(pop, "prefix", source_holdout=True, holdout_frac=0.4, seed=7)
    assert "split" in report
    holdout_sources = {s for s, where in report["split"].items() if where == "holdout"}
    assert set(report["by_source"]).issubset(holdout_sources)


def test_turing_test_oracle_is_indistinguishable_empty_is_obvious():
    pop = make_population(16, seed=4)
    oracle = worldbench.turing_test(pop, "oracle", seed=0)
    empty = worldbench.turing_test(pop, "empty", seed=0)
    assert oracle["indistinguishability"] > empty["indistinguishability"]
    assert oracle["discriminator_accuracy"] <= 0.6
    assert empty["discriminator_accuracy"] > oracle["discriminator_accuracy"]
    assert empty["discriminator_accuracy"] >= 0.7


def test_hybrid_judge_blends_with_rule_scores():
    pop = make_population(8, seed=7)

    def harsh_judge(predicted, target):
        return {"factuality": 0.0, "quality": 0.0}

    rule_only = worldbench.run_bench(pop, "oracle")
    hybrid = worldbench.run_bench(pop, "oracle", judge=harsh_judge, judge_weight=0.5)
    assert rule_only["hybrid"] is False
    assert hybrid["hybrid"] is True
    assert hybrid["overall"]["factuality"] == 0.5
    assert hybrid["overall"]["composite"] < rule_only["overall"]["composite"]


def test_turing_oracle_is_exactly_indistinguishable():
    pop = make_population(16, seed=4)
    oracle = worldbench.turing_test(pop, "oracle", seed=0)
    assert oracle["discriminator_accuracy"] == 0.5
    assert oracle["indistinguishability"] == 1.0


def test_command_predictor_rejects_bad_template():
    import pytest

    with pytest.raises(ValueError):
        worldbench.command_predictor("model {model} reads {prompt_file}")


def test_command_predictor_accepts_prompt_file_template():
    predict = worldbench.command_predictor("cat {prompt_file}")
    assert predict({"history": "hello world"}).strip() == "hello world"


def test_callable_predictor_supported():
    pop = make_population(5, seed=6)
    report = worldbench.run_bench(pop, lambda s: s["target_observation"][:5])
    assert report["n"] == len([1 for _ in report["by_source"]]) or report["n"] >= 1
    assert report["overall"]["composite"] is not None


def test_cli_worldbench_runs(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path / ".episodic"))
    populate_store(8, seed=0)
    rc = main(["worldbench", "--predictor", "oracle", "--turing"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["overall"]["composite"] == 1.0
    assert "turing" in out


def test_cli_worldbench_cmd_requires_execute(tmp_path, monkeypatch):
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path / ".episodic"))
    populate_store(3, seed=0)
    try:
        main(["worldbench", "--cmd", "cat {prompt_file}"])
        assert False, "expected SystemExit"
    except SystemExit as exc:
        assert exc.code == 1


def test_rollout_bench_oracle_is_perfect_and_undrifted():
    pop = make_population(6, seed=8)
    report = worldbench.rollout_bench(pop, "oracle")
    assert report["n_episodes"] == len(pop)
    assert report["n_scored"] == len(pop)
    assert report["mean_composite"] == 1.0
    assert report["mean_drift"] == 0.0
    assert len(report["trajectories"]) == len(pop)


def test_rollout_bench_empty_predictor_scores_lower_than_oracle():
    pop = make_population(6, seed=9)
    oracle = worldbench.rollout_bench(pop, "oracle")
    empty = worldbench.rollout_bench(pop, "empty")
    assert empty["mean_composite"] < oracle["mean_composite"]


def test_rollout_bench_max_turns_flags_truncation():
    pop = make_population(4, seed=10)
    report = worldbench.rollout_bench(pop, "oracle", max_turns=1)
    assert all(t["n"] <= 1 for t in report["trajectories"])
    assert any(t["truncated"] for t in report["trajectories"])


def test_rollout_bench_keep_rollouts_attaches_raw_turns():
    pop = make_population(3, seed=11)
    report = worldbench.rollout_bench(pop, "oracle", keep_rollouts=True)
    assert "rollouts" in report
    assert len(report["rollouts"]) == len(pop)


def test_rollout_bench_rejects_unknown_predictor_name():
    import pytest

    pop = make_population(2, seed=12)
    with pytest.raises(ValueError):
        worldbench.rollout_bench(pop, "not-a-real-predictor")


def test_rollout_bench_supports_callable_predictor():
    pop = make_population(4, seed=13)
    report = worldbench.rollout_bench(pop, lambda sample: sample["target_observation"])
    assert report["mean_composite"] == 1.0


def test_rollout_turing_test_oracle_is_indistinguishable():
    pop = make_population(10, seed=14)
    oracle = worldbench.rollout_turing_test(pop, "oracle", seed=0)
    empty = worldbench.rollout_turing_test(pop, "empty", seed=0)
    assert oracle["discriminator_accuracy"] == 0.5
    assert oracle["indistinguishability"] == 1.0
    assert empty["discriminator_accuracy"] > oracle["discriminator_accuracy"]


def test_rollout_turing_test_skips_episodes_with_no_turns():
    pop = make_population(3, seed=15)
    for ep in pop:
        ep["steps"] = []
    report = worldbench.rollout_turing_test(pop, "oracle")
    assert report["n"] == 0
    assert report["discriminator_accuracy"] is None


def test_cli_worldbench_rollout_mode_runs(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path / ".episodic"))
    populate_store(6, seed=0)
    rc = main(["worldbench", "--predictor", "oracle", "--rollout", "--turing"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["mean_composite"] == 1.0
    assert "turing" in out
    assert out["turing"]["indistinguishability"] == 1.0


def test_cli_worldbench_backend_requires_model_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path / ".episodic"))
    populate_store(3, seed=0)
    try:
        main(["worldbench", "--backend", "mlx"])
        assert False, "expected SystemExit"
    except SystemExit as exc:
        assert exc.code == 1


def test_cli_worldbench_backend_resolves_a_real_predictor(tmp_path, monkeypatch, capsys):
    from episodic.worldmodel import inference as wm_inference

    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path / ".episodic"))
    populate_store(4, seed=0)
    monkeypatch.setattr(wm_inference, "mlx_predictor", lambda model, **kw: (lambda sample: "predicted"))

    rc = main(["worldbench", "--backend", "mlx", "--model-dir", "fake/model"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["overall"]["composite"] is not None


def test_cli_worldbench_backend_unavailable_prints_hint_and_exits_zero(tmp_path, monkeypatch, capsys):
    from episodic import trainers
    from episodic.worldmodel import inference as wm_inference

    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path / ".episodic"))
    populate_store(3, seed=0)

    def raise_unavailable(model, **kw):
        raise trainers.TrainerUnavailable("mlx-sft", "install mlx-lm")

    monkeypatch.setattr(wm_inference, "mlx_predictor", raise_unavailable)
    rc = main(["worldbench", "--backend", "mlx", "--model-dir", "fake/model"])
    assert rc == 0
    assert "install mlx-lm" in capsys.readouterr().err
