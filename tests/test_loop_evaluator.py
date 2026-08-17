import pytest

from episodic.loop import evaluator
from episodic import testing


def test_build_defaults_to_rubric_judge_passthrough():
    judge = object()
    state = evaluator.build({}, judge)
    assert state == {"type": "rubric_judge", "judge": judge, "critic": None}


def test_build_unknown_evaluator_type_falls_back_to_passthrough():
    judge = object()
    state = evaluator.build({"evaluator": {"type": "bogus"}}, judge)
    assert state["type"] == "rubric_judge"
    assert state["judge"] is judge


def test_build_local_critic_without_critic_model_falls_back_to_passthrough():
    judge = object()
    state = evaluator.build({"evaluator": {"type": "local_critic"}}, judge)
    assert state == {"type": "rubric_judge", "judge": judge, "critic": None}


def test_build_trl_reward_without_model_dir_falls_back_to_passthrough():
    judge = object()
    state = evaluator.build({"evaluator": {"type": "trl_reward"}}, judge)
    assert state == {"type": "rubric_judge", "judge": judge, "critic": None}


def test_refresh_is_a_noop_for_rubric_judge():
    judge = object()
    state = evaluator.build({}, judge)
    episodes = testing.make_population(3)
    refreshed = evaluator.refresh(state, {}, episodes, 0, "/tmp/out", None)
    assert refreshed is state


def test_refresh_is_a_noop_without_train_episodes():
    config = {"evaluator": {"type": "local_critic", "critic_model": "tiny/critic"}}
    state = {"type": "local_critic", "judge": None, "critic": None}
    refreshed = evaluator.refresh(state, config, [], 0, "/tmp/out", None)
    assert refreshed is state


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

VOCAB = 64


def _tiny_backbone():
    config = transformers.LlamaConfig(
        vocab_size=VOCAB, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=256,
    )
    return transformers.LlamaModel(config)


class TinyBatch(dict):
    def to(self, device):
        return self


class TinyTokenizer:
    pad_token = "<pad>"
    pad_token_id = 0
    eos_token = "<eos>"
    eos_token_id = 1

    def _ids(self, text):
        return [2 + (ord(c) % (VOCAB - 2)) for c in text[:24]] or [2]

    def __call__(self, texts, return_tensors=None, padding=True, truncation=True, max_length=None):
        rows = [self._ids(text) for text in texts]
        width = max(len(row) for row in rows)
        input_ids = [row + [0] * (width - len(row)) for row in rows]
        attention = [[1] * len(row) + [0] * (width - len(row)) for row in rows]
        return TinyBatch(
            input_ids=torch.tensor(input_ids),
            attention_mask=torch.tensor(attention),
        )

    def save_pretrained(self, out_dir):
        return None


@pytest.fixture
def tiny_models(monkeypatch):
    torch.manual_seed(0)
    monkeypatch.setattr(
        transformers.AutoModel, "from_pretrained",
        staticmethod(lambda name, **kwargs: _tiny_backbone()))
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained",
        staticmethod(lambda name, **kwargs: TinyTokenizer()))


def test_build_local_critic_constructs_a_judge(tiny_models):
    config = {"evaluator": {"type": "local_critic", "critic_model": "tiny/critic", "critic_device": "cpu"}}
    state = evaluator.build(config, judge=None)
    assert state["type"] == "local_critic"
    assert state["critic"] is not None
    satisfied, reason = state["judge"](testing.make_episode("ep_x"), {"desc": "x"})
    assert 0.0 <= satisfied <= 1.0
    assert "critic score" in reason


def test_refresh_local_critic_retrains_the_existing_critic(tmp_path, tiny_models):
    config = {"evaluator": {"type": "local_critic", "critic_model": "tiny/critic", "critic_device": "cpu"}}
    state = evaluator.build(config, judge=None)
    original_critic = state["critic"]
    episodes = testing.make_population(4)

    refreshed = evaluator.refresh(state, config, episodes, 0, tmp_path, None)

    assert refreshed["type"] == "local_critic"
    assert refreshed["critic"] is original_critic
    assert (tmp_path / "evaluator" / "epoch_0" / "dataset" / "reward.jsonl").exists()
    satisfied, _ = refreshed["judge"](episodes[0], {"desc": "x"})
    assert 0.0 <= satisfied <= 1.0


def test_refresh_local_critic_lazily_builds_a_critic_when_none_was_built_yet(tmp_path, tiny_models):
    config = {"evaluator": {"type": "local_critic", "critic_model": "tiny/critic", "critic_device": "cpu"}}
    state = {"type": "local_critic", "judge": None, "critic": None}
    episodes = testing.make_population(4)

    refreshed = evaluator.refresh(state, config, episodes, 0, tmp_path, None)

    assert refreshed["critic"] is not None
    assert callable(refreshed["judge"])


def test_refresh_trl_reward_exports_dpo_trains_and_loads_the_model(tmp_path, monkeypatch):
    from episodic.loop import evaluator as evaluator_mod

    captured = {}

    def fake_train(trainer_name, dataset_path, out_dir, config, cwd=None):
        captured["trainer_name"] = trainer_name
        captured["dataset_path"] = dataset_path
        captured["out_dir"] = out_dir
        captured["config"] = config
        return {"result": {"model_dir": out_dir}}

    class FakeCritic:
        def __init__(self, model_dir):
            self.model_dir = model_dir

        def value(self, texts):
            return [0.5 for _ in texts]

    monkeypatch.setattr(evaluator_mod.trainers, "train", fake_train)
    monkeypatch.setattr(evaluator_mod.critic_mod, "load_trl_reward_model",
                        lambda model_dir, device=None: FakeCritic(model_dir))

    config = {"evaluator": {"type": "trl_reward", "model": "tiny/reward"}}
    state = evaluator.build(config, judge=None)
    assert state["type"] == "rubric_judge"

    episodes = testing.make_population(3)
    refreshed = evaluator.refresh(state, config, episodes, 2, tmp_path, "/repo")

    assert refreshed["type"] == "trl_reward"
    assert captured["trainer_name"] == "trl-reward"
    assert captured["dataset_path"].endswith("dpo.jsonl")
    assert captured["config"]["model"] == "tiny/reward"
    assert refreshed["model_dir"] == captured["out_dir"]
    satisfied, reason = refreshed["judge"](episodes[0], {"desc": "x"})
    assert satisfied == 0.5
    assert "critic score" in reason


def test_refresh_trl_reward_reuses_configured_model_when_no_explicit_model_key(tmp_path, monkeypatch):
    from episodic.loop import evaluator as evaluator_mod

    captured = {}

    def fake_train(trainer_name, dataset_path, out_dir, config, cwd=None):
        captured["config"] = config
        return {"result": {"model_dir": out_dir}}

    monkeypatch.setattr(evaluator_mod.trainers, "train", fake_train)
    monkeypatch.setattr(evaluator_mod.critic_mod, "load_trl_reward_model",
                        lambda model_dir, device=None: object())

    config = {"evaluator": {"type": "trl_reward"}}
    state = {"type": "trl_reward", "judge": None, "critic": None}
    episodes = testing.make_population(2)

    evaluator.refresh(state, config, episodes, 0, tmp_path, None)

    assert captured["config"]["model"] == evaluator_mod.DEFAULT_REWARD_MODEL
