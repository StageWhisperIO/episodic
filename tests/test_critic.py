import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from episodic.trainers import critic as critic_mod

VOCAB = 64


def _tiny_reward_model():
    config = transformers.LlamaConfig(
        vocab_size=VOCAB, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=256, num_labels=1, pad_token_id=0,
    )
    return transformers.LlamaForSequenceClassification(config)


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
def tiny_reward_model(monkeypatch):
    torch.manual_seed(0)
    monkeypatch.setattr(
        transformers.AutoModelForSequenceClassification, "from_pretrained",
        staticmethod(lambda name, **kwargs: _tiny_reward_model()))
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained",
        staticmethod(lambda name, **kwargs: TinyTokenizer()))


def test_load_trl_reward_model_returns_a_value_shaped_object(tiny_reward_model):
    model = critic_mod.load_trl_reward_model("some/model-dir", device="cpu")
    assert model.model_dir == "some/model-dir"
    values = model.value(["fix the bug", "add a feature"])
    assert len(values) == 2
    assert all(0.0 <= v <= 1.0 for v in values)


def test_load_trl_reward_model_composes_with_critic_judge(tiny_reward_model):
    from episodic.core import rubric

    model = critic_mod.load_trl_reward_model("some/model-dir", device="cpu")
    judge = rubric.critic_judge(model, render=lambda episode: episode["text"])
    satisfied, reason = judge({"id": "ep_a", "text": "fix the bug"}, {"desc": "x"})
    assert 0.0 <= satisfied <= 1.0
    assert "critic score" in reason


def test_load_trl_reward_model_requires_torch(monkeypatch):
    monkeypatch.setattr(critic_mod, "_require_torch", critic_mod._require_torch)

    def boom(name):
        raise critic_mod.TrainerUnavailable(name, critic_mod.HINT)

    monkeypatch.setattr(critic_mod, "_require_torch", boom)
    with pytest.raises(critic_mod.TrainerUnavailable):
        critic_mod.load_trl_reward_model("some/model-dir")
