import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from episodic import trainers
from episodic.trainers import critic as critic_mod

VOCAB = 64


def _tiny_causal_lm():
    config = transformers.LlamaConfig(
        vocab_size=VOCAB, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=256,
    )
    return transformers.LlamaForCausalLM(config)


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
        return [2 + (ord(c) % (VOCAB - 2)) for c in text[:24]]

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True, return_tensors=None):
        ids = [1] + self._ids(messages[0].get("content", ""))
        if add_generation_prompt:
            ids.append(2)
        return ids

    def __call__(self, texts, return_tensors=None, padding=True, truncation=True, max_length=None):
        rows = [self._ids(text) or [2] for text in texts]
        width = max(len(row) for row in rows)
        input_ids = [row + [0] * (width - len(row)) for row in rows]
        attention = [[1] * len(row) + [0] * (width - len(row)) for row in rows]
        return TinyBatch(
            input_ids=torch.tensor(input_ids),
            attention_mask=torch.tensor(attention),
        )

    def decode(self, tokens, skip_special_tokens=True):
        return "y" * len(tokens)

    def save_pretrained(self, out_dir):
        return None


def _reward_by_length(prompts=None, completions=None, meta=None, **kwargs):
    return [min(1.0, len(text) / 8.0) for text in (completions or [])]


REWARD_REF = f"{__name__}:_reward_by_length"


def _write_rows(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _dataset(tmp_path):
    path = tmp_path / "sft.jsonl"
    _write_rows(path, [
        {"messages": [{"role": "user", "content": "first prompt"}], "meta": {"episode_id": "ep_a"}},
        {"messages": [{"role": "user", "content": "second prompt"}], "meta": {"episode_id": "ep_b"}},
    ])
    return path


@pytest.fixture
def tiny_models(monkeypatch):
    torch.manual_seed(0)
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained",
        staticmethod(lambda name, **kwargs: _tiny_causal_lm()))
    monkeypatch.setattr(
        transformers.AutoModel, "from_pretrained",
        staticmethod(lambda name, **kwargs: _tiny_backbone()))
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained",
        staticmethod(lambda name, **kwargs: TinyTokenizer()))


def test_trl_sao_trains_and_saves(tmp_path, tiny_models):
    config = {
        "model": "tiny/policy", "device": "cpu", "max_tokens": 8,
        "learning_rate": 1e-4, "reward_funcs": [REWARD_REF],
    }

    result = trainers.get("trl-sao").train(str(_dataset(tmp_path)), str(tmp_path / "out"), config)

    assert result["method"] == "sao"
    assert result["baseline"] == "running_mean"
    assert result["prompts"] == 2
    assert result["steps"] == 2
    assert result["updates"] >= 1
    assert result["mean_reward"] is not None
    for entry in result["history"]:
        if entry["updated"]:
            assert "clip_fraction" in entry
            assert isinstance(entry["loss"], float)
    assert (tmp_path / "out" / "config.json").exists()


def test_trl_sao_with_local_critic(tmp_path, tiny_models):
    config = {
        "model": "tiny/policy", "device": "cpu", "max_tokens": 8,
        "critic_model": "tiny/critic", "critic_device": "cpu", "critic_updates": 2,
        "reward_funcs": [REWARD_REF],
    }

    result = trainers.get("trl-sao").train(str(_dataset(tmp_path)), str(tmp_path / "out"), config)

    assert result["baseline"] == "critic"
    assert result["critic_model"] == "tiny/critic"
    for entry in result["history"]:
        if entry["updated"]:
            assert isinstance(entry["critic_loss"], float)


def test_local_critic_value_update_and_frozen_attention(tiny_models):
    critic = critic_mod.LocalCritic(model_name="tiny/critic", device="cpu", learning_rate=1e-3)

    frozen = [n for n, p in critic.backbone.named_parameters()
              if critic_mod.attention_parameter(n)]
    assert frozen
    assert all(not p.requires_grad for n, p in critic.backbone.named_parameters()
               if critic_mod.attention_parameter(n))

    values = critic.value(["some prompt", "another prompt"])
    assert len(values) == 2

    first = critic.update(["some prompt"] * 4, [0.9] * 4)
    for _ in range(30):
        last = critic.update(["some prompt"] * 4, [0.9] * 4)
    assert last < first


def test_local_critic_pretrain_from_reward_rows(tiny_models):
    rows = [
        {"prompt": "fix the bug", "scalar_reward": 0.8},
        {"prompt": "add a feature", "scalar_reward": 0.3},
        {"prompt": "", "scalar_reward": 0.5},
        {"prompt": "no reward"},
    ]
    pairs = critic_mod.pretrain_pairs_from_reward_rows(rows)
    assert pairs == [("fix the bug", 0.8), ("add a feature", 0.3)]

    critic = critic_mod.LocalCritic(model_name="tiny/critic", device="cpu", learning_rate=1e-3)
    losses = critic.pretrain(pairs, epochs=2, batch_size=1)
    assert len(losses) == 4
    assert all(isinstance(loss, float) for loss in losses)


def test_attention_parameter_name_matching():
    assert critic_mod.attention_parameter("model.layers.0.self_attn.q_proj.weight")
    assert critic_mod.attention_parameter("encoder.attention.output.dense.weight")
    assert not critic_mod.attention_parameter("model.layers.0.mlp.gate_proj.weight")
