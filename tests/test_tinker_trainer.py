import json
import sys
import types as py_types

import pytest

from episodic import trainers

ROLE_TOKENS = {"system": 1, "user": 2, "assistant": 3}
GEN_TOKEN = 99


class FakeFuture:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


class FakeModelInput:
    @classmethod
    def from_ints(cls, ids):
        obj = cls()
        obj.ids = list(ids)
        return obj


class FakeDatum:
    def __init__(self, model_input, loss_fn_inputs):
        self.model_input = model_input
        self.loss_fn_inputs = loss_fn_inputs


class FakeAdamParams:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate


class FakeSamplingParams:
    def __init__(self, max_tokens, temperature):
        self.max_tokens = max_tokens
        self.temperature = temperature


class FakeForwardBackwardOutput:
    def __init__(self, metrics, loss_fn_outputs=None):
        self.metrics = metrics
        self.loss_fn_outputs = loss_fn_outputs or []


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return list(self.values)


class FakeSaveResult:
    def __init__(self, path):
        self.path = path


class FakeSampledSequence:
    def __init__(self, tokens, logprobs):
        self.tokens = tokens
        self.logprobs = logprobs


class FakeSampleResponse:
    def __init__(self, sequences):
        self.sequences = sequences


class FakeTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True):
        ids = []
        for message in messages:
            ids.append(ROLE_TOKENS.get(message["role"], 0))
            ids.extend(100 + (ord(c) % 50) for c in message.get("content", ""))
        if add_generation_prompt:
            ids.append(GEN_TOKEN)
        return ids

    def decode(self, tokens):
        return "x" * len(tokens)


class FakeTrainingClient:
    def __init__(self, calls):
        self.calls = calls
        self.tokenizer = FakeTokenizer()

    def get_tokenizer(self):
        return self.tokenizer

    def forward_backward(self, data, loss_fn, loss_fn_config=None):
        self.calls["forward_backward"].append({"data": data, "loss_fn": loss_fn})
        return FakeFuture(FakeForwardBackwardOutput({"loss:sum": 4.0}))

    def forward(self, data, loss_fn, loss_fn_config=None):
        self.calls["forward"].append({"data": data, "loss_fn": loss_fn})
        shift = self.calls.get("forward_shift", 0.0)
        outputs = [
            {"logprobs": FakeTensor([shift] * len(datum.loss_fn_inputs["target_tokens"]))}
            for datum in data
        ]
        return FakeFuture(FakeForwardBackwardOutput({"loss:sum": 1.0}, outputs))

    def optim_step(self, adam_params):
        self.calls["optim_step"].append(adam_params)
        return FakeFuture(object())

    def save_state(self, name, ttl_seconds=None, overwrite=False):
        self.calls["save_state"].append({"name": name, "ttl_seconds": ttl_seconds, "overwrite": overwrite})
        return FakeFuture(FakeSaveResult(f"tinker://state/{name}"))

    def save_weights_for_sampler(self, name, ttl_seconds=None):
        self.calls["save_weights_for_sampler"].append({"name": name, "ttl_seconds": ttl_seconds})
        return FakeFuture(FakeSaveResult(f"tinker://sampler/{name}"))


class FakeSamplingClient:
    def __init__(self, calls, model_path):
        self.calls = calls
        self.model_path = model_path

    def sample(self, prompt, num_samples, sampling_params):
        self.calls["sample"].append({
            "model_path": self.model_path,
            "num_samples": num_samples,
            "max_tokens": sampling_params.max_tokens,
            "temperature": sampling_params.temperature,
        })
        sequences = [FakeSampledSequence(list(range(i + 1)), [0.1] * (i + 1)) for i in range(num_samples)]
        return FakeFuture(FakeSampleResponse(sequences))


class FakeServiceClient:
    def __init__(self, calls):
        self.calls = calls

    def create_lora_training_client(self, base_model, rank):
        self.calls["create_lora_training_client"].append({"base_model": base_model, "rank": rank})
        return FakeTrainingClient(self.calls)

    def create_training_client_from_state(self, path):
        self.calls["create_training_client_from_state"].append(path)
        return FakeTrainingClient(self.calls)

    def create_sampling_client(self, model_path):
        self.calls["create_sampling_client"].append(model_path)
        return FakeSamplingClient(self.calls, model_path)


def _install_fake_tinker(monkeypatch):
    calls = {key: [] for key in (
        "create_lora_training_client", "create_training_client_from_state",
        "create_sampling_client", "sample", "forward", "forward_backward",
        "optim_step", "save_state", "save_weights_for_sampler",
    )}
    fake_types = py_types.ModuleType("tinker.types")
    fake_types.Datum = FakeDatum
    fake_types.ModelInput = FakeModelInput
    fake_types.AdamParams = FakeAdamParams
    fake_types.SamplingParams = FakeSamplingParams

    fake_tinker = py_types.ModuleType("tinker")
    fake_tinker.types = fake_types
    fake_tinker.ServiceClient = lambda *args, **kwargs: FakeServiceClient(calls)

    monkeypatch.setitem(sys.modules, "tinker", fake_tinker)
    monkeypatch.setitem(sys.modules, "tinker.types", fake_types)
    monkeypatch.setenv("TINKER_API_KEY", "test-key")
    return calls


def _write_rows(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _sft_rows():
    return [
        {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}],
         "meta": {"episode_id": "ep_a"}},
        {"messages": [{"role": "user", "content": "go"}, {"role": "assistant", "content": "ok"}],
         "meta": {"episode_id": "ep_b"}},
    ]


def _sao_rows():
    return [
        {"messages": [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}],
         "meta": {"episode_id": "ep_a"}},
        {"messages": [{"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"}],
         "meta": {"episode_id": "ep_b"}},
    ]


def _length_reward(prompts=None, completions=None, meta=None, **kwargs):
    return [float(len(text)) for text in (completions or [])]


REWARD_REF = f"{__name__}:_length_reward"


def test_tinker_sft_happy_path(tmp_path, monkeypatch):
    calls = _install_fake_tinker(monkeypatch)
    dataset = tmp_path / "sft.jsonl"
    _write_rows(dataset, _sft_rows())
    config = {"model": "fake/sft-model", "epochs": 2, "batch_size": 1, "learning_rate": 5e-5, "lora_rank": 8}

    manifest = trainers.train("tinker-sft", str(dataset), str(tmp_path / "out"), config, cwd=str(tmp_path))
    result = manifest["result"]

    assert manifest["trainer"] == "tinker-sft"
    assert result["backend"] == "tinker"
    assert result["base_model"] == "fake/sft-model"
    assert result["method"] == "lora"
    assert result["lora_rank"] == 8
    assert result["examples"] == 2
    assert result["epochs"] == 2
    assert result["steps"] == 4
    assert result["final_loss"] is not None
    assert result["mean_loss"] is not None
    assert len(result["loss_curve"]) == 4
    assert result["state_path"] == "tinker://state/episodic-sft"
    assert result["sampler_path"] == "tinker://sampler/episodic-sft-4"
    assert len(calls["forward_backward"]) == 4
    assert len(calls["optim_step"]) == 4


def test_tinker_sft_config_knobs_reach_client(tmp_path, monkeypatch):
    calls = _install_fake_tinker(monkeypatch)
    dataset = tmp_path / "sft.jsonl"
    _write_rows(dataset, _sft_rows())
    config = {"model": "fake/knobs", "lora_rank": 16, "learning_rate": 3e-4, "batch_size": 2, "epochs": 1}

    trainers.get("tinker-sft").train(str(dataset), str(tmp_path / "out"), config)

    assert calls["create_lora_training_client"] == [{"base_model": "fake/knobs", "rank": 16}]
    assert len(calls["forward_backward"]) == 1
    assert len(calls["forward_backward"][0]["data"]) == 2
    assert all(params.learning_rate == 3e-4 for params in calls["optim_step"])


def test_tinker_sft_max_rows_limits_dataset(tmp_path, monkeypatch):
    calls = _install_fake_tinker(monkeypatch)
    dataset = tmp_path / "sft.jsonl"
    rows = _sft_rows() + [
        {"messages": [{"role": "user", "content": "third"}, {"role": "assistant", "content": "resp"}]},
    ]
    _write_rows(dataset, rows)
    config = {"max_rows": 1, "batch_size": 5}

    result = trainers.get("tinker-sft").train(str(dataset), str(tmp_path / "out"), config)

    assert result["examples"] == 1
    assert len(calls["forward_backward"]) == 1
    assert len(calls["forward_backward"][0]["data"]) == 1


def test_tinker_sao_single_rollout_happy_path(tmp_path, monkeypatch):
    calls = _install_fake_tinker(monkeypatch)
    dataset = tmp_path / "sft.jsonl"
    _write_rows(dataset, _sao_rows())
    config = {
        "model": "fake/sao-model", "lora_rank": 4, "learning_rate": 2e-5,
        "reward_funcs": [REWARD_REF],
    }

    manifest = trainers.train("tinker-sao", str(dataset), str(tmp_path / "out"), config, cwd=str(tmp_path))
    result = manifest["result"]

    assert result["backend"] == "tinker"
    assert result["base_model"] == "fake/sao-model"
    assert result["method"] == "sao"
    assert result["warm_start"] is False
    assert result["prompts"] == 2
    assert result["steps"] == 2
    assert result["updates"] == 2
    assert len(result["history"]) == 2
    for entry in result["history"]:
        assert entry["updated"] is True
        assert entry["loss_sum"] == 4.0
        assert entry["clip_fraction"] == 0.0
        assert entry["sampler_path"] == "tinker://sampler/sao-0"
    assert result["state_path"] == "tinker://state/episodic-sao"
    assert result["sampler_path"] == "tinker://sampler/episodic-sao-final"
    assert all(call["num_samples"] == 1 for call in calls["sample"])
    assert len(calls["create_sampling_client"]) == 1
    assert len(calls["forward"]) == 2
    assert len(calls["forward_backward"]) == 2
    assert all(call["loss_fn"] == "importance_sampling" for call in calls["forward_backward"])
    assert len(calls["optim_step"]) == 2
    assert all(params.learning_rate == 2e-5 for params in calls["optim_step"])


def test_tinker_sao_config_knobs_reach_client(tmp_path, monkeypatch):
    calls = _install_fake_tinker(monkeypatch)
    dataset = tmp_path / "sft.jsonl"
    _write_rows(dataset, _sao_rows())
    config = {
        "model": "fake/sao-knobs", "lora_rank": 2, "batch_size": 2,
        "max_tokens": 64, "temperature": 0.5, "epsilon_low": 0.1, "epsilon_high": 0.3,
        "baseline_window": 4, "sampler_refresh_steps": 2, "reward_funcs": [REWARD_REF],
    }

    result = trainers.get("tinker-sao").train(str(dataset), str(tmp_path / "out"), config)

    assert calls["create_lora_training_client"] == [{"base_model": "fake/sao-knobs", "rank": 2}]
    assert calls["sample"]
    assert all(call["num_samples"] == 1 for call in calls["sample"])
    assert all(call["max_tokens"] == 64 and call["temperature"] == 0.5 for call in calls["sample"])
    assert result["epsilon_low"] == 0.1
    assert result["epsilon_high"] == 0.3
    assert result["baseline_window"] == 4
    assert result["sampler_refresh_steps"] == 2


def test_tinker_sao_dis_masks_offpolicy_tokens(tmp_path, monkeypatch):
    calls = _install_fake_tinker(monkeypatch)
    calls["forward_shift"] = -5.0
    dataset = tmp_path / "sft.jsonl"
    _write_rows(dataset, _sao_rows())
    config = {"reward_funcs": [REWARD_REF]}

    result = trainers.get("tinker-sao").train(str(dataset), str(tmp_path / "out"), config)

    for entry in result["history"]:
        assert entry["clip_fraction"] == 1.0
    for call in calls["forward_backward"]:
        for datum in call["data"]:
            assert all(value == 0.0 for value in datum.loss_fn_inputs["advantages"])


def test_tinker_sao_running_mean_baseline(tmp_path, monkeypatch):
    calls = _install_fake_tinker(monkeypatch)
    dataset = tmp_path / "sft.jsonl"
    _write_rows(dataset, _sao_rows())
    config = {"batch_size": 2, "reward_funcs": [REWARD_REF]}

    result = trainers.get("tinker-sao").train(str(dataset), str(tmp_path / "out"), config)

    data = calls["forward_backward"][0]["data"]
    assert len(data) == 2
    first_advantages = data[0].loss_fn_inputs["advantages"]
    second_advantages = data[1].loss_fn_inputs["advantages"]
    assert first_advantages[-1] == 1.0
    assert all(value == 0.0 for value in second_advantages)
    assert result["history"][0]["advantage_mean"] == 0.5


def test_tinker_sao_sampler_refresh_steps(tmp_path, monkeypatch):
    calls = _install_fake_tinker(monkeypatch)
    dataset = tmp_path / "sft.jsonl"
    _write_rows(dataset, _sao_rows())
    config = {"sampler_refresh_steps": 1, "reward_funcs": [REWARD_REF]}

    trainers.get("tinker-sao").train(str(dataset), str(tmp_path / "out"), config)

    names = [call["name"] for call in calls["save_weights_for_sampler"]]
    assert names == ["sao-0", "sao-1", "episodic-sao-final"]


@pytest.mark.parametrize("trainer_name", ["tinker-sft", "tinker-sao"])
def test_tinker_missing_api_key_raises(tmp_path, monkeypatch, trainer_name):
    _install_fake_tinker(monkeypatch)
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    dataset = tmp_path / "sft.jsonl"
    _write_rows(dataset, _sft_rows())

    with pytest.raises(trainers.TrainerUnavailable) as info:
        trainers.get(trainer_name).train(str(dataset), str(tmp_path / "out"), {})
    assert "TINKER_API_KEY" in info.value.hint


def test_tinker_sft_sampler_ttl_defaults_to_finite(tmp_path, monkeypatch):
    calls = _install_fake_tinker(monkeypatch)
    dataset = tmp_path / "sft.jsonl"
    _write_rows(dataset, _sft_rows())

    result = trainers.get("tinker-sft").train(str(dataset), str(tmp_path / "out"), {})

    assert calls["save_weights_for_sampler"][-1]["ttl_seconds"] == 7 * 24 * 3600
    assert result["sampler_ttl_seconds"] == 7 * 24 * 3600


def test_tinker_sft_sampler_ttl_explicit_none_is_persistent(tmp_path, monkeypatch):
    calls = _install_fake_tinker(monkeypatch)
    dataset = tmp_path / "sft.jsonl"
    _write_rows(dataset, _sft_rows())

    result = trainers.get("tinker-sft").train(
        str(dataset), str(tmp_path / "out"), {"sampler_ttl_seconds": None})

    assert calls["save_weights_for_sampler"][-1]["ttl_seconds"] is None
    assert result["sampler_ttl_seconds"] is None


def test_tinker_sft_sampler_ttl_knob_applies(tmp_path, monkeypatch):
    calls = _install_fake_tinker(monkeypatch)
    dataset = tmp_path / "sft.jsonl"
    _write_rows(dataset, _sft_rows())

    result = trainers.get("tinker-sft").train(
        str(dataset), str(tmp_path / "out"), {"sampler_ttl_seconds": 1800})

    assert calls["save_weights_for_sampler"][-1]["ttl_seconds"] == 1800
    assert result["sampler_ttl_seconds"] == 1800
    assert result["sampler_path"]
    assert result["state_path"]


def test_tinker_sao_sampler_ttl_defaults_split_refresh_and_final(tmp_path, monkeypatch):
    calls = _install_fake_tinker(monkeypatch)
    dataset = tmp_path / "sft.jsonl"
    _write_rows(dataset, _sao_rows())
    config = {"reward_funcs": [REWARD_REF]}

    result = trainers.get("tinker-sao").train(str(dataset), str(tmp_path / "out"), config)

    ttls = [call["ttl_seconds"] for call in calls["save_weights_for_sampler"]]
    assert ttls == [3600, 7 * 24 * 3600]
    assert result["sampler_ttl_seconds"] == 7 * 24 * 3600


def test_tinker_sao_sampler_ttl_explicit_none_is_persistent(tmp_path, monkeypatch):
    calls = _install_fake_tinker(monkeypatch)
    dataset = tmp_path / "sft.jsonl"
    _write_rows(dataset, _sao_rows())
    config = {"reward_funcs": [REWARD_REF], "sampler_ttl_seconds": None}

    result = trainers.get("tinker-sao").train(str(dataset), str(tmp_path / "out"), config)

    ttls = [call["ttl_seconds"] for call in calls["save_weights_for_sampler"]]
    assert ttls == [None, None]
    assert result["sampler_ttl_seconds"] is None


def test_tinker_sao_sampler_ttl_knob_applies_to_all_checkpoints(tmp_path, monkeypatch):
    calls = _install_fake_tinker(monkeypatch)
    dataset = tmp_path / "sft.jsonl"
    _write_rows(dataset, _sao_rows())
    config = {"reward_funcs": [REWARD_REF], "sampler_ttl_seconds": 900}

    result = trainers.get("tinker-sao").train(str(dataset), str(tmp_path / "out"), config)

    ttls = [call["ttl_seconds"] for call in calls["save_weights_for_sampler"]]
    assert ttls == [900, 900]
    assert result["sampler_ttl_seconds"] == 900
