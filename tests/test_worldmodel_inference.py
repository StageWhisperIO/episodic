import sys
import types

import pytest

from episodic import trainers
from episodic.worldmodel import inference


class FakeMLXTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False):
        return "|".join(f"{m['role']}:{m['content']}" for m in messages)


def _install_fake_mlx(monkeypatch):
    calls = {"load": [], "generate": []}

    def fake_load(model, adapter_path=None):
        calls["load"].append({"model": model, "adapter_path": adapter_path})
        return "FAKE_MODEL", FakeMLXTokenizer()

    def fake_generate(model, tokenizer, prompt, max_tokens=None, sampler=None, verbose=False):
        calls["generate"].append({"prompt": prompt, "max_tokens": max_tokens})
        return f"REPLY::{prompt}"

    fake_sample_utils = types.ModuleType("mlx_lm.sample_utils")
    fake_sample_utils.make_sampler = lambda temp=0.0: f"sampler({temp})"

    fake_mlx_lm = types.ModuleType("mlx_lm")
    fake_mlx_lm.load = fake_load
    fake_mlx_lm.generate = fake_generate
    fake_mlx_lm.sample_utils = fake_sample_utils

    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", fake_sample_utils)
    return calls


def test_mlx_predictor_wraps_history_into_wm_system_messages(monkeypatch):
    calls = _install_fake_mlx(monkeypatch)

    predict = inference.mlx_predictor("fake/model")
    out = predict({"history": "INTENT: fix bug\nACTION: Edit(...)\nOBSERVATION:"})

    assert calls["load"] == [{"model": "fake/model", "adapter_path": None}]
    prompt = calls["generate"][0]["prompt"]
    assert "system:" in prompt
    assert inference.WM_SYSTEM in prompt
    assert "INTENT: fix bug" in prompt
    assert out.startswith("REPLY::")


def test_mlx_predictor_raises_trainer_unavailable_without_mlx_lm(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlx_lm", None)
    with pytest.raises(trainers.TrainerUnavailable):
        inference.mlx_predictor("fake/model")


class FakeFuture:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


class FakeTinkerTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True):
        return [1, 2, 3]

    def decode(self, tokens):
        return "predicted-text"


class FakeSampledSequence:
    def __init__(self, tokens):
        self.tokens = tokens
        self.logprobs = [0.1] * len(tokens)


class FakeSamplingClient:
    def __init__(self, calls, model_path):
        self.calls = calls
        self.model_path = model_path

    def sample(self, prompt, num_samples, sampling_params):
        self.calls["sample"].append({
            "model_path": self.model_path,
            "max_tokens": sampling_params.max_tokens,
            "temperature": sampling_params.temperature,
        })
        return FakeFuture(types.SimpleNamespace(sequences=[FakeSampledSequence([9])]))


class FakeTrainingClient:
    def get_tokenizer(self):
        return FakeTinkerTokenizer()


class FakeServiceClient:
    def __init__(self, calls):
        self.calls = calls

    def create_lora_training_client(self, base_model, rank):
        self.calls["create_lora_training_client"].append({"base_model": base_model, "rank": rank})
        return FakeTrainingClient()

    def create_sampling_client(self, model_path):
        self.calls["create_sampling_client"].append(model_path)
        return FakeSamplingClient(self.calls, model_path)


def _install_fake_tinker(monkeypatch):
    calls = {"create_lora_training_client": [], "create_sampling_client": [], "sample": []}

    class FakeModelInput:
        @classmethod
        def from_ints(cls, ids):
            obj = cls()
            obj.ids = list(ids)
            return obj

    class FakeSamplingParams:
        def __init__(self, max_tokens, temperature):
            self.max_tokens = max_tokens
            self.temperature = temperature

    fake_types = types.ModuleType("tinker.types")
    fake_types.ModelInput = FakeModelInput
    fake_types.SamplingParams = FakeSamplingParams

    fake_tinker = types.ModuleType("tinker")
    fake_tinker.types = fake_types
    fake_tinker.ServiceClient = lambda *a, **k: FakeServiceClient(calls)

    monkeypatch.setitem(sys.modules, "tinker", fake_tinker)
    monkeypatch.setitem(sys.modules, "tinker.types", fake_types)
    monkeypatch.setenv("TINKER_API_KEY", "test-key")
    return calls


def test_tinker_predictor_samples_through_sampling_client(monkeypatch):
    calls = _install_fake_tinker(monkeypatch)

    predict = inference.tinker_predictor("tinker://sampler/abc", base_model="fake/base", max_tokens=50)
    out = predict({"history": "INTENT: ship it\nACTION: Bash(pytest)\nOBSERVATION:"})

    assert calls["create_lora_training_client"] == [{"base_model": "fake/base", "rank": 1}]
    assert calls["create_sampling_client"] == ["tinker://sampler/abc"]
    assert calls["sample"][0]["max_tokens"] == 50
    assert out == "predicted-text"


def test_tinker_predictor_raises_trainer_unavailable_without_api_key(monkeypatch):
    _install_fake_tinker(monkeypatch)
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    with pytest.raises(trainers.TrainerUnavailable):
        inference.tinker_predictor("tinker://sampler/abc")
