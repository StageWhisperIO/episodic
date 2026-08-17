import sys
import types

import pytest

from episodic import trainers
from episodic.trainers.mlx import split_rows


def _row(text):
    return {"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": text}]}


def test_mlx_sft_is_registered():
    assert "mlx-sft" in trainers.available()
    assert trainers.get("mlx-sft").consumes == ("sft",)


def test_split_holds_out_a_fraction():
    train, valid = split_rows([_row(str(i)) for i in range(5)], 0.2)
    assert len(valid) == 1 and len(train) == 4
    assert all("messages" in row and set(row) == {"messages"} for row in train + valid)


def test_split_single_row_reuses_it_for_both():
    train, valid = split_rows([_row("only")], 0.2)
    assert train == valid and len(train) == 1


def test_split_empty_raises():
    with pytest.raises(ValueError):
        split_rows([], 0.2)


class FakeMLXTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False):
        assert tokenize is False
        assert add_generation_prompt is True
        return "|".join(f"{m['role']}:{m['content']}" for m in messages)


def _install_fake_mlx(monkeypatch):
    calls = {"load": [], "generate": [], "samplers": []}

    def fake_load(model, adapter_path=None):
        calls["load"].append({"model": model, "adapter_path": adapter_path})
        return "FAKE_MODEL", FakeMLXTokenizer()

    def fake_generate(model, tokenizer, prompt, max_tokens=None, sampler=None, verbose=False):
        calls["generate"].append({
            "model": model, "prompt": prompt, "max_tokens": max_tokens,
            "sampler": sampler, "verbose": verbose,
        })
        return f"REPLY::{prompt}"

    def fake_make_sampler(temp=0.0):
        calls["samplers"].append(temp)
        return f"sampler(temp={temp})"

    fake_sample_utils = types.ModuleType("mlx_lm.sample_utils")
    fake_sample_utils.make_sampler = fake_make_sampler

    fake_mlx_lm = types.ModuleType("mlx_lm")
    fake_mlx_lm.load = fake_load
    fake_mlx_lm.generate = fake_generate
    fake_mlx_lm.sample_utils = fake_sample_utils

    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", fake_sample_utils)
    return calls


def test_load_predictor_wraps_mlx_lm_load_and_generate(monkeypatch):
    calls = _install_fake_mlx(monkeypatch)
    from episodic.trainers.mlx import load_predictor

    predict_text = load_predictor("fake/model", adapter_path="adapters/x", max_tokens=64, temperature=0.3)
    reply = predict_text([{"role": "system", "content": "sys"}, {"role": "user", "content": "hist"}])

    assert calls["load"] == [{"model": "fake/model", "adapter_path": "adapters/x"}]
    assert calls["samplers"] == [0.3]
    assert calls["generate"][0]["max_tokens"] == 64
    assert calls["generate"][0]["verbose"] is False
    assert "system:sys|user:hist" in calls["generate"][0]["prompt"]
    assert reply.startswith("REPLY::")


def test_load_predictor_requires_mlx_lm(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlx_lm", None)
    from episodic.trainers.mlx import load_predictor

    with pytest.raises(trainers.TrainerUnavailable):
        load_predictor("fake/model")
