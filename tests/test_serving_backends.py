import json
import sys
import types as py_types

import pytest

from episodic.serving import BackendUnavailable
from episodic.serving.openai import OpenAIBackend
from episodic.serving.vllm import VLLMBackend
from episodic.serving.ollama import OllamaBackend
from episodic.serving.tinker import TinkerBackend, _require_tinker


class FakeResponse:
    def __init__(self, json_body=None, lines=None):
        self._json_body = json_body
        self._lines = lines or []

    def read(self):
        return json.dumps(self._json_body).encode("utf-8")

    def __iter__(self):
        return iter(self._lines)


class FakeOpener:
    def __init__(self, response):
        self.calls = []
        self._response = response

    def __call__(self, request, timeout=None):
        self.calls.append({
            "url": request.full_url,
            "method": request.get_method(),
            "authorization": request.get_header("Authorization"),
            "body": json.loads(request.data.decode("utf-8")) if request.data else None,
            "timeout": timeout,
        })
        return self._response


def test_openai_backend_non_streaming_forwards_payload_and_auth():
    upstream_body = {
        "id": "chatcmpl-x", "object": "chat.completion", "model": "gpt-x",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    }
    opener = FakeOpener(FakeResponse(json_body=upstream_body))
    backend = OpenAIBackend(base_url="http://upstream.local/", model="gpt-x", api_key="sk-test", opener=opener)

    result = backend.chat_completions({"messages": [{"role": "user", "content": "hello"}]}, stream=False)

    assert result == upstream_body
    assert len(opener.calls) == 1
    call = opener.calls[0]
    assert call["url"] == "http://upstream.local/v1/chat/completions"
    assert call["method"] == "POST"
    assert call["authorization"] == "Bearer sk-test"
    assert call["body"]["model"] == "gpt-x"
    assert call["body"]["stream"] is False
    assert call["body"]["messages"][0]["content"] == "hello"


def test_openai_backend_streaming_passes_through_upstream_sse_lines():
    lines = [
        b'data: {"id":"1","choices":[{"delta":{"content":"He"}}]}\n',
        b'\n',
        b'data: {"id":"1","choices":[{"delta":{"content":"llo"}}]}\n',
        b'\n',
        b'data: [DONE]\n',
    ]
    opener = FakeOpener(FakeResponse(lines=lines))
    backend = OpenAIBackend(base_url="http://upstream.local", opener=opener)

    chunks = list(backend.chat_completions({"messages": []}, stream=True))

    assert chunks == [line.decode("utf-8") for line in lines]
    assert opener.calls[0]["body"]["stream"] is True


def test_vllm_backend_defaults_base_url_and_requires_no_api_key():
    opener = FakeOpener(FakeResponse(json_body={"choices": []}))
    backend = VLLMBackend(model="local-model", opener=opener)

    backend.chat_completions({"messages": []}, stream=False)

    call = opener.calls[0]
    assert call["url"] == "http://localhost:8000/v1/chat/completions"
    assert call["authorization"] is None


def test_http_backend_requires_base_url_when_no_default():
    from episodic.serving.base import HTTPBackend

    with pytest.raises(ValueError):
        HTTPBackend(base_url=None, opener=lambda *a, **k: None)


def test_openai_backend_models_lists_upstream_ids():
    opener = FakeOpener(FakeResponse(json_body={"data": [{"id": "gpt-a"}, {"id": "gpt-b"}]}))
    backend = OpenAIBackend(base_url="http://upstream.local", opener=opener)

    assert backend.models() == ["gpt-a", "gpt-b"]


def test_ollama_backend_non_streaming_translates_to_openai_shape():
    opener = FakeOpener(FakeResponse(json_body={"message": {"role": "assistant", "content": "hi there"}, "done": True}))
    backend = OllamaBackend(model="llama3", opener=opener)

    result = backend.chat_completions({"messages": [{"role": "user", "content": "hey"}]}, stream=False)

    assert result["object"] == "chat.completion"
    assert result["model"] == "llama3"
    assert result["choices"][0]["message"]["content"] == "hi there"
    assert opener.calls[0]["url"] == "http://localhost:11434/api/chat"
    assert opener.calls[0]["body"] == {"model": "llama3", "messages": [{"role": "user", "content": "hey"}], "stream": False}


def test_ollama_backend_streaming_translates_ndjson_to_sse_chunks():
    lines = [
        json.dumps({"message": {"content": "He"}, "done": False}).encode("utf-8"),
        json.dumps({"message": {"content": "llo"}, "done": False}).encode("utf-8"),
        json.dumps({"message": {"content": ""}, "done": True}).encode("utf-8"),
    ]
    opener = FakeOpener(FakeResponse(lines=lines))
    backend = OllamaBackend(model="llama3", opener=opener)

    chunks = list(backend.chat_completions({"messages": []}, stream=True))

    assert len(chunks) == 4
    assert all(chunk.startswith("data: ") for chunk in chunks)
    payloads = [json.loads(chunk[len("data: "):]) for chunk in chunks[:-1]]
    assert payloads[0]["choices"][0]["delta"]["content"] == "He"
    assert payloads[1]["choices"][0]["delta"]["content"] == "llo"
    assert payloads[2]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1] == "data: [DONE]\n\n"


def test_ollama_backend_models_reads_tags():
    opener = FakeOpener(FakeResponse(json_body={"models": [{"name": "llama3:8b"}]}))
    backend = OllamaBackend(opener=opener)
    assert backend.models() == ["llama3:8b"]


def _install_fake_tinker(monkeypatch, tokens=None, decoded_text="hello from tinker"):
    fake_module = py_types.ModuleType("tinker")
    fake_types_module = py_types.ModuleType("tinker.types")

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

    class FakeFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class FakeSequence:
        def __init__(self, tokens):
            self.tokens = tokens

    class FakeSampleResponse:
        def __init__(self, tokens):
            self.sequences = [FakeSequence(tokens)]

    class FakeTokenizer:
        def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True):
            return [1, 2, 3]

        def decode(self, tokens):
            return decoded_text

    class FakeSamplingClient:
        def get_tokenizer(self):
            return FakeTokenizer()

        def sample(self, prompt, num_samples, sampling_params):
            return FakeFuture(FakeSampleResponse(tokens or [4, 5, 6]))

    class FakeServiceClient:
        def create_sampling_client(self, model_path=None, base_model=None):
            return FakeSamplingClient()

    fake_types_module.ModelInput = FakeModelInput
    fake_types_module.SamplingParams = FakeSamplingParams
    fake_module.types = fake_types_module
    fake_module.ServiceClient = FakeServiceClient

    monkeypatch.setitem(sys.modules, "tinker", fake_module)
    monkeypatch.setitem(sys.modules, "tinker.types", fake_types_module)
    monkeypatch.setenv("TINKER_API_KEY", "test-key")


def test_tinker_backend_samples_via_sdk(monkeypatch):
    _install_fake_tinker(monkeypatch, decoded_text="generated reply")
    backend = TinkerBackend(sampler_path="tinker://run/weights/ckpt-1")

    result = backend.chat_completions({"messages": [{"role": "user", "content": "hi"}]}, stream=False)

    assert result["choices"][0]["message"]["content"] == "generated reply"
    assert result["model"] == "tinker://run/weights/ckpt-1"


def test_tinker_backend_streaming_yields_single_content_chunk_then_done(monkeypatch):
    _install_fake_tinker(monkeypatch, decoded_text="streamed reply")
    backend = TinkerBackend(model="Qwen/Qwen3.5-4B")

    chunks = list(backend.chat_completions({"messages": [{"role": "user", "content": "hi"}]}, stream=True))

    assert chunks[-1] == "data: [DONE]\n\n"
    first = json.loads(chunks[0][len("data: "):])
    assert first["choices"][0]["delta"]["content"] == "streamed reply"


def test_tinker_backend_unavailable_without_sdk(monkeypatch):
    monkeypatch.setitem(sys.modules, "tinker", None)
    with pytest.raises(BackendUnavailable):
        _require_tinker()


def test_tinker_backend_unavailable_without_api_key(monkeypatch):
    _install_fake_tinker(monkeypatch)
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    with pytest.raises(BackendUnavailable):
        _require_tinker()


def test_tinker_backend_requires_model_or_sampler_path():
    with pytest.raises(ValueError):
        TinkerBackend()


def test_tinker_backend_uses_injected_client_factory_bypassing_service_client(monkeypatch):
    _install_fake_tinker(monkeypatch)
    monkeypatch.delenv("TINKER_API_KEY", raising=False)

    class StubSamplingClient:
        def get_tokenizer(self):
            class T:
                def apply_chat_template(self, *a, **k):
                    return [1]

                def decode(self, tokens):
                    return "stub reply"
            return T()

        def sample(self, prompt, num_samples, sampling_params):
            class F:
                def result(self):
                    return py_types.SimpleNamespace(sequences=[py_types.SimpleNamespace(tokens=[9])])
            return F()

    backend = TinkerBackend(model="local", client_factory=lambda: StubSamplingClient())
    result = backend.chat_completions({"messages": []}, stream=False)
    assert result["choices"][0]["message"]["content"] == "stub reply"
