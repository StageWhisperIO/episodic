import os

from . import register, BackendUnavailable
from .base import chat_completion_response, chat_completion_chunk, new_id, sse_frame, sse_done

HINT = (
    "Tinker backend needs the SDK and an API key: pip install tinker, then "
    "export TINKER_API_KEY. Serving samples from a saved sampler/model path "
    "(https://tinker-console.thinkingmachines.ai)."
)


def _require_tinker():
    try:
        import tinker  # noqa: F401
    except ImportError as exc:
        raise BackendUnavailable("tinker", HINT) from exc
    if not os.environ.get("TINKER_API_KEY"):
        raise BackendUnavailable("tinker", HINT)


def _token_ids(rendered):
    ids = rendered["input_ids"] if hasattr(rendered, "keys") else rendered
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return list(ids)


def _strip_reasoning(text):
    marker = "</think>"
    index = text.rfind(marker)
    return text[index + len(marker):].strip() if index != -1 else text.strip()


class TinkerBackend:
    name = "tinker"

    def __init__(self, model=None, sampler_path=None, max_tokens=512, temperature=0.7, client_factory=None):
        if not model and not sampler_path:
            raise ValueError("tinker: model or sampler_path is required")
        self.model = model
        self.sampler_path = sampler_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client_factory = client_factory

    @classmethod
    def from_config(cls, config):
        config = dict(config or {})
        return cls(
            model=config.get("model"),
            sampler_path=config.get("sampler_path"),
            max_tokens=config.get("max_tokens", 512),
            temperature=config.get("temperature", 0.7),
            client_factory=config.get("client_factory"),
        )

    def _sampling_client(self):
        if self._client_factory is not None:
            return self._client_factory()
        _require_tinker()
        import tinker

        service = tinker.ServiceClient()
        if self.sampler_path:
            return service.create_sampling_client(model_path=self.sampler_path)
        return service.create_sampling_client(base_model=self.model)

    def _sample_text(self, messages):
        sampling = self._sampling_client()
        from tinker import types

        tokenizer = sampling.get_tokenizer()
        prompt_ids = _token_ids(tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True))
        response = sampling.sample(
            prompt=types.ModelInput.from_ints(prompt_ids),
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=self.max_tokens, temperature=self.temperature),
        ).result()
        sequence = response.sequences[0]
        return _strip_reasoning(tokenizer.decode(list(sequence.tokens)))

    def chat_completions(self, payload, stream=False):
        messages = payload.get("messages") or []
        model_id = self.sampler_path or self.model
        text = self._sample_text(messages)
        if stream:
            return self._stream(model_id, text)
        return chat_completion_response(model_id, text)

    def _stream(self, model_id, text):
        chunk_id = new_id()
        yield sse_frame(chat_completion_chunk(model_id, {"role": "assistant", "content": text}, chunk_id=chunk_id))
        yield sse_frame(chat_completion_chunk(model_id, {}, finish_reason="stop", chunk_id=chunk_id))
        yield sse_done()

    def models(self):
        return [self.sampler_path or self.model]


register(TinkerBackend)
