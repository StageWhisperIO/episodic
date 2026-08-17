import json

from . import register
from .base import HTTPBackend, chat_completion_response, chat_completion_chunk, new_id, sse_frame, sse_done


class OllamaBackend(HTTPBackend):
    name = "ollama"
    default_base_url = "http://localhost:11434"

    def chat_completions(self, payload, stream=False):
        model = payload.get("model") or self.model
        messages = payload.get("messages") or []
        body = {"model": model, "messages": messages, "stream": stream}
        response = self._post("/api/chat", body)
        if stream:
            return self._stream(response, model)
        data = self._read_json(response)
        content = (data.get("message") or {}).get("content", "")
        return chat_completion_response(model, content)

    def _stream(self, response, model):
        chunk_id = new_id()
        for raw_line in response:
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            content = (data.get("message") or {}).get("content", "")
            if content:
                yield sse_frame(chat_completion_chunk(model, {"content": content}, chunk_id=chunk_id))
            if data.get("done"):
                yield sse_frame(chat_completion_chunk(model, {}, finish_reason="stop", chunk_id=chunk_id))
                yield sse_done()

    def models(self):
        response = self._get("/api/tags")
        data = self._read_json(response)
        return [item.get("name") for item in data.get("models", [])]


register(OllamaBackend)
