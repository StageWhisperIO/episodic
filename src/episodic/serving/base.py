import itertools
import json
import time
import urllib.request

_COUNTER = itertools.count(1)


def new_id(prefix="chatcmpl"):
    return f"{prefix}-{next(_COUNTER):08x}"


def chat_completion_response(model, content, finish_reason="stop", role="assistant", usage=None):
    return {
        "id": new_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": role, "content": content},
            "finish_reason": finish_reason,
        }],
        "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def chat_completion_chunk(model, delta, finish_reason=None, chunk_id=None):
    return {
        "id": chunk_id or new_id(),
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def sse_frame(obj):
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def sse_done():
    return "data: [DONE]\n\n"


class HTTPBackend:
    name = None
    default_base_url = None

    def __init__(self, base_url=None, model=None, api_key=None, headers=None, timeout=60, opener=None):
        resolved = base_url or self.default_base_url
        if not resolved:
            raise ValueError(f"{self.name}: base_url is required")
        self.base_url = resolved.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.headers = dict(headers or {})
        self.timeout = timeout
        self._opener = opener or urllib.request.urlopen

    @classmethod
    def from_config(cls, config):
        config = dict(config or {})
        return cls(
            base_url=config.get("base_url"),
            model=config.get("model"),
            api_key=config.get("api_key"),
            headers=config.get("headers"),
            timeout=config.get("timeout", 60),
            opener=config.get("opener"),
        )

    def _auth_headers(self):
        headers = {"Content-Type": "application/json"}
        headers.update(self.headers)
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post(self, path, payload):
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path, data=body, headers=self._auth_headers(), method="POST",
        )
        return self._opener(request, timeout=self.timeout)

    def _get(self, path):
        request = urllib.request.Request(self.base_url + path, headers=self._auth_headers(), method="GET")
        return self._opener(request, timeout=self.timeout)

    def _read_json(self, response):
        return json.loads(response.read().decode("utf-8"))

    def chat_completions(self, payload, stream=False):
        payload = dict(payload)
        payload.setdefault("model", self.model)
        payload["stream"] = stream
        response = self._post("/v1/chat/completions", payload)
        if stream:
            return self._passthrough_stream(response)
        return self._read_json(response)

    def _passthrough_stream(self, response):
        for raw_line in response:
            yield raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line

    def models(self):
        response = self._get("/v1/models")
        data = self._read_json(response)
        return [item.get("id") for item in data.get("data", [])]
