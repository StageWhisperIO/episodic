import json
import threading
import urllib.error
import urllib.request

import pytest

from episodic.serving.server import build_server


class FakeUpstreamResponse:
    def __init__(self, json_body=None, lines=None):
        self._json_body = json_body
        self._lines = lines or []

    def read(self):
        return json.dumps(self._json_body).encode("utf-8")

    def __iter__(self):
        return iter(self._lines)


def _fake_opener(json_body=None, lines=None):
    def opener(request, timeout=None):
        return FakeUpstreamResponse(json_body=json_body, lines=lines)
    return opener


@pytest.fixture
def running_server():
    servers = []

    def start(config):
        server = build_server("127.0.0.1", 0, config)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append((server, thread))
        host, port = server.server_address
        return f"http://{host}:{port}"

    yield start

    for server, thread in servers:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _post(base_url, path, body):
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(base_url + path, data=data, headers={"Content-Type": "application/json"}, method="POST")
    return urllib.request.urlopen(request)


def _get(base_url, path):
    return urllib.request.urlopen(base_url + path)


def test_chat_completions_non_streaming_round_trip(running_server):
    upstream_body = {
        "id": "chatcmpl-1", "object": "chat.completion", "model": "distilled-1",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    }
    config = {"distilled": {"backend": "openai", "base_url": "http://upstream.local",
                             "model": "distilled-1", "opener": _fake_opener(json_body=upstream_body)}}
    base_url = running_server(config)

    response = _post(base_url, "/v1/chat/completions", {"messages": [{"role": "user", "content": "hey"}]})
    assert response.status == 200
    data = json.loads(response.read())
    assert data["choices"][0]["message"]["content"] == "hi"
    assert data["episodic_tier"] == "distilled"


def test_chat_completions_streaming_round_trip(running_server):
    lines = [
        b'data: {"choices":[{"delta":{"content":"He"}}]}\n',
        b'\n',
        b'data: {"choices":[{"delta":{"content":"llo"}}]}\n',
        b'\n',
        b'data: [DONE]\n',
    ]
    config = {"distilled": {"backend": "openai", "base_url": "http://upstream.local",
                             "opener": _fake_opener(lines=lines)}}
    base_url = running_server(config)

    request = urllib.request.Request(
        base_url + "/v1/chat/completions",
        data=json.dumps({"messages": [], "stream": True}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    response = urllib.request.urlopen(request)
    assert response.status == 200
    assert response.headers.get("Content-Type") == "text/event-stream"
    body = response.read().decode("utf-8")
    assert body == b"".join(lines).decode("utf-8")
    assert "data: [DONE]" in body


def test_chat_completions_routes_frontier_tier_on_explicit_model(running_server):
    distilled_body = {"choices": [{"index": 0, "message": {"role": "assistant", "content": "distilled"}, "finish_reason": "stop"}]}
    frontier_body = {"choices": [{"index": 0, "message": {"role": "assistant", "content": "frontier"}, "finish_reason": "stop"}]}
    config = {
        "distilled": {"backend": "openai", "base_url": "http://d.local", "opener": _fake_opener(json_body=distilled_body)},
        "frontier": {"backend": "openai", "base_url": "http://f.local", "opener": _fake_opener(json_body=frontier_body)},
    }
    base_url = running_server(config)

    response = _post(base_url, "/v1/chat/completions", {"model": "frontier", "messages": []})
    data = json.loads(response.read())
    assert data["choices"][0]["message"]["content"] == "frontier"
    assert data["episodic_tier"] == "frontier"


def test_models_endpoint_lists_configured_tiers(running_server):
    config = {"distilled": {"model": "distilled-1"}, "frontier": {"model": "gpt-4o-mini"}}
    base_url = running_server(config)

    response = _get(base_url, "/v1/models")
    data = json.loads(response.read())
    assert data["object"] == "list"
    ids = {entry["id"] for entry in data["data"]}
    assert ids == {"distilled-1", "gpt-4o-mini"}


def test_unknown_path_returns_404(running_server):
    base_url = running_server({})
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _get(base_url, "/nope")
    assert excinfo.value.code == 404


def test_invalid_json_body_returns_400(running_server):
    base_url = running_server({"distilled": {"base_url": "http://d.local"}})
    request = urllib.request.Request(
        base_url + "/v1/chat/completions", data=b"not-json",
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(request)
    assert excinfo.value.code == 400
    body = json.loads(excinfo.value.read())
    assert body["error"]["type"] == "invalid_request_error"


def test_unknown_backend_name_returns_400(running_server):
    base_url = running_server({"distilled": {"backend": "does-not-exist"}})
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _post(base_url, "/v1/chat/completions", {"messages": []})
    assert excinfo.value.code == 400


def test_backend_construction_error_returns_400(running_server):
    base_url = running_server({"distilled": {"backend": "tinker"}})
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _post(base_url, "/v1/chat/completions", {"messages": []})
    assert excinfo.value.code == 400
    body = json.loads(excinfo.value.read())
    assert body["error"]["type"] == "invalid_request_error"


def test_tinker_backend_unavailable_returns_503(running_server, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "tinker", None)
    base_url = running_server({"distilled": {"backend": "tinker", "model": "Qwen/Qwen3.5-4B"}})
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _post(base_url, "/v1/chat/completions", {"messages": []})
    assert excinfo.value.code == 503
    body = json.loads(excinfo.value.read())
    assert body["error"]["type"] == "backend_unavailable"


def test_escalation_to_unconfigured_frontier_returns_503(running_server):
    config = {"distilled": {"backend": "openai", "base_url": "http://d.local",
                            "opener": _fake_opener(json_body={"choices": []})}}
    base_url = running_server(config)
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _post(base_url, "/v1/chat/completions", {"episodic_escalate": True, "messages": []})
    assert excinfo.value.code == 503
    body = json.loads(excinfo.value.read())
    assert body["error"]["type"] == "backend_unavailable"


def test_upstream_connection_error_returns_502(running_server):
    def failing_opener(request, timeout=None):
        raise urllib.error.URLError("connection refused")

    config = {"distilled": {"backend": "openai", "base_url": "http://d.local", "opener": failing_opener}}
    base_url = running_server(config)
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _post(base_url, "/v1/chat/completions", {"messages": []})
    assert excinfo.value.code == 502
