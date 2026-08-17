import json
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import BackendUnavailable
from .router import Router


class Handler(BaseHTTPRequestHandler):
    config = {}
    start = None

    def log_message(self, format, *args):
        pass

    def _router(self):
        return Router(self.config, start=self.start)

    def _send_json(self, data, code=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code, message, error_type="error"):
        self._send_json({"error": {"message": message, "type": error_type}}, code)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/v1/models":
            self._send_json({"object": "list", "data": self._router().list_models()})
        else:
            self._error(404, "not found", "not_found")

    def do_POST(self):
        path = self.path.split("?")[0]
        if path != "/v1/chat/completions":
            self._error(404, "not found", "not_found")
            return

        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw) if raw else {}
        except ValueError:
            self._error(400, "invalid JSON body", "invalid_request_error")
            return

        try:
            tier, backend, _tier_config = self._router().route(payload)
            if bool(payload.get("stream")):
                self._stream_response(backend, payload, tier)
            else:
                result = backend.chat_completions(payload, stream=False)
                result.setdefault("episodic_tier", tier)
                self._send_json(result)
        except BackendUnavailable as exc:
            self._error(503, exc.hint, "backend_unavailable")
        except (KeyError, ValueError) as exc:
            self._error(400, str(exc), "invalid_request_error")
        except urllib.error.URLError as exc:
            self._error(502, f"upstream request failed: {exc}", "upstream_error")
        except Exception as exc:
            self._error(502, str(exc), "upstream_error")

    def _stream_response(self, backend, payload, tier):
        iterator = iter(backend.chat_completions(payload, stream=True))
        first = next(iterator, None)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        if first is not None:
            self._write_chunk(first)
        try:
            for chunk in iterator:
                self._write_chunk(chunk)
        except Exception:
            pass

    def _write_chunk(self, chunk):
        data = chunk.encode("utf-8") if isinstance(chunk, str) else chunk
        self.wfile.write(data)
        self.wfile.flush()


def _bind(config, start):
    class BoundHandler(Handler):
        pass

    BoundHandler.config = dict(config or {})
    BoundHandler.start = start
    return BoundHandler


def build_server(host="127.0.0.1", port=8000, config=None, start=None):
    return ThreadingHTTPServer((host, port), _bind(config, start))


def serve(host="127.0.0.1", port=8000, config=None, start=None):
    print(f"Episodic serve: http://{host}:{port}/v1")
    server = build_server(host, port, config, start)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
