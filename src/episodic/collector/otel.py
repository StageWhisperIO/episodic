import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .. import store


def attr_value(obj):
    if "stringValue" in obj:
        return obj["stringValue"]
    if "intValue" in obj:
        return int(obj["intValue"])
    if "doubleValue" in obj:
        return float(obj["doubleValue"])
    if "boolValue" in obj:
        return bool(obj["boolValue"])
    return None


def attrs_to_dict(attributes):
    return {a["key"]: attr_value(a["value"]) for a in (attributes or [])}


def iter_data_points(metric):
    for kind in ("sum", "gauge", "histogram"):
        container = metric.get(kind, {})
        if container:
            yield from container.get("dataPoints", [])
            return


def point_number(dp):
    if "asInt" in dp:
        return float(int(dp["asInt"]))
    return float(dp.get("asDouble", 0))


def parse_metrics(otlp_json):
    records = []
    body = otlp_json if isinstance(otlp_json, dict) else {}
    for resource_metrics in body.get("resourceMetrics", []):
        resource_attrs = attrs_to_dict(
            resource_metrics.get("resource", {}).get("attributes", [])
        )
        for scope_metrics in resource_metrics.get("scopeMetrics", []):
            for metric in scope_metrics.get("metrics", []):
                name = metric.get("name", "")
                if name not in ("claude_code.token.usage", "claude_code.cost.usage"):
                    continue
                for dp in iter_data_points(metric):
                    dp_attrs = attrs_to_dict(dp.get("attributes", []))
                    session_id = (
                        dp_attrs.get("session.id")
                        or resource_attrs.get("session.id")
                        or None
                    )
                    records.append({
                        "session_id": session_id,
                        "name": name,
                        "type": dp_attrs.get("type"),
                        "value": point_number(dp),
                    })
    return records


def parse_logs(otlp_json):
    records = []
    body = otlp_json if isinstance(otlp_json, dict) else {}
    for resource_logs in body.get("resourceLogs", []):
        resource_attrs = attrs_to_dict(
            resource_logs.get("resource", {}).get("attributes", [])
        )
        for scope_logs in resource_logs.get("scopeLogs", []):
            for log_record in scope_logs.get("logRecords", []):
                dp_attrs = attrs_to_dict(log_record.get("attributes", []))
                session_id = (
                    dp_attrs.get("session.id")
                    or resource_attrs.get("session.id")
                    or None
                )
                body_val = log_record.get("body", {})
                if isinstance(body_val, dict):
                    body_val = attr_value(body_val) if body_val else None
                records.append({
                    "session_id": session_id,
                    "body": body_val,
                    "attributes": dp_attrs,
                })
    return records


def aggregate_usage(records):
    usage = {}
    for rec in records:
        sid = rec["session_id"]
        if sid not in usage:
            usage[sid] = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
        name = rec["name"]
        value = rec["value"]
        if name == "claude_code.token.usage":
            if rec.get("type") == "input":
                usage[sid]["input_tokens"] += int(value)
            elif rec.get("type") == "output":
                usage[sid]["output_tokens"] += int(value)
        elif name == "claude_code.cost.usage":
            usage[sid]["cost_usd"] += value
    return usage


def apply_usage_to_session(records, start=None):
    aggregated = aggregate_usage(records)
    applied = {}
    for session_id, new_usage in aggregated.items():
        resolved = session_id if session_id is not None else store.get_current(start)
        if resolved is None:
            continue
        existing = store.read_meta(resolved, start).get("usage", {})
        merged = {
            "input_tokens": existing.get("input_tokens", 0) + new_usage["input_tokens"],
            "output_tokens": existing.get("output_tokens", 0) + new_usage["output_tokens"],
            "cost_usd": existing.get("cost_usd", 0.0) + new_usage["cost_usd"],
        }
        store.update_meta(resolved, {"usage": merged}, start)
        applied[resolved] = merged
    return applied


class _OtelHandler(BaseHTTPRequestHandler):
    _start = None

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw) if raw else {}
        except Exception:
            body = {}

        if self.path == "/v1/metrics":
            try:
                records = parse_metrics(body)
                apply_usage_to_session(records, self._start)
            except Exception:
                pass
            self._respond(200, {})
        elif self.path == "/v1/logs":
            try:
                parse_logs(body)
            except Exception:
                pass
            self._respond(200, {})
        else:
            self._respond(404, {})

    def _respond(self, code, data):
        payload = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def serve_otel(host="127.0.0.1", port=4318, start=None):
    handler = type("Handler", (_OtelHandler,), {"_start": start})
    server = ThreadingHTTPServer((host, port), handler)
    print(f"OTel receiver on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 4318
    serve_otel(host, port)
